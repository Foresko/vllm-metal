# SPDX-License-Identifier: Apache-2.0
"""TieringOffloadingSpec with the Metal shared region and worker.

Upstream's region is a ``/dev/shm`` file and its worker copies with a
CUDA-only engine. Everything else, including the scheduler-side manager and
host-pool sizing, is inherited.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, override

import vllm.v1.kv_offload.tiering.spec as _tiering_spec_module
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

if TYPE_CHECKING:
    from vllm_metal.attention.caches.storage import KVCacheStorage
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
    from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker


def route_fs_tiers_to_metal(extra_config: dict[str, Any]) -> None:
    """Point ``fs`` tier configs at the Metal subclass, in place.

    Uses upstream's ``module_path`` hook in ``SecondaryTierFactory``.
    """
    for tier in extra_config.get("secondary_tiers") or []:
        if isinstance(tier, dict) and tier.get("type") == "fs":
            tier["type"] = "MetalFileSystemTierManager"
            tier["module_path"] = "vllm_metal.v1.kv_offload.fs_tier"


@contextmanager
def _metal_shared_region() -> Iterator[None]:
    """Swap in MetalSharedOffloadRegion while upstream's get_manager runs.

    Not re-entrant. The platform requires the uni executor for offloading, so
    only one engine runs per process.
    """
    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

    original_region = _tiering_spec_module.SharedOffloadRegion
    _tiering_spec_module.SharedOffloadRegion = MetalSharedOffloadRegion
    try:
        yield
    finally:
        _tiering_spec_module.SharedOffloadRegion = original_region


class MetalTieringOffloadingSpec(TieringOffloadingSpec):
    """Host pool plus optional ``fs`` disk tiers, on Metal."""

    _metal_worker: MetalKVOffloadWorker | None = None

    def __init__(self, config: OffloadingConfig):
        route_fs_tiers_to_metal(config.extra_config)
        super().__init__(config)

    def _make_region(self, rank: int | None) -> MetalSharedOffloadRegion:
        from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

        return MetalSharedOffloadRegion(
            engine_id=self.config.engine_id,
            num_chunks=self.num_chunks,
            rank=rank,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
        )

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        raise NotImplementedError(
            "Metal builds its worker from KVCacheStorage via "
            "MetalOffloadingConnector.register_kv_caches"
        )

    def get_metal_worker(self, storage: KVCacheStorage) -> MetalKVOffloadWorker:
        if self._metal_worker is None:
            # Imported here so the scheduler side never imports mlx.
            from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

            self._metal_worker = MetalKVOffloadWorker(
                storage,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                # One process, one worker rank (uni executor).
                region=self._make_region(rank=0),
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            )
        elif self._metal_worker.storage is not storage:
            raise RuntimeError("KV offloading worker is bound to another KV storage")
        return self._metal_worker

    @override
    def get_manager(self) -> OffloadingManager:
        """Upstream body with the Metal shared region."""
        with _metal_shared_region():
            return super().get_manager()

    @classmethod
    @override
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        """Resolve tier metrics against the Metal fs tier.

        Upstream resolves tier classes here, outside ``get_manager``, so
        without this the Metal tier's metrics would not be registered.
        """
        route_fs_tiers_to_metal(extra_config)
        return super().build_metric_definitions(extra_config)
