// SPDX-License-Identifier: Apache-2.0
// Hardware facts shared by the MLX and PyTorch launchers.
// Implementations are in metal_device.cpp.
#pragma once

namespace vllm_metal::hardware {

int detected_gpu_core_count();
int gpu_core_count();
void override_detected_gpu_core_count_for_test(int cores);

// MTLGPUFamilyApple10 (M5/A19). Keep the numeric value usable with older SDKs.
constexpr int kApple10 = 1010;

bool nax_supported(bool supports_apple10);

}  // namespace vllm_metal::hardware
