# SPDX-License-Identifier: Apache-2.0

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from tools import stt_sampling_serve_smoke as smoke


@pytest.fixture
def healthy_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_main_rejects_occupied_port(healthy_server, monkeypatch, capsys):
    monkeypatch.setattr(
        smoke.sys,
        "argv",
        [
            "stt-smoke",
            "--audio",
            "unused.wav",
            "--port",
            str(healthy_server.server_port),
        ],
    )

    def unexpected_start(*args, **kwargs):
        pytest.fail(
            "The smoke script must reject an occupied port before starting vLLM"
        )

    monkeypatch.setattr(smoke.subprocess, "Popen", unexpected_start)
    assert smoke.main() == 1
    assert "cannot bind localhost port" in capsys.readouterr().err


@pytest.mark.parametrize("exit_codes", [(7,), (None, 7)])
def test_health_response_cannot_hide_server_exit(healthy_server, exit_codes):
    codes = iter(exit_codes)
    serve = SimpleNamespace(poll=lambda: next(codes))
    base_url = f"http://127.0.0.1:{healthy_server.server_port}"
    assert not smoke._wait_for_health(base_url, 2, serve)


def test_health_wait_accepts_live_server(healthy_server):
    serve = SimpleNamespace(poll=lambda: None)
    base_url = f"http://127.0.0.1:{healthy_server.server_port}"
    assert smoke._wait_for_health(base_url, 2, serve)
