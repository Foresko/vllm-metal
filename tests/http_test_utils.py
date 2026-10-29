# SPDX-License-Identifier: Apache-2.0
"""Loopback HTTP server for smoke-tool tests."""

import socketserver
from http.server import ThreadingHTTPServer


class LoopbackHTTPServer(ThreadingHTTPServer):
    """``ThreadingHTTPServer`` without the reverse-DNS lookup on bind.

    ``HTTPServer.server_bind()`` resolves ``server_name`` with
    ``socket.getfqdn()``, which stalls ~35s on hosted macOS runners. The
    smoke-tool tests only need the port.
    """

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]
