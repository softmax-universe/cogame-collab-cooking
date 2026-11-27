"""Deterministic local HTTP model fixture for CI docker-smoke.

Serves Anthropic-shaped POST /v1/messages responses with valid Collab Cooking
shift plans so CI exercises the real prompt seat without paid network calls.
"""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PLANS = (
    '{"station":"chop","recipe":"soup","zone":"any","handoff":"none","yield_to":"none","say":"Chopping veg for soup","note":"keep pot fed"}',
    '{"station":"pot","recipe":"soup","zone":"any","handoff":"none","yield_to":"none","say":"Cooking soup at the pot","note":"plate when ready"}',
)


def main() -> None:
    port_file = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    calls = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            idx = calls["count"] % len(PLANS)
            calls["count"] += 1
            body = json.dumps(
                {
                    "id": f"msg-fixture-{calls['count']}",
                    "model": "fixture/collab_cooking",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 16, "output_tokens": 24},
                    "content": [{"type": "text", "text": PLANS[idx]}],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    if port_file is not None:
        port_file.write_text(str(server.server_port), encoding="utf-8")
    print(f"PORT={server.server_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
