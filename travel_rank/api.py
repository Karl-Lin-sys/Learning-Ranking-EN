"""Local demonstration API. Standard-library server is NOT production hosting."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from .demo import search


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """O(1) health response."""
        if self.path == "/health":
            self.respond(200, {"status": "ok", "synthetic": True})
        else:
            self.respond(404, {"error": "not found"})

    def do_POST(self):
        """O(S+retrieval) time/space; S <= 4096 request bytes."""
        if self.path != "/search":
            self.respond(404, {"error": "not found"})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("chunked requests unsupported")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 4096:
                raise ValueError("request must be 1..4096 bytes")
            self.connection.settimeout(3)
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("incomplete body")
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) - {"query", "vector", "destination", "budget", "k"}:
                raise ValueError("invalid request fields")
            if not isinstance(payload.get("vector"), list) or len(payload["vector"]) != 3:
                raise ValueError("vector must be a 3-number array")
            if "k" in payload and (type(payload["k"]) is not int):
                raise ValueError("k must be an integer")
            result = search(**payload)
            self.respond(200, result)
        except (ValueError, TypeError, KeyError, OverflowError, TimeoutError) as exc:
            self.respond(400, {"error": "invalid search request"})

    def respond(self, status, body):
        """O(S) serialization time/space."""
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        """O(1); omit query/IP logs in the local demo."""
        return


if __name__ == "__main__":
    print("Synthetic API: http://127.0.0.1:8000 (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
