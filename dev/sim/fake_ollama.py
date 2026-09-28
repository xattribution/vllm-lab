"""Ollama API stand-in on :11434."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODELS = {"llama3.2:3b": 2_019_393_189, "gemma3:12b": 8_149_190_253}
LOADED = {}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def js(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/version":
            self.js({"version": "0.12.3"})
        elif self.path == "/api/tags":
            self.js({"models": [{"name": k, "size": v, "modified_at": "2026-09-01T00:00:00Z", "details": {"family": k.split(":")[0], "parameter_size": k.split(":")[1].upper(), "quantization_level": "Q4_K_M"}} for k, v in MODELS.items()]})
        elif self.path == "/api/ps":
            self.js({"models": [{"name": k, "size": MODELS.get(k, 0), "size_vram": MODELS.get(k, 0), "expires_at": "2026-09-28T00:00:00Z", "context_length": 4096} for k in LOADED]})
        elif self.path == "/v1/models":
            self.js({"object": "list", "data": [{"id": k, "object": "model"} for k in MODELS]})
        else:
            self.js({"error": "not found"}, 404)

    def do_DELETE(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        MODELS.pop(body.get("name"), None)
        LOADED.pop(body.get("name"), None)
        self.js({})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/pull":
            name = body.get("name") or body.get("model")
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def w(o):
                d = (json.dumps(o) + "\n").encode()
                self.wfile.write(f"{len(d):X}\r\n".encode() + d + b"\r\n")
                self.wfile.flush()
            if "nope" in name:
                w({"error": "pull model manifest: file does not exist"})
            else:
                for i in range(0, 101, 20):
                    w({"status": "pulling sha256:abc", "total": 1000, "completed": i * 10})
                    time.sleep(0.2)
                w({"status": "success"})
                MODELS[name if ":" in name else name + ":latest"] = 1_500_000_000
            self.wfile.write(b"0\r\n\r\n")
            return
        if self.path == "/api/generate":
            m = body.get("model")
            if body.get("keep_alive") == 0:
                LOADED.pop(m, None)
            else:
                LOADED[m] = time.time()
            self.js({"model": m, "done": True})
            return
        if self.path == "/v1/chat/completions":
            self.js({"id": "x", "object": "chat.completion", "model": body.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello from Ollama"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}})
            return
        self.js({"error": "not found"}, 404)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 11434), H).serve_forever()
