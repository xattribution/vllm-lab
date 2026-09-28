"""Open WebUI admin API stand-in: sign-in, OpenAI connection config, model list."""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakelib as F  # noqa: E402

args = sys.argv[1:]
port = int(args[args.index("--port") + 1]) if "--port" in args else 58110
CFG = F.SIM / "webui.json"
TOKEN = "tok-admin-123"


def load():
    try:
        return json.loads(CFG.read_text())
    except Exception:
        return {"ENABLE_OPENAI_API": True, "OPENAI_API_BASE_URLS": ["https://api.openai.com/v1"], "OPENAI_API_KEYS": ["sk-user-own"],
                "OPENAI_API_CONFIGS": {"0": {"enable": True}}}


def resolve(url):
    m = re.match(r"http://([a-z0-9_.-]+):(\d+)(/.*)", url)
    if not m:
        return url
    try:
        st = json.loads(F.STATE.read_text())
    except Exception:
        return url
    for c in st.get("containers", {}).values():
        if c["name"] == m.group(1) and c.get("host_port"):
            return f"http://127.0.0.1:{c['host_port']}{m.group(3)}"
    return url


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def js(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def authed(self):
        a = self.headers.get("Authorization") or ""
        return a in (f"Bearer {TOKEN}", "Bearer sk-webui-key")

    def do_GET(self):
        if self.path in ("/", "/health"):
            if self.path == "/health":
                self.js({"status": True})
                return
            raw = b"<!doctype html><title>Open WebUI</title>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        if not self.authed():
            self.js({"detail": "Not authenticated"}, 401)
            return
        if self.path == "/openai/config":
            self.js(load())
            return
        if self.path == "/api/models":
            data = []
            cfg = load()
            for u in cfg["OPENAI_API_BASE_URLS"]:
                if "api.openai.com" in u:
                    continue
                try:
                    with urlopen(resolve(u.rstrip("/")) + "/models", timeout=2) as r:
                        for m in json.loads(r.read()).get("data", []):
                            data.append({"id": m["id"], "name": m["id"], "owned_by": "openai"})
                except Exception:
                    pass
            self.js({"data": data})
            return
        self.js({"detail": "Not Found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/api/v1/auths/signin":
            if body.get("email") == "admin@lab.local" and body.get("password") == "hunter2":
                self.js({"token": TOKEN, "token_type": "Bearer", "role": "admin"})
            else:
                self.js({"detail": "The email or password provided is incorrect."}, 400)
            return
        if not self.authed():
            self.js({"detail": "Not authenticated"}, 401)
            return
        if self.path == "/openai/config/update":
            cfg = {k: body[k] for k in ("ENABLE_OPENAI_API", "OPENAI_API_BASE_URLS", "OPENAI_API_KEYS", "OPENAI_API_CONFIGS") if k in body}
            CFG.write_text(json.dumps(cfg, indent=1))
            self.js(cfg)
            return
        self.js({"detail": "Not Found"}, 404)


ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
