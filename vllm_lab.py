#!/usr/bin/env python3
"""vllm-lab — a control plane for local AI on NVIDIA DGX / GB10 systems.

One file, Python standard library only. Runs as:
  * a CLI           vllm-lab up lightning | vllm-lab logs gemma4 -f | vllm-lab doctor
  * a web console   vllm-lab ui            (http://127.0.0.1:58120)
  * a gateway       one OpenAI-compatible /v1 endpoint that routes by model name
                    across every host, wakes sleeping engines on demand, and puts
                    idle ones back to sleep.

Engines are Docker containers (vLLM or llama.cpp) plus any Ollama runtime found on
a host. Hosts are local or reached over SSH. Nothing here needs pip.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import hmac
import http.client
import json
import math
import os
import queue
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import Request, urlopen

VERSION = "2.0.0"
HOME = Path.home()
CONF_DIR = Path(os.environ.get("VLLM_LAB_HOME", str(HOME / ".config/vllm-lab")))
CONF_FILE = CONF_DIR / "config.json"
ENGINES_FILE = CONF_DIR / "engines.json"
BLUEPRINTS_FILE = CONF_DIR / "blueprints.json"
EVENTS_FILE = CONF_DIR / "events.jsonl"
BENCH_FILE = CONF_DIR / "bench.jsonl"
HFCACHE_FILE = CONF_DIR / "hf-meta.json"
LOCK_FILE = CONF_DIR / ".lock"
PROC_ROOT = os.environ.get("VLLM_LAB_PROC", "/proc")
GIB = 1024**3

LEGACY_IMAGE = os.environ.get("VLLM_IMAGE", "vllm/vllm-openai:v0.27.1")
PATCHED_IMAGE = "vllm-lab/vllm-openai:0.27.1-tf5141"

# Weights from these orgs are blocked (or flagged) by the origin policy. Derivatives
# are caught through the Hub's base_model tags. Editable in Settings.
RESTRICTED_ORGS = [
    "Qwen", "deepseek-ai", "THUDM", "zai-org", "moonshotai", "MiniMaxAI", "01-ai",
    "baichuan-inc", "internlm", "tencent", "baidu", "BAAI", "stepfun-ai", "OpenGVLab",
    "XiaomiMiMo", "ByteDance-Seed", "ByteDance", "inclusionAI", "Alibaba-NLP",
    "openbmb", "meituan-longcat", "Skywork", "iFlytek", "SenseTime", "ZhipuAI",
    "Tencent-Hunyuan", "PaddlePaddle", "AIDC-AI", "Kwai-Kolors", "kwaipilot",
]

DEFAULT_HOSTS = [
    {
        "id": "titan", "label": "Titan", "ssh": "", "ssh_port": None, "bind": "127.0.0.1",
        "human_host": "127.0.0.1", "network": "titan-ai", "enabled": True, "cache": "",
        "reach": "auto", "ports": "", "note": "Fast models. Lightning lives here.",
    },
    {
        "id": "atlas", "label": "Atlas", "ssh": "titan@atlas", "ssh_port": None, "bind": "127.0.0.1",
        "human_host": "atlas", "network": "titan-ai", "enabled": False, "cache": "",
        "reach": "auto", "ports": "", "note": "Long-context / Super.",
    },
]

DEFAULT_CONF: dict = {
    "version": 2,
    "hf_token": "",
    "hf_endpoint": "",
    "vllm_image": LEGACY_IMAGE,
    "llamacpp_image": "ghcr.io/ggml-org/llama.cpp:server-cuda13",
    "container_prefix": "titan-",
    "port_pool": "58100,58102-58109,58112-58119",
    "port_reserved": "58101,58110,58120,58130",
    "headroom_gib": 4.0,
    "auto_flush_cache": True,
    "crash_guard": 3,
    "webui": {
        "url": "http://127.0.0.1:58110", "email": "", "password": "", "api_key": "",
        "container": "titan-webui", "host": "titan", "auto_sync": True, "prune_stopped": True,
        "mode": "direct",
    },
    "gateway": {"enabled": True, "listen": "", "port": 58130, "key": "", "autowake": True, "wake_timeout": 900},
    "policy": {"mode": "block", "orgs": list(RESTRICTED_ORGS)},
    "ui": {"listen": "127.0.0.1", "port": int(os.environ.get("VLLM_LAB_UI_PORT", "58120")), "token": "", "allowed_hosts": []},
    "hosts": DEFAULT_HOSTS,
}

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")
MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(/[A-Za-z0-9_.\-]+)?(:[A-Za-z0-9_.\-]+)?$")
HOST_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")
CONTAINER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
IMAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:@\-]{0,255}$")


class LabError(Exception):
    """An error with a human explanation. `fixes` are action ids the UI can offer."""

    def __init__(self, msg: str, hint: str = "", fixes: list | None = None, code: int = 400, data: dict | None = None):
        super().__init__(msg)
        self.msg = msg
        self.hint = hint
        self.fixes = fixes or []
        self.code = code
        self.data = data or {}

    def as_dict(self) -> dict:
        return {"error": self.msg, "hint": self.hint, "fixes": self.fixes, **({"data": self.data} if self.data else {})}


# ───────────────────────────────────────────────────────────────── small utilities

def now() -> float:
    return time.time()


def fmt_bytes(n) -> str:
    if n is None or n == "":
        return "–"
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(x) < 1024 or u == "TiB":
            return f"{x:.0f} {u}" if u in ("B", "KiB") else f"{x:.1f} {u}"
        x /= 1024
    return "–"


def fmt_params(n) -> str:
    if not n:
        return "–"
    n = float(n)
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return str(int(n))


def fmt_dur(sec) -> str:
    if sec is None or sec < 0:
        return "–"
    sec = int(sec)
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec//60}m {sec%60:02d}s"
    if sec < 86400:
        return f"{sec//3600}h {(sec%3600)//60:02d}m"
    return f"{sec//86400}d {(sec%86400)//3600}h"


def parse_ports(spec: str) -> list[int]:
    out: list[int] = []
    for part in re.split(r"[,\s]+", str(spec or "").strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo, hi = int(a), int(b)
            except ValueError:
                continue
            out.extend(range(lo, min(hi, lo + 500) + 1))
        else:
            with contextlib.suppress(ValueError):
                out.append(int(part))
    return [p for p in out if 1024 < p < 65536]


def deep_merge(base: dict, over: dict) -> dict:
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def as_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on", "y")


def as_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def as_int(v, default=None):
    try:
        if v is None or v == "":
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def atomic_write(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
    with contextlib.suppress(OSError):
        os.chmod(path, mode)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError):
        return default


def short_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def model_slug(model: str) -> str:
    base = (model or "").split("/")[-1].split(":")[0].lower()
    base = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    for junk in ("-instruct", "-it", "-chat", "-hf", "-nvfp4", "-fp8", "-awq", "-gptq", "-gguf", "-bf16"):
        if base.endswith(junk) and len(base) > len(junk) + 2:
            base = base[: -len(junk)]
    return (base or "engine")[:28].strip("-") or "engine"


def model_org(model: str) -> str:
    return model.split("/", 1)[0] if "/" in (model or "") else ""


def hf_cache_folder(model: str) -> str:
    return "models--" + model.split(":")[0].replace("/", "--")


# ───────────────────────────────────────────────────────────────── config store

class Store:
    """Config, engines and blueprints on disk. Safe across the UI service and the CLI:
    writes are atomic and serialized with a lock file, reads re-load on mtime change."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._conf = None
        self._conf_m = -1.0
        self._eng = None
        self._eng_m = -1.0
        CONF_DIR.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(CONF_DIR, 0o700)

    @contextlib.contextmanager
    def flock(self):
        with self.lock:
            CONF_DIR.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(LOCK_FILE), os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    @staticmethod
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return 0.0

    # config ------------------------------------------------------------------
    def conf(self) -> dict:
        with self.lock:
            m = self._mtime(CONF_FILE)
            if self._conf is None or m != self._conf_m:
                self._conf = self._load_conf()
                self._conf_m = m
            return json.loads(json.dumps(self._conf))

    def _load_conf(self) -> dict:
        saved = read_json(CONF_FILE, {})
        if not isinstance(saved, dict):
            saved = {}
        migrated = self.migrate_conf(saved)
        conf = deep_merge(DEFAULT_CONF, {k: v for k, v in migrated.items() if k != "hosts"})
        conf["hosts"] = [self.norm_host(h) for h in (migrated.get("hosts") or DEFAULT_HOSTS)]
        return conf

    @staticmethod
    def migrate_conf(saved: dict) -> dict:
        """v1 kept webui_* at the top level and hosts[].online."""
        out = dict(saved)
        webui = dict(out.get("webui") or {})
        for old, new in (("webui_url", "url"), ("webui_email", "email"), ("webui_password", "password")):
            if old in out:
                if out[old] and not webui.get(new):
                    webui[new] = out[old]
                out.pop(old, None)
        if webui:
            out["webui"] = webui
        return out

    @staticmethod
    def norm_host(h: dict) -> dict:
        base = {
            "id": "", "label": "", "ssh": "", "ssh_port": None, "bind": "127.0.0.1", "human_host": "",
            "network": "titan-ai", "enabled": True, "cache": "", "reach": "auto", "ports": "", "note": "",
        }
        out = {**base, **{k: v for k, v in (h or {}).items() if v is not None or k == "ssh_port"}}
        if "online" in out:
            out["enabled"] = as_bool(out.pop("online"), True)
        out["id"] = str(out.get("id") or "").strip()
        out["label"] = out.get("label") or out["id"].title()
        out["human_host"] = out.get("human_host") or ("127.0.0.1" if not out.get("ssh") else out["ssh"].split("@")[-1])
        out["enabled"] = as_bool(out.get("enabled"), True)
        return out

    def save_conf(self, conf: dict) -> None:
        with self.flock():
            clean = json.loads(json.dumps(conf))
            atomic_write(CONF_FILE, json.dumps(clean, indent=2) + "\n", 0o600)
            self._conf = self._load_conf()
            self._conf_m = self._mtime(CONF_FILE)

    def update_conf(self, fn) -> dict:
        with self.flock():
            conf = self._load_conf()
            fn(conf)
            atomic_write(CONF_FILE, json.dumps(conf, indent=2) + "\n", 0o600)
            self._conf = self._load_conf()
            self._conf_m = self._mtime(CONF_FILE)
            return json.loads(json.dumps(self._conf))

    # engines -----------------------------------------------------------------
    def engines(self) -> dict:
        with self.lock:
            m = self._mtime(ENGINES_FILE)
            if self._eng is None or m != self._eng_m:
                self._eng = self._load_engines()
                self._eng_m = self._mtime(ENGINES_FILE)
            return json.loads(json.dumps(self._eng))

    def _load_engines(self) -> dict:
        raw = read_json(ENGINES_FILE, None)
        if isinstance(raw, dict) and raw.get("version") == 2 and isinstance(raw.get("engines"), dict):
            return {k: normalize_spec({**v, "name": k}) for k, v in raw["engines"].items() if NAME_RE.match(k)}
        # First run or v1 file: seed with the built-in lab recipes, then fold in v1 entries.
        seeded: dict = {}
        for name, rec in LEGACY_RECIPES.items():
            seeded[name] = normalize_spec({**rec, "name": name, "desired": "any", "created": now()})
        if isinstance(raw, dict):
            for name, old in raw.items():
                if not isinstance(old, dict) or not NAME_RE.match(str(name)):
                    continue
                if name in seeded:
                    for k in ("host", "port"):
                        if old.get(k):
                            seeded[name][k] = old[k]
                    continue
                spec = {
                    "name": name, "host": old.get("host") or "titan", "backend": old.get("backend") or "vllm",
                    "model": old.get("model") or "", "port": old.get("port"), "quant": old.get("quant") or "",
                    "util": old.get("util") or 0.5, "max_len": old.get("max_len") or 32768,
                    "extra": old.get("extra") or "", "image": old.get("image") or "",
                    "kv_dtype": "fp8", "trust_remote_code": True, "desired": "any", "created": now(),
                }
                if spec["quant"] == "none":
                    spec["quant"] = ""
                seeded[name] = normalize_spec(spec)
        if raw is not None or not ENGINES_FILE.exists():
            with contextlib.suppress(OSError):
                if ENGINES_FILE.exists():
                    shutil.copy2(ENGINES_FILE, ENGINES_FILE.with_suffix(".v1.json"))
                atomic_write(ENGINES_FILE, json.dumps({"version": 2, "engines": seeded}, indent=2) + "\n", 0o600)
        return seeded

    def save_engines(self, engines: dict) -> None:
        with self.flock():
            atomic_write(ENGINES_FILE, json.dumps({"version": 2, "engines": engines}, indent=2) + "\n", 0o600)
            self._eng = {k: normalize_spec({**v, "name": k}) for k, v in engines.items()}
            self._eng_m = self._mtime(ENGINES_FILE)

    def update_engine(self, name: str, fn) -> dict | None:
        with self.flock():
            self._eng = None
            engines = self.engines()
            spec = engines.get(name)
            if spec is None:
                return None
            fn(spec)
            engines[name] = normalize_spec(spec)
            atomic_write(ENGINES_FILE, json.dumps({"version": 2, "engines": engines}, indent=2) + "\n", 0o600)
            self._eng = engines
            self._eng_m = self._mtime(ENGINES_FILE)
            return engines[name]

    def put_engine(self, spec: dict) -> dict:
        with self.flock():
            self._eng = None
            engines = self.engines()
            spec = normalize_spec(spec)
            engines[spec["name"]] = spec
            atomic_write(ENGINES_FILE, json.dumps({"version": 2, "engines": engines}, indent=2) + "\n", 0o600)
            self._eng = engines
            self._eng_m = self._mtime(ENGINES_FILE)
            return spec

    def drop_engine(self, name: str) -> None:
        with self.flock():
            self._eng = None
            engines = self.engines()
            engines.pop(name, None)
            atomic_write(ENGINES_FILE, json.dumps({"version": 2, "engines": engines}, indent=2) + "\n", 0o600)
            self._eng = engines
            self._eng_m = self._mtime(ENGINES_FILE)

    # blueprints --------------------------------------------------------------
    def user_blueprints(self) -> dict:
        raw = read_json(BLUEPRINTS_FILE, {})
        return raw if isinstance(raw, dict) else {}

    def save_blueprints(self, bps: dict) -> None:
        with self.flock():
            atomic_write(BLUEPRINTS_FILE, json.dumps(bps, indent=2) + "\n", 0o600)


# ───────────────────────────────────────────────────────────────── live bus, events, jobs

class Bus:
    """Fan-out of live updates to every open console (Server-Sent Events)."""

    def __init__(self) -> None:
        self.subs: set[queue.Queue] = set()
        self.lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=400)
        with self.lock:
            self.subs.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            self.subs.discard(q)

    def publish(self, kind: str, data) -> None:
        msg = (kind, data)
        with self.lock:
            subs = list(self.subs)
        for q in subs:
            try:
                q.put_nowait(msg)
            except queue.Full:
                with contextlib.suppress(queue.Empty):
                    q.get_nowait()
                with contextlib.suppress(queue.Full):
                    q.put_nowait(msg)


class Events:
    """Activity feed. Persisted as JSON lines, trimmed to the last 2000 entries."""

    def __init__(self, bus: Bus) -> None:
        self.bus = bus
        self.lock = threading.Lock()
        self.items: list[dict] = []
        try:
            lines = EVENTS_FILE.read_text().splitlines()[-400:]
            for ln in lines:
                with contextlib.suppress(json.JSONDecodeError):
                    self.items.append(json.loads(ln))
        except OSError:
            pass

    def add(self, level: str, source: str, msg: str, **data) -> dict:
        ev = {"id": uuid.uuid4().hex[:10], "t": now(), "level": level, "source": source, "msg": msg}
        if data:
            ev["data"] = data
        with self.lock:
            self.items.append(ev)
            self.items = self.items[-400:]
            try:
                with open(EVENTS_FILE, "a") as fh:
                    fh.write(json.dumps(ev) + "\n")
                if EVENTS_FILE.stat().st_size > 1_500_000:
                    tail = EVENTS_FILE.read_text().splitlines()[-2000:]
                    atomic_write(EVENTS_FILE, "\n".join(tail) + "\n")
            except OSError:
                pass
        self.bus.publish("event", ev)
        return ev

    def recent(self, n: int = 150) -> list[dict]:
        with self.lock:
            return list(self.items[-n:])


class Cancelled(Exception):
    pass


class Job:
    def __init__(self, jobs: "Jobs", kind: str, title: str, target: str = "") -> None:
        self.jobs = jobs
        self.id = uuid.uuid4().hex[:10]
        self.kind = kind
        self.title = title
        self.target = target
        self.state = "running"
        self.progress: float | None = None
        self.stage = ""
        self.lines: list[list] = []
        self.result = ""
        self.error: dict | None = None
        self.t0 = now()
        self.t1: float | None = None
        self.cancelled = False
        self.data: dict = {}

    def log(self, msg: str) -> None:
        self.lines.append([round(now() - self.t0, 1), str(msg)])
        self.lines = self.lines[-300:]
        self.jobs.push(self)

    def step(self, stage: str, progress: float | None = None) -> None:
        self.stage = stage
        if progress is not None:
            self.progress = max(0.0, min(1.0, progress))
        self.log(stage)

    def set_progress(self, p: float | None, stage: str | None = None, quiet: bool = True) -> None:
        self.progress = None if p is None else max(0.0, min(1.0, p))
        if stage:
            self.stage = stage
        self.jobs.push(self, throttle=quiet)

    def check(self) -> None:
        if self.cancelled:
            raise Cancelled("cancelled")

    def as_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "title": self.title, "target": self.target, "state": self.state,
            "progress": self.progress, "stage": self.stage, "lines": self.lines[-60:], "result": self.result,
            "error": self.error, "t0": self.t0, "t1": self.t1, "data": self.data,
        }


class Jobs:
    def __init__(self, bus: Bus, events: Events) -> None:
        self.bus = bus
        self.events = events
        self.items: dict[str, Job] = {}
        self.lock = threading.Lock()
        self._last_push: dict[str, float] = {}

    def push(self, job: Job, throttle: bool = False) -> None:
        t = now()
        if throttle and t - self._last_push.get(job.id, 0) < 0.4:
            return
        self._last_push[job.id] = t
        self.bus.publish("job", job.as_dict())

    def start(self, kind: str, title: str, target: str, fn, background: bool = True) -> Job:
        job = Job(self, kind, title, target)
        with self.lock:
            self.items[job.id] = job
            if len(self.items) > 60:
                for jid in sorted(self.items, key=lambda k: self.items[k].t0)[:-60]:
                    if self.items[jid].state != "running":
                        self.items.pop(jid, None)
        self.push(job)

        def runner() -> None:
            try:
                res = fn(job)
                job.result = str(res or "done")
                job.state = "ok"
                job.progress = 1.0
            except Cancelled:
                job.state = "cancelled"
                job.result = "cancelled"
            except LabError as e:
                job.state = "error"
                job.error = e.as_dict()
                job.result = e.msg
            except Exception as e:  # noqa: BLE001 — shown to the operator verbatim
                job.state = "error"
                job.error = {"error": f"{type(e).__name__}: {e}", "hint": "", "fixes": [], "trace": traceback.format_exc()[-2000:]}
                job.result = str(e)
            job.t1 = now()
            self.push(job)
            if job.state == "error":
                self.events.add("error", job.target or job.kind, f"{job.title} failed: {job.result}")

        if background:
            threading.Thread(target=runner, daemon=True, name=f"job-{kind}").start()
        else:
            runner()
        return job

    def get(self, jid: str) -> Job | None:
        return self.items.get(jid)

    def list(self) -> list[dict]:
        with self.lock:
            return [j.as_dict() for j in sorted(self.items.values(), key=lambda j: j.t0)][-40:]

    def busy(self, target: str) -> Job | None:
        for j in list(self.items.values()):
            if j.state == "running" and j.target == target:
                return j
        return None


# ───────────────────────────────────────────────────────────────── command execution

@dataclass
class Res:
    code: int
    out: str = ""
    err: str = ""

    @property
    def ok(self) -> bool:
        return self.code == 0

    @property
    def text(self) -> str:
        return ((self.out or "") + ("\n" if self.out and self.err else "") + (self.err or "")).strip()


def run_local(argv: list[str], timeout: float = 30, input_text: str | None = None, env: dict | None = None) -> Res:
    try:
        p = subprocess.run(
            argv, text=True, capture_output=True, timeout=timeout, input=input_text,
            env={**os.environ, **env} if env else None, errors="replace",
        )
        return Res(p.returncode, p.stdout or "", p.stderr or "")
    except FileNotFoundError:
        return Res(127, "", f"{argv[0] if argv else '?'}: command not found")
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return Res(124, out, f"timed out after {timeout:.0f}s: {' '.join(argv[:4])}")
    except OSError as e:
        return Res(126, "", str(e))


def _cm_dir() -> str:
    d = str(CONF_DIR)
    if len(d) > 60:
        d = f"/tmp/vllm-lab-{os.getuid()}"
    os.makedirs(d, mode=0o700, exist_ok=True)
    return d


class Host:
    """A machine that runs engines. Local hosts use subprocess; remote hosts use SSH with a
    persistent control connection. Every remote argument is shell-quoted."""

    def __init__(self, spec: dict, lab: "Lab") -> None:
        self.spec = spec
        self.lab = lab
        self.id = spec["id"]
        self._docker: list[str] | None = None
        self._docker_t = 0.0
        self._reach: tuple[bool, str] | None = None
        self._reach_t = 0.0
        self._home: str | None = None
        self.lock = threading.RLock()

    # identity ----------------------------------------------------------------
    @property
    def local(self) -> bool:
        return not (self.spec.get("ssh") or "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.spec.get("enabled", True))

    @property
    def label(self) -> str:
        return self.spec.get("label") or self.id

    @property
    def network(self) -> str:
        return self.spec.get("network") or "titan-ai"

    @property
    def bind(self) -> str:
        return self.spec.get("bind") or "127.0.0.1"

    @property
    def human_host(self) -> str:
        return self.spec.get("human_host") or ("127.0.0.1" if self.local else self.spec["ssh"].split("@")[-1])

    @property
    def reach(self) -> str:
        r = self.spec.get("reach") or "auto"
        if self.local:
            return "local"
        if r == "auto":
            return "ssh" if self.bind in ("127.0.0.1", "localhost", "::1") else "direct"
        return r

    def ssh_base(self, control: bool = True) -> list[str]:
        base = [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new", "-o", "LogLevel=ERROR",
        ]
        if control:
            base += ["-o", f"ControlPath={_cm_dir()}/cm-%C", "-o", "ControlMaster=auto", "-o", "ControlPersist=300"]
        else:
            base += ["-o", "ControlMaster=no", "-o", "ControlPath=none"]
        if self.spec.get("ssh_port"):
            base += ["-p", str(int(self.spec["ssh_port"]))]
        return base

    # execution ---------------------------------------------------------------
    def argv(self, cmd: list[str], tty: bool = False) -> list[str]:
        if self.local:
            return list(cmd)
        remote = " ".join(shlex.quote(str(c)) for c in cmd)
        # -tt: the remote process gets SIGHUP when we hang up (live log views, cancellations)
        return self.ssh_base() + (["-tt"] if tty else []) + [self.spec["ssh"].strip(), "--", remote]

    def run(self, cmd: list[str], timeout: float = 30, input_text: str | None = None, env: dict | None = None) -> Res:
        if not self.local and env:
            cmd = ["env"] + [f"{k}={v}" for k, v in env.items()] + list(cmd)
            env = None
        return run_local(self.argv(cmd), timeout=timeout, input_text=input_text, env=env)

    def sh(self, script: str, timeout: float = 30, input_text: str | None = None) -> Res:
        return self.run(["sh", "-c", script], timeout=timeout, input_text=input_text)

    def reachable(self, force: bool = False) -> tuple[bool, str]:
        if self.local:
            return True, "local"
        if not self.enabled:
            return False, "disabled"
        with self.lock:
            if not force and self._reach and now() - self._reach_t < (10 if self._reach[0] else 20):
                return self._reach
            r = self.run(["true"], timeout=8)
            self._reach = (r.ok, "ok" if r.ok else (r.err.strip().splitlines() or ["ssh failed"])[-1][:200])
            self._reach_t = now()
            if not r.ok:
                self._docker = None
            return self._reach

    def require(self) -> None:
        ok, why = self.reachable()
        if not ok:
            if why == "disabled":
                raise LabError(f"{self.label} is disabled", "Enable it on the Hosts page once SSH works.", ["host-enable"])
            raise LabError(f"Cannot reach {self.label} over SSH", f"ssh {self.spec.get('ssh')}: {why}", ["host-test"])

    def home(self) -> str:
        if self.local:
            return str(HOME)
        if self._home is None:
            r = self.sh('printf %s "$HOME"', timeout=10)
            if r.ok and r.out.strip():
                self._home = r.out.strip()
        return self._home or "/root"

    def cache_dir(self) -> str:
        c = (self.spec.get("cache") or "").strip()
        if c:
            return c
        if self.local:
            return os.environ.get("HF_HOME", str(HOME / ".cache/huggingface"))
        return self.home().rstrip("/") + "/.cache/huggingface"

    def state_dir(self) -> str:
        return str(CONF_DIR) if self.local else self.home().rstrip("/") + "/.config/vllm-lab"

    def proc_expr(self) -> str:
        """Shell expression for procfs on this host (overridable with VLLM_LAB_PROC for testing)."""
        return shlex.quote(PROC_ROOT) if self.local else '"${VLLM_LAB_PROC:-/proc}"'

    def write_file(self, path: str, content: str) -> Res:
        d = path.rsplit("/", 1)[0]
        script = f"umask 077 && mkdir -p {shlex.quote(d)} && cat > {shlex.quote(path)}"
        return self.sh(script, timeout=15, input_text=content)

    # docker ------------------------------------------------------------------
    def docker(self, force: bool = False) -> list[str]:
        with self.lock:
            if self._docker is not None and not force and now() - self._docker_t < 600:
                return self._docker
            probe = self.run(["docker", "version", "--format", "{{.Server.Version}}"], timeout=15)
            if probe.ok:
                self._docker = ["docker"]
            else:
                probe2 = self.run(["sudo", "-n", "docker", "version", "--format", "{{.Server.Version}}"], timeout=15)
                self._docker = ["sudo", "-n", "docker"] if probe2.ok else ["docker"]
            self._docker_t = now()
            return self._docker

    def dk(self, *args: str, timeout: float = 60, input_text: str | None = None, env: dict | None = None) -> Res:
        return self.run(self.docker() + [str(a) for a in args], timeout=timeout, input_text=input_text, env=env)

    def dk_merged(self, *args: str, timeout: float = 60) -> Res:
        """docker … 2>&1 so stdout and stderr stay interleaved in order (logs)."""
        script = " ".join(shlex.quote(x) for x in self.docker() + [str(a) for a in args]) + " 2>&1"
        return self.sh(script, timeout=timeout)

    def docker_access(self) -> tuple[bool, str]:
        r = self.run(self.docker(force=True) + ["version", "--format", "{{.Server.Version}}"], timeout=15)
        if r.ok:
            return True, r.out.strip()
        err = r.text
        if "permission denied" in err.lower():
            return False, "Permission denied on the Docker socket. Add the user to the docker group (then re-login) or run the service with SupplementaryGroups=docker."
        if "command not found" in err or r.code == 127:
            return False, "Docker is not installed on this host."
        if "Cannot connect" in err or "failed to connect" in err or "Is the docker daemon running" in err:
            return False, "The Docker daemon is not running."
        return False, err[:300]

    # http data plane -----------------------------------------------------------
    def api_base(self, port: int) -> str:
        """Where the manager itself reaches an engine on this host."""
        if self.local:
            return f"http://127.0.0.1:{port}"
        if self.reach == "direct":
            return f"http://{self.human_host}:{port}"
        lport = self.lab.tunnels.ensure(self, int(port))
        return f"http://127.0.0.1:{lport}" if lport else ""

    def human_url(self, port) -> str:
        return f"http://{self.human_host}:{port}/v1" if port else ""


class Tunnels:
    """SSH local forwards so the manager can talk to engines on remote hosts that bind loopback."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.procs: dict[tuple[str, int], tuple[subprocess.Popen, int, float]] = {}
        self.failed: dict[tuple[str, int], float] = {}
        self.keylocks: dict[tuple[str, int], threading.Lock] = {}

    @staticmethod
    def _free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    def ensure(self, host: Host, rport: int) -> int | None:
        key = (host.id, rport)
        with self.lock:
            cur = self.procs.get(key)
            if cur and cur[0].poll() is None:
                return cur[1]
            if now() - self.failed.get(key, 0) < 15:
                return None
            klock = self.keylocks.setdefault(key, threading.Lock())
        if not klock.acquire(timeout=15):
            return None
        try:
            with self.lock:
                cur = self.procs.get(key)
                if cur and cur[0].poll() is None:
                    return cur[1]
                self.procs.pop(key, None)
            ok, _ = host.reachable()
            if not ok:
                return None
            lport = self._free_port()
            argv = host.ssh_base(control=False) + [
                "-N", "-o", "ExitOnForwardFailure=yes", "-L", f"127.0.0.1:{lport}:127.0.0.1:{rport}", host.spec["ssh"].strip(),
            ]
            try:
                p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                with self.lock:
                    self.failed[key] = now()
                return None
            for _ in range(40):
                if p.poll() is not None:
                    with self.lock:
                        self.failed[key] = now()
                    return None
                try:
                    socket.create_connection(("127.0.0.1", lport), timeout=0.2).close()
                    break
                except OSError:
                    time.sleep(0.1)
            with self.lock:
                self.procs[key] = (p, lport, now())
            return lport
        finally:
            klock.release()

    def close(self, host_id: str, rport: int) -> None:
        with self.lock:
            cur = self.procs.pop((host_id, rport), None)
        if cur:
            with contextlib.suppress(Exception):
                cur[0].terminate()

    def list(self) -> list[dict]:
        with self.lock:
            return [
                {"host": k[0], "remote": k[1], "local": v[1], "alive": v[0].poll() is None, "since": v[2]}
                for k, v in self.procs.items()
            ]

    def close_all(self) -> None:
        with self.lock:
            items = list(self.procs.values())
            self.procs.clear()
        for p, _, _ in items:
            with contextlib.suppress(Exception):
                p.terminate()


# ───────────────────────────────────────────────────────────────── http helpers

def http_json(url: str, method: str = "GET", body=None, headers: dict | None = None, timeout: float = 10):
    data = None if body is None else json.dumps(body).encode()
    h = {"Accept": "application/json", "User-Agent": f"vllm-lab/{VERSION}"}
    if data is not None:
        h["Content-Type"] = "application/json"
    h.update(headers or {})
    req = Request(url, data=data, headers=h, method=method)
    with urlopen(req, timeout=timeout) as r:
        raw = r.read().decode("utf-8", "replace")
    if not raw.strip():
        return {}
    return json.loads(raw)


def http_text(url: str, timeout: float = 3, headers: dict | None = None) -> tuple[int, str]:
    try:
        req = Request(url, headers={"User-Agent": f"vllm-lab/{VERSION}", **(headers or {})})
        with urlopen(req, timeout=timeout) as r:
            return r.status, r.read(2_000_000).decode("utf-8", "replace")
    except HTTPError as e:
        try:
            body = e.read(4000).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            body = ""
        return e.code, body
    except Exception as e:  # noqa: BLE001
        return 0, f"{type(e).__name__}: {e}"


# ───────────────────────────────────────────────────────────────── engine specs & blueprints

# The three engines this lab already runs. Seeded into engines.json on first start so the
# existing titan-* containers are adopted as-is.
LEGACY_RECIPES = {
    "lightning": {
        "host": "titan", "backend": "vllm", "model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "port": 58100, "quant": "", "util": 0.35, "max_len": 131072, "kv_dtype": "fp8", "trust_remote_code": True,
        "extra": (
            "--moe-backend marlin --enable-prefix-caching --enable-auto-tool-choice "
            "--tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 "
            "--mamba-backend flashinfer --mamba-cache-mode align --mamba-ssm-cache-dtype float16 "
            "--speculative_config.method dspark --speculative_config.num_speculative_tokens 3 "
            "--speculative_config.model nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark"
        ),
        "note": "Primary Titan chat model.", "blueprint": "nemotron-lightning",
    },
    "gemma4": {
        "host": "titan", "backend": "vllm", "model": "google/gemma-4-E4B-it", "port": 58102, "quant": "",
        "util": 0.18, "max_len": 32768, "image": PATCHED_IMAGE, "kv_dtype": "fp8", "trust_remote_code": True,
        "extra": "--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4",
        "max_num_seqs": 4, "note": "Needs the patched runtime (Transformers 5.14.1); built automatically.",
        "blueprint": "gemma4-e4b",
    },
    "super": {
        "host": "atlas", "backend": "vllm", "model": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        "port": 58101, "quant": "", "util": 0.85, "max_len": 65536, "kv_dtype": "fp8", "trust_remote_code": True,
        "extra": "", "note": "Atlas long-context engine.", "blueprint": "nemotron-super",
    },
}

BUILTIN_BLUEPRINTS = {
    "nemotron-lightning": {
        "title": "Nemotron 3.5 Lightning 30B-A3B", "family": "Nemotron", "maker": "NVIDIA", "tags": ["chat", "tools", "reasoning", "fast"],
        "spec": {k: v for k, v in LEGACY_RECIPES["lightning"].items() if k not in ("host", "port", "note", "blueprint")},
    },
    "nemotron-super": {
        "title": "Nemotron 3 Super 120B-A12B", "family": "Nemotron", "maker": "NVIDIA", "tags": ["chat", "long-context"],
        "spec": {k: v for k, v in LEGACY_RECIPES["super"].items() if k not in ("host", "port", "note", "blueprint")},
    },
    "gemma4-e4b": {
        "title": "Gemma 4 E4B", "family": "Gemma", "maker": "Google", "tags": ["chat", "small", "tools"],
        "spec": {k: v for k, v in LEGACY_RECIPES["gemma4"].items() if k not in ("host", "port", "note", "blueprint")},
    },
    "gpt-oss-120b": {
        "title": "gpt-oss 120B", "family": "gpt-oss", "maker": "OpenAI", "tags": ["chat", "reasoning", "tools"],
        "spec": {"backend": "vllm", "model": "openai/gpt-oss-120b", "util": 0.7, "max_len": 131072, "kv_dtype": "auto",
                 "extra": "--enable-auto-tool-choice --tool-call-parser openai"},
    },
    "gpt-oss-20b": {
        "title": "gpt-oss 20B", "family": "gpt-oss", "maker": "OpenAI", "tags": ["chat", "reasoning", "small"],
        "spec": {"backend": "vllm", "model": "openai/gpt-oss-20b", "util": 0.25, "max_len": 65536, "kv_dtype": "auto",
                 "extra": "--enable-auto-tool-choice --tool-call-parser openai"},
    },
    "llama-3.3-70b-fp4": {
        "title": "Llama 3.3 70B Instruct (NVFP4)", "family": "Llama", "maker": "Meta / NVIDIA", "tags": ["chat"],
        "spec": {"backend": "vllm", "model": "nvidia/Llama-3.3-70B-Instruct-FP4", "util": 0.6, "max_len": 65536, "kv_dtype": "fp8",
                 "extra": "--enable-prefix-caching"},
    },
    "llama-3.1-8b": {
        "title": "Llama 3.1 8B Instruct", "family": "Llama", "maker": "Meta", "tags": ["chat", "small"],
        "spec": {"backend": "vllm", "model": "meta-llama/Llama-3.1-8B-Instruct", "util": 0.2, "max_len": 32768, "kv_dtype": "fp8",
                 "extra": "--enable-prefix-caching --enable-auto-tool-choice --tool-call-parser llama3_json"},
    },
    "mistral-small-3.2": {
        "title": "Mistral Small 3.2 24B", "family": "Mistral", "maker": "Mistral AI", "tags": ["chat", "vision", "tools"],
        "spec": {"backend": "vllm", "model": "mistralai/Mistral-Small-3.2-24B-Instruct-2506", "util": 0.4, "max_len": 32768,
                 "kv_dtype": "fp8", "extra": "--tokenizer-mode mistral --config-format mistral --load-format mistral "
                 "--enable-auto-tool-choice --tool-call-parser mistral"},
    },
    "phi-4": {
        "title": "Phi-4 14B", "family": "Phi", "maker": "Microsoft", "tags": ["chat", "reasoning", "small"],
        "spec": {"backend": "vllm", "model": "microsoft/phi-4", "util": 0.25, "max_len": 16384, "kv_dtype": "fp8"},
    },
    "gpt-oss-20b-gguf": {
        "title": "gpt-oss 20B (llama.cpp GGUF)", "family": "gpt-oss", "maker": "OpenAI / ggml-org", "tags": ["chat", "gguf"],
        "spec": {"backend": "llamacpp", "model": "ggml-org/gpt-oss-20b-GGUF", "max_len": 32768, "max_num_seqs": 4},
    },
}

SPEC_DEFAULTS = {
    "name": "", "host": "titan", "backend": "vllm", "model": "", "port": None, "image": "", "util": 0.5,
    "max_len": 32768, "kv_dtype": "fp8", "quant": "", "trust_remote_code": False, "max_num_seqs": None,
    "served_name": "", "extra": "", "env": {}, "autostart": True, "idle_sleep_min": 0, "wake": True,
    "note": "", "blueprint": "", "created": 0, "desired": "any", "gguf": "", "halted": "",
}

MANAGED_FLAGS = {
    "--host", "--port", "--gpu-memory-utilization", "--max-model-len", "--kv-cache-dtype", "--trust-remote-code",
    "--quantization", "-q", "--served-model-name", "--max-num-seqs", "--model",
}
BOOL_FLAGS = {"--trust-remote-code"}


def normalize_spec(s: dict) -> dict:
    out = {**SPEC_DEFAULTS, **{k: v for k, v in (s or {}).items() if k in SPEC_DEFAULTS}}
    out["name"] = str(out["name"] or "").strip().lower()
    out["backend"] = out["backend"] if out["backend"] in ("vllm", "llamacpp") else "vllm"
    out["model"] = str(out["model"] or "").strip()
    out["port"] = as_int(out["port"], None)
    out["util"] = round(min(0.98, max(0.02, as_float(out["util"], 0.5))), 3)
    out["max_len"] = max(256, as_int(out["max_len"], 32768))
    out["max_num_seqs"] = as_int(out["max_num_seqs"], None) or None
    out["idle_sleep_min"] = max(0.0, round(as_float(out["idle_sleep_min"], 0) or 0, 2))
    out["trust_remote_code"] = as_bool(out["trust_remote_code"])
    out["autostart"] = as_bool(out["autostart"], True)
    out["wake"] = as_bool(out["wake"], True)
    out["quant"] = "" if str(out["quant"] or "").lower() in ("", "none", "auto") else str(out["quant"]).strip()
    out["kv_dtype"] = str(out["kv_dtype"] or "auto").strip() or "auto"
    out["env"] = {str(k): str(v) for k, v in (out["env"] or {}).items() if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", str(k))}
    out["extra"] = str(out["extra"] or "").strip()
    out["image"] = str(out["image"] or "").strip()
    out["served_name"] = str(out["served_name"] or "").strip()
    out["halted"] = str(out.get("halted") or "")[:64]
    out["desired"] = out["desired"] if out["desired"] in ("running", "stopped", "sleeping", "any") else "any"
    return out


def validate_spec(spec: dict) -> list[str]:
    errs = []
    if not NAME_RE.match(spec["name"]):
        errs.append("Name: lowercase letters, digits, dash, dot or underscore; 1–40 characters.")
    if not spec["model"] or not MODEL_RE.match(spec["model"]):
        errs.append("Model: a Hugging Face id like org/model.")
    if spec["image"] and not IMAGE_RE.match(spec["image"]):
        errs.append("Image: not a valid image reference.")
    if spec["port"] is not None and not (1024 < spec["port"] < 65536):
        errs.append("Port must be between 1025 and 65535.")
    if spec["served_name"] and not MODEL_RE.match(spec["served_name"]):
        errs.append("Served name: letters, digits, dash, dot, underscore, one slash.")
    try:
        shlex.split(spec["extra"])
    except ValueError as e:
        errs.append(f"Extra args: {e}")
    return errs


def split_extra(extra: str) -> tuple[list[str], list[str]]:
    """Return (args, dropped managed flags)."""
    try:
        parts = shlex.split(extra or "")
    except ValueError:
        parts = (extra or "").split()
    out, dropped = [], []
    i = 0
    while i < len(parts):
        tok = parts[i]
        flag = tok.split("=", 1)[0]
        if flag in MANAGED_FLAGS:
            dropped.append(flag)
            if "=" not in tok and flag not in BOOL_FLAGS and i + 1 < len(parts) and not parts[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        out.append(tok)
        i += 1
    return out, dropped


def extra_models(extra: str) -> list[str]:
    """Model ids referenced inside extra args (speculative drafts etc.)."""
    found = []
    parts, _ = split_extra(extra)
    for i, tok in enumerate(parts):
        key, _, val = tok.partition("=")
        if key.endswith("model") or key.endswith(".model") or key in ("--tokenizer",):
            v = val or (parts[i + 1] if i + 1 < len(parts) else "")
            if "/" in v and MODEL_RE.match(v):
                found.append(v)
        m = re.search(r'"model"\s*:\s*"([^"]+/[^"]+)"', tok)
        if m:
            found.append(m.group(1))
    return found


def served_id(spec: dict) -> str:
    return spec.get("served_name") or spec.get("model") or spec.get("name")


def resolve_image(spec: dict, conf: dict) -> str:
    if spec.get("image"):
        return spec["image"]
    return conf.get("llamacpp_image") if spec["backend"] == "llamacpp" else conf.get("vllm_image") or LEGACY_IMAGE


def backend_args(spec: dict) -> list[str]:
    if spec["backend"] == "llamacpp":
        ref = spec["model"] + (f":{spec['gguf']}" if spec.get("gguf") and ":" not in spec["model"] else "")
        args = ["-hf", ref, "--host", "0.0.0.0", "--port", "8000", "-ngl", "999", "-c", str(spec["max_len"]),
                "--jinja", "--metrics", "--alias", served_id(spec)]
        if spec.get("max_num_seqs"):
            args += ["-np", str(spec["max_num_seqs"])]
        extra, _ = split_extra(spec["extra"])
        return args + extra
    args = [spec["model"], "--host", "0.0.0.0", "--port", "8000",
            "--gpu-memory-utilization", f"{spec['util']:.3g}", "--max-model-len", str(spec["max_len"])]
    if spec.get("served_name"):
        args += ["--served-model-name", spec["served_name"]]
    if spec.get("kv_dtype") and spec["kv_dtype"] != "auto":
        args += ["--kv-cache-dtype", spec["kv_dtype"]]
    if spec.get("trust_remote_code"):
        args += ["--trust-remote-code"]
    if spec.get("quant"):
        args += ["--quantization", spec["quant"]]
    if spec.get("max_num_seqs"):
        args += ["--max-num-seqs", str(spec["max_num_seqs"])]
    extra, _ = split_extra(spec["extra"])
    return args + extra


def spec_fingerprint(spec: dict, image: str, host: Host) -> str:
    keys = ("backend", "model", "port", "util", "max_len", "kv_dtype", "quant", "trust_remote_code",
            "max_num_seqs", "served_name", "extra", "env", "autostart", "gguf")
    return short_hash({**{k: spec.get(k) for k in keys}, "image": image, "net": host.network, "bind": host.bind})


def container_name(conf: dict, name: str) -> str:
    return f"{conf.get('container_prefix') or ''}{name}"


def docker_run_argv(spec: dict, host: Host, conf: dict, image: str, env_file: str | None) -> list[str]:
    cname = container_name(conf, spec["name"])
    cache = host.cache_dir()
    argv = [
        "run", "-d", "--name", cname,
        "--label", f"vllm-lab.engine={spec['name']}",
        "--label", f"vllm-lab.spec={spec_fingerprint(spec, image, host)}",
        "--label", f"vllm-lab.backend={spec['backend']}",
        "--label", f"vllm-lab.model={spec['model']}",
        "--restart", "unless-stopped" if spec.get("autostart", True) else "no",
        "--gpus", "all", "--ipc=host",
        "--network", host.network,
        "-p", (f"[{host.bind}]" if ":" in host.bind else host.bind) + f":{spec['port']}:8000",
    ]
    if spec["backend"] == "llamacpp":
        argv += ["-v", f"{cache.rstrip('/')}/llama.cpp:/root/.cache/llama.cpp", "-e", "LLAMA_CACHE=/root/.cache/llama.cpp"]
    else:
        argv += ["-v", f"{cache}:/root/.cache/huggingface"]
    if env_file:
        argv += ["--env-file", env_file]
    for k, v in (spec.get("env") or {}).items():
        argv += ["-e", f"{k}={v}"]
    if spec["backend"] == "llamacpp":
        argv += ["--entrypoint", "/app/llama-server"]
    return argv + [image] + backend_args(spec)


def origin_check(model: str, conf: dict, base_models: list[str] | None = None) -> dict:
    pol = conf.get("policy") or {}
    orgs = {o.lower() for o in (pol.get("orgs") or [])}
    mode = pol.get("mode") or "block"
    hit = ""
    org = model_org(model).lower()
    if org and org in orgs:
        hit = model
    for b in base_models or []:
        if model_org(b).lower() in orgs:
            hit = hit or b
    return {"restricted": bool(hit), "via": hit, "mode": mode if hit else "", "blocked": bool(hit) and mode == "block"}


# ───────────────────────────────────────────────────────────────── log reading: boot stages & crash decoder

BOOT_STAGES = [
    ("pull", "Image"), ("download", "Weights"), ("load", "Load"), ("compile", "Compile"),
    ("graphs", "CUDA graphs"), ("serve", "Serve"),
]
STAGE_IDX = {k: i for i, (k, _) in enumerate(BOOT_STAGES)}

_RE_PCT = re.compile(r"(\d{1,3})%\|")
_RE_SHARDS = re.compile(r"Loading safetensors checkpoint shards:\s+(\d+)% Completed \|\s*(\d+)/(\d+)")
_RE_KVTOK = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_RE_MAXCONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x")
_RE_WEIGHTS = re.compile(r"Model loading took ([\d.]+)\s*Gi?B")
_RE_DLFILE = re.compile(r"([\w./+-]+\.(?:safetensors|bin|gguf|pt))[^\n%]*?:\s*(\d{1,3})%\|")
_RE_TIMESTAMP = re.compile(r"^\S+\s+\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?\s+\[[^\]]+\]\s*")


def read_boot(log: str, backend: str = "vllm") -> dict:
    """Infer boot stage + details from the tail of an engine's log."""
    stage, pct, detail = "", None, ""
    info: dict = {}
    lines = re.split(r"[\r\n]+", log or "")
    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        if backend == "llamacpp":
            if "downloading" in ln.lower() or "curl_perform" in ln:
                stage, detail = "download", "Downloading GGUF"
                m = _RE_PCT.search(ln) or re.search(r"(\d{1,3})%", ln)
                pct = int(m.group(1)) if m else pct
            elif "load_tensors" in ln or "llama_model_load" in ln or "loading model" in ln:
                stage, detail, pct = "load", "Loading tensors", None
            elif "warming up" in ln.lower():
                stage, detail, pct = "graphs", "Warming up", None
            elif "server is listening" in ln or "all slots are idle" in ln or "model loaded" in ln:
                stage, detail, pct = "serve", "Listening", None
            continue
        m = _RE_DLFILE.search(ln)
        if m or "Downloading" in ln or "Fetching" in ln and "files" in ln:
            if STAGE_IDX.get(stage, -1) <= STAGE_IDX["download"]:
                stage = "download"
                pm = _RE_PCT.search(ln)
                pct = int(pm.group(1)) if pm else pct
                detail = (m.group(1).split("/")[-1] if m else "Fetching files")
            continue
        m = _RE_SHARDS.search(ln)
        if m:
            stage, pct, detail = "load", int(m.group(1)), f"shard {m.group(2)}/{m.group(3)}"
            continue
        if "Loading weights took" in ln or "Loading model weights" in ln or "Starting to load model" in ln:
            if STAGE_IDX.get(stage, -1) < STAGE_IDX["load"]:
                stage, pct, detail = "load", None, "Loading weights"
        m = _RE_WEIGHTS.search(ln)
        if m:
            info["weights_gib"] = float(m.group(1))
            stage, pct, detail = "load", 100, f"{m.group(1)} GiB in memory"
        if "torch.compile" in ln or "Compiling a graph" in ln or "Dynamo bytecode transform" in ln or "compile range" in ln.lower():
            if STAGE_IDX.get(stage, -1) <= STAGE_IDX["compile"]:
                stage, pct, detail = "compile", None, "torch.compile"
        m = _RE_KVTOK.search(ln)
        if m:
            info["kv_tokens"] = int(m.group(1).replace(",", ""))
        m = _RE_MAXCONC.search(ln)
        if m:
            info["max_conc"] = float(m.group(2))
            info["max_conc_len"] = int(m.group(1).replace(",", ""))
        if "Capturing CUDA graph" in ln or "Capturing cudagraph" in ln or "CUDA graph" in ln and "%|" in ln:
            stage = "graphs"
            pm = _RE_PCT.search(ln)
            pct = int(pm.group(1)) if pm else pct
            detail = "Capturing graphs"
        if "Starting vLLM API server" in ln or "Application startup complete" in ln or "Uvicorn running" in ln:
            stage, pct, detail = "serve", None, "API up"
    return {"stage": stage, "pct": pct, "detail": detail, **info}


CRASH_RULES = [
    (r"Free memory on device.*?less than desired GPU memory utilization|less than desired GPU memory",
     "Not enough free memory at startup",
     "Another engine (or the page cache) is holding unified memory. Free some, or start this one with a smaller memory share.",
     ["make-room", "flush-cache", "fit-util"]),
    (r"estimated maximum model length is (\d+)",
     "Context window does not fit in the KV cache",
     "The memory share leaves too little room for the KV cache at this max context. Lower the context or raise the share.",
     ["set-context", "fit-util"]),
    (r"No available memory for the cache blocks|larger than the available KV cache memory|Insufficient memory for KV cache",
     "KV cache does not fit",
     "Weights fit but nothing is left for the KV cache. Raise the memory share or lower max context.",
     ["fit-util", "lower-context"]),
    (r"CUDA out of memory|torch\.OutOfMemoryError|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED",
     "Ran out of GPU memory",
     "The engine asked for more memory than was free. Stop another engine or lower this one's share.",
     ["make-room", "fit-util"]),
    (r"GatedRepoError|Cannot access gated repo|Access to model .* is restricted|awaiting a review",
     "Gated model — access not granted",
     "Accept the model's terms on Hugging Face with the account that owns your token, then start again.",
     ["open-hf", "set-token"]),
    (r"401 Client Error|Invalid user token|Invalid credentials in Authorization header",
     "Hugging Face rejected the token",
     "The token is missing, expired, or lacks read access.",
     ["set-token"]),
    (r"RepositoryNotFoundError|is not a local folder and is not a valid model identifier|404 Client Error.*huggingface",
     "Model id not found on Hugging Face",
     "Check the spelling of org/model, or that the token can see a private repo.",
     ["edit"]),
    (r"AmbiguousGlobalPerLayerAttributeError|per-layer.*head_dim|head_dim.*per-layer",
     "This vLLM build is too old for Gemma 4",
     "Gemma 4 needs Transformers 5.14.1. The patched runtime image fixes it.",
     ["gemma-runtime"]),
    (r"User-specified max_model_len \((\d+)\) is greater than the derived max_model_len.*?=(\d+)",
     "Context longer than the model supports",
     "Lower max context to the model's native limit.",
     ["set-context"]),
    (r"Model architectures? \[?'?([\w]+)'?\]? (?:are|is) not supported|Cannot find model module|not supported for now",
     "Architecture not supported by this vLLM image",
     "Try a newer vLLM image, a different quant of the same model, or llama.cpp with a GGUF build.",
     ["edit"]),
    (r"Unknown quantization method|quantization method .* is not supported|Cannot find the config file for (?:awq|gptq)",
     "Quantization flag does not match the weights",
     "Leave quantization on Auto — vLLM reads it from the checkpoint.",
     ["quant-auto"]),
    (r"unrecognized arguments: (.+)|error: argument (--[\w.-]+)|invalid choice: '([^']+)'",
     "vLLM rejected an argument",
     "An extra argument is not valid for this vLLM version. Edit the engine's extra args.",
     ["edit"]),
    (r"no matching manifest for linux/arm64|exec format error|image's platform \(linux/amd64\)",
     "Image is not built for ARM64",
     "GB10 is ARM64 (aarch64). Pick an image that publishes linux/arm64, or build one on the box.",
     ["edit"]),
    (r"could not select device driver .* capabilities: \[\[gpu\]\]|nvidia-container-cli|unknown or invalid runtime name: nvidia",
     "Docker cannot see the GPU",
     "The NVIDIA Container Toolkit is missing or not configured for Docker on this host.",
     ["doctor"]),
    (r"port is already allocated|address already in use|bind: address already in use",
     "Port already in use",
     "Something else holds this engine's host port. Move it to a free port.",
     ["change-port"]),
    (r"pull access denied|manifest unknown|repository does not exist|not found: manifest",
     "Image not found",
     "The image name or tag does not exist, or needs a registry login.",
     ["edit"]),
    (r"Please pass the argument `trust_remote_code=True`|requires you to execute the configuration file|contains custom code which must be executed",
     "Model needs remote code",
     "This repo ships its own modeling code. Turn on Trust remote code for this engine.",
     ["trust-remote-code"]),
    (r"No space left on device",
     "Disk is full",
     "The weights cache or Docker storage ran out of space. Free space in Library.",
     ["library"]),
    (r"Temporary failure in name resolution|MaxRetryError|ConnectionError.*huggingface|Network is unreachable|LocalEntryNotFoundError",
     "No network path to Hugging Face",
     "The box cannot download. If the weights are already cached, set HF_HUB_OFFLINE=1 on the engine.",
     ["offline"]),
]

EXIT_HELP = {
    1: "The process exited with an error.",
    125: "Docker could not create the container.",
    126: "The command in the image is not executable.",
    127: "The command was not found in the image.",
    137: "Killed (SIGKILL) — usually memory pressure.",
    139: "Segmentation fault inside the engine.",
}


def decode_crash(log: str, exit_code=None, oom: bool = False, backend: str = "vllm") -> dict | None:
    text = log or ""
    if oom:
        return {"title": "Killed by the kernel's OOM killer", "hint": "The box ran out of unified memory while this engine was loading or serving. Give it less memory, or stop another engine first.",
                "fixes": ["make-room", "fit-util"], "line": "", "exit": exit_code}
    tail = text[-60000:]
    last_err = ""
    for ln in re.split(r"[\r\n]+", tail):
        s = _RE_TIMESTAMP.sub("", ln.strip())
        if re.match(r"^[\w.]*(Error|Exception|Exit)\b[:(]", s) or re.match(r"^\w+(\.\w+)*(Error|Exception): ", s):
            last_err = s[:400]
    for rx, title, hint, fixes in CRASH_RULES:
        m = None
        for m in re.finditer(rx, tail):
            pass
        if m:
            data = {}
            if "estimated maximum model length" in rx:
                data["max_len"] = int(m.group(1))
            elif "User-specified max_model_len" in rx:
                data["max_len"] = int(m.group(2))
            elif "unrecognized arguments" in rx:
                data["args"] = next((g for g in m.groups() if g), "")
            return {"title": title, "hint": hint, "fixes": fixes, "line": last_err or m.group(0)[:300], "exit": exit_code, "data": data}
    if exit_code not in (None, 0, 143) or last_err:
        return {"title": EXIT_HELP.get(int(exit_code or 1), f"Exited with code {exit_code}"), "hint": "Open the log for the full traceback.",
                "fixes": ["logs"], "line": last_err, "exit": exit_code}
    return None


# ───────────────────────────────────────────────────────────────── prometheus metrics

_RE_PROM = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEnaNinfINF]+)")


def parse_prom(text: str) -> dict:
    out: dict[str, float] = {}
    labels: dict[str, str] = {}
    for line in (text or "").splitlines():
        if not line or line[0] == "#":
            continue
        m = _RE_PROM.match(line)
        if not m:
            continue
        name, lab, val = m.group(1), m.group(2) or "", m.group(3)
        try:
            v = float(val)
        except ValueError:
            continue
        if math.isnan(v) or math.isinf(v):
            continue
        if name.endswith("_bucket"):
            continue
        out[name] = out.get(name, 0.0) + v
        if name == "vllm:cache_config_info":
            labels["cache_config"] = lab
    if "cache_config" in labels:
        lab = labels["cache_config"]
        b = re.search(r'num_gpu_blocks="(\d+)"', lab)
        s = re.search(r'block_size="(\d+)"', lab)
        if b and s:
            out["_kv_capacity_tokens"] = float(int(b.group(1)) * int(s.group(1)))
    return out


def metric(m: dict, *names: str, default=None):
    for n in names:
        if n in m:
            return m[n]
    return default


class Rates:
    """Turns counters into rates between scrapes."""

    def __init__(self) -> None:
        self.prev: dict[str, tuple[float, dict]] = {}

    def update(self, key: str, m: dict) -> dict:
        t = now()
        prev = self.prev.get(key)
        self.prev[key] = (t, m)
        out = {}
        if not prev:
            return out
        dt = t - prev[0]
        if dt <= 0.2:
            return out
        pm = prev[1]

        def delta(*names):
            a = metric(m, *names)
            b = metric(pm, *names)
            if a is None or b is None:
                return None
            d = a - b
            return d if d >= 0 else a  # counter reset → use absolute

        g = delta("vllm:generation_tokens_total", "llamacpp:tokens_predicted_total")
        p = delta("vllm:prompt_tokens_total", "llamacpp:prompt_tokens_total")
        r = delta("vllm:request_success_total", "llamacpp:requests_total")
        if g is not None:
            out["gen_tps"] = g / dt
        if p is not None:
            out["prompt_tps"] = p / dt
        if r is not None:
            out["req_rate"] = r / dt
            out["req_delta"] = r
        ts, tc = delta("vllm:time_to_first_token_seconds_sum"), delta("vllm:time_to_first_token_seconds_count")
        if ts is not None and tc:
            out["ttft"] = ts / tc
        es, ec = delta("vllm:e2e_request_latency_seconds_sum"), delta("vllm:e2e_request_latency_seconds_count")
        if es is not None and ec:
            out["e2e"] = es / ec
        return out


# ───────────────────────────────────────────────────────────────── engine operations

def stream_process(argv: list[str], on_line, timeout: float = 3600, input_text: str | None = None, job: Job | None = None) -> Res:
    """Run a command, feeding each output line (split on \\r and \\n) to on_line."""
    try:
        p = subprocess.Popen(argv, stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        return Res(127, "", f"{argv[0]}: command not found")
    if input_text is not None:
        with contextlib.suppress(BrokenPipeError):
            p.stdin.write(input_text.encode())
            p.stdin.close()
    buf = b""
    lines: list[str] = []
    t0 = now()
    assert p.stdout is not None
    os.set_blocking(p.stdout.fileno(), False)
    while True:
        if job and job.cancelled:
            p.kill()
            raise Cancelled()
        if now() - t0 > timeout:
            p.kill()
            return Res(124, "\n".join(lines[-200:]), "timed out")
        chunk = None
        with contextlib.suppress(BlockingIOError):
            chunk = p.stdout.read(65536)
        if chunk:
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for part in parts:
                s = part.decode("utf-8", "replace").rstrip()
                if s:
                    lines.append(s)
                    lines = lines[-2000:]
                    with contextlib.suppress(Exception):
                        on_line(s)
            continue
        if p.poll() is not None:
            rest = p.stdout.read() or b""
            for part in re.split(rb"[\r\n]", buf + rest):
                s = part.decode("utf-8", "replace").rstrip()
                if s:
                    lines.append(s)
                    with contextlib.suppress(Exception):
                        on_line(s)
            break
        time.sleep(0.1)
    return Res(p.returncode, "\n".join(lines[-400:]), "")


OLLAMA_ORGS = {"qwen": "Qwen", "deepseek": "deepseek-ai", "yi": "01-ai", "glm": "THUDM", "internlm": "internlm", "baichuan": "baichuan-inc",
               "minicpm": "openbmb", "codegeex": "THUDM", "kimi": "moonshotai", "minimax": "MiniMaxAI", "hunyuan": "tencent"}


def ollama_origin(tag: str) -> str:
    """Ollama tags carry no org; map well-known families to their Hub org for the origin policy."""
    base = tag.split(":")[0].split("/")[-1].lower()
    for k, org in OLLAMA_ORGS.items():
        if base.startswith(k):
            return f"{org}/{base}"
    return tag.split(":")[0]


class OpsMixin:
    # these attributes are provided by Lab
    store: Store
    events: Events
    jobs: Jobs

    # create / update --------------------------------------------------------------
    def allocate_port(self, hid: str, exclude: str = "", want: int | None = None) -> int:
        conf = self.conf()
        h = self.host(hid)
        pool = parse_ports(h.spec.get("ports") or conf.get("port_pool") or "58100-58119")
        reserved = set(parse_ports(conf.get("port_reserved") or ""))
        used = {s["port"] for n, s in self.store.engines().items() if s["host"] == hid and n != exclude and s.get("port")}
        for c in self.containers.get(hid, []):
            if c["name"] == self.cname(exclude):
                continue
            for m in re.finditer(r":(\d+)->", c.get("ports") or ""):
                used.add(int(m.group(1)))
        if want and want not in used:
            return want
        for p in pool:
            if p not in used and p not in reserved:
                return p
        raise LabError("No free port left in the pool", "Widen the port pool in Settings.", ["settings"])

    def upsert_engine(self, data: dict, create: bool = False, old_name: str = "") -> dict:
        engines = self.store.engines()
        name = str(data.get("name") or "").strip().lower()
        if not name and data.get("model"):
            name = model_slug(data["model"])
            base, i = name, 2
            while name in engines:
                name = f"{base}-{i}"
                i += 1
        data = {**data, "name": name}
        existing = engines.get(old_name or name)
        if create and name in engines:
            raise LabError(f"An engine named '{name}' already exists", "Pick another name.", code=409)
        if old_name and old_name != name and name in engines:
            raise LabError(f"An engine named '{name}' already exists", code=409)
        bp = data.get("blueprint")
        base = {}
        if bp and not existing:
            b = self.blueprints().get(bp)
            if b:
                base = dict(b.get("spec") or {})
        spec = normalize_spec({**base, **(existing or {}), **data})
        if not existing:
            spec["created"] = now()
            spec["desired"] = "stopped"
        errs = validate_spec(spec)
        if errs:
            raise LabError(errs[0], " ".join(errs[1:]), code=422)
        self.host(spec["host"])
        conf = self.conf()
        pol = origin_check(spec["model"], conf)
        for m in extra_models(spec["extra"]):
            p2 = origin_check(m, conf)
            if p2["restricted"]:
                pol = p2
        if pol["blocked"]:
            raise LabError(f"{pol['via']} is blocked by the origin policy", "Change the policy in Settings if this is intentional.", ["policy"], code=403)
        if spec["backend"] == "vllm" and ("gguf" in spec["model"].lower()):
            raise LabError("GGUF repos run on llama.cpp, not vLLM", "Switch Backend to llama.cpp.", ["to-llamacpp"], code=422)
        port_taken = spec.get("port") and any(
            s["port"] == spec["port"] and s["host"] == spec["host"] and n not in (name, old_name) for n, s in engines.items())
        if not spec.get("port") or port_taken:
            spec["port"] = self.allocate_port(spec["host"], exclude=old_name or name)
        if old_name and old_name != name:
            if (self.engine_state.get(old_name) or {}).get("state") in ("ready", "booting", "crashed", "stopped", "sleeping"):
                raise LabError("Remove the container before renaming", "Rename changes the container name.", code=409)
            self.store.drop_engine(old_name)
        self.store.put_engine(spec)
        self.events.add("info", name, f"{'Created' if not existing else 'Updated'} engine {name} ({spec['model']})")
        return spec

    # container helpers --------------------------------------------------------------
    def inspect_one(self, h: Host, cname: str) -> dict | None:
        r = h.dk("inspect", cname, timeout=15)
        if not r.ok:
            return None
        with contextlib.suppress(json.JSONDecodeError, IndexError):
            return json.loads(r.out)[0]
        return None

    def ensure_network(self, h: Host) -> None:
        if not h.dk("network", "inspect", h.network, timeout=15).ok:
            r = h.dk("network", "create", h.network, timeout=20)
            if not r.ok and "already exists" not in r.text:
                raise LabError(f"Could not create Docker network {h.network}", r.text[:300])

    def ensure_env_file(self, h: Host) -> str | None:
        tok = self.hf_token()
        if not tok:
            return None
        path = h.state_dir().rstrip("/") + "/engine.env"
        content = f"HF_TOKEN={tok}\nHUGGING_FACE_HUB_TOKEN={tok}\n"
        r = h.write_file(path, content)
        if not r.ok:
            raise LabError(f"Could not write the token file on {h.label}", r.text[:300])
        return path

    def hf_token(self) -> str:
        tok = os.environ.get("HF_TOKEN") or self.conf().get("hf_token") or ""
        if tok:
            return tok.strip()
        p = HOME / ".cache/huggingface/token"
        with contextlib.suppress(OSError):
            return p.read_text().strip()
        return ""

    def image_present(self, h: Host, image: str) -> bool:
        return h.dk("image", "inspect", "--format", "{{.Id}}", image, timeout=20).ok

    def ensure_image(self, h: Host, image: str, job: Job | None = None) -> str:
        if self.image_present(h, image):
            return image
        if image == PATCHED_IMAGE:
            return self.build_patched(h, job)
        if job:
            job.step(f"Pulling {image}", 0.0)
        layers: dict[str, str] = {}

        def on_line(s: str) -> None:
            m = re.match(r"^([0-9a-f]{12}): (.+)$", s)
            if m:
                layers[m.group(1)] = m.group(2)
                done = sum(1 for v in layers.values() if v.startswith(("Pull complete", "Already exists")))
                if job:
                    job.set_progress(done / max(1, len(layers)), f"Pulling {image} · {done}/{len(layers)} layers")
        r = stream_process(h.argv(h.docker() + ["pull", image]), on_line, timeout=7200, job=job)
        if not r.ok:
            crash = decode_crash(r.out, 1)
            raise LabError(f"Could not pull {image}", (crash or {}).get("hint") or r.out[-400:], (crash or {}).get("fixes") or ["edit"])
        return image

    def build_patched(self, h: Host, job: Job | None = None) -> str:
        base = self.conf().get("vllm_image") or LEGACY_IMAGE
        if job:
            job.step(f"Building patched runtime from {base}")
        self.ensure_image(h, base, job)
        dockerfile = f"FROM {base}\nRUN python3 -m pip install --no-cache-dir 'transformers==5.14.1'\n"

        def on_line(s: str) -> None:
            if job and (s.startswith("#") or s.startswith("Step")):
                job.set_progress(None, s[:120])
        r = stream_process(h.argv(h.docker() + ["build", "-t", PATCHED_IMAGE, "-"]), on_line, timeout=3600, input_text=dockerfile, job=job)
        if not r.ok:
            raise LabError("Patched image build failed", r.out[-800:])
        return PATCHED_IMAGE

    # memory fit --------------------------------------------------------------------
    def fit_check(self, spec: dict, h: Host, util: float | None = None) -> dict:
        hs = self.host_state.get(h.id) or {}
        if not hs.get("mem"):
            self.collect_host(h)
            hs = self.host_state.get(h.id) or {}
        mem = hs.get("mem") or {}
        conf = self.conf()
        total = int(mem.get("gpu_total") or mem.get("total") or 0)
        if not total:
            return {"known": False, "fits": True}
        headroom = int(float(conf.get("headroom_gib") or 4) * GIB)
        util = spec["util"] if util is None else util
        if spec["backend"] == "llamacpp":
            meta = (self.hf_meta_cache.get(spec["model"]) or (0, {}))[1]
            need = int((meta.get("gguf_bytes") or meta.get("weights_bytes") or 0) * 1.15 + 1.5 * GIB) if meta else 0
        else:
            need = int(util * total)
        own = self.engine_state.get(spec["name"]) or {}
        own_back = own.get("reserve", 0) if own.get("state") in ("ready", "booting") else 0
        if mem.get("gpu_total"):
            free = available = total - int(mem.get("gpu_used") or 0)
            cached = 0
        else:
            free, available, cached = int(mem.get("free") or 0), int(mem.get("available") or 0), int(mem.get("cached") or 0)
        free_eff = free - headroom + own_back
        avail_eff = available - headroom + own_back
        others = []
        for n, e in self.engine_state.items():
            if n == spec["name"] or e.get("host") != h.id or e.get("state") not in ("ready", "booting"):
                continue
            la = max(self.activity.get(n) or 0, e.get("started_ts") or 0)
            others.append({"name": n, "reserve": e.get("reserve") or 0, "state": e.get("state"), "idle": now() - la if la else None,
                           "busy": ((e.get("metrics") or {}).get("running") or 0) > 0})
        others.sort(key=lambda o: (o["busy"], -(o["idle"] or 0), -o["reserve"]))
        evict, gain = [], 0
        for o in others:
            if need <= avail_eff + gain:
                break
            evict.append(o["name"])
            gain += o["reserve"]
        fit_util = math.floor(max(0, avail_eff) / total * 100) / 100
        return {
            "known": True, "need": need, "total": total, "free": free, "available": available, "cached": cached,
            "headroom": headroom, "fits": need <= free_eff, "fits_after_flush": need <= avail_eff,
            "short": max(0, need - avail_eff), "others": others,
            "suggest": {"evict": evict if need <= avail_eff + gain else [], "util": fit_util if fit_util >= 0.05 else None},
        }

    def flush_cache(self, h: Host) -> tuple[bool, str]:
        target = h.proc_expr() + "/sys/vm/drop_caches"
        r = h.sh(f"sync; echo 3 | sudo -n tee {target} >/dev/null", timeout=60)
        if r.ok:
            return True, "Flushed the page cache"
        r2 = h.sh(f"sync; echo 3 > {target}", timeout=60)
        if r2.ok:
            return True, "Flushed the page cache"
        user = "titan"
        with contextlib.suppress(Exception):
            user = h.sh("id -un", timeout=5).out.strip() or user
        return False, (f"Needs passwordless sudo for one command. Add with visudo:\n"
                       f"{user} ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches")

    # start / stop ------------------------------------------------------------------
    def op_start(self, name: str, job: Job | None = None, on_conflict: str = "ask", evict: list | None = None,
                 util: float | None = None, recreate: bool = False, wait: bool = True, wait_timeout: float = 1800) -> str:
        job = job or Job(self.jobs, "start", name, name)
        with self.engine_lock(name):
            spec = self.spec(name)
            h = self.host(spec["host"])
            job.step(f"Checking {h.label}")
            h.require()
            conf = self.conf()
            pol = origin_check(spec["model"], conf)
            if pol["blocked"]:
                raise LabError(f"{spec['model']} is blocked by the origin policy", "", ["policy"], code=403)
            if util is not None:
                spec = self.store.update_engine(name, lambda s: s.__setitem__("util", float(util))) or spec
            image = resolve_image(spec, conf)
            cname = self.cname(name)
            obj = self.inspect_one(h, cname)
            fp = spec_fingerprint(spec, image, h)
            cur_fp = ((obj or {}).get("Config") or {}).get("Labels", {}) or {}
            cur_fp = cur_fp.get("vllm-lab.spec")
            running = bool(obj and (obj.get("State") or {}).get("Running"))
            legacy = bool(obj) and cur_fp is None
            if running and not recreate and (cur_fp == fp or legacy):
                self.store.update_engine(name, lambda s: s.__setitem__("desired", "running"))
                return f"{name} is already running"
            reuse = bool(obj) and not running and not recreate and (cur_fp == fp or (legacy and (obj.get("State") or {}).get("ExitCode") in (0, 143, 137)))
            # memory ----------------------------------------------------------
            job.step("Checking memory")
            if running:
                h.dk("stop", "-t", "20", cname, timeout=60)
                self.refresh([h.id])
            fit = self.fit_check(spec, h)
            if fit.get("known") and not fit["fits_after_flush"]:
                if on_conflict in ("evict", "solo"):
                    victims = evict if evict else (fit["suggest"]["evict"] if on_conflict == "evict" else [o["name"] for o in fit["others"]])
                    if not victims:
                        raise LabError("Stopping other engines would still not free enough memory", f"Short by {fmt_bytes(fit['short'])}.",
                                       ["fit-util"], code=409, data={"fit": fit})
                    for v in victims:
                        job.step(f"Putting {v} to sleep to make room")
                        self.op_stop(v, None, sleeping=True, reason=f"to make room for {name}")
                    self.refresh([h.id])
                    fit = self.fit_check(spec, h)
                elif on_conflict == "shrink" and fit["suggest"].get("util"):
                    u = fit["suggest"]["util"]
                    spec = self.store.update_engine(name, lambda s: s.__setitem__("util", u)) or spec
                    job.log(f"Memory share lowered to {u:.2f}")
                    image = resolve_image(spec, conf)
                    reuse = False
                    fit = self.fit_check(spec, h)
                elif on_conflict != "force":
                    raise LabError(
                        f"{name} needs {fmt_bytes(fit['need'])} but only {fmt_bytes(max(0, fit['available'] - fit['headroom']))} is free on {h.label}",
                        "Stop another engine, shrink this one, or start anyway.", ["make-room", "fit-util", "force"], code=409, data={"fit": fit})
            if fit.get("known") and not fit["fits"] and fit["fits_after_flush"] and on_conflict != "force":
                if conf.get("auto_flush_cache", True):
                    ok, msg = self.flush_cache(h)
                    job.log("Flushed the page cache so CUDA sees the free memory" if ok else "Page cache could not be flushed (no passwordless sudo) — start may fail")
                    if ok:
                        self.refresh([h.id])
                else:
                    job.log(f"{fmt_bytes(fit['cached'])} of page cache may look like used memory to CUDA")
            self.stopping.pop(name, None)
            cid = ""
            if reuse:
                cid = (obj.get("Id") or "")[:12]
                job.step("Starting existing container")
                r = h.dk("start", cname, timeout=60)
                if not r.ok:
                    crash = decode_crash(r.text, 125)
                    raise LabError(f"docker start failed: {(crash or {}).get('title') or r.text[:200]}", (crash or {}).get("hint", r.text[-300:]), (crash or {}).get("fixes", []))
            else:
                if obj:
                    job.step("Removing old container")
                    h.dk("rm", "-f", cname, timeout=60)
                env_file = self.ensure_env_file(h)
                if not env_file and spec["backend"] == "vllm":
                    job.log("No Hugging Face token set — public models only")
                job.step("Preparing network and image")
                self.ensure_network(h)
                image = self.ensure_image(h, image, job)
                # port sanity
                for c in self.containers.get(h.id, []):
                    if c["name"] != cname and re.search(rf":{spec['port']}->", c.get("ports") or ""):
                        raise LabError(f"Port {spec['port']} is already used by {c['name']}", "Move this engine to a free port.", ["change-port"], code=409)
                argv = docker_run_argv(spec, h, conf, image, env_file)
                job.step("Creating container")
                r = h.dk(*argv, timeout=180)
                cid = (r.out.strip().splitlines() or [""])[-1][:12] if r.ok else ""
                if not r.ok:
                    h.dk("rm", "-f", cname, timeout=30)
                    crash = decode_crash(r.text, 125)
                    raise LabError(f"docker run failed: {(crash or {}).get('title') or r.text.splitlines()[-1][:200] if r.text else 'unknown error'}",
                                   (crash or {}).get("hint") or r.text[-500:], (crash or {}).get("fixes") or ["logs"])
            self.store.update_engine(name, lambda s: s.update(desired="running", halted=""))
            self.activity[name] = now()
            self.boot_seen.pop(name, None)
            self.log_cache.pop(name, None)
            with self.lock:
                prev = dict(self.engine_state.get(name) or {})
                self.engine_state[name] = {**prev, "state": "booting", "crash": None, "metrics": {}, "reserve": prev.get("reserve") or 0,
                                           "boot": {"stage": "pull", "elapsed": 0}, "container": {"id": cid, "status": "running", "restarts": 0},
                                           "started_ts": now()}
            self.events.add("info", name, f"Started {name} on {h.label}")
        if wait:
            return self.wait_ready(name, job, wait_timeout, cid=cid)
        return f"{name} started"

    def wait_ready(self, name: str, job: Job, timeout: float = 1800, cid: str = "") -> str:
        t0 = now()
        last = ""
        seen_boot = False
        while now() - t0 < timeout:
            job.check()
            if not self.serve:
                self.refresh([self.spec(name)["host"]])
            e = self.engine_state.get(name) or {}
            st = e.get("state")
            ecid = (e.get("container") or {}).get("id") or ""
            if cid and ecid and not (ecid.startswith(cid) or cid.startswith(ecid)):
                time.sleep(0.5)
                continue   # still looking at the previous container
            if st == "ready":
                took = now() - t0
                return f"{name} is ready in {fmt_dur(took)}"
            if st == "booting":
                seen_boot = True
                b = e.get("boot") or {}
                label = dict(BOOT_STAGES).get(b.get("stage"), "Booting")
                txt = f"{label}" + (f" · {b['detail']}" if b.get("detail") else "") + (f" · {b['pct']}%" if b.get("pct") is not None else "")
                idx = STAGE_IDX.get(b.get("stage") or "", 0)
                prog = (idx + (b.get("pct") or 0) / 100) / len(BOOT_STAGES)
                if txt != last:
                    job.step(txt, prog)
                    last = txt
                else:
                    job.set_progress(prog, txt)
            if st == "crashed" or (st == "booting" and e.get("crash") and (e.get("container") or {}).get("restarts", 0) >= 1):
                cr = e.get("crash") or {}
                self.halt_crashloop(name)
                raise LabError(f"{name} crashed while booting: {cr.get('title', 'see logs')}", cr.get("hint", ""), cr.get("fixes") or ["logs"],
                               data={"line": cr.get("line"), "crash": cr})
            if st in ("stopped", "absent") and seen_boot:
                raise LabError(f"{name} stopped while booting", "Open the log for details.", ["logs"])
            time.sleep(1.5 if self.serve else 3)
        raise LabError(f"{name} is still booting after {fmt_dur(timeout)}", "It may still come up — watch the engine card.", ["logs"])

    def halt_crashloop(self, name: str) -> None:
        """A failed boot should not keep restarting and grabbing memory; keep the container for its log."""
        with contextlib.suppress(LabError):
            spec = self.spec(name)
            h = self.host(spec["host"])
            cn = self.cname(name)
            h.dk("update", "--restart=no", cn, timeout=20)
            h.dk("stop", "-t", "3", cn, timeout=30)
            cid = ((self.engine_state.get(name) or {}).get("container") or {}).get("id") or ""
            self.store.update_engine(name, lambda s: s.__setitem__("halted", cid))
            self.guarded.add(((self.engine_state.get(name) or {}).get("container") or {}).get("id") or "")

    def op_stop(self, name: str, job: Job | None = None, sleeping: bool = False, reason: str = "") -> str:
        spec = self.spec(name)
        h = self.host(spec["host"])
        h.require()
        with self.engine_lock(name) if job else contextlib.nullcontext():
            self.stopping[name] = now()
            self.store.update_engine(name, lambda s: s.__setitem__("desired", "sleeping" if sleeping else "stopped"))
            if job:
                job.step("Stopping container")
            reserve = (self.engine_state.get(name) or {}).get("reserve") or 0
            r = h.dk("stop", "-t", "30", self.cname(name), timeout=90)
            if not r.ok and "No such container" not in r.text:
                raise LabError("docker stop failed", r.text[:300])
            if h.reach == "ssh" and spec.get("port"):
                self.tunnels.close(h.id, int(spec["port"]))
            self.pool.submit(self.refresh, [h.id])
            freed = f" (freed {fmt_bytes(reserve)})" if reserve else ""
            if sleeping:
                self.events.add("info", name, f"{name} went to sleep {reason or 'after idling'}{freed}")
            return f"{'Slept' if sleeping else 'Stopped'} {name}{freed}"

    def op_restart(self, name: str, job: Job) -> str:
        spec = self.spec(name)
        h = self.host(spec["host"])
        h.require()
        job.step("Restarting container")
        r = h.dk("restart", "-t", "20", self.cname(name), timeout=120)
        if not r.ok:
            raise LabError("docker restart failed", r.text[:300], ["recreate"])
        self.store.update_engine(name, lambda s: s.__setitem__("desired", "running"))
        self.boot_seen.pop(name, None)
        self.refresh([h.id])
        return self.wait_ready(name, job)

    def op_remove(self, name: str, job: Job | None = None, delete_spec: bool = False) -> str:
        spec = self.spec(name)
        h = self.host(spec["host"])
        ok, _ = h.reachable()
        if ok:
            self.stopping[name] = now()
            h.dk("rm", "-f", self.cname(name), timeout=90)
        if spec.get("port") and h.reach == "ssh":
            self.tunnels.close(h.id, int(spec["port"]))
        if delete_spec:
            self.store.drop_engine(name)
            with self.lock:
                self.engine_state.pop(name, None)
                self.hist.pop(name, None)
            self.events.add("warn", name, f"Deleted engine {name}")
        else:
            self.store.update_engine(name, lambda s: s.__setitem__("desired", "stopped"))
            self.events.add("info", name, f"Removed container for {name} (config kept)")
        self.schedule_webui_sync(1.0)
        self.pool.submit(self.refresh, [h.id])
        return f"{'Deleted' if delete_spec else 'Removed container for'} {name}"

    # fixes -----------------------------------------------------------------------------
    def op_fix(self, name: str, fix: str, job: Job, params: dict | None = None) -> str:
        params = params or {}
        spec = self.spec(name)
        h = self.host(spec["host"])
        if fix == "gemma-runtime":
            self.build_patched(h, job)
            self.store.update_engine(name, lambda s: s.__setitem__("image", PATCHED_IMAGE))
            return self.op_start(name, job, recreate=True)
        if fix == "trust-remote-code":
            self.store.update_engine(name, lambda s: s.__setitem__("trust_remote_code", True))
            return self.op_start(name, job, recreate=True)
        if fix == "quant-auto":
            self.store.update_engine(name, lambda s: s.__setitem__("quant", ""))
            return self.op_start(name, job, recreate=True)
        if fix in ("set-context", "lower-context"):
            hint = (((self.engine_state.get(name) or {}).get("crash") or {}).get("data") or {}).get("max_len")
            new = as_int(params.get("max_len")) or (int(hint) // 1024 * 1024 if hint else 0) or max(2048, int(spec["max_len"] // 2))
            self.store.update_engine(name, lambda s: s.__setitem__("max_len", new))
            job.log(f"Max context → {new:,}")
            return self.op_start(name, job, recreate=True)
        if fix == "fit-util":
            fit = self.fit_check(spec, h)
            u = as_float(params.get("util")) or (fit.get("suggest") or {}).get("util")
            if not u:
                raise LabError("There is no free memory to fit this engine into", "Stop another engine first.", ["make-room"])
            self.store.update_engine(name, lambda s: s.__setitem__("util", u))
            job.log(f"Memory share → {u:.2f}")
            return self.op_start(name, job, recreate=True)
        if fix == "make-room":
            return self.op_start(name, job, on_conflict="evict", evict=params.get("evict"), recreate=True)
        if fix == "force":
            return self.op_start(name, job, on_conflict="force", recreate=True)
        if fix == "flush-cache":
            ok, msg = self.flush_cache(h)
            if not ok:
                raise LabError("Could not flush the page cache", msg)
            return self.op_start(name, job, recreate=True)
        if fix == "change-port":
            p = self.allocate_port(h.id, exclude=name)
            self.store.update_engine(name, lambda s: s.__setitem__("port", p))
            job.log(f"Port → {p}")
            return self.op_start(name, job, recreate=True)
        if fix == "to-llamacpp":
            self.store.update_engine(name, lambda s: s.update(backend="llamacpp", image=""))
            return self.op_start(name, job, recreate=True)
        if fix == "offline":
            self.store.update_engine(name, lambda s: s.__setitem__("env", {**(s.get("env") or {}), "HF_HUB_OFFLINE": "1"}))
            return self.op_start(name, job, recreate=True)
        if fix in ("recreate", "restart"):
            return self.op_start(name, job, recreate=True)
        raise LabError(f"Unknown fix '{fix}'")

    # logs, details ---------------------------------------------------------------
    def logs(self, name: str, tail: int = 400, since: str = "") -> dict:
        spec = self.spec(name)
        h = self.host(spec["host"])
        h.require()
        args = ["logs", "--tail", str(max(10, min(5000, tail))), "--timestamps"]
        if since:
            args += ["--since", since]
        r = h.dk_merged(*args, self.cname(name), timeout=30)
        text = r.out or ""
        if not r.ok and "No such container" in text:
            return {"text": "", "exists": False}
        return {"text": text[-400_000:], "exists": True}

    def follow_logs(self, name: str, tail: int = 200):
        """Generator of log lines (for the CLI `logs -f` and the live log view)."""
        spec = self.spec(name)
        h = self.host(spec["host"])
        argv = h.argv(h.docker() + ["logs", "-f", "--tail", str(tail), self.cname(name)], tty=True)
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            assert p.stdout is not None
            for raw in iter(p.stdout.readline, b""):
                yield raw.decode("utf-8", "replace").rstrip("\r\n")
        finally:
            with contextlib.suppress(Exception):
                p.kill()

    def run_command(self, name: str) -> dict:
        spec = self.spec(name)
        h = self.host(spec["host"])
        conf = self.conf()
        image = resolve_image(spec, conf)
        argv = docker_run_argv(spec, h, conf, image, (h.state_dir().rstrip("/") + "/engine.env") if self.hf_token() else None)
        _, dropped = split_extra(spec["extra"])
        return {"argv": h.docker() + argv, "shell": " ".join(shlex.quote(a) for a in h.docker() + argv),
                "dropped": dropped, "image": image, "fingerprint": spec_fingerprint(spec, image, h)}

    def preview(self, data: dict) -> dict:
        spec = normalize_spec({**data, "name": data.get("name") or model_slug(data.get("model") or "") or "preview"})
        errs = validate_spec(spec)
        try:
            h = self.host(spec["host"])
        except LabError as e:
            return {"errors": [e.msg]}
        conf = self.conf()
        if not spec.get("port"):
            with contextlib.suppress(LabError):
                spec["port"] = self.allocate_port(h.id, exclude=spec["name"])
        image = resolve_image(spec, conf)
        argv = docker_run_argv(spec, h, conf, image, (h.state_dir().rstrip("/") + "/engine.env") if self.hf_token() else None)
        _, dropped = split_extra(spec["extra"])
        pol = origin_check(spec["model"], conf)
        fit = self.fit_check(spec, h) if spec["backend"] == "vllm" else {}
        return {"shell": " ".join(shlex.quote(a) for a in h.docker() + argv), "errors": errs, "dropped": dropped,
                "port": spec["port"], "name": spec["name"], "policy": pol, "fit": fit, "human_url": h.human_url(spec["port"]),
                "docker_url": f"http://{self.cname(spec['name'])}:8000/v1", "image_present": None}

    # adopt a container that was started outside the manager -------------------------
    def adopt(self, hid: str, container: str) -> dict:
        if not CONTAINER_RE.match(container):
            raise LabError("Bad container name")
        h = self.host(hid)
        obj = self.inspect_one(h, container)
        if not obj:
            raise LabError(f"No container {container} on {h.label}", code=404)
        cfg = obj.get("Config") or {}
        cmd = list(cfg.get("Cmd") or [])
        image = cfg.get("Image") or ""
        prefix = self.conf().get("container_prefix") or ""
        name = container[len(prefix):] if prefix and container.startswith(prefix) else container
        name = re.sub(r"[^a-z0-9_.-]", "-", name.lower())[:40] or "adopted"
        spec: dict = {"name": name, "host": hid, "image": image, "desired": "running" if (obj.get("State") or {}).get("Running") else "stopped"}
        backend = "llamacpp" if "llama" in image and "vllm" not in image else "vllm"
        spec["backend"] = backend
        rest: list[str] = []
        i = 0
        flags_with_val = {"--gpu-memory-utilization": "util", "--max-model-len": "max_len", "--kv-cache-dtype": "kv_dtype",
                          "--quantization": "quant", "--served-model-name": "served_name", "--max-num-seqs": "max_num_seqs",
                          "--model": "model", "-hf": "model", "-c": "max_len", "--alias": "served_name", "-np": "max_num_seqs"}
        skip = {"--host", "--port", "-ngl", "--metrics", "--jinja"}
        while i < len(cmd):
            tok = cmd[i]
            key, eq, val = tok.partition("=")
            if key in flags_with_val:
                v = val if eq else (cmd[i + 1] if i + 1 < len(cmd) else "")
                spec[flags_with_val[key]] = v
                i += 1 if eq else 2
                continue
            if key in skip:
                i += 1 if (eq or key in ("--metrics", "--jinja")) else 2
                continue
            if key == "--trust-remote-code":
                spec["trust_remote_code"] = True
                i += 1
                continue
            if not tok.startswith("-") and "model" not in spec and "/" in tok:
                spec["model"] = tok
                i += 1
                continue
            rest.append(tok)
            i += 1
        spec["extra"] = " ".join(shlex.quote(x) for x in rest)
        ports = ((obj.get("NetworkSettings") or {}).get("Ports") or {}).get("8000/tcp") or []
        if ports:
            spec["port"] = as_int(ports[0].get("HostPort"))
        if image == (self.conf().get("vllm_image")) or image == (self.conf().get("llamacpp_image")):
            spec["image"] = ""
        spec.setdefault("kv_dtype", "auto")
        if not spec.get("model"):
            raise LabError("Could not find a model id in that container's command", "Create the engine by hand instead.")
        spec["note"] = f"Adopted from {container}"
        if name in self.store.engines():
            raise LabError(f"An engine named {name} already exists", code=409)
        s = normalize_spec({**spec, "created": now()})
        if container != self.cname(s["name"]):
            raise LabError(f"Adopting needs the container to be named {self.cname(s['name'])}",
                           "Rename the container (docker rename) or change the container prefix in Settings.")
        self.store.put_engine(s)
        self.events.add("info", s["name"], f"Adopted {container} as engine {s['name']}")
        return s

    # probes & benchmarks -----------------------------------------------------------
    def engine_base(self, name: str) -> tuple[str, dict]:
        spec = self.spec(name)
        h = self.host(spec["host"])
        h.require()
        base = h.api_base(int(spec["port"])) if spec.get("port") else ""
        if not base:
            raise LabError(f"No route to {name}", "The SSH tunnel is down.", ["host-test"])
        return base, spec

    def probe_chat(self, name: str, prompt: str = "Reply with the single word OK.") -> dict:
        base, spec = self.engine_base(name)
        model = self._engine_model_id(base) or served_id(spec)
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0, "stream": True,
                "stream_options": {"include_usage": True}}
        t0 = now()
        ttft = None
        text = ""
        usage = {}
        try:
            for ev in stream_openai(base + "/v1/chat/completions", body, timeout=120):
                if ev.get("choices"):
                    d = (ev["choices"][0].get("delta") or {})
                    piece = d.get("content") or d.get("reasoning_content") or d.get("reasoning") or ""
                    if piece and ttft is None:
                        ttft = now() - t0
                    text += d.get("content") or ""
                if ev.get("usage"):
                    usage = ev["usage"]
        except LabError as e:
            return {"ok": False, "error": e.msg, "hint": e.hint, "model": model, "url": base + "/v1/chat/completions"}
        dt = now() - t0
        self.activity[name] = now()
        return {"ok": True, "model": model, "text": text.strip()[:400], "ttft": ttft, "total": dt, "usage": usage,
                "url": base + "/v1/chat/completions"}

    def _engine_model_id(self, base: str) -> str:
        code, body = http_text(base + "/v1/models", timeout=5)
        if code == 200:
            with contextlib.suppress(Exception):
                data = json.loads(body).get("data") or []
                if data:
                    return data[0].get("id") or ""
        return ""

    def op_bench(self, name: str, job: Job, concurrency: int = 4, requests: int = 16, max_tokens: int = 256,
                 prompt: str = "", prompt_tokens: int = 0) -> str:
        base, spec = self.engine_base(name)
        model = self._engine_model_id(base) or served_id(spec)
        concurrency = max(1, min(128, int(concurrency)))
        requests = max(1, min(1000, int(requests)))
        max_tokens = max(8, min(8192, int(max_tokens)))
        prompt = prompt or "Write a detailed, well-structured technical explanation of how a transformer language model generates text, step by step."
        if prompt_tokens:
            filler = ("The quick brown fox jumps over the lazy dog near the riverbank. " * (prompt_tokens // 12 + 1))
            prompt = filler[: prompt_tokens * 4] + "\n\n" + prompt
        results: list[dict] = []
        lock = threading.Lock()
        idx = {"n": 0}
        t_start = now()

        def worker():
            while True:
                job.check()
                with lock:
                    if idx["n"] >= requests:
                        return
                    idx["n"] += 1
                body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                        "temperature": 0.7, "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True}
                t0 = now()
                ttft = None
                toks = 0
                usage = {}
                err = ""
                try:
                    for ev in stream_openai(base + "/v1/chat/completions", body, timeout=600):
                        if ev.get("choices"):
                            d = ev["choices"][0].get("delta") or {}
                            if (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                                toks += 1
                                if ttft is None:
                                    ttft = now() - t0
                        if ev.get("usage"):
                            usage = ev["usage"]
                except LabError as e:
                    err = e.msg
                t1 = now()
                out_toks = (usage.get("completion_tokens") or toks)
                with lock:
                    results.append({"ttft": ttft, "dur": t1 - t0, "out": out_toks, "in": usage.get("prompt_tokens"), "err": err, "end": t1})
                    done = len(results)
                job.set_progress(done / requests, f"{done}/{requests} requests")

        job.step(f"Benchmark {name}: {requests} requests, {concurrency} at a time, {max_tokens} tokens each", 0)
        threads = [threading.Thread(target=worker, daemon=True) for _ in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = now() - t_start
        ok = [r for r in results if not r["err"]]
        errs = [r for r in results if r["err"]]
        if not ok:
            raise LabError(f"Every request failed: {errs[0]['err'] if errs else 'no results'}", "", ["logs"])
        ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)
        per = sorted((r["out"] / max(0.001, r["dur"] - (r["ttft"] or 0))) for r in ok if r["out"])

        def pct(arr, p):
            if not arr:
                return None
            k = (len(arr) - 1) * p
            f, c = math.floor(k), math.ceil(k)
            return arr[f] if f == c else arr[f] + (arr[c] - arr[f]) * (k - f)
        total_out = sum(r["out"] for r in ok)
        res = {
            "engine": name, "model": model, "t": now(), "concurrency": concurrency, "requests": requests, "max_tokens": max_tokens,
            "ok": len(ok), "errors": len(errs), "wall": wall, "out_tokens": total_out, "agg_tps": total_out / wall if wall else 0,
            "ttft_p50": pct(ttfts, 0.5), "ttft_p95": pct(ttfts, 0.95), "tps_p50": pct(per, 0.5), "tps_min": per[0] if per else None,
            "prompt_tokens": ok[0].get("in"), "host": spec["host"], "util": spec["util"], "image": resolve_image(spec, self.conf()),
        }
        job.data = res
        with contextlib.suppress(OSError):
            with open(BENCH_FILE, "a") as fh:
                fh.write(json.dumps(res) + "\n")
        self.activity[name] = now()
        self.events.add("ok", name, f"Bench {name}: {res['agg_tps']:.0f} tok/s aggregate at {concurrency}×, TTFT p50 {res['ttft_p50'] or 0:.2f}s")
        return f"{res['agg_tps']:.0f} tok/s aggregate · {res['tps_p50'] or 0:.1f} tok/s per stream · TTFT p50 {res['ttft_p50'] or 0:.2f}s"

    def bench_history(self, name: str = "") -> list[dict]:
        out = []
        with contextlib.suppress(OSError):
            for ln in BENCH_FILE.read_text().splitlines()[-500:]:
                with contextlib.suppress(json.JSONDecodeError):
                    r = json.loads(ln)
                    if not name or r.get("engine") == name:
                        out.append(r)
        return out[-100:]

    # ollama ----------------------------------------------------------------------
    def ollama_op(self, hid: str, op: str, model: str, job: Job | None = None) -> str:
        h = self.host(hid)
        h.require()
        if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.:/\-]{0,120}$", model or ""):
            raise LabError("Bad Ollama model name")
        pol = origin_check(ollama_origin(model), self.conf())
        if op in ("pull", "load") and pol["blocked"]:
            raise LabError(f"{model} is blocked by the origin policy", "", ["policy"], code=403)
        base = h.api_base(11434)
        if not base:
            raise LabError("No route to Ollama on that host")
        if op == "pull":
            req = Request(base + "/api/pull", data=json.dumps({"name": model, "stream": True}).encode(),
                          headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urlopen(req, timeout=7200) as r:
                    for raw in r:
                        if job:
                            job.check()
                        with contextlib.suppress(json.JSONDecodeError):
                            ev = json.loads(raw)
                            if ev.get("error"):
                                raise LabError(f"Ollama: {ev['error']}")
                            if job:
                                tot, comp = ev.get("total"), ev.get("completed")
                                job.set_progress((comp / tot) if tot and comp else None, ev.get("status", ""))
            except (URLError, OSError) as e:
                raise LabError(f"Ollama pull failed: {e}")
            self.events.add("ok", f"ollama@{hid}", f"Pulled {model} into Ollama on {h.label}")
            return f"pulled {model}"
        if op in ("load", "unload"):
            body = {"model": model, "keep_alive": "30m" if op == "load" else 0}
            try:
                http_json(base + "/api/generate", "POST", body, timeout=600)
            except Exception as e:  # noqa: BLE001
                raise LabError(f"Ollama {op} failed: {e}")
            return f"{op}ed {model}"
        if op == "delete":
            try:
                http_json(base + "/api/delete", "DELETE", {"name": model}, timeout=60)
            except Exception as e:  # noqa: BLE001
                raise LabError(f"Ollama delete failed: {e}")
            self.events.add("warn", f"ollama@{hid}", f"Deleted {model} from Ollama on {h.label}")
            return f"deleted {model}"
        raise LabError("Unknown Ollama action")

    # raw docker (containers page) -------------------------------------------------
    def docker_op(self, hid: str, op: str, target: str) -> dict:
        h = self.host(hid)
        h.require()
        if op in ("start", "stop", "restart", "rm", "logs", "inspect", "pause", "unpause"):
            if not CONTAINER_RE.match(target or ""):
                raise LabError("Bad container name")
            webui_c = (self.conf().get("webui") or {}).get("container")
            if op == "logs":
                r = h.dk("logs", "--tail", "500", "--timestamps", target, timeout=30)
                return {"text": (r.out + r.err)[-300_000:]}
            if op == "inspect":
                r = h.dk("inspect", target, timeout=20)
                if not r.ok:
                    raise LabError(r.text[:300])
                return {"json": json.loads(r.out)[0]}
            args = [op, target] if op != "rm" else ["rm", "-f", target]
            if op == "stop":
                args = ["stop", "-t", "20", target]
            r = h.dk(*args, timeout=120)
            if not r.ok:
                raise LabError(f"docker {op} failed", r.text[:400])
            for n in self.store.engines():
                if self.cname(n) == target and op in ("stop", "rm"):
                    self.stopping[n] = now()
                    self.store.update_engine(n, lambda s: s.__setitem__("desired", "stopped"))
            if target == webui_c:
                self.events.add("info", "webui", f"Open WebUI container: {op}")
            self.pool.submit(self.refresh, [hid])
            return {"msg": f"{op} {target}"}
        if op == "rmi":
            if not IMAGE_RE.match(target or ""):
                raise LabError("Bad image name")
            r = h.dk("rmi", target, timeout=120)
            if not r.ok:
                raise LabError("Could not remove image", r.text[:300])
            return {"msg": f"removed {target}"}
        if op == "prune-images":
            r = h.dk("image", "prune", "-f", timeout=300)
            return {"msg": (r.out.strip().splitlines() or ["pruned"])[-1]}
        raise LabError(f"Unknown docker action '{op}'")

    def docker_fleet(self, hid: str) -> dict:
        h = self.host(hid)
        h.require()
        ok, msg = h.docker_access()
        if not ok:
            return {"containers": [], "images": [], "networks": [], "error": msg}
        imgs = []
        r = h.dk("images", "--format", "{{json .}}", timeout=30)
        used = {c["image"] for c in self.containers.get(hid, [])}
        for ln in r.out.splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                o = json.loads(ln)
                ref = f"{o.get('Repository')}:{o.get('Tag')}"
                imgs.append({"ref": ref, "id": o.get("ID"), "size": o.get("Size"), "created": o.get("CreatedSince"),
                             "used": ref in used or o.get("ID") in used, "dangling": o.get("Repository") == "<none>"})
        nets = []
        r = h.dk("network", "ls", "--format", "{{json .}}", timeout=20)
        for ln in r.out.splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                o = json.loads(ln)
                nets.append({"name": o.get("Name"), "driver": o.get("Driver"), "scope": o.get("Scope")})
        stats = self.stats.get(hid) or {}
        conts = []
        eng_by_c = {self.cname(n): n for n in self.store.engines()}
        for c in self.containers.get(hid, []):
            st = stats.get(c["name"]) or {}
            conts.append({**c, "cpu": st.get("cpu"), "mem": st.get("mem"), "engine": eng_by_c.get(c["name"]) or c["labels"].get("vllm-lab.engine"),
                          "download": c["labels"].get("vllm-lab.download")})
        return {"containers": conts, "images": imgs, "networks": nets, "error": ""}


# ───────────────────────────────────────────────────────────────── OpenAI streaming client

def _conn_for(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    u = urlparse(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    return conn, (u.path or "/") + (f"?{u.query}" if u.query else "")


def stream_openai(url: str, body: dict, timeout: float = 300, headers: dict | None = None):
    """Yield parsed SSE JSON events from an OpenAI-compatible streaming endpoint."""
    try:
        conn, path = _conn_for(url, timeout)
        conn.request("POST", path, body=json.dumps(body), headers={"Content-Type": "application/json", "Accept": "text/event-stream", **(headers or {})})
        resp = conn.getresponse()
    except (OSError, http.client.HTTPException) as e:
        raise LabError(f"Could not connect: {e}")
    if resp.status != 200:
        raw = resp.read(4000).decode("utf-8", "replace")
        msg = raw
        with contextlib.suppress(Exception):
            j = json.loads(raw)
            msg = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else (j.get("error") or j.get("message") or raw)
        conn.close()
        raise LabError(f"HTTP {resp.status}: {str(msg)[:300]}")
    try:
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            with contextlib.suppress(json.JSONDecodeError):
                yield json.loads(data)
    except (OSError, http.client.HTTPException) as e:
        raise LabError(f"Stream broke: {e}")
    finally:
        conn.close()


# ───────────────────────────────────────────────────────────────── Hugging Face, planner, library

QUANT_WORDS = ("nvfp4", "fp4", "mxfp4", "fp8", "int8", "w8a8", "w4a16", "awq", "gptq", "gguf", "bf16", "fp16", "modelopt", "exl2", "bnb")


def kv_geometry(cfg: dict) -> dict:
    tc = cfg.get("text_config") or cfg.get("llm_config") or cfg.get("language_config") or cfg
    if not isinstance(tc, dict):
        tc = cfg

    def g(*keys, default=None):
        for k in keys:
            if tc.get(k) is not None:
                return tc[k]
            if cfg.get(k) is not None:
                return cfg[k]
        return default
    layers = int(g("num_hidden_layers", "n_layer", "num_layers", default=0) or 0)
    heads = int(g("num_attention_heads", "n_head", default=0) or 0)
    kvh = int(g("num_key_value_heads", "n_head_kv", default=heads) or heads or 0)
    hidden = int(g("hidden_size", "n_embd", "d_model", default=0) or 0)
    hd = int(g("head_dim", default=0) or (hidden // heads if heads else 0))
    full, sliding = layers, 0
    mamba_layers = 0
    pattern = g("hybrid_override_pattern")
    if isinstance(pattern, str) and pattern:
        full = pattern.count("*")
        mamba_layers = pattern.count("M")
    lt = g("layer_types")
    if isinstance(lt, list) and lt:
        full = sum(1 for x in lt if "full" in str(x))
        sliding = sum(1 for x in lt if "sliding" in str(x))
        mamba_layers = mamba_layers or sum(1 for x in lt if "mamba" in str(x) or "linear" in str(x))
    shared = int(g("num_kv_shared_layers", default=0) or 0)
    if shared and (full + sliding):
        ratio = max(0.0, 1 - shared / max(1, layers))
        full = round(full * ratio)
        sliding = round(sliding * ratio)
    sw = int(g("sliding_window", default=0) or 0)
    if sliding and not sw:
        sw = 4096
    if not sliding and sw and g("use_sliding_window", default=False):
        sliding, full = full, 0
    mamba_bytes = 0
    if mamba_layers:
        mh = int(g("mamba_num_heads", default=0) or 0)
        md = int(g("mamba_head_dim", default=0) or 0)
        ds = int(g("ssm_state_size", "mamba_d_state", default=128) or 128)
        ng = int(g("n_groups", "mamba_n_groups", default=8) or 8)
        ck = int(g("conv_kernel", "mamba_d_conv", default=4) or 4)
        inner = mh * md if mh and md else int(g("mamba_expand", default=2) or 2) * hidden
        mamba_bytes = mamba_layers * (inner * ds * 2 + (inner + 2 * ng * ds) * ck * 2)
    dtype = str(g("torch_dtype", "dtype", default="bfloat16"))
    qc = cfg.get("quantization_config") or tc.get("quantization_config") or {}
    ctx = int(g("max_position_embeddings", "max_seq_len", "seq_length", "n_positions", default=0) or 0)
    rs = g("rope_scaling") or {}
    if isinstance(rs, dict) and rs.get("original_max_position_embeddings") and rs.get("factor"):
        ctx = max(ctx, int(rs["original_max_position_embeddings"] * rs["factor"]))
    return {
        "layers": layers, "attn_full": full, "attn_sliding": sliding, "kv_heads": kvh, "head_dim": hd, "sliding_window": sw,
        "mamba_layers": mamba_layers, "mamba_bytes_per_seq": mamba_bytes, "dtype": dtype,
        "kv_elems_full": 2 * full * kvh * hd, "kv_elems_sliding": 2 * sliding * kvh * hd,
        "ctx_max": ctx, "quant_method": (qc or {}).get("quant_method") or "", "arch": (cfg.get("architectures") or [""])[0],
        "model_type": cfg.get("model_type") or tc.get("model_type") or "", "known": bool(layers and kvh and hd),
    }


class IntelMixin:
    # hub plumbing ------------------------------------------------------------------
    def hf_base(self) -> str:
        return (self.conf().get("hf_endpoint") or os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")

    def hf_headers(self) -> dict:
        tok = self.hf_token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    def hf_get(self, path: str, timeout: float = 15):
        try:
            return http_json(self.hf_base() + path, headers=self.hf_headers(), timeout=timeout)
        except HTTPError as e:
            if e.code in (401, 403):
                raise LabError("Hugging Face refused the request", "Gated or private — check the token and that you accepted the model's terms.", ["set-token", "open-hf"], code=e.code)
            if e.code == 404:
                raise LabError("Not found on Hugging Face", code=404)
            raise LabError(f"Hugging Face returned HTTP {e.code}")
        except (URLError, OSError, socket.timeout) as e:
            raise LabError("Hugging Face is not reachable from this box", f"{e}. Search works offline only for weights already in the cache.", code=503)

    def hf_whoami(self) -> dict:
        if not self.hf_token():
            return {"ok": False, "why": "no token"}
        try:
            d = self.hf_get("/api/whoami-v2", timeout=8)
            return {"ok": True, "name": d.get("name"), "orgs": [o.get("name") for o in (d.get("orgs") or [])],
                    "role": ((d.get("auth") or {}).get("accessToken") or {}).get("role")}
        except LabError as e:
            return {"ok": False, "why": e.msg}

    def hf_search(self, q: str, hid: str = "", kind: str = "", limit: int = 30, sort: str = "downloads") -> list[dict]:
        params = [("search", q or ""), ("sort", sort if sort in ("downloads", "likes", "lastModified", "trendingScore") else "downloads"),
                  ("direction", "-1"), ("limit", str(max(1, min(60, limit))))]
        for ex in ("downloads", "likes", "gated", "pipeline_tag", "tags", "safetensors", "lastModified", "library_name"):
            params.append(("expand[]", ex))
        if kind == "gguf":
            params.append(("filter", "gguf"))
        elif kind == "text":
            params.append(("pipeline_tag", "text-generation"))
        listing = self.hf_get("/api/models?" + urlencode(params), timeout=20)
        conf = self.conf()
        cached = self._cached_models(hid) if hid else set()
        rows = []
        for m in listing or []:
            mid = m.get("id") or m.get("modelId") or ""
            tags = [str(t) for t in (m.get("tags") or [])]
            blob = (mid + " " + " ".join(tags)).lower()
            quants = [k for k in QUANT_WORDS if k in blob]
            quants = [k for k in quants if not any(k != o and k in o for o in quants)]
            bases = [t.split(":", 1)[1] for t in tags if t.startswith("base_model:") and t.count(":") == 1]
            bases += [t.split(":", 2)[2] for t in tags if t.startswith("base_model:") and t.count(":") == 2]
            st = m.get("safetensors") or {}
            rows.append({
                "id": mid, "downloads": m.get("downloads") or 0, "likes": m.get("likes") or 0,
                "params": st.get("total") if isinstance(st, dict) else None, "gated": bool(m.get("gated")),
                "pipeline": m.get("pipeline_tag") or "", "quant": quants, "gguf": "gguf" in quants,
                "modified": m.get("lastModified") or "", "policy": origin_check(mid, conf, bases),
                "base": bases[:3], "cached": mid in cached, "library": m.get("library_name") or "",
            })
        return rows

    def hf_meta(self, model: str, hid: str = "", refresh: bool = False) -> dict:
        model = model.strip()
        if not MODEL_RE.match(model):
            raise LabError("Not a model id")
        repo = model.split(":")[0]
        hit = self.hf_meta_cache.get(repo)
        if hit and not refresh and now() - hit[0] < 6 * 3600:
            return hit[1]
        disk = read_json(HFCACHE_FILE, {})
        if not refresh and isinstance(disk, dict) and repo in disk and now() - disk[repo].get("_t", 0) < 86400:
            self.hf_meta_cache[repo] = (now(), disk[repo])
            return disk[repo]
        meta: dict = {"id": repo, "source": "hub"}
        try:
            info = self.hf_get(f"/api/models/{repo}", timeout=15)
            meta.update(self._meta_from_info(info))
            tree = []
            with contextlib.suppress(LabError):
                tree = self.hf_get(f"/api/models/{repo}/tree/main?recursive=true", timeout=20) or []
            files = [f for f in tree if isinstance(f, dict) and f.get("type") == "file"]
            meta.update(self._meta_from_files(files))
            cfg = {}
            if not meta.get("gguf") or meta.get("weights_bytes"):
                with contextlib.suppress(Exception):
                    cfg = http_json(f"{self.hf_base()}/{repo}/resolve/main/config.json", headers=self.hf_headers(), timeout=12)
            if cfg:
                meta["kv"] = kv_geometry(cfg)
        except LabError as e:
            meta = self.local_meta(repo, hid) or {"id": repo, "source": "none", "error": e.msg, "hint": e.hint}
        meta["policy"] = origin_check(repo, self.conf(), meta.get("base") or [])
        meta["_t"] = now()
        self.hf_meta_cache[repo] = (now(), meta)
        if meta.get("source") == "hub":
            disk = read_json(HFCACHE_FILE, {})
            if not isinstance(disk, dict):
                disk = {}
            disk[repo] = meta
            if len(disk) > 300:
                for k in sorted(disk, key=lambda k: disk[k].get("_t", 0))[:-300]:
                    disk.pop(k, None)
            with contextlib.suppress(OSError):
                atomic_write(HFCACHE_FILE, json.dumps(disk))
        return meta

    @staticmethod
    def _meta_from_info(info: dict) -> dict:
        st = info.get("safetensors") or {}
        tags = [str(t) for t in (info.get("tags") or [])]
        card = info.get("cardData") or {}
        bases = card.get("base_model") or []
        if isinstance(bases, str):
            bases = [bases]
        bases += [t.split(":")[-1] for t in tags if t.startswith("base_model:")]
        cfgi = info.get("config") or {}
        return {
            "params": st.get("total") if isinstance(st, dict) else None,
            "param_dtypes": st.get("parameters") if isinstance(st, dict) else None,
            "gated": info.get("gated") or False, "private": bool(info.get("private")),
            "license": card.get("license") or next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), ""),
            "pipeline": info.get("pipeline_tag") or "", "downloads": info.get("downloads"), "likes": info.get("likes"),
            "modified": info.get("lastModified"), "tags": tags[:40], "base": list(dict.fromkeys(b for b in bases if b))[:5],
            "arch": (cfgi.get("architectures") or [""])[0], "model_type": cfgi.get("model_type") or "",
            "author": info.get("author") or "",
        }

    @staticmethod
    def _meta_from_files(files: list[dict]) -> dict:
        def size(f):
            return int((f.get("lfs") or {}).get("size") or f.get("size") or 0)
        st = [f for f in files if f.get("path", "").endswith(".safetensors")]
        bins = [f for f in files if re.search(r"\.(bin|pt|pth)$", f.get("path", "")) and "training_args" not in f["path"]]
        ggufs = [f for f in files if f.get("path", "").endswith(".gguf")]
        out: dict = {"files": len(files), "repo_bytes": sum(size(f) for f in files)}
        if st:
            out["weights_bytes"] = sum(size(f) for f in st)
            out["format"] = "safetensors"
        elif bins:
            out["weights_bytes"] = sum(size(f) for f in bins)
            out["format"] = "bin"
        if ggufs:
            groups: dict[str, int] = {}
            for f in ggufs:
                p = f["path"].split("/")[-1]
                if "mmproj" in p.lower():
                    continue
                m = re.search(r"[-._]((?:I?Q\d[\w]*|F16|BF16|F32|MXFP4\w*|FP8\w*))(?:-\d{5}-of-\d{5})?\.gguf$", p, re.I)
                q = (m.group(1) if m else p.rsplit(".", 1)[0]).upper()
                groups[q] = groups.get(q, 0) + size(f)
            out["gguf"] = [{"quant": q, "bytes": b} for q, b in sorted(groups.items(), key=lambda kv: kv[1])]
            pick = next((g for g in out["gguf"] if g["quant"].startswith("Q4_K_M")), None) or (out["gguf"][len(out["gguf"]) // 2] if out["gguf"] else None)
            if pick:
                out["gguf_default"] = pick["quant"]
                out["gguf_bytes"] = pick["bytes"]
            if not st and not bins:
                out["format"] = "gguf"
        return out

    def local_meta(self, repo: str, hid: str = "") -> dict | None:
        hosts = [self.host(hid)] if hid else [h for h in self.hosts() if h.reachable()[0]]
        for h in hosts:
            folder = h.cache_dir().rstrip("/") + "/hub/" + hf_cache_folder(repo)
            r = h.sh(f"d={shlex.quote(folder)}; [ -d \"$d\" ] || exit 3; du -sbL \"$d/blobs\" 2>/dev/null | cut -f1; "
                     f"echo @@cfg; cat \"$(ls -d \"$d\"/snapshots/*/ 2>/dev/null | head -n1)config.json\" 2>/dev/null", timeout=20)
            if r.code == 3 or not r.out.strip():
                continue
            size_s, _, cfg_s = r.out.partition("@@cfg")
            meta = {"id": repo, "source": "local", "host": h.id, "weights_bytes": as_int(size_s.strip().splitlines()[0] if size_s.strip() else 0, 0)}
            with contextlib.suppress(json.JSONDecodeError):
                cfg = json.loads(cfg_s.strip() or "{}")
                if cfg:
                    meta["kv"] = kv_geometry(cfg)
                    meta["arch"] = meta["kv"]["arch"]
            return meta
        return None

    def plan(self, model: str, hid: str, extra: str = "") -> dict:
        meta = self.hf_meta(model, hid)
        drafts = []
        for m in extra_models(extra):
            with contextlib.suppress(LabError):
                dm = self.hf_meta(m, hid)
                drafts.append({"id": m, "bytes": dm.get("weights_bytes") or 0})
        h = self.host(hid)
        hs = self.host_state.get(h.id) or {}
        mem = hs.get("mem") or {}
        total = int(mem.get("gpu_total") or mem.get("total") or 0)
        others = [{"name": n, "reserve": e.get("reserve") or 0, "state": e.get("state"), "backend": e.get("backend")}
                  for n, e in self.engine_state.items() if e.get("host") == hid and e.get("state") in ("ready", "booting")]
        cached = model.split(":")[0] in self._cached_models(hid)
        return {"meta": meta, "drafts": drafts, "host": hid, "total": total, "free": mem.get("free"), "available": mem.get("available"),
                "cached_bytes": mem.get("cached"), "uma": mem.get("uma"), "others": others, "headroom": int(float(self.conf().get("headroom_gib") or 4) * GIB),
                "on_disk": cached, "overhead": int(2.5 * GIB)}

    # library ----------------------------------------------------------------------
    def _cached_models(self, hid: str) -> set[str]:
        lib = getattr(self, "_lib_cache", {}).get(hid)
        if lib and now() - lib[0] < 60:
            return {w["model"] for w in lib[1]}
        try:
            return {w["model"] for w in self.scan_weights(hid)}
        except LabError:
            return set()

    def scan_weights(self, hid: str) -> list[dict]:
        h = self.host(hid)
        h.require()
        hub = h.cache_dir().rstrip("/") + "/hub"
        script = (f"cd {shlex.quote(hub)} 2>/dev/null || exit 0; for d in models--*; do [ -d \"$d\" ] || continue; "
                  "s=$(du -sbL \"$d/blobs\" 2>/dev/null | cut -f1); i=$(find \"$d/blobs\" -name '*.incomplete' 2>/dev/null | wc -l); "
                  "m=$(stat -c %Y \"$d\" 2>/dev/null); printf '%s\\t%s\\t%s\\t%s\\n' \"$d\" \"${s:-0}\" \"$i\" \"${m:-0}\"; done")
        r = h.sh(script, timeout=120)
        out = []
        eng = self.store.engines()
        for ln in r.out.splitlines():
            parts = ln.split("\t")
            if len(parts) != 4 or not parts[0].startswith("models--"):
                continue
            model = parts[0][len("models--"):].replace("--", "/", 1)
            used = [n for n, s in eng.items() if s["host"] == hid and (s["model"].split(":")[0] == model or model in extra_models(s["extra"]))]
            running = [n for n in used if (self.engine_state.get(n) or {}).get("state") in ("ready", "booting")]
            out.append({"model": model, "folder": parts[0], "bytes": as_int(parts[1], 0), "incomplete": as_int(parts[2], 0),
                        "modified": as_int(parts[3], 0), "used_by": used, "running": running,
                        "policy": origin_check(model, self.conf())})
        out.sort(key=lambda w: -w["bytes"])
        if not hasattr(self, "_lib_cache"):
            self._lib_cache = {}
        self._lib_cache[hid] = (now(), out)
        return out

    def scan_gguf(self, hid: str) -> list[dict]:
        h = self.host(hid)
        d = h.cache_dir().rstrip("/") + "/llama.cpp"
        r = h.sh(f"cd {shlex.quote(d)} 2>/dev/null || exit 0; for f in *.gguf; do [ -f \"$f\" ] && printf '%s\\t%s\\n' \"$f\" $(stat -c %s \"$f\"); done", timeout=30)
        return [{"file": p[0], "bytes": as_int(p[1], 0)} for p in (ln.split("\t") for ln in r.out.splitlines()) if len(p) == 2]

    def library(self, hid: str) -> dict:
        h = self.host(hid)
        weights = self.scan_weights(hid)
        hs = self.host_state.get(hid) or {}
        return {"host": hid, "cache": h.cache_dir(), "weights": weights, "gguf": self.scan_gguf(hid), "disk": hs.get("disk"),
                "downloads": hs.get("downloads") or [], "dl_sizes": hs.get("dl_sizes") or {}, "ollama": self.ollama.get(hid) or {}}

    def op_download(self, model: str, hid: str, job: Job) -> str:
        if not MODEL_RE.match(model) or ":" in model:
            raise LabError("Not a Hugging Face model id")
        conf = self.conf()
        pol = origin_check(model, conf)
        h = self.host(hid)
        h.require()
        meta = {}
        with contextlib.suppress(LabError):
            meta = self.hf_meta(model, hid)
        pol = origin_check(model, conf, meta.get("base") or [])
        if pol["blocked"]:
            raise LabError(f"{pol['via']} is blocked by the origin policy", "", ["policy"], code=403)
        total = int(meta.get("weights_bytes") or meta.get("repo_bytes") or 0)
        image = conf.get("vllm_image") or LEGACY_IMAGE
        job.step(f"Preparing download of {model}")
        image = self.ensure_image(h, image, job)
        env_file = self.ensure_env_file(h)
        cname = f"{conf.get('container_prefix') or ''}dl-{model_slug(model)}"[:60]
        h.dk("rm", "-f", cname, timeout=30)
        patterns = None
        if meta.get("format") == "safetensors":
            patterns = ["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "*.tiktoken", "*.jinja", "tokenizer*", "*.md"]
        code = ("import os,sys\nfrom huggingface_hub import snapshot_download\n"
                "p=os.environ.get('VL_PATTERNS')\n"
                "snapshot_download(os.environ['VL_MODEL'], allow_patterns=(p.split('|') if p else None))\nprint('DONE')\n")
        argv = ["run", "-d", "--name", cname, "--label", f"vllm-lab.download={model}", "--label", f"vllm-lab.bytes={total}",
                "-v", f"{h.cache_dir()}:/root/.cache/huggingface", "-e", f"VL_MODEL={model}", "-e", "HF_HUB_ENABLE_HF_TRANSFER=0"]
        if patterns:
            argv += ["-e", "VL_PATTERNS=" + "|".join(patterns)]
        if env_file:
            argv += ["--env-file", env_file]
        argv += ["--entrypoint", "python3", image, "-c", code]
        r = h.dk(*argv, timeout=120)
        if not r.ok:
            raise LabError("Could not start the download container", r.text[-400:])
        self.events.add("info", f"library@{hid}", f"Downloading {model} to {h.label}" + (f" ({fmt_bytes(total)})" if total else ""))
        folder = h.cache_dir().rstrip("/") + "/hub/" + hf_cache_folder(model)
        last_size, last_t = 0, now()
        while True:
            job.check()
            time.sleep(2)
            st = h.dk("inspect", "-f", "{{.State.Status}} {{.State.ExitCode}}", cname, timeout=15).out.split()
            sz = as_int(h.sh(f"du -sbL {shlex.quote(folder)}/blobs 2>/dev/null | cut -f1", timeout=20).out.strip(), 0) or 0
            rate = (sz - last_size) / max(0.1, now() - last_t)
            last_size, last_t = sz, now()
            label = f"{fmt_bytes(sz)}" + (f" / {fmt_bytes(total)}" if total else "") + (f" · {fmt_bytes(rate)}/s" if rate > 0 else "")
            job.set_progress((sz / total) if total else None, label)
            if not st or st[0] in ("exited", "dead"):
                code_s = st[1] if len(st) > 1 else "1"
                logs = h.dk("logs", "--tail", "80", cname, timeout=20)
                h.dk("rm", "-f", cname, timeout=30)
                if code_s == "0":
                    self._lib_cache = {}
                    self.events.add("ok", f"library@{hid}", f"Downloaded {model} ({fmt_bytes(sz)})")
                    return f"Downloaded {model} · {fmt_bytes(sz)}"
                crash = decode_crash(logs.text, as_int(code_s, 1))
                raise LabError(f"Download failed: {(crash or {}).get('title') or 'see log'}", (crash or {}).get("hint") or logs.text[-500:], (crash or {}).get("fixes") or [])

    def delete_weights(self, model: str, hid: str) -> str:
        if not MODEL_RE.match(model):
            raise LabError("Not a model id")
        h = self.host(hid)
        h.require()
        running = [n for n, e in self.engine_state.items() if e.get("host") == hid and e.get("state") in ("ready", "booting")
                   and (e.get("model") == model or model in extra_models(self.store.engines().get(n, {}).get("extra", "")))]
        if running:
            raise LabError(f"{model} is in use by {', '.join(running)}", "Stop those engines first.", code=409)
        folder = hf_cache_folder(model)
        if not re.match(r"^models--[A-Za-z0-9_.\-]+(--[A-Za-z0-9_.\-]+)?$", folder):
            raise LabError("Refusing an odd folder name")
        path = h.cache_dir().rstrip("/") + "/hub/" + folder
        r = h.sh(f"[ -d {shlex.quote(path)} ] && rm -rf -- {shlex.quote(path)}", timeout=300)
        if not r.ok:
            r2 = h.sh(f"sudo -n rm -rf -- {shlex.quote(path)}", timeout=300)
            if not r2.ok:
                raise LabError("Could not delete the folder", (r.err or r2.err or "permission denied")[:300] +
                               " — files written by containers are often owned by root.")
        self._lib_cache = {}
        self.events.add("warn", f"library@{hid}", f"Deleted weights {model} from {h.label}")
        return f"Deleted {model}"

    # blueprints ---------------------------------------------------------------------
    def blueprints(self) -> dict:
        out = {k: {**v, "builtin": True, "id": k} for k, v in BUILTIN_BLUEPRINTS.items()}
        for k, v in self.store.user_blueprints().items():
            out[k] = {**v, "builtin": False, "id": k}
        return out

    def save_blueprint(self, bid: str, title: str, spec: dict, tags: list | None = None) -> dict:
        bid = re.sub(r"[^a-z0-9-]", "-", (bid or model_slug(spec.get("model", ""))).lower()).strip("-")[:40]
        if not bid:
            raise LabError("Blueprint needs an id")
        if bid in BUILTIN_BLUEPRINTS:
            bid = bid + "-custom"
        keep = {k: v for k, v in normalize_spec(spec).items() if k not in ("name", "host", "port", "created", "desired", "note", "blueprint")}
        bps = self.store.user_blueprints()
        bps[bid] = {"title": title or keep.get("model"), "spec": keep, "tags": tags or [], "created": now(), "family": "", "maker": "You"}
        self.store.save_blueprints(bps)
        return {**bps[bid], "id": bid}

    def delete_blueprint(self, bid: str) -> None:
        bps = self.store.user_blueprints()
        bps.pop(bid, None)
        self.store.save_blueprints(bps)


# ───────────────────────────────────────────────────────────────── Open WebUI

class WebUIMixin:
    _wtoken: tuple[float, str] | None = None

    def webui_conf(self) -> dict:
        return self.conf().get("webui") or {}

    def webui_auth(self) -> str:
        w = self.webui_conf()
        if w.get("api_key"):
            return w["api_key"]
        if not (w.get("email") and w.get("password")):
            raise LabError("Open WebUI login is not saved", "Add an API key (Settings → Account in Open WebUI) or admin email + password.", ["webui-login"])
        if self._wtoken and now() - self._wtoken[0] < 1800:
            return self._wtoken[1]
        try:
            data = http_json(w["url"].rstrip("/") + "/api/v1/auths/signin", "POST", {"email": w["email"], "password": w["password"]}, timeout=15)
        except HTTPError as e:
            raise LabError("Open WebUI sign-in failed", f"HTTP {e.code} — check the email and password.", ["webui-login"])
        except (URLError, OSError) as e:
            raise LabError("Open WebUI is not answering", str(e), ["webui-open"])
        except json.JSONDecodeError:
            raise LabError("Open WebUI returned HTML instead of JSON", "Is the URL pointing at Open WebUI?", ["webui-login"])
        tok = data.get("token") or data.get("access_token")
        if not tok:
            raise LabError("Open WebUI sign-in returned no token")
        self._wtoken = (now(), tok)
        return tok

    def webui_call(self, path: str, body=None, method: str = "GET"):
        w = self.webui_conf()
        tok = self.webui_auth()
        try:
            return http_json(w["url"].rstrip("/") + path, method, body, headers={"Authorization": f"Bearer {tok}", "Cookie": f"token={tok}"}, timeout=20)
        except HTTPError as e:
            if e.code == 401:
                self._wtoken = None
            raise LabError(f"Open WebUI {path} → HTTP {e.code}")
        except json.JSONDecodeError:
            raise LabError(f"Open WebUI {path} returned HTML")
        except (URLError, OSError) as e:
            raise LabError("Open WebUI is not answering", str(e))

    def webui_get_config(self) -> tuple[dict, str]:
        last = None
        for p in ("/openai/config", "/api/v1/openai/config"):
            try:
                return self.webui_call(p), p
            except LabError as e:
                last = e
        raise last or LabError("Could not read Open WebUI connections")

    def webui_set_config(self, cfg: dict, path: str) -> None:
        body = {"ENABLE_OPENAI_API": cfg.get("ENABLE_OPENAI_API", True), "OPENAI_API_BASE_URLS": cfg["OPENAI_API_BASE_URLS"],
                "OPENAI_API_KEYS": cfg["OPENAI_API_KEYS"], "OPENAI_API_CONFIGS": cfg.get("OPENAI_API_CONFIGS") or {}}
        self.webui_call(path + "/update", body, "POST")

    def webui_desired(self) -> dict[str, str]:
        """URL → engine name for every engine Open WebUI should list, as reachable *from the WebUI container*."""
        w = self.webui_conf()
        conf = self.conf()
        out = {}
        if w.get("mode") == "gateway":
            gw = conf.get("gateway") or {}
            addr = self.gateway_addr_for_containers()
            if addr:
                out[f"http://{addr}:{gw.get('port') or 58130}/v1"] = "*gateway"
            return out
        whost = w.get("host") or "titan"
        prune = w.get("prune_stopped", True)
        for n, s in self.store.engines().items():
            st = (self.engine_state.get(n) or {}).get("state")
            if prune and st not in ("ready", "booting"):
                continue
            if not prune and st in (None, "absent"):
                continue
            h = self.host(s["host"])
            if s["host"] == whost:
                out[f"http://{self.cname(n)}:8000/v1"] = n
            elif h.reach == "direct" and s.get("port"):
                out[f"http://{h.human_host}:{s['port']}/v1"] = n
        return out

    def owned_urls(self) -> set[str]:
        conf = self.conf()
        owned = set()
        for n, s in self.store.engines().items():
            owned.add(f"http://{self.cname(n)}:8000/v1")
            with contextlib.suppress(LabError):
                h = self.host(s["host"])
                if s.get("port"):
                    owned.add(f"http://{h.human_host}:{s['port']}/v1")
        gw = conf.get("gateway") or {}
        addr = self.gateway_addr_for_containers()
        if addr:
            owned.add(f"http://{addr}:{gw.get('port') or 58130}/v1")
        return owned | set(self.webui_last.get("owned") or [])

    def _is_owned(self, url: str, owned: set[str]) -> bool:
        if url in owned:
            return True
        pre = self.conf().get("container_prefix") or ""
        return bool(pre) and re.match(rf"^http://{re.escape(pre)}[a-z0-9_.-]+:8000/v1/?$", url or "") is not None

    def webui_sync(self, prune_dead: bool = True) -> dict:
        cfg, path = self.webui_get_config()
        urls = list(cfg.get("OPENAI_API_BASE_URLS") or [])
        keys = list(cfg.get("OPENAI_API_KEYS") or [])
        confs = dict(cfg.get("OPENAI_API_CONFIGS") or {})
        while len(keys) < len(urls):
            keys.append("")
        desired = self.webui_desired()
        owned = self.owned_urls()
        new_urls, new_keys, new_confs = [], [], {}
        removed, added = [], []
        for i, u in enumerate(urls):
            norm = u.rstrip("/")
            if self._is_owned(norm, owned) and norm not in desired and prune_dead:
                removed.append(norm)
                continue
            if norm in [x.rstrip("/") for x in new_urls]:
                removed.append(norm)
                continue
            c = confs.get(str(i)) or confs.get(i)
            if c is not None:
                new_confs[str(len(new_urls))] = c
            new_urls.append(u)
            new_keys.append(keys[i] if i < len(keys) else "")
        present = {u.rstrip("/") for u in new_urls}
        gw_key = ((self.conf().get("gateway") or {}).get("key") or "local")
        for u, n in desired.items():
            if u not in present:
                new_confs[str(len(new_urls))] = {"enable": True, "tags": [], "prefix_id": "", "model_ids": [], "connection_type": "local"}
                new_urls.append(u)
                new_keys.append(gw_key if n == "*gateway" else "local")
                added.append(u)
        if added or removed:
            cfg.update({"ENABLE_OPENAI_API": True, "OPENAI_API_BASE_URLS": new_urls, "OPENAI_API_KEYS": new_keys, "OPENAI_API_CONFIGS": new_confs})
            self.webui_set_config(cfg, path)
        self.webui_last = {"t": now(), "added": added, "removed": removed, "owned": sorted(set(desired))}
        return {"added": added, "removed": removed, "urls": new_urls}

    def webui_probe_from_container(self, url: str) -> tuple[bool, str]:
        w = self.webui_conf()
        try:
            h = self.host(w.get("host") or "titan")
        except LabError as e:
            return False, e.msg
        script = ("import urllib.request,sys\nu=sys.argv[1].rstrip('/')+'/models'\n"
                  "try:\n r=urllib.request.urlopen(u,timeout=6); print('OK',r.status,r.read(200).decode('utf-8','replace').replace(chr(10),' '))\n"
                  "except Exception as e:\n print('FAIL',type(e).__name__,e)\n")
        r = h.dk("exec", w.get("container") or "titan-webui", "python3", "-c", script, url, timeout=20)
        text = r.text.strip()
        if not r.ok and not text.startswith(("OK", "FAIL")):
            return False, "Open WebUI container is not running (or has no python3)"
        return text.startswith("OK"), text[:300]

    def webui_overview(self, deep: bool = True) -> dict:
        w = self.webui_conf()
        base = (w.get("url") or "").rstrip("/")
        out: dict = {"url": base, "login": "api_key" if w.get("api_key") else ("password" if w.get("email") and w.get("password") else ""),
                     "email": w.get("email") or "", "mode": w.get("mode") or "direct", "auto_sync": w.get("auto_sync", True),
                     "prune_stopped": w.get("prune_stopped", True), "container": w.get("container"), "host": w.get("host"),
                     "checks": [], "connections": [], "models": [], "desired": self.webui_desired() if deep else {}}
        code, _ = http_text(base + "/health", timeout=4)
        if code == 0:
            code, _ = http_text(base + "/", timeout=4)
        out["up"] = 0 < code < 500
        out["checks"].append({"ok": out["up"], "title": "Open WebUI answers", "detail": f"{base} → HTTP {code or 'no answer'}"})
        if not out["up"] or not out["login"]:
            if not out["login"]:
                out["checks"].append({"ok": False, "title": "Login not saved", "detail": "Needed to register engines automatically.", "fix": "webui-login"})
            return out
        try:
            cfg, _ = self.webui_get_config()
            out["checks"].append({"ok": True, "title": "Admin API", "detail": "signed in"})
        except LabError as e:
            out["checks"].append({"ok": False, "title": "Admin API", "detail": e.msg + (" — " + e.hint if e.hint else ""), "fix": "webui-login"})
            return out
        owned = self.owned_urls()
        desired = out["desired"]
        urls = cfg.get("OPENAI_API_BASE_URLS") or []
        keys = cfg.get("OPENAI_API_KEYS") or []
        futs = {}
        if deep:
            for u in urls:
                futs[u] = self.probe_pool.submit(self.webui_probe_from_container, u)
        for i, u in enumerate(urls):
            ok, detail = (None, "")
            if u in futs:
                with contextlib.suppress(Exception):
                    ok, detail = futs[u].result(timeout=25)
            out["connections"].append({"url": u, "owned": self._is_owned(u.rstrip("/"), owned), "engine": desired.get(u.rstrip("/")),
                                       "key": bool(keys[i]) if i < len(keys) else False, "ok": ok, "detail": detail})
        missing = [u for u in desired if u not in {x.rstrip("/") for x in urls}]
        stale = [c["url"] for c in out["connections"] if c["owned"] and c["url"].rstrip("/") not in desired]
        out["missing"], out["stale"] = missing, stale
        out["checks"].append({"ok": not missing and not stale, "title": "Connections match running engines",
                              "detail": (f"missing {len(missing)} · stale {len(stale)}" if missing or stale else "in sync"), "fix": "webui-sync" if missing or stale else ""})
        with contextlib.suppress(LabError):
            payload = self.webui_call("/api/models")
            data = payload.get("data") if isinstance(payload, dict) else payload
            out["models"] = [str(m.get("id") or m.get("name")) for m in (data or []) if isinstance(m, dict)][:80]
        return out

    def save_webui(self, data: dict) -> None:
        def fn(conf):
            w = conf.setdefault("webui", {})
            for k in ("url", "email", "container", "host", "mode"):
                if k in data and data[k] is not None:
                    w[k] = str(data[k]).strip()
            for k in ("password", "api_key"):
                if data.get(k):
                    w[k] = str(data[k])
                if data.get(f"clear_{k}"):
                    w[k] = ""
            for k in ("auto_sync", "prune_stopped"):
                if k in data:
                    w[k] = as_bool(data[k])
        self.store.update_conf(fn)
        self._wtoken = None


# ───────────────────────────────────────────────────────────────── doctor, hosts, settings, gateway routing

class AdminMixin:
    # doctor -------------------------------------------------------------------------
    def doctor(self, deep: bool = True) -> dict:
        checks: list[dict] = []

        def add(group, level, title, detail="", fix="", target="", host=""):
            checks.append({"group": group, "level": level, "title": title, "detail": detail, "fix": fix, "target": target, "host": host})

        conf = self.conf()
        try:
            mode = CONF_FILE.stat().st_mode & 0o777
            add("Setup", "ok" if mode & 0o077 == 0 else "fail", "Config file is private" if mode & 0o077 == 0 else "Config file is readable by others",
                f"{CONF_FILE} mode {oct(mode)}", "" if mode & 0o077 == 0 else "chmod-config")
        except OSError:
            add("Setup", "warn", "No config file yet", "It is created on the first save.")
        if not self.hf_token():
            add("Setup", "warn", "No Hugging Face token", "Public models work; gated ones (Llama, Gemma) will fail.", "set-token")
        elif deep:
            who = self.hf_whoami()
            if who.get("ok"):
                add("Setup", "ok", f"Hugging Face token works ({who.get('name')})", f"role: {who.get('role') or 'read'}")
            elif "not reachable" in (who.get("why") or ""):
                add("Setup", "warn", "Hugging Face is not reachable", "Downloads and search need internet; cached weights still start with HF_HUB_OFFLINE=1.")
            else:
                add("Setup", "fail", "Hugging Face rejected the token", who.get("why", ""), "set-token")
        engines = self.store.engines()
        for h in self.hosts():
            g = f"Host · {h.label}"
            if not h.enabled:
                add(g, "info", "Disabled", "Enable it when SSH works.", "host-enable", host=h.id)
                continue
            ok, why = h.reachable(force=deep)
            if not ok:
                add(g, "fail", "SSH does not connect", f"ssh {h.spec.get('ssh')}: {why}. Set up a key with ssh-copy-id.", "host-test", host=h.id)
                continue
            if not h.local:
                add(g, "ok", "SSH connects", f"{h.spec.get('ssh')} · reach: {h.reach}", host=h.id)
            dok, dmsg = h.docker_access()
            add(g, "ok" if dok else "fail", f"Docker {dmsg}" if dok else "Docker is not usable", "" if dok else dmsg, "" if dok else "docker-group", host=h.id)
            if not dok:
                continue
            v = self.versions.get(h.id) or {}
            if not v:
                self._collect_versions(h)
                v = self.versions.get(h.id) or {}
            hs = self.host_state.get(h.id) or {}
            if not hs.get("mem"):
                self.collect_host(h)
                hs = self.host_state.get(h.id) or {}
            gpus = hs.get("gpus") or []
            add(g, "ok" if gpus else "fail", f"GPU: {gpus[0]['name']}" if gpus else "nvidia-smi found no GPU",
                f"driver {v.get('driver') or '?'} · CUDA {v.get('cuda') or '?'} · {v.get('arch') or ''}" if gpus else "Install or repair the NVIDIA driver.", host=h.id)
            add(g, "ok" if v.get("nvidia_runtime") else "warn", "NVIDIA container runtime registered" if v.get("nvidia_runtime") else "Docker lists no 'nvidia' runtime",
                "" if v.get("nvidia_runtime") else "--gpus all may still work through CDI. If engines fail with 'could not select device driver', run: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
                host=h.id)
            d = hs.get("disk") or {}
            if d:
                free = d.get("free") or 0
                lvl = "ok" if free > 80 * GIB else ("warn" if free > 20 * GIB else "fail")
                add(g, lvl, f"{fmt_bytes(free)} free for weights", f"{d.get('path')} on {d.get('mount')}", "" if lvl == "ok" else "library", host=h.id)
            mem = hs.get("mem") or {}
            if mem.get("uma") and (mem.get("cached") or 0) > 16 * GIB:
                add(g, "info", f"{fmt_bytes(mem['cached'])} is page cache", "On unified memory, CUDA may count page cache as used. Flushing it before a big start avoids false out-of-memory errors.",
                    "flush-cache", host=h.id)
            if not v.get("sudo"):
                add(g, "info", "No passwordless sudo", "Page-cache flush before starts is skipped. Optional sudoers line: <user> ALL=(root) NOPASSWD: /usr/bin/tee /proc/sys/vm/drop_caches", host=h.id)
            if not h.dk("network", "inspect", h.network, timeout=10).ok:
                add(g, "warn", f"Docker network {h.network} is missing", "It is created automatically on the next start.", "create-network", host=h.id)
            img = conf.get("vllm_image")
            if deep and img and not self.image_present(h, img):
                add(g, "info", f"{img} is not pulled yet", "The first vLLM start pulls it (≈10–20 GB).", "pull-image", target=img, host=h.id)
            total = int(mem.get("gpu_total") or mem.get("total") or 0)
            boot = [(n, s) for n, s in engines.items() if s["host"] == h.id and s.get("autostart") and s["backend"] == "vllm"
                    and (self.engine_state.get(n) or {}).get("state") in ("ready", "booting", "crashed", "stopped")
                    and s.get("desired") != "stopped"]
            if total and boot:
                sum_u = sum(s["util"] for _, s in boot)
                if sum_u > 0.92:
                    add(g, "warn", "Engines that auto-start at boot would not all fit", f"{', '.join(n for n, _ in boot)} reserve {sum_u:.0%} together. After a reboot one of them will fail.",
                        "", host=h.id)
            for u in self.unmanaged.get(h.id) or []:
                add(g, "info", f"Unmanaged container {u['name']}", f"{u['image']} · {u['status']}", "adopt", target=u["name"], host=h.id)
        ports: dict = {}
        for n, s in engines.items():
            key = (s["host"], s.get("port"))
            if key in ports:
                add("Engines", "fail", f"{n} and {ports[key]} share port {s.get('port')}", "", "change-port", target=n)
            ports[key] = n
            e = self.engine_state.get(n) or {}
            st = e.get("state")
            if st == "crashed":
                cr = e.get("crash") or {}
                add("Engines", "fail", f"{n}: {cr.get('title', 'crashed')}", cr.get("hint", ""), (cr.get("fixes") or ["logs"])[0], target=n)
            if e.get("drift") and st in ("ready", "booting", "stopped"):
                add("Engines", "warn", f"{n} runs an older config", "Settings changed since this container was created. Recreate to apply.", "recreate", target=n)
            if s.get("desired") == "running" and st == "absent":
                add("Engines", "warn", f"{n} should be running but has no container", "", "start", target=n)
            pol = origin_check(s["model"], conf)
            if pol["restricted"]:
                add("Engines", "fail" if pol["blocked"] else "warn", f"{n} uses restricted-origin weights", pol["via"], "", target=n)
        if not any(c["group"] == "Engines" for c in checks):
            add("Engines", "ok", "All engines look healthy")
        w = self.webui_conf()
        if deep and w.get("url"):
            try:
                ov = self.webui_overview(deep=False)
                for c in ov["checks"]:
                    add("Open WebUI", "ok" if c["ok"] else ("warn" if "Login" in c["title"] else "fail"), c["title"], c.get("detail", ""), c.get("fix", ""))
            except Exception as e:  # noqa: BLE001
                add("Open WebUI", "warn", "Could not check Open WebUI", str(e))
        gw = conf.get("gateway") or {}
        if gw.get("enabled"):
            add("Gateway", "ok", "Gateway is on", f"OpenAI-compatible endpoint at http://127.0.0.1:{conf['ui'].get('port')}/v1" +
                (f" and {self.gateway_listen_addr()}:{gw.get('port')}" if self.gateway_listen_addr() else ""))
        order = {"fail": 0, "warn": 1, "info": 2, "ok": 3}
        summary = {k: sum(1 for c in checks if c["level"] == k) for k in order}
        return {"checks": checks, "summary": summary, "t": now()}

    def doctor_fix(self, fix: str, host: str = "", target: str = "", job: Job | None = None) -> str:
        if fix == "chmod-config":
            os.chmod(CONF_FILE, 0o600)
            return "Config is now private (600)"
        if fix == "create-network":
            self.ensure_network(self.host(host))
            return "Network created"
        if fix == "flush-cache":
            ok, msg = self.flush_cache(self.host(host))
            if not ok:
                raise LabError("Could not flush the page cache", msg)
            return msg
        if fix == "pull-image":
            self.ensure_image(self.host(host), target, job)
            return f"Pulled {target}"
        if fix == "adopt":
            s = self.adopt(host, target)
            return f"Adopted as {s['name']}"
        if fix == "webui-sync":
            r = self.webui_sync()
            return f"Open WebUI: +{len(r['added'])} −{len(r['removed'])}"
        if fix == "host-enable":
            self.upsert_host({"id": host, "enabled": True})
            return "Enabled"
        if target and fix in ("recreate", "start"):
            return self.op_start(target, job, recreate=fix == "recreate")
        if target:
            return self.op_fix(target, fix, job or Job(self.jobs, "fix", fix, target))
        raise LabError(f"'{fix}' needs you — see the hint")

    # hosts ------------------------------------------------------------------------------
    def upsert_host(self, data: dict, old_id: str = "") -> dict:
        hid = str(data.get("id") or old_id or "").strip().lower()
        if not HOST_ID_RE.match(hid):
            raise LabError("Host id: lowercase letters, digits, dash; up to 24 characters", code=422)
        ssh = str(data.get("ssh") or "").strip()
        if ssh and not re.match(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*@[A-Za-z0-9_\[][A-Za-z0-9_.\-:\[\]]*$|^[A-Za-z0-9_][A-Za-z0-9_.\-]*$", ssh):
            raise LabError("SSH target should look like user@host", code=422)
        for k in ("human_host", "bind"):
            v = str(data.get(k) or "").strip()
            if v and not re.match(r"^[A-Za-z0-9_.\-:\[\]]+$", v):
                raise LabError(f"{k}: not a hostname or IP", code=422)
        if data.get("cache") and not str(data["cache"]).startswith("/"):
            raise LabError("Weights cache must be an absolute path", code=422)
        if data.get("network") and not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$", str(data["network"])):
            raise LabError("Network name is not valid", code=422)
        result = {}

        def fn(conf):
            hosts = conf["hosts"]
            cur = next((h for h in hosts if h["id"] == (old_id or hid)), None)
            if old_id and old_id != hid and any(h["id"] == hid for h in hosts):
                raise LabError(f"A host named {hid} already exists", code=409)
            if cur is None:
                cur = {"id": hid}
                hosts.append(cur)
            for k in ("label", "ssh", "bind", "human_host", "network", "cache", "reach", "ports", "note"):
                if k in data and data[k] is not None:
                    cur[k] = str(data[k]).strip()
            if "ssh_port" in data:
                cur["ssh_port"] = as_int(data["ssh_port"], None)
            if "enabled" in data:
                cur["enabled"] = as_bool(data["enabled"], True)
            if old_id and old_id != hid:
                cur["id"] = hid
            conf["hosts"] = [Store.norm_host(h) for h in hosts]
            result.update(next(h for h in conf["hosts"] if h["id"] == hid))

        self.store.update_conf(fn)
        if old_id and old_id != hid:
            for n, s in self.store.engines().items():
                if s["host"] == old_id:
                    self.store.update_engine(n, lambda sp: sp.__setitem__("host", hid))
        self.events.add("info", f"host@{hid}", f"Saved host {hid}")
        return result

    def remove_host(self, hid: str) -> None:
        used = [n for n, s in self.store.engines().items() if s["host"] == hid]
        if used:
            raise LabError(f"Engines still live on {hid}: {', '.join(used)}", "Move or delete them first.", code=409)
        self.store.update_conf(lambda c: c.__setitem__("hosts", [h for h in c["hosts"] if h["id"] != hid]))
        self.events.add("warn", f"host@{hid}", f"Removed host {hid}")

    def op_test_host(self, hid: str, job: Job) -> str:
        h = self.host(hid)
        steps = []
        job.step("Connecting")
        ok, why = h.reachable(force=True) if h.enabled else (h.local, "disabled")
        if not h.local and not h.enabled:
            h.spec = {**h.spec, "enabled": True}
            h._reach = None
            ok, why = h.reachable(force=True)
        steps.append(("SSH" if not h.local else "Local", ok, why))
        if not ok:
            raise LabError(f"Cannot reach {h.label}", f"{why}\nFrom this box: ssh-copy-id {h.spec.get('ssh')}")
        job.step("Checking Docker")
        dok, dmsg = h.docker_access()
        steps.append(("Docker", dok, dmsg))
        if not dok:
            raise LabError("Docker is not usable on that host", dmsg)
        job.step("Checking GPU")
        self._collect_versions(h)
        st = self.collect_host(h)
        gpus = st.get("gpus") or []
        steps.append(("GPU", bool(gpus), gpus[0]["name"] if gpus else "none"))
        job.step("Checking weights cache")
        cache = h.cache_dir()
        r = h.sh(f"mkdir -p {shlex.quote(cache)}/hub && test -w {shlex.quote(cache)} && echo ok", timeout=15)
        steps.append(("Cache", r.out.strip() == "ok", cache))
        for s in steps:
            job.log(f"{'✓' if s[1] else '✗'} {s[0]}: {s[2]}")
        bad = [s for s in steps if not s[1]]
        if bad:
            raise LabError(f"{bad[0][0]} check failed on {h.label}", str(bad[0][2]))
        return f"{h.label}: SSH, Docker, GPU and cache all good"

    # settings ------------------------------------------------------------------------
    def settings_public(self) -> dict:
        c = self.conf()
        w = c.get("webui") or {}
        return {
            "version": VERSION, "conf_dir": str(CONF_DIR),
            "hf_token_set": bool(self.hf_token()), "hf_token_env": bool(os.environ.get("HF_TOKEN")),
            "hf_endpoint": c.get("hf_endpoint") or "", "vllm_image": c.get("vllm_image"), "llamacpp_image": c.get("llamacpp_image"),
            "container_prefix": c.get("container_prefix"), "port_pool": c.get("port_pool"), "port_reserved": c.get("port_reserved"),
            "headroom_gib": c.get("headroom_gib"), "auto_flush_cache": c.get("auto_flush_cache"), "crash_guard": c.get("crash_guard"),
            "policy": c.get("policy"), "gateway": {**(c.get("gateway") or {}), "key": "", "key_set": bool((c.get("gateway") or {}).get("key"))},
            "ui": {**(c.get("ui") or {}), "token": "", "token_set": bool((c.get("ui") or {}).get("token"))},
            "webui": {k: v for k, v in w.items() if k not in ("password", "api_key")} | {"password_set": bool(w.get("password")), "api_key_set": bool(w.get("api_key"))},
            "hosts": c.get("hosts"),
        }

    def update_settings(self, data: dict) -> dict:
        def fn(conf):
            if "hf_token" in data:
                tok = str(data["hf_token"] or "").strip()
                if tok and not re.match(r"^hf_[A-Za-z0-9]{10,}$", tok):
                    raise LabError("That does not look like a Hugging Face token (hf_…)", code=422)
                conf["hf_token"] = tok
            for k in ("hf_endpoint", "vllm_image", "llamacpp_image", "container_prefix", "port_pool", "port_reserved"):
                if k in data and data[k] is not None:
                    v = str(data[k]).strip()
                    if k.endswith("image") and v and not IMAGE_RE.match(v):
                        raise LabError(f"{k}: not a valid image", code=422)
                    if k == "hf_endpoint" and v and not re.match(r"^https?://", v):
                        raise LabError("HF endpoint must start with http(s)://", code=422)
                    if k == "container_prefix" and not re.match(r"^[a-z0-9][a-z0-9_.-]{0,20}$|^$", v):
                        raise LabError("Container prefix: lowercase letters, digits, dash", code=422)
                    if k in ("port_pool", "port_reserved") and v and not parse_ports(v):
                        raise LabError(f"{k}: use a list like 58100,58102-58109", code=422)
                    conf[k] = v
            if "headroom_gib" in data:
                conf["headroom_gib"] = max(0.0, min(64.0, as_float(data["headroom_gib"], 4.0)))
            if "crash_guard" in data:
                conf["crash_guard"] = max(0, min(50, as_int(data["crash_guard"], 3)))
            if "auto_flush_cache" in data:
                conf["auto_flush_cache"] = as_bool(data["auto_flush_cache"], True)
            if isinstance(data.get("policy"), dict):
                p = conf.setdefault("policy", {})
                if data["policy"].get("mode") in ("block", "warn", "off"):
                    p["mode"] = data["policy"]["mode"]
                if isinstance(data["policy"].get("orgs"), list):
                    p["orgs"] = sorted({str(o).strip() for o in data["policy"]["orgs"] if re.match(r"^[A-Za-z0-9_.\-]{1,64}$", str(o).strip())}, key=str.lower)
            if isinstance(data.get("gateway"), dict):
                g = conf.setdefault("gateway", {})
                gd = data["gateway"]
                for k in ("enabled", "autowake"):
                    if k in gd:
                        g[k] = as_bool(gd[k])
                if "port" in gd:
                    g["port"] = as_int(gd["port"], 58130)
                if "wake_timeout" in gd:
                    g["wake_timeout"] = max(30, min(7200, as_int(gd["wake_timeout"], 900)))
                if "listen" in gd:
                    v = str(gd["listen"] or "").strip()
                    if v and v != "docker" and not re.match(r"^[0-9a-fA-F.:]+$", v):
                        raise LabError("Gateway listen: an IP address, 'docker', or empty", code=422)
                    g["listen"] = v
                if gd.get("key"):
                    g["key"] = str(gd["key"])
                if gd.get("clear_key"):
                    g["key"] = ""
                if gd.get("generate_key"):
                    g["key"] = "sk-lab-" + secrets.token_urlsafe(24)
            if isinstance(data.get("ui"), dict):
                u = conf.setdefault("ui", {})
                if "allowed_hosts" in data["ui"]:
                    u["allowed_hosts"] = [str(x).strip() for x in data["ui"]["allowed_hosts"] if str(x).strip()]
                if data["ui"].get("token"):
                    u["token"] = str(data["ui"]["token"])
                if data["ui"].get("clear_token"):
                    u["token"] = ""
        self.store.update_conf(fn)
        self.events.add("info", "settings", "Settings saved")
        out = self.settings_public()
        if isinstance(data.get("gateway"), dict) and data["gateway"].get("generate_key"):
            out["new_gateway_key"] = (self.conf().get("gateway") or {}).get("key")
        return out

    # gateway routing ------------------------------------------------------------------
    def gateway_listen_addr(self) -> str:
        gw = self.conf().get("gateway") or {}
        v = (gw.get("listen") or "").strip()
        if v == "docker":
            return self.docker_gateway_ip() or ""
        return v

    def docker_gateway_ip(self) -> str:
        cached = getattr(self, "_dgw", None)
        if cached and now() - cached[0] < 300:
            return cached[1]
        ip = ""
        with contextlib.suppress(Exception):
            w = self.webui_conf()
            h = self.host(w.get("host") or "titan")
            r = h.dk("network", "inspect", "-f", "{{range .IPAM.Config}}{{.Gateway}} {{end}}", h.network, timeout=10)
            ip = (r.out.split() or [""])[0] if r.ok else ""
        self._dgw = (now(), ip)
        return ip

    def gateway_addr_for_containers(self) -> str:
        gw = self.conf().get("gateway") or {}
        v = (gw.get("listen") or "").strip()
        if v == "docker":
            return self.docker_gateway_ip()
        if v and v not in ("127.0.0.1", "::1", "0.0.0.0"):
            return v
        return ""

    def gw_catalog(self) -> list[dict]:
        conf = self.conf()
        gw = conf.get("gateway") or {}
        out = []
        engines = self.store.engines()
        seen: dict[str, int] = {}
        for n, s in engines.items():
            seen[served_id(s)] = seen.get(served_id(s), 0) + 1
        for n, s in sorted(engines.items()):
            e = self.engine_state.get(n) or {}
            st = e.get("state") or "unknown"
            wakeable = gw.get("autowake", True) and s.get("wake", True) and st in ("stopped", "sleeping", "absent")
            if st not in ("ready", "booting") and not wakeable:
                continue
            mid = served_id(s) if seen[served_id(s)] == 1 else n
            out.append({"id": mid, "engine": n, "host": s["host"], "state": st, "backend": s["backend"], "model": s["model"],
                        "max_len": s.get("max_len")})
        for hid, info in self.ollama.items():
            if not info.get("ok"):
                continue
            for m in info.get("models") or []:
                out.append({"id": m["name"], "engine": f"ollama@{hid}", "host": hid, "state": "ready", "backend": "ollama", "model": m["name"]})
        return out

    def gw_resolve(self, model: str) -> dict | None:
        model = (model or "").strip()
        engines = self.store.engines()
        if model in engines:
            return {"engine": model}
        if "/" in model:
            hid, _, name = model.partition("/")
            if name in engines and engines[name]["host"] == hid:
                return {"engine": name}
        cands = [n for n, s in engines.items() if model in (served_id(s), s["model"])]
        if cands:
            ready = [n for n in cands if (self.engine_state.get(n) or {}).get("state") == "ready"]
            return {"engine": (ready or cands)[0]}
        tag = model[len("ollama:"):] if model.startswith("ollama:") else model
        for hid, info in self.ollama.items():
            for m in info.get("models") or []:
                if m["name"] == tag or m["name"].split(":")[0] == tag:
                    return {"ollama": hid, "model": m["name"]}
        return None

    def gw_wake(self, name: str, timeout: float) -> None:
        e = self.engine_state.get(name) or {}
        if e.get("state") == "ready":
            return
        spec = self.spec(name)
        if not spec.get("wake", True):
            raise LabError(f"{name} is not running and wake-on-request is off for it", code=503)
        job = self.jobs.busy(name)
        if job is None:
            h = self.host(spec["host"])
            fit = self.fit_check(spec, h)
            evict = []
            if fit.get("known") and not fit["fits_after_flush"]:
                idle = [o for o in fit["others"] if not o["busy"]]
                gain = 0
                for o in idle:
                    if fit["need"] <= fit["available"] - fit["headroom"] + gain:
                        break
                    evict.append(o["name"])
                    gain += o["reserve"]
                if fit["need"] > fit["available"] - fit["headroom"] + gain:
                    raise LabError(f"{name} does not fit next to the engines that are busy right now", code=503)
            self.events.add("info", name, f"Waking {name} for a gateway request" + (f" (sleeping {', '.join(evict)})" if evict else ""))
            job = self.jobs.start("wake", f"Wake {name}", name,
                                  lambda j: self.op_start(name, j, on_conflict="evict" if evict else "ask", evict=evict or None, wait=True))
        t0 = now()
        while now() - t0 < timeout:
            if job.state != "running":
                if job.state == "ok" and (self.engine_state.get(name) or {}).get("state") == "ready":
                    return
                if job.state == "error":
                    raise LabError(f"Could not wake {name}: {job.result}", code=503)
            if (self.engine_state.get(name) or {}).get("state") == "ready":
                return
            time.sleep(1)
        raise LabError(f"{name} did not become ready within {fmt_dur(timeout)}", code=504)


# ───────────────────────────────────────────────────────────────── the lab: state + collector

HOST_SCRIPT = r"""
P={proc}
echo @@meminfo; head -n 40 "$P/meminfo" 2>/dev/null
echo @@loadavg; cat "$P/loadavg" 2>/dev/null
echo @@uptime; cat "$P/uptime" 2>/dev/null
echo @@cpus; nproc 2>/dev/null || getconf _NPROCESSORS_ONLN
echo @@gpu; nvidia-smi --query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw,power.limit,memory.used,memory.total,clocks.sm --format=csv,noheader,nounits 2>/dev/null || echo NOSMI
echo @@disk; df -PB1 {cache} 2>/dev/null | tail -n 1 || df -PB1 "$HOME" | tail -n 1
echo @@ps; {docker} ps -a --no-trunc --format '{{{{json .}}}}' 2>&1
echo @@inspect; ids=$({docker} ps -aq --filter label=vllm-lab.engine 2>/dev/null; {docker} ps -aq --filter name='^{prefix}' 2>/dev/null)
ids=$(printf '%s\n' $ids | sort -u)
[ -n "$ids" ] && {docker} inspect $ids 2>/dev/null
echo @@end
"""

VERSION_SCRIPT = r"""
echo @@kernel; uname -srm
echo @@os; . /etc/os-release 2>/dev/null && echo "$PRETTY_NAME"
echo @@driver; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1
echo @@cuda; nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -n1
echo @@docker; {docker} version --format '{{{{.Server.Version}}}}' 2>/dev/null
echo @@runtimes; {docker} info --format '{{{{json .Runtimes}}}}' 2>/dev/null
echo @@sudo; sudo -n true 2>/dev/null && echo yes || echo no
echo @@end
"""


def split_sections(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    cur = None
    buf: list[str] = []
    for line in (text or "").splitlines():
        if line.startswith("@@") and re.match(r"^@@[a-z]+$", line.strip()):
            if cur:
                out[cur] = "\n".join(buf)
            cur = line.strip()[2:]
            buf = []
        else:
            buf.append(line)
    if cur and cur != "end":
        out[cur] = "\n".join(buf)
    return out


def parse_meminfo(text: str) -> dict:
    kv = {}
    for line in text.splitlines():
        m = re.match(r"^(\w+):\s+(\d+)\s*kB", line)
        if m:
            kv[m.group(1)] = int(m.group(2)) * 1024
    total = kv.get("MemTotal", 0)
    avail = kv.get("MemAvailable", kv.get("MemFree", 0))
    free = kv.get("MemFree", 0)
    cached = kv.get("Cached", 0) + kv.get("Buffers", 0) + kv.get("SReclaimable", 0)
    return {"total": total, "available": avail, "free": free, "cached": cached, "swap_total": kv.get("SwapTotal", 0),
            "swap_free": kv.get("SwapFree", 0)}


def parse_gpu(text: str) -> list[dict]:
    gpus = []
    if not text or "NOSMI" in text:
        return gpus
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue

        def num(x):
            try:
                return float(x)
            except ValueError:
                return None
        gpus.append({
            "index": parts[0], "name": parts[1], "util": num(parts[2]), "temp": num(parts[3]), "power": num(parts[4]),
            "power_limit": num(parts[5]), "mem_used": num(parts[6]), "mem_total": num(parts[7]), "clock": num(parts[8]),
        })
    return gpus


def parse_labels(s) -> dict:
    if isinstance(s, dict):
        return s
    out = {}
    for part in str(s or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class Lab(OpsMixin, IntelMixin, WebUIMixin, AdminMixin):
    FAST = 2.0

    def __init__(self, serve: bool = False) -> None:
        self.store = Store()
        self.bus = Bus()
        self.events = Events(self.bus)
        self.jobs = Jobs(self.bus, self.events)
        self.tunnels = Tunnels()
        self.rates = Rates()
        self.serve = serve
        self.lock = threading.RLock()
        self._hosts: dict[str, Host] = {}
        self.host_state: dict[str, dict] = {}
        self.engine_state: dict[str, dict] = {}
        self.stats: dict[str, dict] = {}
        self.versions: dict[str, dict] = {}
        self.ollama: dict[str, dict] = {}
        self.unmanaged: dict[str, list] = {}
        self.containers: dict[str, list] = {}
        self.inspects: dict[str, dict] = {}
        self.hist: dict[str, list] = {}
        self.hhist: dict[str, list] = {}
        self.activity: dict[str, float] = {}
        self.gw_stats: dict[str, dict] = {}
        self.boot_seen: dict[str, dict] = {}
        self.log_cache: dict[str, tuple[str, str]] = {}
        self.guarded: set[str] = set()
        self.prev_state: dict[str, str] = {}
        self.stopping: dict[str, float] = {}
        self.engine_locks: dict[str, threading.Lock] = {}
        self.pool = ThreadPoolExecutor(max_workers=12, thread_name_prefix="bg")
        self.probe_pool = ThreadPoolExecutor(max_workers=24, thread_name_prefix="probe")
        self._sync_timer: threading.Timer | None = None
        self._last_publish = 0.0
        self._stop = threading.Event()
        self.hf_meta_cache: dict[str, tuple[float, dict]] = {}
        self.webui_last: dict = {}

    # hosts -----------------------------------------------------------------------
    def conf(self) -> dict:
        return self.store.conf()

    def hosts(self) -> list[Host]:
        specs = self.conf()["hosts"]
        with self.lock:
            out = []
            for s in specs:
                h = self._hosts.get(s["id"])
                if h is None or h.spec != s:
                    old = h
                    h = Host(s, self)
                    if old and old.spec.get("ssh") == s.get("ssh"):
                        h._docker, h._docker_t, h._home = old._docker, old._docker_t, old._home
                    self._hosts[s["id"]] = h
                out.append(h)
            for hid in list(self._hosts):
                if hid not in {s["id"] for s in specs}:
                    self._hosts.pop(hid, None)
            return out

    def host(self, hid: str) -> Host:
        for h in self.hosts():
            if h.id == hid:
                return h
        raise LabError(f"Unknown host '{hid}'", "Add it on the Hosts page.", code=404)

    def engine_lock(self, name: str) -> threading.Lock:
        with self.lock:
            return self.engine_locks.setdefault(name, threading.Lock())

    def spec(self, name: str) -> dict:
        s = self.store.engines().get(name)
        if not s:
            raise LabError(f"No engine named '{name}'", code=404)
        return s

    def cname(self, name: str) -> str:
        return container_name(self.conf(), name)

    def total_mem(self, hid: str) -> int:
        hs = self.host_state.get(hid) or {}
        mem = hs.get("mem") or {}
        return int(mem.get("gpu_total") or mem.get("total") or 0)

    # collector -------------------------------------------------------------------
    def start_collector(self) -> None:
        threading.Thread(target=self._supervisor, daemon=True, name="collector").start()

    def _supervisor(self) -> None:
        running: dict[str, threading.Thread] = {}
        while not self._stop.is_set():
            for h in self.hosts():
                t = running.get(h.id)
                if t is None or not t.is_alive():
                    t = threading.Thread(target=self._host_loop, args=(h.id,), daemon=True, name=f"host-{h.id}")
                    running[h.id] = t
                    t.start()
            self._stop.wait(5)

    def _host_loop(self, hid: str) -> None:
        tick = 0
        while not self._stop.is_set():
            try:
                h = next((x for x in self.hosts() if x.id == hid), None)
                if h is None:
                    with self.lock:
                        self.host_state.pop(hid, None)
                    self.publish(force=True)
                    return
                self.collect_host(h, tick)
                if self.serve:
                    self.autopilot(h)
            except Exception as e:  # noqa: BLE001
                with self.lock:
                    self.host_state.setdefault(hid, {})["error"] = f"{type(e).__name__}: {e}"
                if os.environ.get("VLLM_LAB_DEBUG"):
                    traceback.print_exc()
            self.publish()
            tick += 1
            booting = any(e.get("state") == "booting" for e in self.engine_state.values() if e.get("host") == hid)
            hs = self.host_state.get(hid) or {}
            delay = self.FAST if (booting or hs.get("online")) else 6.0
            if not hs.get("enabled", True):
                delay = 10.0
            self._stop.wait(delay)

    def collect_host(self, h: Host, tick: int = 0) -> dict:
        conf = self.conf()
        engines = {n: s for n, s in self.store.engines().items() if s["host"] == h.id}
        prefix = conf.get("container_prefix") or ""
        st: dict = {"id": h.id, "label": h.label, "enabled": h.enabled, "local": h.local, "reach": h.reach,
                    "ssh": h.spec.get("ssh") or "", "human_host": h.human_host, "t": now(), "online": False, "why": ""}
        ok, why = h.reachable()
        if not ok:
            st["why"] = why
            with self.lock:
                self.host_state[h.id] = st
                for n, s in engines.items():
                    self.engine_state[n] = self._offline_engine(s, h, why)
            return st
        dk = " ".join(shlex.quote(x) for x in h.docker())
        script = HOST_SCRIPT.format(proc=h.proc_expr(), cache=shlex.quote(h.cache_dir()), docker=dk,
                                    prefix=re.escape(prefix) if prefix else "vllm-lab-none")
        r = h.sh(script, timeout=25)
        sec = split_sections(r.out)
        if not sec:
            st["why"] = (r.err or "no output from host probe").strip()[:300]
            with self.lock:
                self.host_state[h.id] = st
            return st
        st["online"] = True
        mem = parse_meminfo(sec.get("meminfo", ""))
        gpus = parse_gpu(sec.get("gpu", ""))
        uma = bool(gpus) and (gpus[0].get("mem_total") in (None, 0) or "GB10" in (gpus[0].get("name") or "") or "GH200" in (gpus[0].get("name") or ""))
        mem["uma"] = uma or not gpus
        if gpus and not uma and gpus[0].get("mem_total"):
            mem["gpu_total"] = int(sum((g.get("mem_total") or 0) for g in gpus) * 1024 * 1024)
            mem["gpu_used"] = int(sum((g.get("mem_used") or 0) for g in gpus) * 1024 * 1024)
        st["mem"] = mem
        st["gpus"] = gpus
        try:
            la = sec.get("loadavg", "").split()
            st["load"] = [float(x) for x in la[:3]]
        except ValueError:
            st["load"] = []
        st["cpus"] = as_int((sec.get("cpus") or "").strip(), None)
        st["uptime"] = as_float((sec.get("uptime") or "0").split()[0] if sec.get("uptime") else 0, 0)
        d = (sec.get("disk") or "").split()
        if len(d) >= 6:
            st["disk"] = {"total": as_int(d[1], 0), "used": as_int(d[2], 0), "free": as_int(d[3], 0), "mount": d[5], "path": h.cache_dir()}
        # containers
        rows = []
        docker_err = ""
        for line in (sec.get("ps") or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if not line.startswith("{"):
                docker_err = docker_err or line
                continue
            with contextlib.suppress(json.JSONDecodeError):
                c = json.loads(line)
                rows.append({
                    "id": c.get("ID", "")[:12], "name": c.get("Names", ""), "image": c.get("Image", ""),
                    "state": c.get("State", ""), "status": c.get("Status", ""), "ports": c.get("Ports", ""),
                    "labels": parse_labels(c.get("Labels")), "created": c.get("CreatedAt", ""), "running_for": c.get("RunningFor", ""),
                })
        st["docker"] = {"ok": not docker_err, "msg": docker_err[:300]}
        insp: dict[str, dict] = {}
        raw_insp = (sec.get("inspect") or "").strip()
        if raw_insp.startswith("["):
            with contextlib.suppress(json.JSONDecodeError):
                for obj in json.loads(raw_insp):
                    insp[(obj.get("Name") or "").lstrip("/")] = obj
        st["containers"] = len(rows)
        st["running"] = sum(1 for c in rows if c["state"] == "running")
        with self.lock:
            self.containers[h.id] = rows
            self.inspects[h.id] = insp
        # engines
        new_states = {}
        probes = {}
        for name, spec in engines.items():
            cn = container_name(conf, name)
            obj = insp.get(cn)
            if obj is None:
                for o in insp.values():
                    if (o.get("Config", {}).get("Labels") or {}).get("vllm-lab.engine") == name:
                        obj = o
                        break
            new_states[name] = (spec, obj)
            if obj and (obj.get("State") or {}).get("Running") and spec.get("port"):
                probes[name] = self.probe_pool.submit(self._probe_engine, h, spec)
        for name, (spec, obj) in new_states.items():
            probe = None
            if name in probes:
                with contextlib.suppress(Exception):
                    probe = probes[name].result(timeout=8)
            est = self._derive_engine(h, spec, obj, probe, st)
            with self.lock:
                self.engine_state[name] = est
            self._track_history(name, est)
        # engines whose host changed or were deleted
        with self.lock:
            live = set(self.store.engines())
            for n in list(self.engine_state):
                if n not in live:
                    self.engine_state.pop(n, None)
        # unmanaged containers with our prefix → adoptable
        managed = {container_name(conf, n) for n in engines} | {container_name(conf, n) for n in self.store.engines()}
        webui_c = (conf.get("webui") or {}).get("container") or ""
        um = []
        for c in rows:
            lab = c["labels"]
            if c["name"] in managed or c["name"] == webui_c or lab.get("vllm-lab.download"):
                continue
            if (prefix and c["name"].startswith(prefix)) or lab.get("vllm-lab.engine"):
                um.append({"name": c["name"], "image": c["image"], "state": c["state"], "status": c["status"]})
        with self.lock:
            self.unmanaged[h.id] = um
        # downloads in progress
        st["downloads"] = [
            {"container": c["name"], "model": c["labels"].get("vllm-lab.download"), "state": c["state"], "status": c["status"],
             "total": as_int(c["labels"].get("vllm-lab.bytes"), 0)}
            for c in rows if c["labels"].get("vllm-lab.download")
        ]
        # slower cadences
        if tick % 4 == 0:
            self.pool.submit(self._collect_stats, h)
        if tick % 3 == 0:
            self.pool.submit(self._collect_ollama, h)
        if tick % 30 == 0 or h.id not in self.versions:
            self.pool.submit(self._collect_versions, h)
        if st.get("downloads"):
            st["dl_sizes"] = self._measure_downloads(h, st["downloads"])
        st["stats_t"] = (self.stats.get(h.id) or {}).get("_t")
        with self.lock:
            self.host_state[h.id] = st
        self._track_host_history(h.id, st)
        return st

    def _offline_engine(self, spec: dict, h: Host, why: str) -> dict:
        return {
            "name": spec["name"], "host": h.id, "host_label": h.label, "backend": spec["backend"], "model": spec["model"],
            "served": served_id(spec), "port": spec.get("port"), "state": "offline", "desired": spec.get("desired"),
            "why": why, "human_url": h.human_url(spec.get("port")), "docker_url": self._docker_url(spec), "container": {},
            "metrics": {}, "boot": {}, "crash": None, "drift": False, "reserve": 0, "idle_sleep_min": spec.get("idle_sleep_min"),
            "wake": spec.get("wake"), "note": spec.get("note"),
        }

    def _docker_url(self, spec: dict) -> str:
        return f"http://{self.cname(spec['name'])}:8000/v1"

    def _probe_engine(self, h: Host, spec: dict) -> dict:
        base = h.api_base(int(spec["port"]))
        if not base:
            return {"health": 0, "err": "no route (tunnel down)"}
        code, _ = http_text(base + "/health", timeout=2)
        out = {"health": code, "base": base}
        if code == 200:
            mcode, mtext = http_text(base + "/metrics", timeout=2.5)
            if mcode == 200:
                out["metrics"] = parse_prom(mtext)
        return out

    def _derive_engine(self, h: Host, spec: dict, obj: dict | None, probe: dict | None, hs: dict) -> dict:
        name = spec["name"]
        conf = self.conf()
        mem = hs.get("mem") or {}
        total = int(mem.get("gpu_total") or mem.get("total") or 0)
        est = {
            "name": name, "host": h.id, "host_label": h.label, "backend": spec["backend"], "model": spec["model"],
            "served": served_id(spec), "port": spec.get("port"), "desired": spec.get("desired"),
            "human_url": h.human_url(spec.get("port")), "docker_url": self._docker_url(spec),
            "idle_sleep_min": spec.get("idle_sleep_min"), "wake": spec.get("wake"), "note": spec.get("note"),
            "util": spec.get("util"), "max_len": spec.get("max_len"), "blueprint": spec.get("blueprint"),
            "metrics": {}, "boot": {}, "crash": None, "drift": False, "legacy": False, "reserve": 0, "why": "",
            "container": {},
        }
        if not obj:
            est["state"] = "sleeping" if spec.get("desired") == "sleeping" else "absent"
            return est
        s = obj.get("State") or {}
        labels = (obj.get("Config") or {}).get("Labels") or {}
        cid = (obj.get("Id") or "")[:12]
        status = s.get("Status") or ""
        restarts = int(obj.get("RestartCount") or 0)
        c = {
            "id": cid, "status": status, "exit": s.get("ExitCode"), "oom": bool(s.get("OOMKilled")), "error": s.get("Error") or "",
            "restarts": restarts, "started": s.get("StartedAt") or "", "finished": s.get("FinishedAt") or "",
            "image": (obj.get("Config") or {}).get("Image") or "", "policy": ((obj.get("HostConfig") or {}).get("RestartPolicy") or {}).get("Name"),
        }
        est["container"] = c
        started_ts = _parse_ts(c["started"])
        est["started_ts"] = started_ts
        fp = labels.get("vllm-lab.spec")
        if not fp:
            # legacy container: trust the port it actually publishes so the human URL stays right
            bound = (((obj.get("NetworkSettings") or {}).get("Ports") or {}).get("8000/tcp") or [{}])[0].get("HostPort")
            if bound and as_int(bound) and as_int(bound) != spec.get("port"):
                self.store.update_engine(name, lambda s, p=as_int(bound): s.__setitem__("port", p))
                spec = {**spec, "port": as_int(bound)}
                est["port"], est["human_url"] = spec["port"], h.human_url(spec["port"])
        if fp:
            est["drift"] = fp != spec_fingerprint(spec, resolve_image(spec, conf), h)
        else:
            est["legacy"] = True
        if spec["backend"] == "vllm" and total:
            est["reserve"] = int(spec["util"] * total)
        stat = (self.stats.get(h.id) or {}).get(self.cname(name)) or {}
        if stat:
            est["mem_usage"] = stat.get("mem")
            est["cpu"] = stat.get("cpu")
        running = bool(s.get("Running")) and status == "running"
        if running:
            health = (probe or {}).get("health", 0)
            if health == 200:
                est["state"] = "ready"
                m = (probe or {}).get("metrics") or {}
                est["metrics"] = self._engine_metrics(name, m)
                seen = self.boot_seen.get(name)
                if seen and seen.get("cid") == cid:
                    est["boot"] = {**seen.get("info", {}), "took": seen.get("took")}
                if spec["backend"] == "llamacpp" and not est["reserve"]:
                    est["reserve"] = self._mem_bytes(stat.get("mem")) if stat else 0
            else:
                est["state"] = "booting"
                log = self._tail_log(h, name, cid, 400)
                b = read_boot(log, spec["backend"])
                if not b["stage"]:
                    b["stage"] = "download" if "Downloading" in log else ("load" if log.strip() else "pull")
                est["boot"] = {**b, "elapsed": now() - started_ts if started_ts else None}
                self.boot_seen[name] = {"cid": cid, "info": {k: v for k, v in b.items() if k in ("kv_tokens", "max_conc", "max_conc_len", "weights_gib")},
                                        "t0": started_ts}
                if restarts >= 1:
                    crash = decode_crash(log, c["exit"], c["oom"], spec["backend"])
                    if crash:
                        est["crash"] = crash
        elif status == "restarting":
            est["state"] = "crashed"
            log = self._tail_log(h, name, f"{cid}:{restarts}", 400)
            est["crash"] = decode_crash(log, c["exit"], c["oom"], spec["backend"]) or {"title": "Restarting in a loop", "hint": "", "fixes": ["logs"]}
        else:
            code = c["exit"]
            user_stop = spec.get("desired") in ("stopped", "sleeping") or (name in self.stopping and now() - self.stopping[name] < 120)
            clean = code in (0, 143) or (code == 137 and not c["oom"] and user_stop)
            if status in ("created",):
                est["state"] = "stopped"
            elif clean and not user_stop and spec.get("desired") == "running" and spec.get("halted") and cid.startswith(spec["halted"][:12]):
                # stopped by the crash guard (or halted after a failed boot): keep the diagnosis visible
                log = self._tail_log(h, name, f"{cid}:{c['finished']}", 400)
                crash = decode_crash(log, None, c["oom"], spec["backend"])
                est["state"] = "crashed" if crash else "stopped"
                est["crash"] = crash
            elif user_stop or clean:
                est["state"] = "sleeping" if spec.get("desired") == "sleeping" else "stopped"
            else:
                est["state"] = "crashed"
                log = self._tail_log(h, name, f"{cid}:{c['finished']}", 400)
                est["crash"] = decode_crash(log, code, c["oom"], spec["backend"]) or {"title": f"Exited with code {code}", "hint": "", "fixes": ["logs"]}
        if est["state"] in ("ready", "booting"):
            est["uptime"] = now() - started_ts if started_ts else None
        else:
            est["reserve"] = 0
        if name in self.boot_seen and est["state"] == "ready" and "took" not in self.boot_seen[name]:
            t0 = self.boot_seen[name].get("t0")
            if t0:
                self.boot_seen[name]["took"] = now() - t0
                est["boot"]["took"] = self.boot_seen[name]["took"]
        est["last_active"] = self.activity.get(name)
        return est

    @staticmethod
    def _mem_bytes(s: str | None) -> int:
        if not s:
            return 0
        m = re.match(r"\s*([\d.]+)\s*([KMGT]?i?B)", s.split("/")[0])
        if not m:
            return 0
        mult = {"B": 1, "KB": 1e3, "KiB": 1024, "MB": 1e6, "MiB": 1024**2, "GB": 1e9, "GiB": GIB, "TB": 1e12, "TiB": 1024**4}.get(m.group(2), 1)
        return int(float(m.group(1)) * mult)

    def _tail_log(self, h: Host, name: str, key: str, n: int = 400) -> str:
        cached = self.log_cache.get(name)
        booting_key = key and ":" not in key
        if cached and cached[0] == key and not booting_key:
            return cached[1]
        r = h.dk_merged("logs", "--tail", str(n), self.cname(name), timeout=15)
        text = r.out or ""
        self.log_cache[name] = (key, text)
        return text

    def _engine_metrics(self, name: str, m: dict) -> dict:
        rates = self.rates.update(name, m)
        out = {
            "running": metric(m, "vllm:num_requests_running", "llamacpp:requests_processing"),
            "waiting": metric(m, "vllm:num_requests_waiting", "llamacpp:requests_deferred"),
            "kv": metric(m, "vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "llamacpp:kv_cache_usage_ratio"),
            "gen_total": metric(m, "vllm:generation_tokens_total", "llamacpp:tokens_predicted_total"),
            "req_total": metric(m, "vllm:request_success_total"),
            "kv_capacity": m.get("_kv_capacity_tokens"),
            **{k: round(v, 3) for k, v in rates.items()},
        }
        hits, queries = metric(m, "vllm:prefix_cache_hits_total", "vllm:gpu_prefix_cache_hits_total"), metric(m, "vllm:prefix_cache_queries_total", "vllm:gpu_prefix_cache_queries_total")
        if hits is not None and queries:
            out["prefix_hit"] = hits / queries
        acc, drafts = metric(m, "vllm:spec_decode_num_accepted_tokens_total"), metric(m, "vllm:spec_decode_num_draft_tokens_total")
        if acc is not None and drafts:
            out["spec_accept"] = acc / drafts
        busy = (out.get("running") or 0) > 0 or (out.get("waiting") or 0) > 0 or (rates.get("req_delta") or 0) > 0 or (rates.get("gen_tps") or 0) > 0.5
        if busy:
            self.activity[name] = now()
        return {k: v for k, v in out.items() if v is not None}

    def _track_history(self, name: str, est: dict) -> None:
        m = est.get("metrics") or {}
        pt = [round(now(), 1), round(m.get("gen_tps") or 0, 1), round((m.get("kv") or 0) * 100, 1), m.get("running") or 0, m.get("waiting") or 0]
        with self.lock:
            buf = self.hist.setdefault(name, [])
            if est.get("state") in ("ready", "booting"):
                buf.append(pt)
                del buf[:-300]

    def _track_host_history(self, hid: str, st: dict) -> None:
        mem = st.get("mem") or {}
        g = (st.get("gpus") or [{}])[0] if st.get("gpus") else {}
        total = mem.get("total") or 1
        pt = [round(now(), 1), round(100 * (1 - (mem.get("available") or 0) / total), 1) if mem.get("total") else None,
              g.get("util"), g.get("temp"), g.get("power")]
        with self.lock:
            buf = self.hhist.setdefault(hid, [])
            buf.append(pt)
            del buf[:-300]

    def _collect_stats(self, h: Host) -> None:
        r = h.dk("stats", "--no-stream", "--format", "{{json .}}", timeout=25)
        out: dict = {"_t": now()}
        for line in r.out.splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                o = json.loads(line)
                out[o.get("Name", "")] = {"cpu": o.get("CPUPerc", ""), "mem": o.get("MemUsage", ""), "memp": o.get("MemPerc", ""),
                                          "net": o.get("NetIO", ""), "block": o.get("BlockIO", ""), "pids": o.get("PIDs", "")}
        with self.lock:
            self.stats[h.id] = out

    def _collect_versions(self, h: Host) -> None:
        dk = " ".join(shlex.quote(x) for x in h.docker())
        r = h.sh(VERSION_SCRIPT.format(docker=dk), timeout=25)
        sec = split_sections(r.out)
        v = {k: (sec.get(k) or "").strip() for k in ("kernel", "os", "driver", "cuda", "docker", "sudo")}
        rt = sec.get("runtimes") or ""
        v["nvidia_runtime"] = "nvidia" in rt
        v["sudo"] = v.get("sudo") == "yes"
        v["arch"] = "arm64" if "aarch64" in v.get("kernel", "") or "arm64" in v.get("kernel", "") else ("x86_64" if "x86_64" in v.get("kernel", "") else "")
        with self.lock:
            self.versions[h.id] = v

    def _collect_ollama(self, h: Host) -> None:
        base = h.api_base(11434)
        info: dict = {"ok": False, "t": now()}
        if base:
            code, body = http_text(base + "/api/version", timeout=1.5)
            if code == 200:
                info["ok"] = True
                with contextlib.suppress(Exception):
                    info["version"] = json.loads(body).get("version")
                c2, tags = http_text(base + "/api/tags", timeout=3)
                c3, ps = http_text(base + "/api/ps", timeout=3)
                with contextlib.suppress(Exception):
                    info["models"] = [
                        {"name": m.get("name"), "policy": origin_check(ollama_origin(m.get("name") or ""), self.conf()), "size": m.get("size"), "family": (m.get("details") or {}).get("family"),
                         "params": (m.get("details") or {}).get("parameter_size"), "quant": (m.get("details") or {}).get("quantization_level"),
                         "modified": m.get("modified_at")}
                        for m in (json.loads(tags).get("models") or [])
                    ] if c2 == 200 else []
                with contextlib.suppress(Exception):
                    info["loaded"] = [
                        {"name": m.get("name"), "size": m.get("size"), "vram": m.get("size_vram"), "expires": m.get("expires_at"),
                         "ctx": m.get("context_length")}
                        for m in (json.loads(ps).get("models") or [])
                    ] if c3 == 200 else []
        with self.lock:
            self.ollama[h.id] = info

    def _measure_downloads(self, h: Host, downloads: list[dict]) -> dict:
        paths = []
        for d in downloads:
            if d.get("model"):
                paths.append(h.cache_dir().rstrip("/") + "/hub/" + hf_cache_folder(d["model"]))
        if not paths:
            return {}
        r = h.sh("du -sbL " + " ".join(shlex.quote(p) for p in paths) + " 2>/dev/null", timeout=20)
        sizes = {}
        for line in r.out.splitlines():
            parts = line.split("\t")
            if len(parts) == 2:
                sizes[parts[1].rstrip("/").rsplit("/", 1)[-1]] = as_int(parts[0], 0)
        return sizes

    # snapshot & publish ------------------------------------------------------------
    def snapshot(self) -> dict:
        tunnels = self.tunnels.list()
        with self.lock:
            hosts = {}
            for h in self.hosts():
                st = dict(self.host_state.get(h.id) or {"id": h.id, "label": h.label, "enabled": h.enabled, "online": False, "why": "starting"})
                st["versions"] = self.versions.get(h.id) or {}
                st["ollama"] = self.ollama.get(h.id) or {}
                st["unmanaged"] = self.unmanaged.get(h.id) or []
                st["reserved"] = sum(e.get("reserve") or 0 for e in self.engine_state.values()
                                     if e.get("host") == h.id and e.get("state") in ("ready", "booting"))
                hosts[h.id] = st
            engines = {}
            for n, e in self.engine_state.items():
                e = dict(e)
                if n in self.gw_stats:
                    e["gateway"] = self.gw_stats[n]
                la = max(self.activity.get(n) or 0, (e.get("started_ts") or 0) if e.get("state") in ("ready", "booting") else 0)
                e["last_active"] = la or None
                busy = self.jobs.busy(n)
                if busy:
                    e["job"] = {"id": busy.id, "kind": busy.kind, "stage": busy.stage, "progress": busy.progress}
                engines[n] = e
            self.store.conf()
            return {"t": now(), "hosts": hosts, "engines": engines, "tunnels": tunnels, "specs": self.store.engines(),
                    "conf_m": self.store._conf_m, "gateway": {k: v for k, v in self.gw_stats.items()}}

    def publish(self, force: bool = False) -> None:
        if not self.serve:
            return
        t = now()
        if not force and t - self._last_publish < 0.8:
            return
        self._last_publish = t
        self.bus.publish("snapshot", self.snapshot())

    def refresh(self, host_ids: list[str] | None = None) -> None:
        """One synchronous collection pass (CLI, or right after an action)."""
        hs = [h for h in self.hosts() if not host_ids or h.id in host_ids]

        def one(h):
            with contextlib.suppress(Exception):
                self.collect_host(h)
        threads = [threading.Thread(target=one, args=(h,), daemon=True) for h in hs]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)
        self.publish(force=True)

    def history(self) -> dict:
        with self.lock:
            return {"engines": {k: v[-300:] for k, v in self.hist.items()}, "hosts": {k: v[-300:] for k, v in self.hhist.items()}}

    # autopilot: transitions, crash guard, idle sleep, webui sync ------------------------
    def autopilot(self, h: Host) -> None:
        conf = self.conf()
        for name, e in list(self.engine_state.items()):
            if e.get("host") != h.id:
                continue
            st = e.get("state")
            prev = self.prev_state.get(name)
            self.prev_state[name] = st
            if prev and prev != st:
                self._on_transition(name, prev, st, e)
            c = e.get("container") or {}
            guard = int(conf.get("crash_guard") or 0)
            key = f"{c.get('id')}"
            if (guard and c.get("restarts", 0) >= guard and st in ("crashed", "booting") and e.get("crash") and key not in self.guarded
                    and c.get("status") in ("restarting", "running")):
                self.guarded.add(key)
                h.dk("update", "--restart=no", self.cname(name), timeout=20)
                h.dk("stop", "-t", "5", self.cname(name), timeout=40)
                self.store.update_engine(name, lambda s, c=c.get("id") or "": s.__setitem__("halted", c))
                self.events.add("error", name, f"{name} crash-looped {c.get('restarts')}× — stopped it so it stops fighting for memory. {e['crash'].get('title')}",
                                fixes=e["crash"].get("fixes"))
            idle = float(e.get("idle_sleep_min") or 0)
            if idle and st == "ready" and not self.jobs.busy(name):
                la = max(self.activity.get(name) or 0, e.get("started_ts") or 0)
                if la and now() - la > idle * 60:
                    self.activity[name] = now()
                    self.jobs.start("sleep", f"Idle sleep {name}", name, lambda job, n=name, m=idle: self.op_stop(n, job, sleeping=True, reason=f"after {m:g} min idle"))

    def _on_transition(self, name: str, prev: str, st: str, e: dict) -> None:
        if st == "ready":
            took = (e.get("boot") or {}).get("took")
            self.events.add("ok", name, f"{name} is ready" + (f" (boot {fmt_dur(took)})" if took and prev == "booting" else ""))
            self.schedule_webui_sync()
        elif st == "crashed":
            cr = e.get("crash") or {}
            self.events.add("error", name, f"{name} crashed: {cr.get('title', 'unknown cause')}", fixes=cr.get("fixes"), line=cr.get("line"))
            self.schedule_webui_sync()
        elif st in ("stopped", "sleeping", "absent") and prev in ("ready", "booting"):
            self.events.add("info", name, f"{name} {'is asleep' if st == 'sleeping' else 'stopped'}")
            self.schedule_webui_sync()
        elif st == "booting" and prev in ("stopped", "absent", "sleeping", "crashed"):
            self.events.add("info", name, f"{name} is booting")

    def schedule_webui_sync(self, delay: float = 3.0) -> None:
        w = self.conf().get("webui") or {}
        if not self.serve or not w.get("auto_sync") or not (w.get("api_key") or (w.get("email") and w.get("password"))):
            return
        if self._sync_timer:
            self._sync_timer.cancel()

        def go():
            try:
                res = self.webui_sync()
                if res.get("added") or res.get("removed"):
                    self.events.add("info", "webui", f"Open WebUI synced: +{len(res['added'])} −{len(res['removed'])}")
            except Exception as e:  # noqa: BLE001
                self.events.add("warn", "webui", f"Open WebUI sync failed: {e}")

        self._sync_timer = threading.Timer(delay, go)
        self._sync_timer.daemon = True
        self._sync_timer.start()


def _parse_ts(s: str) -> float | None:
    if not s or s.startswith("0001-"):
        return None
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)?", s)
    if not m:
        return None
    import calendar
    y, mo, d, hh, mm, ss = (int(m.group(i)) for i in range(1, 7))
    t = calendar.timegm((y, mo, d, hh, mm, ss, 0, 0, 0))
    frac = m.group(7)
    if frac:
        t += float("0." + frac[:6])
    tz = m.group(8)
    if tz and tz != "Z":
        sign = 1 if tz[0] == "+" else -1
        t -= sign * (int(tz[1:3]) * 3600 + int(tz[4:6]) * 60)
    return t


# ───────────────────────────────────────────────────────────────── HTTP server: console API, live stream, gateway

MAX_BODY = 64 * 1024 * 1024
LOOPBACK = {"127.0.0.1", "localhost", "::1", "[::1]"}


def session_cookie(token: str) -> str:
    return hmac.new(token.encode(), b"vllm-lab-session", hashlib.sha256).hexdigest()


LOGIN_PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vllm-lab</title><style>
:root{color-scheme:dark}body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0a0c0f;color:#e7eaee;font:15px/1.4 system-ui,sans-serif}
form{display:grid;gap:12px;width:min(320px,90vw)}b{font:600 13px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:#f0a53a}
input,button{font:inherit;padding:11px 12px;border:1px solid #2a3039;background:#12161b;color:inherit}button{background:#f0a53a;color:#111;border:0;font-weight:600;cursor:pointer}
p{color:#ff7a7a;margin:0;min-height:1.2em;font-size:13px}</style>
<form method="post" action="/login"><b>vllm-lab</b><input type="password" name="token" placeholder="Access token" autofocus required><button>Enter</button><p>%s</p></form></html>"""


class LabServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, handler, lab: Lab, gateway_only: bool = False):
        self.lab = lab
        self.gateway_only = gateway_only
        family = socket.AF_INET6 if ":" in addr[0] else socket.AF_INET
        self.address_family = family
        super().__init__(addr, handler)


class Handler(BaseHTTPRequestHandler):
    server_version = f"vllm-lab/{VERSION}"
    protocol_version = "HTTP/1.1"

    # plumbing -------------------------------------------------------------------------
    @property
    def lab(self) -> Lab:
        return self.server.lab  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # noqa: D401 — quiet by default
        if os.environ.get("VLLM_LAB_DEBUG"):
            sys.stderr.write("[http] " + (fmt % args) + "\n")

    def _headers_common(self, ctype: str, length: int | None = None, extra: dict | None = None) -> None:
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for k, v in (extra or {}).items():
            self.send_header(k, v)

    def _unread_body(self) -> bool:
        return self.command in ("POST", "PUT", "DELETE") and int(self.headers.get("Content-Length") or 0) > 0 and not getattr(self, "_body_read", False)

    def send_json(self, obj, code: int = 200, extra: dict | None = None) -> None:
        raw = json.dumps(obj, default=str).encode()
        if self._unread_body():
            self.close_connection = True
            extra = {**(extra or {}), "Connection": "close"}
        self.send_response(code)
        self._headers_common("application/json; charset=utf-8", len(raw), extra)
        self.end_headers()
        self.wfile.write(raw)

    def send_html(self, html: str, code: int = 200, extra: dict | None = None) -> None:
        raw = html.encode()
        self.send_response(code)
        csp = ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        self._headers_common("text/html; charset=utf-8", len(raw), {"Content-Security-Policy": csp, "X-Frame-Options": "DENY", **(extra or {})})
        self.end_headers()
        self.wfile.write(raw)

    def send_err(self, e: Exception) -> None:
        if isinstance(e, LabError):
            self.send_json(e.as_dict(), e.code if 400 <= e.code < 600 else 400)
        else:
            if os.environ.get("VLLM_LAB_DEBUG"):
                traceback.print_exc()
            self.send_json({"error": f"{type(e).__name__}: {e}", "hint": "", "fixes": []}, 500)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            raise LabError("Request too large", code=413)
        raw = self.rfile.read(n) if n else b""
        self._body_read = True
        if not raw:
            return {}
        ctype = self.headers.get("Content-Type") or ""
        if "application/json" in ctype:
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                raise LabError("Body is not valid JSON", code=400)
            return data if isinstance(data, dict) else {"_": data}
        form = parse_qs(raw.decode("utf-8", "replace"))
        return {k: v[0] if v else "" for k, v in form.items()}

    def start_stream(self, ctype: str = "text/event-stream") -> None:
        self.send_response(200)
        self._headers_common(ctype, None, {"Transfer-Encoding": "chunked", "X-Accel-Buffering": "no", "Connection": "keep-alive"})
        self.end_headers()

    def chunk(self, data: bytes) -> None:
        if not data:
            return
        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def end_chunks(self) -> None:
        with contextlib.suppress(Exception):
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    # security ----------------------------------------------------------------------------
    def _host_ok(self) -> bool:
        conf = self.lab.conf()
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        allowed = set(LOOPBACK) | {h.lower() for h in (conf.get("ui") or {}).get("allowed_hosts") or []}
        listen = (conf.get("ui") or {}).get("listen") or "127.0.0.1"
        allowed.add(listen.lower())
        gl = self.lab.gateway_listen_addr()
        if gl:
            allowed.add(gl.lower())
        if getattr(self.server, "gateway_only", False):
            return True
        return name in allowed

    def _authed(self) -> bool:
        tok = ((self.lab.conf().get("ui") or {}).get("token") or "").strip()
        if not tok:
            return True
        auth = self.headers.get("Authorization") or ""
        if auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), tok):
            return True
        cookie = self.headers.get("Cookie") or ""
        m = re.search(r"(?:^|;\s*)vllm_lab=([0-9a-f]{64})", cookie)
        return bool(m) and hmac.compare_digest(m.group(1), session_cookie(tok))

    def _mutation_ok(self) -> bool:
        if (self.headers.get("X-Lab-Client") or "") == "1":
            origin = self.headers.get("Origin")
            if origin:
                o = urlparse(origin).netloc.lower()
                return o == (self.headers.get("Host") or "").lower()
            return True
        auth = self.headers.get("Authorization") or ""
        return auth.startswith("Bearer ") and bool(((self.lab.conf().get("ui") or {}).get("token") or "").strip())

    # dispatch ------------------------------------------------------------------------
    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")

    def do_OPTIONS(self):
        if self.path.startswith("/v1/") and self._cors():
            self.send_response(204)
            self._headers_common("text/plain", 0, {"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Authorization, Content-Type",
                                                   "Access-Control-Allow-Methods": "GET, POST, OPTIONS"})
            self.end_headers()
            return
        self.send_json({"error": "not allowed"}, 405)

    def _dispatch(self, method: str) -> None:
        self._body_read = False
        try:
            u = urlparse(self.path)
            path = u.path.rstrip("/") or "/"
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            if not self._host_ok():
                self.send_json({"error": "Host header not allowed (DNS rebinding protection). Add it to ui.allowed_hosts."}, 421)
                return
            if path.startswith("/v1/") or path == "/v1":
                self.gateway(method, path, u.query)
                return
            if getattr(self.server, "gateway_only", False):
                self.send_json({"error": "gateway listener serves /v1 only"}, 404)
                return
            if path == "/healthz":
                self.send_json({"ok": True, "version": VERSION})
                return
            if path == "/login":
                self.login(method)
                return
            if not self._authed():
                if path == "/":
                    self.send_html(LOGIN_PAGE % "")
                else:
                    self.send_json({"error": "login required"}, 401)
                return
            if path == "/":
                self.send_html(UI_HTML.replace("__VERSION__", VERSION))
                return
            if path == "/favicon.svg" or path == "/favicon.ico":
                raw = FAVICON.encode()
                self.send_response(200)
                self._headers_common("image/svg+xml", len(raw))
                self.end_headers()
                self.wfile.write(raw)
                return
            if not path.startswith("/api/"):
                self.send_json({"error": "not found"}, 404)
                return
            if method in ("POST", "DELETE") and not self._mutation_ok():
                self.send_json({"error": "Missing X-Lab-Client header (cross-site request blocked)"}, 403)
                return
            self.api(method, path[4:], q)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001
            with contextlib.suppress(Exception):
                self.send_err(e)

    def login(self, method: str) -> None:
        tok = ((self.lab.conf().get("ui") or {}).get("token") or "").strip()
        if method != "POST":
            self.send_html(LOGIN_PAGE % "")
            return
        given = str(self.body().get("token") or "")
        if tok and hmac.compare_digest(given, tok):
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie", f"vllm_lab={session_cookie(tok)}; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        time.sleep(0.6)
        self.send_html(LOGIN_PAGE % "Wrong token", 401)

    # api ---------------------------------------------------------------------------------
    def api(self, method: str, path: str, q: dict) -> None:
        lab = self.lab
        parts = [p for p in path.split("/") if p]
        b = self.body() if method in ("POST", "DELETE") else {}
        head = parts[0] if parts else ""

        def job(kind, title, target, fn):
            j = lab.jobs.start(kind, title, target, fn)
            self.send_json({"job": j.id, "title": title})

        if method == "GET" and head == "bootstrap":
            self.send_json({"version": VERSION, "settings": lab.settings_public(), "specs": lab.store.engines(), "blueprints": lab.blueprints(),
                            "snapshot": lab.snapshot(), "jobs": lab.jobs.list(), "events": lab.events.recent(150), "history": lab.history(),
                            "stages": BOOT_STAGES})
            return
        if method == "GET" and head == "stream":
            self.stream_events()
            return
        if method == "GET" and head == "snapshot":
            self.send_json(lab.snapshot())
            return
        if method == "GET" and head == "status":  # v1 compatibility
            snap = lab.snapshot()
            self.send_json({"rows": list(snap["engines"].values()), "hosts": lab.conf()["hosts"]})
            return
        if method == "GET" and head == "history":
            self.send_json(lab.history())
            return
        if method == "GET" and head == "events":
            self.send_json({"events": lab.events.recent(int(q.get("n") or 200))})
            return
        if head == "jobs":
            if method == "GET" and len(parts) == 1:
                self.send_json({"jobs": lab.jobs.list()})
                return
            if len(parts) == 3 and parts[2] == "cancel" and method == "POST":
                j = lab.jobs.get(parts[1])
                if not j:
                    raise LabError("No such job", code=404)
                j.cancelled = True
                self.send_json({"msg": "cancelling"})
                return
        # engines -------------------------------------------------------------------
        if head == "engines":
            if len(parts) == 1:
                if method == "GET":
                    self.send_json({"specs": lab.store.engines()})
                    return
                spec = lab.upsert_engine(b.get("spec") or {}, create=bool(b.get("create")), old_name=b.get("old_name") or "")
                if b.get("launch"):
                    name = spec["name"]
                    j = lab.jobs.start("start", f"Start {name}", name,
                                       lambda jb: lab.op_start(name, jb, on_conflict=b.get("on_conflict") or "ask", evict=b.get("evict"), recreate=True))
                    self.send_json({"spec": spec, "job": j.id})
                else:
                    self.send_json({"spec": spec})
                return
            name = parts[1]
            if not NAME_RE.match(name):
                raise LabError("Bad engine name", code=400)
            sub = parts[2] if len(parts) > 2 else ""
            if method == "GET":
                if sub == "logs" and len(parts) > 3 and parts[3] == "stream":
                    self.stream_logs(name, int(q.get("tail") or 300))
                    return
                if sub == "logs":
                    self.send_json(lab.logs(name, int(q.get("tail") or 600), q.get("since") or ""))
                    return
                if sub == "command":
                    self.send_json(lab.run_command(name))
                    return
                if sub == "inspect":
                    spec = lab.spec(name)
                    obj = lab.inspect_one(lab.host(spec["host"]), lab.cname(name))
                    self.send_json({"json": obj})
                    return
                if sub == "bench":
                    self.send_json({"runs": lab.bench_history(name)})
                    return
                if not sub:
                    self.send_json({"spec": lab.spec(name), "state": lab.engine_state.get(name)})
                    return
            if method == "POST":
                lab.spec(name)
                busy = lab.jobs.busy(name)
                if sub in ("start", "recreate", "restart", "fix", "bench") and busy and not b.get("force_parallel"):
                    raise LabError(f"{name} is busy: {busy.title}", "Wait for it to finish or cancel it.", code=409, data={"job": busy.id})
                if sub == "start":
                    job("start", f"Start {name}", name, lambda jb: lab.op_start(
                        name, jb, on_conflict=b.get("on_conflict") or "ask", evict=b.get("evict"), util=as_float(b.get("util")), recreate=as_bool(b.get("recreate"))))
                    return
                if sub == "recreate":
                    job("start", f"Recreate {name}", name, lambda jb: lab.op_start(name, jb, on_conflict=b.get("on_conflict") or "ask", evict=b.get("evict"), recreate=True))
                    return
                if sub == "stop":
                    job("stop", f"Stop {name}", name, lambda jb: lab.op_stop(name, jb, sleeping=as_bool(b.get("sleep"))))
                    return
                if sub == "restart":
                    job("restart", f"Restart {name}", name, lambda jb: lab.op_restart(name, jb))
                    return
                if sub == "remove":
                    self.send_json({"msg": lab.op_remove(name, None, delete_spec=as_bool(b.get("delete")))})
                    return
                if sub == "fix":
                    fix = str(b.get("fix") or "")
                    job("fix", f"Fix {name}: {fix}", name, lambda jb: lab.op_fix(name, fix, jb, b.get("params") or {}))
                    return
                if sub == "probe":
                    self.send_json(lab.probe_chat(name, b.get("prompt") or "Reply with the single word OK."))
                    return
                if sub == "bench":
                    job("bench", f"Bench {name}", name, lambda jb: lab.op_bench(
                        name, jb, as_int(b.get("concurrency"), 4), as_int(b.get("requests"), 16), as_int(b.get("max_tokens"), 256),
                        str(b.get("prompt") or ""), as_int(b.get("prompt_tokens"), 0)))
                    return
                if sub == "fit":
                    spec = lab.spec(name)
                    self.send_json(lab.fit_check(spec, lab.host(spec["host"]), as_float(b.get("util"))))
                    return
                if sub == "blueprint":
                    self.send_json(lab.save_blueprint(b.get("id") or name, b.get("title") or "", lab.spec(name), b.get("tags")))
                    return
        if method == "POST" and head == "preview":
            self.send_json(lab.preview(b.get("spec") or b))
            return
        if method == "POST" and head == "fit":
            spec = normalize_spec({**(b.get("spec") or {}), "name": (b.get("spec") or {}).get("name") or "preview"})
            self.send_json(lab.fit_check(spec, lab.host(spec["host"])))
            return
        # hugging face / planner --------------------------------------------------------
        if method == "GET" and head == "hf":
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "search":
                self.send_json({"rows": lab.hf_search(q.get("q", ""), q.get("host", ""), q.get("kind", ""), int(q.get("limit") or 30), q.get("sort") or "downloads")})
                return
            if sub == "model":
                self.send_json(lab.hf_meta(q.get("id", ""), q.get("host", ""), as_bool(q.get("refresh"))))
                return
            if sub == "whoami":
                self.send_json(lab.hf_whoami())
                return
        if method == "GET" and head == "plan":
            self.send_json(lab.plan(q.get("model", ""), q.get("host") or "titan", q.get("extra") or ""))
            return
        # library ------------------------------------------------------------------------
        if head == "library":
            if method == "GET":
                self.send_json(lab.library(q.get("host") or "titan"))
                return
            sub = parts[1] if len(parts) > 1 else ""
            if sub == "download":
                model, hid = str(b.get("model") or ""), str(b.get("host") or "titan")
                job("download", f"Download {model}", f"library@{hid}", lambda jb: lab.op_download(model, hid, jb))
                return
            if sub == "delete":
                self.send_json({"msg": lab.delete_weights(str(b.get("model") or ""), str(b.get("host") or "titan"))})
                return
        if method == "POST" and head == "ollama":
            hid, op, model = str(b.get("host") or "titan"), str(b.get("op") or ""), str(b.get("model") or "")
            if op in ("pull", "load") and origin_check(ollama_origin(model), lab.conf())["blocked"]:
                raise LabError(f"{model} is blocked by the origin policy", "", ["policy"], code=403)
            if op == "pull":
                job("ollama", f"Ollama pull {model}", f"ollama@{hid}", lambda jb: lab.ollama_op(hid, op, model, jb))
            else:
                self.send_json({"msg": lab.ollama_op(hid, op, model)})
                lab.pool.submit(lab._collect_ollama, lab.host(hid))
            return
        # blueprints ----------------------------------------------------------------------
        if head == "blueprints":
            if method == "GET":
                self.send_json({"blueprints": lab.blueprints()})
                return
            if len(parts) > 1 and parts[1] == "delete":
                lab.delete_blueprint(str(b.get("id") or ""))
                self.send_json({"blueprints": lab.blueprints()})
                return
            if len(parts) > 1 and parts[1] == "import":
                items = b.get("blueprints") or {}
                n = 0
                for bid, bp in (items.items() if isinstance(items, dict) else []):
                    if isinstance(bp, dict) and isinstance(bp.get("spec"), dict):
                        lab.save_blueprint(bid, bp.get("title") or bid, bp["spec"], bp.get("tags"))
                        n += 1
                self.send_json({"imported": n, "blueprints": lab.blueprints()})
                return
            self.send_json(lab.save_blueprint(b.get("id") or "", b.get("title") or "", b.get("spec") or {}, b.get("tags")))
            return
        # hosts ---------------------------------------------------------------------------
        if head == "hosts":
            if method == "GET" and len(parts) == 1:
                self.send_json({"hosts": lab.conf()["hosts"]})
                return
            if method == "POST" and len(parts) == 1:
                h = lab.upsert_host(b.get("host") or b, old_id=str(b.get("old_id") or ""))
                lab.pool.submit(lab.refresh, [h["id"]])
                self.send_json({"host": h})
                return
            hid = parts[1]
            sub = parts[2] if len(parts) > 2 else ""
            if method == "GET" and sub == "containers":
                self.send_json(lab.docker_fleet(hid))
                return
            if method == "POST" and sub == "docker":
                self.send_json(lab.docker_op(hid, str(b.get("op") or ""), str(b.get("target") or b.get("container") or "")))
                return
            if method == "POST" and sub == "pull":
                img = str(b.get("image") or "")
                if not IMAGE_RE.match(img):
                    raise LabError("Bad image name")
                job("pull", f"Pull {img}", f"host@{hid}", lambda jb: lab.ensure_image(lab.host(hid), img, jb) and f"Pulled {img}")
                return
            if method == "POST" and sub == "test":
                job("host-test", f"Test {hid}", f"host@{hid}", lambda jb: lab.op_test_host(hid, jb))
                return
            if method == "POST" and sub == "remove":
                lab.remove_host(hid)
                self.send_json({"msg": f"removed {hid}"})
                return
            if method == "POST" and sub == "flush":
                ok, msg = lab.flush_cache(lab.host(hid))
                if not ok:
                    raise LabError("Could not flush the page cache", msg)
                lab.pool.submit(lab.refresh, [hid])
                self.send_json({"msg": msg})
                return
            if method == "POST" and sub == "adopt":
                self.send_json({"spec": lab.adopt(hid, str(b.get("container") or ""))})
                return
        # doctor ----------------------------------------------------------------------------
        if method == "GET" and head == "doctor":
            self.send_json(lab.doctor(deep=as_bool(q.get("deep"), True)))
            return
        if method == "POST" and head == "doctor":
            fix, hid, target = str(b.get("fix") or ""), str(b.get("host") or ""), str(b.get("target") or "")
            if fix in ("pull-image", "recreate", "start") or (target and fix not in ("adopt",)):
                job("fix", f"Fix: {fix}", target or f"host@{hid}", lambda jb: lab.doctor_fix(fix, hid, target, jb))
            else:
                self.send_json({"msg": lab.doctor_fix(fix, hid, target)})
            return
        # open webui ------------------------------------------------------------------------
        if head == "webui":
            sub = parts[1] if len(parts) > 1 else ""
            if method == "GET":
                self.send_json(lab.webui_overview(deep=as_bool(q.get("deep"), True)))
                return
            if sub == "save":
                lab.save_webui(b)
                self.send_json({"msg": "saved", "settings": lab.settings_public()})
                return
            if sub == "sync":
                r = lab.webui_sync(prune_dead=as_bool(b.get("prune"), True))
                lab.events.add("info", "webui", f"Open WebUI synced: +{len(r['added'])} −{len(r['removed'])}")
                self.send_json(r)
                return
        # settings --------------------------------------------------------------------------
        if head == "settings":
            if method == "GET":
                self.send_json(lab.settings_public())
                return
            self.send_json(lab.update_settings(b))
            return
        if method == "GET" and head == "catalog":
            self.send_json({"models": lab.gw_catalog()})
            return
        raise LabError(f"No route {method} /api/{path}", code=404)

    # live streams ---------------------------------------------------------------------------
    def stream_events(self) -> None:
        lab = self.lab
        q = lab.bus.subscribe()
        try:
            self.start_stream()
            self.chunk(b"retry: 2000\n\n")
            self.chunk(("event: snapshot\ndata: " + json.dumps(lab.snapshot(), default=str) + "\n\n").encode())
            while True:
                try:
                    kind, data = q.get(timeout=15)
                except queue.Empty:
                    self.chunk(b": ping\n\n")
                    continue
                self.chunk((f"event: {kind}\ndata: " + json.dumps(data, default=str) + "\n\n").encode())
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            lab.bus.unsubscribe(q)
            self.close_connection = True

    def stream_logs(self, name: str, tail: int) -> None:
        lab = self.lab
        spec = lab.spec(name)
        h = lab.host(spec["host"])
        h.require()
        argv = h.argv(h.docker() + ["logs", "-f", "--timestamps", "--tail", str(max(0, min(5000, tail))), lab.cname(name)], tty=True)
        p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        lines: queue.Queue = queue.Queue(maxsize=5000)

        def pump():
            assert p.stdout is not None
            buf = b""
            while True:
                chunk = p.stdout.read1(65536) if hasattr(p.stdout, "read1") else p.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                parts = re.split(rb"\r?\n|\r", buf)
                buf = parts.pop()
                for part in parts:
                    with contextlib.suppress(queue.Full):
                        lines.put_nowait(part.decode("utf-8", "replace"))
            lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        try:
            self.start_stream()
            while True:
                batch = []
                try:
                    item = lines.get(timeout=10)
                except queue.Empty:
                    self.chunk(b": ping\n\n")
                    continue
                if item is None:
                    self.chunk(b"event: end\ndata: {}\n\n")
                    break
                batch.append(item)
                while len(batch) < 400:
                    try:
                        item = lines.get_nowait()
                    except queue.Empty:
                        break
                    if item is None:
                        lines.put(None)
                        break
                    batch.append(item)
                self.chunk(("data: " + json.dumps(batch) + "\n\n").encode())
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with contextlib.suppress(Exception):
                p.kill()
            self.end_chunks()
            self.close_connection = True

    # gateway -------------------------------------------------------------------------------
    def _cors(self) -> dict:
        return {"Access-Control-Allow-Origin": "*"} if (self.lab.conf().get("gateway") or {}).get("cors") else {}

    def _oai_error(self, code: int, msg: str, typ: str = "invalid_request_error") -> None:
        self.send_json({"error": {"message": msg, "type": typ, "code": code}}, code, self._cors())

    def gateway(self, method: str, path: str, query: str) -> None:
        lab = self.lab
        conf = lab.conf()
        gw = conf.get("gateway") or {}
        if not gw.get("enabled", True):
            self._oai_error(404, "The vllm-lab gateway is turned off (Settings → Gateway).")
            return
        key = (gw.get("key") or "").strip()
        auth = self.headers.get("Authorization") or ""
        console = (self.headers.get("X-Lab-Client") == "1" and self._authed() and not getattr(self.server, "gateway_only", False))
        if key and not console and not (auth.startswith("Bearer ") and hmac.compare_digest(auth[7:].strip(), key)):
            self._oai_error(401, "Missing or wrong API key for the vllm-lab gateway.", "authentication_error")
            return
        if not key and not console and not self._authed() and not getattr(self.server, "gateway_only", False):
            self._oai_error(401, "Login required.", "authentication_error")
            return
        origin = self.headers.get("Origin")
        if origin and not key and not self._cors() and urlparse(origin).netloc.lower() != (self.headers.get("Host") or "").lower():
            self._oai_error(403, "Cross-site requests to the gateway are blocked. Set a gateway API key to allow browser apps.", "permission_error")
            return
        if method == "GET" and path == "/v1/models":
            data = [{"id": m["id"], "object": "model", "created": 0, "owned_by": f"vllm-lab/{m['host']}",
                     "meta": {"engine": m["engine"], "state": m["state"], "backend": m["backend"], "model": m["model"], "max_model_len": m.get("max_len")}}
                    for m in lab.gw_catalog()]
            self.send_json({"object": "list", "data": data}, extra=self._cors())
            return
        if method != "POST":
            self._oai_error(405, "Use POST.")
            return
        if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
            self._oai_error(415, "Send Content-Type: application/json.")
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            self._oai_error(413, "Request too large.")
            return
        raw = self.rfile.read(n) if n else b"{}"
        self._body_read = True
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._oai_error(400, "Body is not JSON.")
            return
        model = str(body.get("model") or "")
        route = lab.gw_resolve(model)
        if not route:
            self._oai_error(404, f"No engine serves '{model}'. GET /v1/models lists what is available.")
            return
        stream = bool(body.get("stream"))
        wait_timeout = float(gw.get("wake_timeout") or 900)
        started_stream = False
        if "engine" in route:
            name = route["engine"]
            spec = lab.spec(name)
            st = (lab.engine_state.get(name) or {}).get("state")
            if st != "ready":
                if not gw.get("autowake", True):
                    self._oai_error(503, f"{name} is {st or 'not running'} and auto-wake is off.", "server_error")
                    return
                result: dict = {}

                def waker():
                    try:
                        lab.gw_wake(name, wait_timeout)
                        result["ok"] = True
                    except LabError as e:
                        result["err"] = e
                t = threading.Thread(target=waker, daemon=True)
                t.start()
                if stream:
                    self.start_stream()
                    started_stream = True
                    while t.is_alive():
                        t.join(5)
                        e = lab.engine_state.get(name) or {}
                        b = e.get("boot") or {}
                        self.chunk(f": waking {name} · {e.get('state')} {b.get('stage') or ''} {b.get('pct') or ''}\n\n".encode())
                else:
                    t.join(wait_timeout + 30)
                if "err" in result or "ok" not in result:
                    err = result.get("err")
                    msg = err.msg if err else f"{name} did not wake in time"
                    if started_stream:
                        self.chunk(("data: " + json.dumps({"error": {"message": msg, "type": "server_error"}}) + "\n\ndata: [DONE]\n\n").encode())
                        self.end_chunks()
                    else:
                        self._oai_error(503, msg, "server_error")
                    return
            h = lab.host(spec["host"])
            base = h.api_base(int(spec["port"]))
            body["model"] = served_id(spec)
            key_name = name
        else:
            h = lab.host(route["ollama"])
            base = h.api_base(11434)
            body["model"] = route["model"]
            key_name = f"ollama@{route['ollama']}"
        if not base:
            self._oai_error(502, "No route to that engine (SSH tunnel down).", "server_error")
            return
        st = lab.gw_stats.setdefault(key_name, {"requests": 0, "errors": 0, "last": 0})
        st["requests"] += 1
        st["last"] = now()
        lab.activity[key_name] = now()
        try:
            conn, _ = _conn_for(base, timeout=float(gw.get("request_timeout") or 900))
            fwd_path = path + (f"?{query}" if query else "")
            conn.request("POST", fwd_path, body=json.dumps(body), headers={"Content-Type": "application/json",
                                                                             "Accept": self.headers.get("Accept") or "application/json"})
            resp = conn.getresponse()
        except (OSError, http.client.HTTPException) as e:
            st["errors"] += 1
            if started_stream:
                self.chunk(("data: " + json.dumps({"error": {"message": f"upstream: {e}"}}) + "\n\n").encode())
                self.end_chunks()
            else:
                self._oai_error(502, f"Upstream engine did not answer: {e}", "server_error")
            return
        if resp.status >= 400:
            st["errors"] += 1
        ctype = resp.getheader("Content-Type") or "application/json"
        try:
            if not started_stream:
                self.send_response(resp.status)
                self._headers_common(ctype, None, {"Transfer-Encoding": "chunked", "X-Vllm-Lab-Engine": key_name, **self._cors()})
                self.end_headers()
            while True:
                chunk = resp.read1(65536) if hasattr(resp, "read1") else resp.read(8192)
                if not chunk:
                    break
                self.chunk(chunk)
                lab.activity[key_name] = now()
            self.end_chunks()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()
            self.close_connection = started_stream or self.close_connection


# ───────────────────────────────────────────────────────────────── serve

def serve(listen: str | None = None, port: int | None = None, open_browser: bool = True) -> None:
    lab = Lab(serve=True)
    conf = lab.conf()
    ui = conf.get("ui") or {}
    listen = listen or ui.get("listen") or "127.0.0.1"
    port = port or int(ui.get("port") or 58120)
    if listen not in LOOPBACK and not (ui.get("token") or "").strip():
        print("refusing to listen on a non-loopback address without an access token.\n"
              "  set one:  vllm-lab config --ui-token <long-random-string>", file=sys.stderr)
        raise SystemExit(2)
    try:
        httpd = LabServer((listen, port), Handler, lab)
    except OSError as e:
        print(f"cannot listen on {listen}:{port}: {e}", file=sys.stderr)
        raise SystemExit(1)
    lab.start_collector()
    servers = [httpd]
    gw = conf.get("gateway") or {}
    gl = lab.gateway_listen_addr() if gw.get("enabled", True) else ""
    if gl and not (gw.get("key") or "").strip():
        print(f"gateway listener {gl} not started: set a gateway API key first (Settings → Gateway)", file=sys.stderr)
        lab.events.add("warn", "gateway", f"Extra gateway listener on {gl} needs an API key — not started")
        gl = ""
    if gl:
        try:
            gsrv = LabServer((gl, int(gw.get("port") or 58130)), Handler, lab, gateway_only=True)
            servers.append(gsrv)
            threading.Thread(target=gsrv.serve_forever, daemon=True, name="gateway").start()
            print(f"gateway   http://{gl}:{gw.get('port') or 58130}/v1")
        except OSError as e:
            print(f"gateway listener {gl}: {e}", file=sys.stderr)
            lab.events.add("warn", "gateway", f"Could not listen on {gl}:{gw.get('port')}: {e}")
    host_disp = f"[{listen}]" if ":" in listen else listen
    print(f"vllm-lab {VERSION}")
    print(f"console   http://{host_disp}:{port}")
    print(f"gateway   http://{host_disp}:{port}/v1")
    lab.events.add("info", "system", f"vllm-lab {VERSION} console started on {host_disp}:{port}")

    def shutdown(*_):
        lab._stop.set()
        lab.tunnels.close_all()
        for s in servers:
            threading.Thread(target=s.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, shutdown)
    if open_browser and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and not os.environ.get("INVOCATION_ID"):
        with contextlib.suppress(Exception):
            import webbrowser
            threading.Timer(0.8, lambda: webbrowser.open(f"http://{host_disp}:{port}")).start()
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


# ───────────────────────────────────────────────────────────────── planner math (mirrored in the console)

def kv_bytes_per_el(kv_dtype: str, model_dtype: str = "bfloat16") -> float:
    k = (kv_dtype or "auto").lower()
    if "fp8" in k or "int8" in k:
        return 1.0
    if "fp4" in k:
        return 0.5
    if k in ("float32", "fp32"):
        return 4.0
    if k == "auto" and "float32" in (model_dtype or ""):
        return 4.0
    return 2.0


def plan_numbers(p: dict, ctx: int, seqs: int = 1, kv_dtype: str = "fp8") -> dict:
    meta = p.get("meta") or {}
    kv = meta.get("kv") or {}
    bpe = kv_bytes_per_el(kv_dtype, kv.get("dtype", ""))
    weights = int(meta.get("weights_bytes") or meta.get("gguf_bytes") or 0) + sum(d.get("bytes") or 0 for d in p.get("drafts") or [])
    sw = kv.get("sliding_window") or ctx
    per_seq = (kv.get("kv_elems_full") or 0) * bpe * ctx + (kv.get("kv_elems_sliding") or 0) * bpe * min(ctx, sw) + (kv.get("mamba_bytes_per_seq") or 0)
    overhead = int(p.get("overhead") or 2.5 * GIB)
    need = weights + overhead + per_seq * max(1, seqs)
    total = p.get("total") or 0
    util = need / total if total else None
    return {"weights": weights, "kv_per_seq": per_seq, "kv_total": per_seq * max(1, seqs), "overhead": overhead, "need": need,
            "util": util, "util_rec": min(0.95, math.ceil((util or 0) * 1.06 * 100) / 100) if util else None,
            "kv_known": bool(kv.get("known")), "tok_bytes": (kv.get("kv_elems_full") or 0) * bpe}


# ───────────────────────────────────────────────────────────────── CLI

class C:
    on = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    @staticmethod
    def w(code: str, s) -> str:
        return f"\033[{code}m{s}\033[0m" if C.on else str(s)

    dim = staticmethod(lambda s: C.w("2", s))
    b = staticmethod(lambda s: C.w("1", s))
    ok = staticmethod(lambda s: C.w("32", s))
    bad = staticmethod(lambda s: C.w("31", s))
    warn = staticmethod(lambda s: C.w("33", s))
    acc = staticmethod(lambda s: C.w("38;5;214", s))
    blue = staticmethod(lambda s: C.w("36", s))


STATE_COLOR = {"ready": C.ok, "booting": C.blue, "crashed": C.bad, "offline": C.dim, "stopped": C.dim, "sleeping": C.dim, "absent": C.dim}


def run_job_cli(lab: Lab, kind: str, title: str, target: str, fn) -> int:
    q = lab.bus.subscribe()
    job = lab.jobs.start(kind, title, target, fn)
    last_line = 0
    inline = False
    try:
        while True:
            try:
                k, data = q.get(timeout=0.5)
            except queue.Empty:
                if job.state != "running":
                    break
                continue
            if k != "job" or data.get("id") != job.id:
                continue
            lines = data.get("lines") or []
            for t, msg in lines[last_line:] if len(lines) >= last_line else lines:
                if inline:
                    sys.stdout.write("\r\033[K" if C.on else "\n")
                    inline = False
                print(f"  {C.dim(f'{t:>6.1f}s')}  {msg}")
            last_line = len(lines)
            if data.get("state") == "running" and data.get("progress") is not None and C.on:
                p = data["progress"]
                bar = "█" * int(p * 24) + "░" * (24 - int(p * 24))
                sys.stdout.write(f"\r\033[K  {C.acc(bar)} {p*100:5.1f}%  {C.dim(data.get('stage') or '')[:80]}")
                sys.stdout.flush()
                inline = True
            if data.get("state") != "running":
                break
    except KeyboardInterrupt:
        job.cancelled = True
        print("\n" + C.warn("detached") + " — the engine keeps going. Check with: vllm-lab status")
        return 130
    finally:
        lab.bus.unsubscribe(q)
    while job.state == "running":
        time.sleep(0.2)
    if inline:
        print()
    if job.state == "ok":
        print(C.ok("✓ ") + job.result)
        return 0
    err = job.error or {}
    print(C.bad("✗ ") + (err.get("error") or job.result))
    if err.get("hint"):
        print("  " + err["hint"])
    if err.get("fixes"):
        print("  " + C.dim("fixes: " + ", ".join(err["fixes"]) + "   (vllm-lab fix NAME <fix>)"))
    fit = (err.get("data") or {}).get("fit")
    if fit and fit.get("others"):
        print("  " + C.dim("running: " + ", ".join(f"{o['name']} {fmt_bytes(o['reserve'])}" for o in fit["others"])))
        print("  " + C.dim("retry with --solo (stop others), --shrink (smaller share) or --force"))
    return 1


def print_status(lab: Lab) -> None:
    lab.refresh()
    snap = lab.snapshot()
    for hid, hs in snap["hosts"].items():
        mem = hs.get("mem") or {}
        g = (hs.get("gpus") or [{}])[0] if hs.get("gpus") else {}
        if not hs.get("enabled", True):
            print(f"{C.b(hs.get('label'))} {C.dim('disabled')}")
            continue
        if not hs.get("online"):
            print(f"{C.b(hs.get('label'))} {C.bad('offline')} {C.dim(hs.get('why') or '')}")
            continue
        tot = mem.get("total") or 0
        used = tot - (mem.get("available") or 0)
        bar = ""
        if tot:
            n = 30
            k = int(n * used / tot)
            bar = C.acc("▮" * k) + C.dim("▯" * (n - k))
        gdesc = g.get("name") or ""
        if g.get("util") is not None and g.get("temp") is not None:
            gdesc += f" · {g['util']:.0f}% · {g['temp']:.0f}°C"
        print(f"{C.b(hs.get('label'))}  {bar} {fmt_bytes(used)} / {fmt_bytes(tot)}  {C.dim(gdesc)}")
    rows = sorted(snap["engines"].values(), key=lambda e: (e["host"], e["name"]))
    if not rows:
        print(C.dim("no engines — create one: vllm-lab up NAME --model org/model"))
        return
    print()
    print(C.dim(f"{'ENGINE':<16} {'HOST':<8} {'STATE':<10} {'TOK/S':>7} {'MEM':>9}  {'URL':<30} MODEL"))
    for e in rows:
        st = e.get("state") or "?"
        m = e.get("metrics") or {}
        tps = f"{m['gen_tps']:.1f}" if m.get("gen_tps") else "–"
        extra = ""
        if st == "booting":
            b = e.get("boot") or {}
            extra = C.blue(f"  {dict(BOOT_STAGES).get(b.get('stage'), '')} {b.get('detail') or ''} {str(b.get('pct')) + '%' if b.get('pct') is not None else ''}")
        if st == "crashed":
            extra = C.bad(f"  {(e.get('crash') or {}).get('title', '')}")
        if e.get("drift"):
            extra += C.warn("  config changed → recreate")
        color = STATE_COLOR.get(st, str)
        print(f"{C.b(e['name']):<{16 + (9 if C.on else 0)}} {e['host']:<8} {color(f'{st:<10}')} {tps:>7} {fmt_bytes(e.get('reserve') or 0) if e.get('reserve') else '–':>9}  "
              f"{(e.get('human_url') or '–'):<30} {C.dim(e.get('model'))}{extra}")


CLI_HELP = """vllm-lab — local AI control plane for DGX / GB10

  vllm-lab                         status of every host and engine
  vllm-lab ui [--listen A] [--port N] [--no-browser]
  vllm-lab up NAME [--model ID] [--host H] [--backend vllm|llamacpp] [--util F] [--ctx N]
                   [--blueprint B] [--port P] [--quant Q] [--solo|--shrink|--force] [--no-wait]
  vllm-lab solo NAME               start NAME, stopping other engines on its host
  vllm-lab stop|down NAME          stop (frees memory, keeps the container)
  vllm-lab sleep NAME              stop and mark as wakeable by the gateway
  vllm-lab restart|recreate NAME
  vllm-lab rm NAME [--delete]      remove container (and the engine config with --delete)
  vllm-lab logs NAME [-n 200] [-f]
  vllm-lab fix NAME FIX            apply a suggested fix (fit-util, make-room, set-context …)
  vllm-lab probe NAME              one short chat request, with timing
  vllm-lab bench NAME [-c 4] [-n 16] [--max-tokens 256]
  vllm-lab plan MODEL [--host H] [--ctx N] [--seqs N] [--kv fp8|auto]
  vllm-lab search QUERY [--gguf]
  vllm-lab pull MODEL [--host H]   pre-download weights
  vllm-lab doctor
  vllm-lab hosts | host ID [--ssh U@H] [--human-host A] [--enabled|--online true|false] [--label L]
  vllm-lab adopt CONTAINER [--host H]
  vllm-lab webui-sync
  vllm-lab config [--hf-token T] [--webui-url U] [--webui-email E] [--webui-password P]
                  [--webui-key K] [--ui-token T] [--gateway-key K]
  vllm-lab export > lab.json  |  vllm-lab import lab.json
"""


def _flags(args: list[str], spec: dict) -> tuple[dict, list[str]]:
    """Tiny flag parser: spec maps flag → ('val'|'bool', dest)."""
    out: dict = {}
    pos: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        key, eq, val = a.partition("=")
        if key in spec:
            kind, dest = spec[key]
            if kind == "bool":
                out[dest] = True
                i += 1
                continue
            if eq:
                out[dest] = val
                i += 1
                continue
            if i + 1 >= len(args):
                raise LabError(f"{key} needs a value")
            out[dest] = args[i + 1]
            i += 2
            continue
        if a.startswith("-") and a not in ("-",):
            raise LabError(f"unknown option {a}")
        pos.append(a)
        i += 1
    return out, pos


def main(argv: list[str]) -> int:
    if argv and argv[0] in ("-h", "--help", "help"):
        print(CLI_HELP)
        return 0
    if argv and argv[0] in ("version", "--version", "-V"):
        print(VERSION)
        return 0
    if argv and argv[0] == "ui":
        f, _ = _flags(argv[1:], {"--listen": ("val", "listen"), "--port": ("val", "port"), "--no-browser": ("bool", "nob")})
        serve(f.get("listen"), as_int(f.get("port")), not f.get("nob"))
        return 0
    lab = Lab(serve=False)
    try:
        if not argv or argv[0] in ("list", "status", "ls", "ps"):
            print_status(lab)
            return 0
        cmd, rest = argv[0], argv[1:]
        if cmd in ("up", "solo", "start"):
            f, pos = _flags(rest, {
                "--model": ("val", "model"), "--host": ("val", "host"), "--backend": ("val", "backend"), "--quant": ("val", "quant"),
                "--port": ("val", "port"), "--util": ("val", "util"), "--ctx": ("val", "max_len"), "--blueprint": ("val", "blueprint"),
                "--image": ("val", "image"), "--extra": ("val", "extra"), "--solo": ("bool", "solo"), "--shrink": ("bool", "shrink"),
                "--force": ("bool", "force"), "--no-wait": ("bool", "nowait"), "--recreate": ("bool", "recreate"),
            })
            if not pos:
                raise LabError("need an engine NAME")
            name = pos[0]
            exists = name in lab.store.engines()
            patch = {k: f[k] for k in ("model", "host", "backend", "quant", "port", "util", "max_len", "blueprint", "image", "extra") if k in f}
            if not exists and not (patch.get("model") or patch.get("blueprint")):
                raise LabError(f"no engine named {name}", f"create it: vllm-lab up {name} --model org/model  (or --blueprint {', '.join(list(BUILTIN_BLUEPRINTS)[:3])} …)")
            if patch or not exists:
                lab.upsert_engine({"name": name, **patch}, create=not exists, old_name=name if exists else "")
            mode = "solo" if (cmd == "solo" or f.get("solo")) else ("shrink" if f.get("shrink") else ("force" if f.get("force") else "ask"))
            recreate = bool(f.get("recreate") or (exists and patch))
            lab.refresh([lab.spec(name)["host"]])
            return run_job_cli(lab, "start", f"Start {name}", name,
                               lambda j: lab.op_start(name, j, on_conflict=mode, recreate=recreate, wait=not f.get("nowait")))
        if cmd in ("stop", "down", "sleep"):
            if not rest:
                raise LabError("need NAME")
            lab.refresh([lab.spec(rest[0])["host"]])
            print(lab.op_stop(rest[0], None, sleeping=cmd == "sleep"))
            return 0
        if cmd in ("restart", "recreate"):
            if not rest:
                raise LabError("need NAME")
            name = rest[0]
            lab.refresh([lab.spec(name)["host"]])
            if cmd == "restart":
                return run_job_cli(lab, "restart", f"Restart {name}", name, lambda j: lab.op_restart(name, j))
            return run_job_cli(lab, "start", f"Recreate {name}", name, lambda j: lab.op_start(name, j, recreate=True))
        if cmd in ("rm", "remove"):
            f, pos = _flags(rest, {"--delete": ("bool", "delete")})
            if not pos:
                raise LabError("need NAME")
            print(lab.op_remove(pos[0], None, delete_spec=bool(f.get("delete"))))
            return 0
        if cmd == "logs":
            f, pos = _flags(rest, {"-n": ("val", "n"), "--tail": ("val", "n"), "-f": ("bool", "follow"), "--follow": ("bool", "follow")})
            if not pos:
                raise LabError("need NAME")
            n = as_int(f.get("n"), 200)
            if f.get("follow"):
                try:
                    for line in lab.follow_logs(pos[0], n):
                        print(line)
                except KeyboardInterrupt:
                    pass
                return 0
            print(lab.logs(pos[0], n)["text"])
            return 0
        if cmd == "fix":
            if len(rest) < 2:
                raise LabError("usage: vllm-lab fix NAME FIX")
            name, fix = rest[0], rest[1]
            lab.refresh([lab.spec(name)["host"]])
            return run_job_cli(lab, "fix", f"Fix {name}: {fix}", name, lambda j: lab.op_fix(name, fix, j))
        if cmd == "probe":
            if not rest:
                raise LabError("need NAME")
            r = lab.probe_chat(rest[0])
            if r.get("ok"):
                print(C.ok("✓ ") + f"{r['model']} answered in {r['total']:.2f}s (first token {r['ttft'] or 0:.2f}s): {r['text']!r}")
                return 0
            print(C.bad("✗ ") + r.get("error", "") + (f"\n  {r['hint']}" if r.get("hint") else ""))
            return 1
        if cmd == "bench":
            f, pos = _flags(rest, {"-c": ("val", "c"), "-n": ("val", "n"), "--max-tokens": ("val", "mt"), "--prompt-tokens": ("val", "pt")})
            if not pos:
                raise LabError("need NAME")
            name = pos[0]
            return run_job_cli(lab, "bench", f"Bench {name}", name, lambda j: lab.op_bench(
                name, j, as_int(f.get("c"), 4), as_int(f.get("n"), 16), as_int(f.get("mt"), 256), "", as_int(f.get("pt"), 0)))
        if cmd == "plan":
            f, pos = _flags(rest, {"--host": ("val", "host"), "--ctx": ("val", "ctx"), "--seqs": ("val", "seqs"), "--kv": ("val", "kv")})
            if not pos:
                raise LabError("need MODEL")
            hid = f.get("host") or lab.hosts()[0].id
            lab.refresh([hid])
            p = lab.plan(pos[0], hid)
            meta = p["meta"]
            if meta.get("error"):
                raise LabError(meta["error"], meta.get("hint", ""))
            kv = meta.get("kv") or {}
            ctx = as_int(f.get("ctx"), None) or min(kv.get("ctx_max") or 32768, 131072)
            seqs = as_int(f.get("seqs"), 1)
            n = plan_numbers(p, ctx, seqs, f.get("kv") or "fp8")
            print(C.b(meta["id"]) + C.dim(f"  {meta.get('arch') or ''} · {fmt_params(meta.get('params'))} params · source {meta.get('source')}"))
            if meta.get("policy", {}).get("restricted"):
                print(C.bad(f"  origin policy: {meta['policy']['via']} ({meta['policy']['mode']})"))
            print(f"  weights      {fmt_bytes(n['weights'])}")
            if n["kv_known"]:
                print(f"  KV cache     {fmt_bytes(n['tok_bytes'])}/token → {fmt_bytes(n['kv_total'])} for {seqs}×{ctx:,} tokens")
            else:
                print(C.warn("  KV cache     unknown (no config.json) — estimate is weights-only"))
            print(f"  overhead     {fmt_bytes(n['overhead'])}")
            print(f"  needs        {C.b(fmt_bytes(n['need']))} of {fmt_bytes(p['total'])}" + (f"  →  --util {n['util_rec']:.2f}" if n.get("util_rec") else ""))
            if kv.get("ctx_max"):
                print(C.dim(f"  native context {kv['ctx_max']:,} tokens"))
            return 0
        if cmd == "search":
            f, pos = _flags(rest, {"--gguf": ("bool", "gguf"), "--host": ("val", "host")})
            q = " ".join(pos)
            if not q:
                raise LabError("need a QUERY")
            for r in lab.hf_search(q, f.get("host") or "", "gguf" if f.get("gguf") else ""):
                flag = C.bad(" ⛔") if r["policy"]["blocked"] else (C.warn(" ⚠") if r["policy"]["restricted"] else "")
                gated = C.warn(" gated") if r["gated"] else ""
                print(f"{fmt_params(r['params']):>7}  {r['downloads']:>9,}↓  {','.join(r['quant'])[:14]:<14} {r['id']}{gated}{flag}")
            return 0
        if cmd == "pull":
            f, pos = _flags(rest, {"--host": ("val", "host")})
            if not pos:
                raise LabError("need MODEL")
            hid = f.get("host") or lab.hosts()[0].id
            return run_job_cli(lab, "download", f"Download {pos[0]}", f"library@{hid}", lambda j: lab.op_download(pos[0], hid, j))
        if cmd == "doctor":
            lab.refresh()
            d = lab.doctor()
            icon = {"ok": C.ok("✓"), "warn": C.warn("!"), "fail": C.bad("✗"), "info": C.blue("i")}
            group = None
            for c in d["checks"]:
                if c["group"] != group:
                    group = c["group"]
                    print("\n" + C.b(group))
                print(f"  {icon[c['level']]} {c['title']}" + (C.dim(f"  → {c['fix']}") if c.get("fix") else ""))
                if c.get("detail") and c["level"] != "ok":
                    for ln in str(c["detail"]).splitlines():
                        print("      " + C.dim(ln))
            s = d["summary"]
            print(f"\n{s['fail']} failing · {s['warn']} warnings · {s['ok']} ok")
            return 1 if s["fail"] else 0
        if cmd == "hosts":
            for h in lab.conf()["hosts"]:
                print(f"{h['id']:<10} {('on' if h['enabled'] else 'off'):<4} {h.get('ssh') or 'local':<24} human={h['human_host']} bind={h['bind']} net={h['network']}")
            return 0
        if cmd == "host":
            if not rest:
                raise LabError("need host ID")
            f, pos = _flags(rest[1:], {"--ssh": ("val", "ssh"), "--online": ("val", "enabled"), "--enabled": ("val", "enabled"),
                                       "--human-host": ("val", "human_host"), "--label": ("val", "label"), "--bind": ("val", "bind"),
                                       "--cache": ("val", "cache"), "--network": ("val", "network"), "--ssh-port": ("val", "ssh_port"),
                                       "--reach": ("val", "reach"), "--remove": ("bool", "remove")})
            if f.get("remove"):
                lab.remove_host(rest[0])
                print(f"removed {rest[0]}")
                return 0
            h = lab.upsert_host({"id": rest[0], **f}, old_id=rest[0] if rest[0] in {x["id"] for x in lab.conf()["hosts"]} else "")
            print(json.dumps(h, indent=2))
            return 0
        if cmd == "adopt":
            f, pos = _flags(rest, {"--host": ("val", "host")})
            if not pos:
                raise LabError("need CONTAINER")
            s = lab.adopt(f.get("host") or lab.hosts()[0].id, pos[0])
            print(f"adopted as {s['name']}: {s['model']}")
            return 0
        if cmd in ("webui-sync", "webui-add"):
            lab.refresh()
            r = lab.webui_sync()
            print(f"Open WebUI: added {len(r['added'])}, removed {len(r['removed'])}")
            for u in r["added"]:
                print("  + " + u)
            for u in r["removed"]:
                print("  − " + u)
            return 0
        if cmd == "config":
            f, _ = _flags(rest, {"--hf-token": ("val", "hf_token"), "--webui-url": ("val", "url"), "--webui-email": ("val", "email"),
                                 "--webui-password": ("val", "password"), "--webui-key": ("val", "api_key"), "--ui-token": ("val", "ui_token"),
                                 "--gateway-key": ("val", "gw_key"), "--image": ("val", "vllm_image"), "--hf-endpoint": ("val", "hf_endpoint")})
            if not f:
                print(json.dumps(lab.settings_public(), indent=2))
                return 0
            web = {k: f[k] for k in ("url", "email", "password", "api_key") if k in f}
            if web:
                lab.save_webui(web)
            s: dict = {k: f[k] for k in ("hf_token", "vllm_image", "hf_endpoint") if k in f}
            if "ui_token" in f:
                s["ui"] = {"token": f["ui_token"]}
            if "gw_key" in f:
                s["gateway"] = {"key": f["gw_key"]}
            if s:
                lab.update_settings(s)
            print(f"wrote {CONF_FILE}")
            return 0
        if cmd == "export":
            print(json.dumps({"vllm_lab": VERSION, "engines": lab.store.engines(), "blueprints": lab.store.user_blueprints()}, indent=2))
            return 0
        if cmd == "import":
            if not rest:
                raise LabError("need FILE")
            data = json.loads(Path(rest[0]).read_text())
            n = 0
            for name, spec in (data.get("engines") or {}).items():
                if name not in lab.store.engines():
                    lab.upsert_engine({**spec, "name": name}, create=True)
                    n += 1
            for bid, bp in (data.get("blueprints") or {}).items():
                lab.save_blueprint(bid, bp.get("title") or bid, bp.get("spec") or {}, bp.get("tags"))
            print(f"imported {n} engine(s), {len(data.get('blueprints') or {})} blueprint(s)")
            return 0
        print(CLI_HELP)
        return 2
    except LabError as e:
        print(C.bad("error: ") + e.msg, file=sys.stderr)
        if e.hint:
            print("  " + e.hint, file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        lab.tunnels.close_all()


FAVICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" fill="#0b0d10"/>
<rect x="5" y="19" width="5" height="8" fill="#f0a53a"/><rect x="13.5" y="12" width="5" height="15" fill="#f0a53a"/><rect x="22" y="5" width="5" height="22" fill="#f0a53a"/></svg>"""


# ───────────────────────────────────────────────────────────────── console (single page app)

UI_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>vllm-lab</title><link rel="icon" href="/favicon.svg" type="image/svg+xml"><meta name="color-scheme" content="dark light">
<style>
:root{
  color-scheme:dark;
  --bg:#0a0c0f;--s1:#0e1115;--s2:#13171c;--s3:#191e25;--s4:#212833;
  --line:#1f252d;--line2:#2b333d;--line3:#3a4450;
  --tx:#e8ebef;--tx2:#aab3be;--tx3:#737e8b;--tx4:#4f5864;
  --acc:#ff8a3d;--acc2:#ffae75;--acc-ink:#140a02;--acc-dim:#3a2415;
  --ok:#3ecf8e;--warn:#f2c94c;--crit:#ff5f56;--boot:#5aa9ff;--sleep:#a193ff;
  --e1:#3987e5;--e2:#199e70;--e3:#d55181;--e4:#9085e9;--e5:#d95926;--e6:#2f9b2f;--e7:#e66767;--e8:#c98500;
  --ghost:repeating-linear-gradient(135deg,transparent 0 5px,color-mix(in srgb,var(--acc) 55%,transparent) 5px 7px);
  --cache:repeating-linear-gradient(135deg,transparent 0 3px,var(--line2) 3px 4px);
  --mono:ui-monospace,"JetBrains Mono","SF Mono","Cascadia Mono",Menlo,Consolas,monospace;
  --sans:"Inter",system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;
  --r:2px;
}
:root[data-theme=light]{
  color-scheme:light;
  --bg:#f3f2ef;--s1:#fbfaf8;--s2:#ffffff;--s3:#f0eeea;--s4:#e5e2dc;
  --line:#e2dfd8;--line2:#d2cec5;--line3:#b9b4a9;
  --tx:#15171a;--tx2:#4a4f57;--tx3:#737882;--tx4:#a2a6ad;
  --acc-dim:#ffe4d1;
  --ok:#128a57;--warn:#9a7500;--crit:#d23a31;--boot:#1f6fd1;--sleep:#6a55e0;
  --e1:#2a78d6;--e2:#1baf7a;--e3:#e87ba4;--e4:#4a3aa7;--e5:#eb6834;--e6:#008300;--e7:#e34948;--e8:#eda100;
}
:root[data-accent=ion]{--acc:#39c6f0;--acc2:#8adcf6;--acc-ink:#021016;--acc-dim:#11303a}
:root[data-accent=nominal]{--acc:#8bd11a;--acc2:#b6e66a;--acc-ink:#0b1300;--acc-dim:#243312}
:root[data-accent=violet]{--acc:#b18cff;--acc2:#d0b9ff;--acc-ink:#12082a;--acc-dim:#2a2140}
:root[data-accent=signal]{--acc:#ff8a3d;--acc2:#ffae75;--acc-ink:#140a02;--acc-dim:#3a2415}
:root[data-theme=light][data-accent=ion]{--acc:#0a8fbd;--acc-ink:#fff;--acc-dim:#d3eef8}
:root[data-theme=light][data-accent=nominal]{--acc:#4f8a00;--acc-ink:#fff;--acc-dim:#e3f0cd}
:root[data-theme=light][data-accent=violet]{--acc:#6d44d8;--acc-ink:#fff;--acc-dim:#ebe3ff}
:root[data-theme=light][data-accent=signal],:root[data-theme=light]:not([data-accent]){--acc:#d9631a;--acc-ink:#fff;--acc-dim:#ffe4d1}

*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--bg);color:var(--tx);font:13.5px/1.45 var(--sans);-webkit-font-smoothing:antialiased;overflow:hidden}
a{color:inherit}
button,input,select,textarea{font:inherit;color:inherit}
::selection{background:var(--acc);color:var(--acc-ink)}
.mono,code,kbd,pre{font-family:var(--mono);font-size:12px}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}
.dim{color:var(--tx2)}.faint{color:var(--tx3)}.ghosttx{color:var(--tx4)}
.ok{color:var(--ok)}.warn{color:var(--warn)}.crit{color:var(--crit)}.boot{color:var(--boot)}.sleep{color:var(--sleep)}.acc{color:var(--acc)}
.cap{font:600 10.5px/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--tx3)}
.hide{display:none!important}
.row{display:flex;align-items:center;gap:10px}.row.wrap{flex-wrap:wrap}.row.top{align-items:flex-start}.row.end{justify-content:flex-end}
.grow{flex:1;min-width:0}.nowrap{white-space:nowrap}.ellip{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sp{height:14px}.sp2{height:26px}
hr{border:0;border-top:1px solid var(--line);margin:18px 0}

/* ── shell ─────────────────────────────────────────── */
#app{display:grid;grid-template-columns:216px 1fr;height:100vh}
#rail{background:var(--s1);border-right:1px solid var(--line);display:flex;flex-direction:column;min-height:0}
.brand{display:flex;align-items:center;gap:10px;padding:16px 16px 14px;border-bottom:1px solid var(--line)}
.brand svg{flex:none}
.brand b{font:700 13px/1 var(--mono);letter-spacing:.06em}
.brand small{display:block;font:500 10px/1.2 var(--mono);color:var(--tx3);margin-top:3px;letter-spacing:.04em}
nav.main{padding:10px 8px;display:flex;flex-direction:column;gap:1px;overflow:auto}
nav.main a{display:flex;align-items:center;gap:11px;padding:7px 10px;text-decoration:none;color:var(--tx2);border-radius:var(--r);font-weight:500;position:relative}
nav.main a svg{width:16px;height:16px;flex:none;opacity:.85}
nav.main a:hover{background:var(--s3);color:var(--tx)}
nav.main a.on{background:var(--acc);color:var(--acc-ink)}
nav.main a.on svg{opacity:1}
nav.main a .badge{margin-left:auto;font:600 10px/1 var(--mono);padding:3px 5px;background:var(--crit);color:#fff;border-radius:var(--r)}
nav.main a.on .badge{background:var(--acc-ink);color:var(--acc)}
nav.main .sep{height:1px;background:var(--line);margin:8px 6px}
.railhosts{margin-top:auto;border-top:1px solid var(--line);padding:12px 12px 10px;display:flex;flex-direction:column;gap:10px}
.rh{display:grid;grid-template-columns:auto 1fr auto;gap:4px 8px;align-items:center;cursor:pointer;text-decoration:none}
.rh .nm{font-weight:600;font-size:12.5px}
.rh .v{font:500 11px var(--mono);color:var(--tx3)}
.rh .mbar{grid-column:1/-1;height:4px;background:var(--s4);display:flex;overflow:hidden}
.rh .mbar i{display:block;height:100%}
.railfoot{padding:10px 12px;border-top:1px solid var(--line);display:flex;align-items:center;gap:8px;color:var(--tx3);font:500 11px var(--mono)}
.railfoot .grow{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

#main{display:flex;flex-direction:column;min-width:0;min-height:0}
#top{height:50px;flex:none;display:flex;align-items:center;gap:14px;padding:0 22px;border-bottom:1px solid var(--line);background:var(--bg)}
#top h1{font:650 15px/1 var(--sans);margin:0;letter-spacing:.01em;display:flex;align-items:center;gap:10px;min-width:0}
#top h1 .crumb{color:var(--tx3);font-weight:500}
#top .fleet{margin-left:auto;display:flex;align-items:center;gap:16px}
.pill{display:inline-flex;align-items:center;gap:6px;font:500 11.5px/1 var(--mono);color:var(--tx2)}
.kbd,kbd{font:500 10.5px/1 var(--mono);padding:3px 5px;border:1px solid var(--line2);border-bottom-width:2px;color:var(--tx2);border-radius:var(--r);background:var(--s2)}
.search svg{width:14px;height:14px;flex:none}
.search{display:flex;align-items:center;gap:8px;padding:6px 10px;border:1px solid var(--line2);background:var(--s1);color:var(--tx3);cursor:pointer;min-width:220px;border-radius:var(--r)}
.search:hover{border-color:var(--line3);color:var(--tx2)}
#view{flex:1;overflow:auto;padding:22px 26px 80px;scroll-behavior:smooth}
.page{max-width:1480px;margin:0 auto}
.conn{position:fixed;left:50%;top:10px;transform:translateX(-50%);z-index:80;background:var(--crit);color:#fff;padding:6px 12px;font:600 12px var(--mono);border-radius:var(--r)}

/* ── controls ─────────────────────────────────────── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:7px;height:30px;padding:0 12px;border:1px solid var(--line2);background:var(--s2);color:var(--tx);border-radius:var(--r);cursor:pointer;font-weight:550;font-size:12.5px;white-space:nowrap;text-decoration:none;user-select:none}
.btn:hover{border-color:var(--line3);background:var(--s3)}
.btn:active{transform:translateY(1px)}
.btn svg{width:14px;height:14px}
.btn.pri{background:var(--acc);border-color:var(--acc);color:var(--acc-ink)}
.btn.pri:hover{background:var(--acc2);border-color:var(--acc2)}
.btn.ghost{background:transparent;border-color:transparent;color:var(--tx2)}
.btn.ghost:hover{background:var(--s3);color:var(--tx)}
.btn.danger{color:var(--crit)}
.btn.danger:hover{background:var(--crit);border-color:var(--crit);color:#fff}
.btn.sm{height:25px;padding:0 9px;font-size:12px}
.btn.xs{height:21px;padding:0 7px;font-size:11px;gap:4px}
.btn.icon{width:30px;padding:0}.btn.sm.icon{width:25px}
.btn[disabled]{opacity:.45;pointer-events:none}
.btn.busy{pointer-events:none;opacity:.8}
.btn.busy::after{content:"";width:10px;height:10px;border:2px solid currentColor;border-right-color:transparent;border-radius:50%;animation:spin .7s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
input[type=text],input[type=password],input[type=number],input[type=search],input:not([type]),select,textarea{
  height:31px;padding:0 10px;border:1px solid var(--line2);background:var(--s1);border-radius:var(--r);outline:none;min-width:0;width:100%}
textarea{height:auto;padding:8px 10px;resize:vertical;font-family:var(--mono);font-size:12px;line-height:1.5}
input:focus,select:focus,textarea:focus{border-color:var(--acc)}
input.mono{font-family:var(--mono);font-size:12px}
select{appearance:none;background-image:linear-gradient(45deg,transparent 50%,var(--tx3) 50%),linear-gradient(135deg,var(--tx3) 50%,transparent 50%);background-position:calc(100% - 13px) 13px,calc(100% - 9px) 13px;background-size:4px 4px;background-repeat:no-repeat;padding-right:26px}
input[type=radio]{accent-color:var(--acc)}
input[type=checkbox]{accent-color:var(--acc);width:15px;height:15px;margin:0}
input[type=range]{accent-color:var(--acc);width:100%}
label.f{display:flex;flex-direction:column;gap:5px;min-width:0}
label.f>span{min-height:14px;font:600 10.5px/1 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);display:flex;align-items:center;gap:6px}
label.chk{display:flex;align-items:center;gap:8px;cursor:pointer;color:var(--tx2)}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:14px 16px}
.fgrid .w2{grid-column:span 2}.fgrid .wall{grid-column:1/-1}
.seg{flex:none;display:inline-flex;border:1px solid var(--line2);border-radius:var(--r);overflow:hidden;background:var(--s1)}
.seg button{border:0;background:transparent;padding:0 11px;height:28px;cursor:pointer;color:var(--tx2);font-weight:550;font-size:12px;border-right:1px solid var(--line2)}
.seg button:last-child{border-right:0}
.seg button:hover{color:var(--tx);background:var(--s3)}
.seg button.on{background:var(--acc);color:var(--acc-ink)}
.tabs{display:flex;gap:2px;border-bottom:1px solid var(--line);margin-bottom:18px}
.tabs button{border:0;background:transparent;padding:9px 14px;cursor:pointer;color:var(--tx2);font-weight:600;font-size:12.5px;border-bottom:2px solid transparent;margin-bottom:-1px}
.tabs button:hover{color:var(--tx)}
.tabs button.on{color:var(--tx);border-bottom-color:var(--acc)}
.tag{display:inline-flex;align-items:center;gap:4px;height:19px;padding:0 6px;font:600 10.5px/1 var(--mono);border:1px solid var(--line2);color:var(--tx2);border-radius:var(--r);white-space:nowrap}
.tag.solid{background:var(--s4);border-color:var(--s4)}
.tag.acc{border-color:var(--acc);color:var(--acc)}
.tag.crit{border-color:var(--crit);color:var(--crit)}
.tag.warn{border-color:var(--warn);color:var(--warn)}
.tag.ok{border-color:var(--ok);color:var(--ok)}
.tag.boot{border-color:var(--boot);color:var(--boot)}
.tag.sleep{border-color:var(--sleep);color:var(--sleep)}
.i-tip{display:inline-grid;place-items:center;width:14px;height:14px;border:1px solid var(--line3);border-radius:50%;font:700 9px/1 var(--sans);color:var(--tx3);cursor:help;text-transform:none;letter-spacing:0;position:relative;flex:none}
.i-tip:hover{color:var(--acc);border-color:var(--acc)}
#tip{position:fixed;z-index:100;max-width:320px;background:var(--s4);color:var(--tx);border:1px solid var(--line3);padding:8px 10px;font:500 12px/1.45 var(--sans);pointer-events:none;box-shadow:0 8px 28px #0008;white-space:pre-line}
#tip .mono{font-size:11.5px}

/* ── status LEDs ──────────────────────────────────── */
.led{display:inline-block;width:8px;height:8px;border-radius:50%;flex:none;background:var(--tx4);box-shadow:0 0 0 2px color-mix(in srgb,var(--tx4) 25%,transparent)}
.led.ready,.led.ok{background:var(--ok);box-shadow:0 0 0 2px color-mix(in srgb,var(--ok) 22%,transparent),0 0 10px color-mix(in srgb,var(--ok) 60%,transparent)}
.led.booting{background:var(--boot);animation:pulse 1.1s ease-in-out infinite}
.led.crashed{background:var(--crit);box-shadow:0 0 0 2px color-mix(in srgb,var(--crit) 25%,transparent),0 0 10px color-mix(in srgb,var(--crit) 60%,transparent)}
.led.sleeping{background:transparent;box-shadow:inset 0 0 0 1.5px var(--sleep)}
.led.offline{background:transparent;box-shadow:inset 0 0 0 1.5px var(--tx4)}
.led.warn{background:var(--warn)}
@keyframes pulse{0%,100%{box-shadow:0 0 0 2px color-mix(in srgb,var(--boot) 25%,transparent)}50%{box-shadow:0 0 0 5px color-mix(in srgb,var(--boot) 10%,transparent),0 0 12px var(--boot)}}

/* ── section headers ─────────────────────────────── */
.sh{display:flex;align-items:center;gap:10px;margin:0 0 12px}
.sh h2{margin:0;font:650 12px/1 var(--sans);letter-spacing:.12em;text-transform:uppercase;color:var(--tx2)}
.sh .rule{flex:1;height:1px;background:var(--line)}
.sec{margin-bottom:30px}

/* ── the tank (unified-memory budget) ───────────── */
.hostband{display:grid;grid-template-columns:minmax(0,1fr);gap:10px;padding:16px 0 18px;border-bottom:1px solid var(--line)}
.hostband:first-child{padding-top:0}
.hb-head{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.hb-name{font:700 18px/1 var(--sans);letter-spacing:.01em;display:flex;align-items:center;gap:9px;text-decoration:none}
.hb-meta{display:flex;gap:18px;flex-wrap:wrap;margin-left:auto}
.gauge{display:flex;flex-direction:column;gap:3px;min-width:74px}
.gauge .v{font:600 15px/1 var(--mono);font-variant-numeric:tabular-nums}
.gauge .v small{font-size:11px;color:var(--tx3);font-weight:500;margin-left:2px}
.gauge .spk{height:16px}
.tank{position:relative;height:40px;background:var(--s2);display:flex;overflow:hidden;border:1px solid var(--line)}
.tank .seg-e{position:relative;height:100%;display:flex;align-items:center;padding:0 8px;min-width:0;transition:width .6s cubic-bezier(.2,.8,.2,1);border-right:2px solid var(--bg);cursor:pointer;overflow:hidden}
.tank .seg-e span{font:650 11.5px/1.1 var(--mono);color:#fff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-shadow:0 1px 2px #0006}
.tank .seg-e span small{display:block;font-weight:500;opacity:.85;font-size:10.5px}
.tank .seg-e.booting{background-image:linear-gradient(90deg,transparent,#ffffff26,transparent);background-size:200% 100%;animation:sweep 1.6s linear infinite}
@keyframes sweep{from{background-position:150% 0}to{background-position:-50% 0}}
.tank .seg-os{background:var(--s4);border-right:2px solid var(--bg);height:100%;display:flex;align-items:center;padding:0 8px;overflow:hidden}
.tank .seg-os span,.tank .seg-cache span,.tank .seg-free span{font:600 10.5px/1 var(--mono);color:var(--tx3);white-space:nowrap}
.tank .seg-cache{background:var(--cache);height:100%;display:flex;align-items:center;padding:0 8px;overflow:hidden;border-right:2px solid var(--bg)}
.tank .seg-free{flex:1;display:flex;align-items:center;justify-content:flex-end;padding:0 10px;overflow:hidden}
.tank .seg-ghost{height:100%;background:var(--ghost);border:1.5px dashed var(--acc);display:flex;align-items:center;padding:0 8px;overflow:hidden;transition:width .4s}
.tank .seg-ghost span{font:700 11px var(--mono);color:var(--acc);white-space:nowrap;text-shadow:0 0 6px var(--bg)}
.tank .seg-ghost.over{border-color:var(--crit);background:repeating-linear-gradient(135deg,transparent 0 5px,color-mix(in srgb,var(--crit) 55%,transparent) 5px 7px)}
.tank .seg-ghost.over span{color:var(--crit)}
.tank .seg-e.evict{opacity:.35;background-image:repeating-linear-gradient(135deg,transparent 0 6px,#0005 6px 9px)!important}
.tank-scale{display:flex;justify-content:space-between;font:500 10px var(--mono);color:var(--tx4);margin-top:2px}
.tank.sm{height:22px}.tank.sm .seg-e span small{display:none}

/* ── engine modules ──────────────────────────────── */
.mods{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.mod{background:var(--s1);border:1px solid var(--line);display:flex;flex-direction:column;min-width:0;position:relative;transition:border-color .15s}
.mod:hover{border-color:var(--line2)}
.mod .stripe{position:absolute;left:0;top:0;bottom:0;width:3px}
.mod .mh{display:flex;align-items:center;gap:9px;padding:12px 14px 4px 16px}
.mod .mh .nm{font:700 14.5px/1.1 var(--sans);text-decoration:none}
.mod .mh .nm:hover{color:var(--acc)}
.mod .mm{padding:0 14px 10px 16px;font:500 11.5px/1.3 var(--mono);color:var(--tx3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mod .mbody{padding:4px 14px 12px 16px;min-height:74px;display:flex;flex-direction:column;justify-content:center;gap:8px}
.mod .mf{display:flex;align-items:center;gap:6px;padding:8px 10px 8px 16px;border-top:1px solid var(--line);background:var(--s2)}
.mod.crashed{border-color:color-mix(in srgb,var(--crit) 45%,var(--line))}
.mod.off .mh .nm,.mod.off .mm{opacity:.75}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.metric .v{font:650 16px/1.05 var(--mono);font-variant-numeric:tabular-nums;white-space:nowrap}
.metric .v small{font-size:10.5px;color:var(--tx3);font-weight:500;margin-left:2px}
.metric .l{font:600 9.5px/1 var(--sans);letter-spacing:.1em;text-transform:uppercase;color:var(--tx3);margin-top:4px}
.spark-row{display:flex;align-items:flex-end;gap:10px}
.spark-row svg{flex:1;height:30px;min-width:0}
.kvbar{height:4px;background:var(--s4);position:relative;overflow:hidden}
.kvbar i{position:absolute;left:0;top:0;bottom:0;background:var(--tx2)}

/* boot pipeline */
.pipe{display:grid;grid-template-columns:repeat(6,1fr);gap:3px}
.pipe .st{height:5px;background:var(--s4);position:relative;overflow:hidden}
.pipe .st.done{background:var(--boot)}
.pipe .st.cur{background:color-mix(in srgb,var(--boot) 30%,var(--s4))}
.pipe .st.cur i{position:absolute;left:0;top:0;bottom:0;background:var(--boot);transition:width .5s}
.pipe .st.cur.indet i{width:35%!important;animation:indet 1.2s ease-in-out infinite}
@keyframes indet{from{left:-35%}to{left:100%}}
.pipe-lab{display:grid;grid-template-columns:repeat(6,1fr);gap:3px;font:600 9px/1 var(--sans);letter-spacing:.08em;text-transform:uppercase;color:var(--tx4);margin-top:5px}
.pipe-lab .cur{color:var(--boot)}.pipe-lab .done{color:var(--tx3)}
.bootline{display:flex;justify-content:space-between;gap:8px;font:500 11.5px var(--mono);color:var(--tx2)}

/* crash card */
.crash{border-left:2px solid var(--crit);padding:2px 0 2px 10px;display:flex;flex-direction:column;gap:5px}
.crash b{color:var(--crit);font-size:13px}
.crash .h{color:var(--tx2);font-size:12px;line-height:1.4}
.crash .ln{font:500 11px/1.4 var(--mono);color:var(--tx3);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.fixes{display:flex;gap:6px;flex-wrap:wrap}

/* ── tables ──────────────────────────────────────── */
.tbl{width:100%;border-collapse:collapse}
.tbl th{text-align:left;font:600 10px/1 var(--sans);letter-spacing:.11em;text-transform:uppercase;color:var(--tx3);padding:9px 10px;border-bottom:1px solid var(--line2);white-space:nowrap;position:sticky;top:0;background:var(--bg);z-index:1}
.tbl td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:middle}
.tbl tr:hover td{background:var(--s1)}
.tbl tr.sel td{background:var(--s2)}
.tbl td.r,.tbl th.r{text-align:right}
.tbl .acts{display:flex;gap:4px;justify-content:flex-end;opacity:.55;transition:opacity .1s}
.tbl tr:hover .acts{opacity:1}
.sizebar{height:3px;background:var(--s4);margin-top:5px;max-width:160px}
.sizebar i{display:block;height:100%;background:var(--tx3)}
.empty{padding:40px 20px;text-align:center;color:var(--tx3);border:1px dashed var(--line2)}
.empty b{display:block;color:var(--tx2);font-size:14px;margin-bottom:6px}

/* ── two-column layouts ─────────────────────────── */
.cols{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:30px}
.cols.launch{grid-template-columns:minmax(0,1.05fr) minmax(0,1fr)}
.cols.half{grid-template-columns:1fr 1fr}
@media (max-width:1180px){.cols,.cols.launch,.cols.half{grid-template-columns:1fr}}
.panel{border-left:1px solid var(--line);padding-left:24px}
@media (max-width:1180px){.panel{border-left:0;padding-left:0}}

/* feed */
.feed{display:flex;flex-direction:column}
.ev{display:grid;grid-template-columns:48px 8px 1fr;gap:8px;padding:7px 0;border-bottom:1px solid var(--line);align-items:start;font-size:12.5px}
.ev .t{font:500 10.5px/1.6 var(--mono);color:var(--tx4)}
.ev .d{width:6px;height:6px;margin-top:6px;border-radius:50%;background:var(--tx3)}
.ev.ok .d{background:var(--ok)}.ev.warn .d{background:var(--warn)}.ev.error .d{background:var(--crit)}
.ev .m{color:var(--tx2);min-width:0;overflow-wrap:anywhere}
.ev .m b{color:var(--tx);font-weight:600}

/* job tray */
#tray{position:fixed;right:18px;bottom:18px;z-index:60;display:flex;flex-direction:column;gap:8px;width:360px;max-width:calc(100vw - 36px)}
.job{background:var(--s2);border:1px solid var(--line2);box-shadow:0 10px 30px #0007;padding:11px 12px 10px;display:flex;flex-direction:column;gap:7px;animation:rise .2s ease-out}
@keyframes rise{from{transform:translateY(8px);opacity:0}}
.job .jt{display:flex;align-items:center;gap:8px;font-weight:600;font-size:12.5px}
.job .js{font:500 11.5px/1.35 var(--mono);color:var(--tx2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.job .pb{height:3px;background:var(--s4);position:relative;overflow:hidden}
.job .pb i{position:absolute;left:0;top:0;bottom:0;background:var(--acc);transition:width .3s}
.job .pb.indet i{width:30%;animation:indet 1.1s ease-in-out infinite}
.job.ok{border-color:color-mix(in srgb,var(--ok) 50%,var(--line2))}
.job.error{border-color:var(--crit)}
.job.error .js{white-space:normal;color:var(--tx)}
.job .hint{font-size:12px;color:var(--tx2);line-height:1.4}
.toast{background:var(--s3);border:1px solid var(--line2);padding:10px 12px;font-size:12.5px;box-shadow:0 10px 30px #0007;animation:rise .2s ease-out;display:flex;gap:9px;align-items:flex-start}
.toast.err{border-color:var(--crit)}
.toast .x{margin-left:auto;cursor:pointer;color:var(--tx3)}

/* modal */
#modal{position:fixed;inset:0;background:#05070acc;z-index:90;display:grid;place-items:center;padding:24px;animation:fade .15s}
@keyframes fade{from{opacity:0}}
.dlg{background:var(--s1);border:1px solid var(--line2);width:min(640px,100%);max-height:calc(100vh - 48px);overflow:auto;box-shadow:0 24px 80px #000a}
.dlg header{padding:18px 20px 0;display:flex;align-items:flex-start;gap:12px}
.dlg header h3{margin:0;font:650 16px/1.3 var(--sans)}
.dlg .body{padding:14px 20px}
.dlg footer{padding:14px 20px;border-top:1px solid var(--line);display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;background:var(--s2)}
.choice{display:flex;gap:12px;align-items:flex-start;padding:12px;border:1px solid var(--line2);cursor:pointer;margin-bottom:8px}
.choice:hover{border-color:var(--line3);background:var(--s2)}
.choice.on{border-color:var(--acc);background:var(--acc-dim)}
.choice b{display:block;font-size:13px;margin-bottom:2px}
.choice .d{font-size:12px;color:var(--tx2)}

/* palette */
#pal{position:fixed;inset:0;z-index:95;background:#05070a99;display:flex;justify-content:center;padding-top:12vh;animation:fade .1s}
.palbox{width:min(620px,94vw);background:var(--s1);border:1px solid var(--line3);box-shadow:0 30px 90px #000c;align-self:flex-start;display:flex;flex-direction:column;max-height:64vh}
.palbox input{height:48px;border:0;border-bottom:1px solid var(--line2);background:transparent;font-size:15px;padding:0 16px}
.palbox input:focus{border-color:var(--line2)}
.pallist{overflow:auto;padding:6px}
.pi{display:flex;align-items:center;gap:10px;padding:8px 10px;cursor:pointer;border-radius:var(--r)}
.pi .k{margin-left:auto;font:500 11px var(--mono);color:var(--tx3)}
.pi svg{width:15px;height:15px;color:var(--tx3)}
.pi.on{background:var(--acc);color:var(--acc-ink)}
.pi.on svg,.pi.on .k{color:var(--acc-ink)}
.pi .g{font:500 10.5px var(--mono);color:var(--tx4);min-width:60px}
.pi.on .g{color:var(--acc-ink)}

/* logs */
.logbar{display:flex;gap:8px;align-items:center;margin-bottom:10px;flex-wrap:wrap}
.logview{background:var(--s1);border:1px solid var(--line);height:calc(100vh - 290px);min-height:320px;overflow:auto;font:12px/1.55 var(--mono);padding:8px 0;counter-reset:ln}
.logview .l{padding:0 14px 0 64px;white-space:pre-wrap;word-break:break-word;position:relative;color:var(--tx2)}
.logview .l::before{counter-increment:ln;content:counter(ln);position:absolute;left:0;width:48px;text-align:right;color:var(--tx4);font-size:10.5px}
.logview .l.err{color:var(--crit);background:color-mix(in srgb,var(--crit) 7%,transparent)}
.logview .l.warn{color:var(--warn)}
.logview .l.ok{color:var(--ok)}
.logview .l.prog{color:var(--boot)}
.logview .l .ts{color:var(--tx4)}
.logview mark{background:var(--acc);color:var(--acc-ink)}

/* code / connect */
.code{background:var(--s1);border:1px solid var(--line);padding:12px 14px;position:relative;overflow:auto}
.code pre{margin:0;white-space:pre;color:var(--tx2);line-height:1.55}
.code .cp{position:absolute;right:6px;top:6px}
.urlrow{display:grid;grid-template-columns:130px 1fr auto;gap:10px;align-items:center;padding:9px 0;border-bottom:1px solid var(--line)}
.urlrow code{color:var(--acc);font-size:12.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* chart */
.chart{position:relative}
.chart svg{display:block;width:100%;overflow:visible}
.chart .hov{position:absolute;pointer-events:none;background:var(--s4);border:1px solid var(--line3);padding:5px 8px;font:500 11.5px/1.4 var(--mono);white-space:nowrap;transform:translate(-50%,-110%);z-index:2}
.axis text{font:500 10px var(--mono);fill:var(--tx4)}
.grid line{stroke:var(--line);stroke-width:1}

/* hub results */
.hits{display:flex;flex-direction:column;border-top:1px solid var(--line)}
.hit{display:grid;grid-template-columns:1fr auto;gap:4px 12px;padding:11px 10px;border-bottom:1px solid var(--line);cursor:pointer}
.hit:hover{background:var(--s1)}
.hit.on{background:var(--s2);box-shadow:inset 3px 0 0 var(--acc)}
.hit .id{font:600 13px/1.3 var(--mono);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hit .id .org{color:var(--tx3);font-weight:500}
.hit .meta{display:flex;gap:6px;flex-wrap:wrap;align-items:center;grid-column:1/-1}
.hit .num{color:var(--tx3);font-size:11.5px}
.hit.blocked{opacity:.55}
.bp{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:8px}
.bpc{border:1px solid var(--line);padding:11px 12px;cursor:pointer;display:flex;flex-direction:column;gap:5px;background:var(--s1)}
.bpc:hover{border-color:var(--line3)}
.bpc.on{border-color:var(--acc);box-shadow:inset 0 0 0 1px var(--acc)}
.bpc b{font-size:13px}
.bpc .mk{font:500 11px var(--mono);color:var(--tx3)}

.verdict{display:flex;gap:10px;align-items:center;padding:11px 12px;border:1px solid var(--line2);background:var(--s1)}
.verdict.good{border-color:color-mix(in srgb,var(--ok) 55%,var(--line2))}
.verdict.bad{border-color:color-mix(in srgb,var(--crit) 55%,var(--line2))}
.verdict.meh{border-color:color-mix(in srgb,var(--warn) 55%,var(--line2))}
.verdict .big{font:700 20px/1 var(--mono)}
.breakdown{display:grid;grid-template-columns:auto 1fr auto;gap:6px 12px;font-size:12.5px;align-items:center}
.breakdown .bar{height:6px;background:var(--s4)}
.breakdown .bar i{display:block;height:100%}

/* chat */
.chatwrap{display:grid;grid-template-columns:repeat(var(--n,1),minmax(0,1fr));gap:14px;height:calc(100vh - 250px);min-height:380px}
.chatcol{display:flex;flex-direction:column;border:1px solid var(--line);background:var(--s1);min-height:0}
.chatcol header{display:flex;align-items:center;gap:8px;padding:8px 10px;border-bottom:1px solid var(--line)}
.msgs{flex:1;overflow:auto;padding:14px 16px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:92%;white-space:pre-wrap;word-wrap:break-word;line-height:1.55}
.msg.user{align-self:flex-end;background:var(--s3);padding:8px 12px}
.msg.assistant{align-self:flex-start}
.msg .think{color:var(--tx3);font-style:italic;border-left:2px solid var(--line2);padding-left:9px;margin-bottom:6px;font-size:12.5px;max-height:160px;overflow:auto}
.msg .stat{font:500 10.5px var(--mono);color:var(--tx4);margin-top:5px}
.msg.err{color:var(--crit)}
.composer{display:flex;gap:10px;margin-top:12px;align-items:flex-end}
.composer textarea{min-height:44px;max-height:200px;font-family:var(--sans);font-size:13.5px}
.caret{display:inline-block;width:7px;height:14px;background:var(--acc);vertical-align:-2px;animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}

.hostcard{border:1px solid var(--line);background:var(--s1);padding:16px 18px;display:flex;flex-direction:column;gap:12px}
.kvl{display:grid;grid-template-columns:130px 1fr;gap:6px 12px;font-size:12.5px}
.kvl dt{color:var(--tx3)}.kvl dd{margin:0;font-family:var(--mono);font-size:12px;overflow-wrap:anywhere}
.check{display:grid;grid-template-columns:18px 1fr auto;gap:10px;padding:10px 0;border-bottom:1px solid var(--line);align-items:start}
.check .ic{font:700 13px/1.3 var(--mono);text-align:center}
.check .t{font-weight:600}
.check .dd{color:var(--tx2);font-size:12.5px;margin-top:2px;white-space:pre-line}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:6px;height:24px;padding:0 4px 0 9px;border:1px solid var(--line2);font:500 11.5px var(--mono);background:var(--s2)}
.chip button{border:0;background:transparent;color:var(--tx3);cursor:pointer;width:18px;height:18px;display:grid;place-items:center}
.chip button:hover{color:var(--crit)}
.swatches{display:flex;gap:8px}
.sw{width:30px;height:30px;border:2px solid var(--line2);cursor:pointer;border-radius:var(--r)}
.sw.on{border-color:var(--tx);box-shadow:0 0 0 2px var(--bg) inset}
.diagram{display:grid;grid-template-columns:1fr 26px 1fr 26px 1fr;align-items:center;gap:0;font-size:12px;margin:6px 0 14px}
.diagram .n{border:1px solid var(--line2);padding:9px 10px;background:var(--s1)}
.diagram .n b{display:block;font-size:12px}
.diagram .n code{font-size:11px;color:var(--acc)}
.diagram .a{height:1px;background:var(--line3);position:relative}
.diagram .a::after{content:"";position:absolute;right:-1px;top:-3px;border:3.5px solid transparent;border-left:5px solid var(--line3)}
::-webkit-scrollbar{width:10px;height:10px}::-webkit-scrollbar-thumb{background:var(--s4);border:2px solid var(--bg)}::-webkit-scrollbar-track{background:transparent}
#navbtn{display:none}
@media (max-width:860px){#app{grid-template-columns:1fr}#rail{position:fixed;left:0;top:0;bottom:0;width:240px;z-index:70;transform:translateX(-102%);transition:transform .2s;box-shadow:0 0 40px #000a}body.navopen #rail{transform:none}#navbtn{display:inline-flex}#view{padding:16px}.metrics{grid-template-columns:repeat(2,1fr)}.search{min-width:0}.search .grow,.search kbd{display:none}#top{padding:0 12px;gap:10px}#top .fleet .pill:not(:first-child){display:none}.hb-meta{margin-left:0}}

.row>input,.row>select{flex:1}
.pi svg,.btn svg,nav.main a svg{flex:none}

</style></head>
<body><div id="app"></div><div id="tray"><div id="jobs"></div></div><div id="tip" class="hide"></div>
<script>
'use strict';
/* ── utilities ───────────────────────────────────────────── */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const GIB = 1024 ** 3;
function fmtB(n, d) {
  if (n == null || n === '' || isNaN(n)) return '–';
  n = Number(n); const u = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; let i = 0;
  while (Math.abs(n) >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i < 2 ? Math.round(n) : n.toFixed(d ?? (n >= 100 ? 0 : 1))) + ' ' + u[i];
}
function fmtGiB(n) { return n == null ? '–' : (n / GIB).toFixed(n / GIB >= 100 ? 0 : 1); }
function fmtN(n, d = 0) { if (n == null || isNaN(n)) return '–'; return Number(n).toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d }); }
function fmtK(n) { if (n == null) return '–'; n = Number(n); if (n >= 1e9) return (n / 1e9).toFixed(1) + 'B'; if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M'; if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k'; return String(Math.round(n)); }
function fmtDur(s) {
  if (s == null || s < 0 || isNaN(s)) return '–'; s = Math.floor(s);
  if (s < 60) return s + 's'; if (s < 3600) return Math.floor(s / 60) + 'm ' + String(s % 60).padStart(2, '0') + 's';
  if (s < 86400) return Math.floor(s / 3600) + 'h ' + String(Math.floor(s % 3600 / 60)).padStart(2, '0') + 'm';
  return Math.floor(s / 86400) + 'd ' + Math.floor(s % 86400 / 3600) + 'h';
}
function fmtAgo(t) { if (!t) return '–'; const s = Date.now() / 1000 - t; if (s < 5) return 'now'; if (s < 60) return Math.floor(s) + 's'; if (s < 3600) return Math.floor(s / 60) + 'm'; if (s < 86400) return Math.floor(s / 3600) + 'h'; return Math.floor(s / 86400) + 'd'; }
function fmtClock(t) { const d = new Date(t * 1000); return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' }); }
function fmtParams(n) { if (!n) return '–'; return n >= 1e9 ? (n / 1e9).toFixed(n >= 1e11 ? 0 : 1) + 'B' : (n / 1e6).toFixed(0) + 'M'; }
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const store = {
  get(k, d) { try { const v = localStorage.getItem('vllm-lab:' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('vllm-lab:' + k, JSON.stringify(v)); } catch { /* private mode */ } },
};
function tip(text) { return `<span class="i-tip" data-tip="${esc(text)}">i</span>`; }
function copyText(t) {
  const done = () => toast('Copied');
  if (navigator.clipboard && window.isSecureContext) return navigator.clipboard.writeText(t).then(done).catch(() => fallback());
  fallback();
  function fallback() { const ta = document.createElement('textarea'); ta.value = t; ta.style.position = 'fixed'; ta.style.opacity = '0'; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); done(); } catch { toast('Copy failed', true); } ta.remove(); }
}
function download(name, text, type = 'application/json') { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
function hfUrl(id) { return 'https://huggingface.co/' + String(id || '').split(':')[0]; }
function splitId(id) { id = String(id || ''); const i = id.indexOf('/'); return i < 0 ? ['', id] : [id.slice(0, i + 1), id.slice(i + 1)]; }

/* ── icons (16px, stroke) ────────────────────────────────── */
const P = d => `<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">${d}</svg>`;
const I = {
  deck: P('<rect x="2" y="2" width="5" height="6"/><rect x="9" y="2" width="5" height="3"/><rect x="9" y="7" width="5" height="7"/><rect x="2" y="10" width="5" height="4"/>'),
  engines: P('<rect x="4" y="4" width="8" height="8"/><path d="M6 1.5v2.5M10 1.5v2.5M6 12v2.5M10 12v2.5M1.5 6H4M1.5 10H4M12 6h2.5M12 10h2.5"/><rect x="6.5" y="6.5" width="3" height="3"/>'),
  launch: P('<path d="M8 2.5v11M2.5 8h11"/>'),
  play: P('<path d="M2.5 3.5h11v7h-6l-3 2.5v-2.5h-2z"/>'),
  library: P('<path d="M2.5 13.5V3M5.5 13.5V3M8.5 13.5V5M11 13.5l2.5-8"/>'),
  hosts: P('<rect x="2" y="2.5" width="12" height="4.5"/><rect x="2" y="9" width="12" height="4.5"/><path d="M4.5 4.75h.01M4.5 11.25h.01M8 4.75h3.5M8 11.25h3.5"/>'),
  containers: P('<path d="M8 1.8l5.5 3v6.4L8 14.2l-5.5-3V4.8z"/><path d="M2.5 4.8L8 7.8l5.5-3M8 7.8v6.4"/>'),
  webui: P('<rect x="1.8" y="2.5" width="12.4" height="11"/><path d="M1.8 5.5h12.4M4 4h.01M5.8 4h.01"/>'),
  doctor: P('<path d="M1.5 8.5h3l1.5-4 3 8 1.5-4h4"/>'),
  settings: P('<path d="M3 2v4M3 9v5M8 2v7M8 12v2M13 2v2M13 7v7"/><path d="M1.5 7.5h3M6.5 10.5h3M11.5 5.5h3"/>'),
  activity: P('<path d="M2 3.5h12M2 8h12M2 12.5h8"/>'),
  start: P('<path d="M5 3.2l7.5 4.8L5 12.8z" fill="currentColor"/>'),
  stop: P('<rect x="4" y="4" width="8" height="8" fill="currentColor"/>'),
  restart: P('<path d="M13 8a5 5 0 1 1-1.6-3.7"/><path d="M13 2.5v3h-3"/>'),
  logs: P('<path d="M3 3.5h10M3 6.5h10M3 9.5h7M3 12.5h5"/>'),
  more: P('<path d="M3.5 8h.01M8 8h.01M12.5 8h.01" stroke-width="2.4"/>'),
  copy: P('<rect x="5" y="5" width="8.5" height="8.5"/><path d="M11 5V2.5H2.5V11H5"/>'),
  trash: P('<path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5h6.6L12 4"/>'),
  moon: P('<path d="M13 9.5A5.5 5.5 0 0 1 6.5 3a5.5 5.5 0 1 0 6.5 6.5z"/>'),
  bolt: P('<path d="M9 1.5L3.5 9h4l-1 5.5L12 7H8z"/>'),
  search: P('<circle cx="7" cy="7" r="4.5"/><path d="M10.5 10.5l3 3"/>'),
  ext: P('<path d="M9 2.5h4.5V7M13.5 2.5L7.5 8.5M11 9.5v4H2.5V5h4"/>'),
  x: P('<path d="M4 4l8 8M12 4l-8 8"/>'),
  check: P('<path d="M3 8.5l3.2 3L13 4.5"/>'),
  edit: P('<path d="M10.5 2.5l3 3L6 13H3v-3z"/>'),
  gauge: P('<path d="M2.5 11a5.5 5.5 0 1 1 11 0"/><path d="M8 11l2.5-3.5"/>'),
  link: P('<path d="M6.5 9.5l3-3M7 4.5l1.3-1.3a2.5 2.5 0 0 1 3.5 3.5L10.5 8M9 11.5l-1.3 1.3a2.5 2.5 0 0 1-3.5-3.5L5.5 8"/>'),
  down: P('<path d="M8 2.5v8.5M4.5 7.5L8 11l3.5-3.5M2.5 13.5h11"/>'),
  plug: P('<path d="M5.5 1.5v3M10.5 1.5v3M3.5 4.5h9v3a4.5 4.5 0 0 1-9 0zM8 12v2.5"/>'),
  sun: P('<circle cx="8" cy="8" r="3"/><path d="M8 1v1.5M8 13.5V15M1 8h1.5M13.5 8H15M3 3l1 1M12 12l1 1M3 13l1-1M12 4l1-1"/>'),
  bench: P('<path d="M2 13.5h12M4 13.5V9M7 13.5V5M10 13.5V7M13 13.5V3"/>'),
  send: P('<path d="M2 8l11.5-5.5L9 14l-1.5-5z"/>'),
};
const BRAND = `<svg width="26" height="26" viewBox="0 0 32 32"><rect width="32" height="32" fill="var(--acc)"/><rect x="6" y="18" width="4.5" height="8" fill="var(--acc-ink)"/><rect x="13.75" y="12" width="4.5" height="14" fill="var(--acc-ink)"/><rect x="21.5" y="6" width="4.5" height="20" fill="var(--acc-ink)"/></svg>`;

/* ── state ───────────────────────────────────────────────── */
const S = {
  version: '', settings: {}, specs: {}, bps: {}, stages: [], snap: { hosts: {}, engines: {} },
  jobs: {}, mine: new Set(), events: [], hist: { e: {}, h: {} }, route: { view: 'deck', arg: '', sub: '', q: {} },
  connected: false, confM: null, jobSeen: {}, dismissed: new Set(),
};
const STATE_ORDER = { ready: 0, booting: 1, crashed: 2, sleeping: 3, stopped: 4, absent: 5, offline: 6 };
const STATE_LABEL = { ready: 'Ready', booting: 'Booting', crashed: 'Crashed', sleeping: 'Asleep', stopped: 'Stopped', absent: 'Not started', offline: 'Unreachable' };
function engineColor(name) {
  const host = (S.specs[name] || {}).host;
  const names = Object.keys(S.specs).filter(n => S.specs[n].host === host).sort((a, b) => ((S.specs[a].created || 0) - (S.specs[b].created || 0)) || a.localeCompare(b));
  const i = names.indexOf(name);
  return i < 0 ? 'var(--acc)' : `var(--e${i % 8 + 1})`;
}
function eng(name) { return S.snap.engines[name] || null; }
function hostOf(id) { return S.snap.hosts[id] || null; }
function hostList() { return (S.settings.hosts || []).map(h => ({ ...h, st: S.snap.hosts[h.id] || {} })); }
function hostTotal(hid) { const m = (S.snap.hosts[hid] || {}).mem || {}; return m.gpu_total || m.total || 0; }

/* ── api ─────────────────────────────────────────────────── */
async function api(path, opt = {}) {
  const o = { method: opt.method || 'GET', headers: { 'X-Lab-Client': '1' } };
  if (opt.body !== undefined) { o.method = opt.method || 'POST'; o.headers['Content-Type'] = 'application/json'; o.body = JSON.stringify(opt.body); }
  if (opt.signal) o.signal = opt.signal;
  let r;
  try { r = await fetch(path, o); } catch (e) { const er = new Error('The console server is not answering'); er.data = {}; throw er; }
  const t = await r.text(); let d;
  try { d = t ? JSON.parse(t) : {}; } catch { d = { error: t.slice(0, 300) || ('HTTP ' + r.status) }; }
  if (r.status === 401 && !path.startsWith('/v1')) { location.reload(); }
  if (!r.ok) { const e = new Error(d.error || ('HTTP ' + r.status)); e.data = d; e.status = r.status; throw e; }
  return d;
}
const post = (p, b = {}) => api(p, { body: b });
async function startJob(path, body = {}, opts = {}) {
  try {
    const d = await post(path, body);
    if (d.job) { S.mine.add(d.job); if (opts.onDone) S.jobHooks[d.job] = opts.onDone; }
    return d;
  } catch (e) { showError(e); throw e; }
}
S.jobHooks = {};
function showError(e, title) {
  const d = e.data || {};
  toast((title ? title + ': ' : '') + (e.message || e) + (d.hint ? ' — ' + d.hint : ''), true);
}

/* ── tiny DOM morph: keeps focus, scroll and hover while live data changes ── */
function morph(el, html) {
  const t = document.createElement(el.tagName === 'svg' ? 'div' : el.tagName);
  t.innerHTML = html;
  morphKids(el, t);
}
function sameNode(a, b) {
  if (a.nodeType !== b.nodeType || a.nodeName !== b.nodeName) return false;
  if (a.nodeType === 1 && (a.getAttribute('data-key') || '') !== (b.getAttribute('data-key') || '')) return false;
  return true;
}
function morphKids(a, b) {
  const bk = [...b.childNodes];
  for (let i = 0; i < bk.length; i++) {
    const bn = bk[i]; const an = a.childNodes[i];
    if (!an) { a.appendChild(bn); continue; }
    if (!sameNode(an, bn)) {
      // try to find a keyed match further along before replacing
      const key = bn.nodeType === 1 && bn.getAttribute('data-key');
      if (key) { const found = [...a.childNodes].slice(i + 1).find(n => n.nodeType === 1 && n.getAttribute('data-key') === key && n.nodeName === bn.nodeName); if (found) { a.insertBefore(found, an); morphNode(found, bn); continue; } }
      a.replaceChild(bn, an); continue;
    }
    morphNode(an, bn);
  }
  while (a.childNodes.length > bk.length) a.removeChild(a.lastChild);
}
function morphNode(an, bn) {
  if (an.nodeType === 3 || an.nodeType === 8) { if (an.nodeValue !== bn.nodeValue) an.nodeValue = bn.nodeValue; return; }
  if (an.nodeType !== 1) return;
  for (const at of [...an.attributes]) if (!bn.hasAttribute(at.name) && at.name !== 'open') an.removeAttribute(at.name);
  for (const at of [...bn.attributes]) if (an.getAttribute(at.name) !== at.value) an.setAttribute(at.name, at.value);
  if (an.hasAttribute('data-keep')) return;
  const tag = an.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA') {
    if (document.activeElement !== an) { if (an.type === 'checkbox' || an.type === 'radio') an.checked = bn.hasAttribute('checked'); else if (bn.hasAttribute('value') && an.value !== bn.getAttribute('value')) an.value = bn.getAttribute('value'); }
    if (tag === 'TEXTAREA' && document.activeElement !== an && an.value !== bn.value) an.value = bn.value;
    return;
  }
  morphKids(an, bn);
  if (tag === 'SELECT' && document.activeElement !== an) { const sel = [...bn.options].find(o => o.hasAttribute('selected')); if (sel) an.value = sel.value; }
}

/* ── overlays: toast, modal, menu, tooltip ───────────────── */
function toast(msg, err) {
  const tr = $('#tray');
  const el = document.createElement('div');
  el.className = 'toast' + (err ? ' err' : '');
  el.innerHTML = `<span class="${err ? 'crit' : 'ok'}">${err ? '✕' : '✓'}</span><span>${esc(msg)}</span><span class="x">×</span>`;
  el.querySelector('.x').onclick = () => el.remove();
  tr.appendChild(el);
  setTimeout(() => el.remove(), err ? 9000 : 3200);
}
let modalResolve = null;
function modal(html, opts = {}) {
  closeModal();
  const m = document.createElement('div');
  m.id = 'modal';
  m.innerHTML = `<div class="dlg" style="${opts.width ? 'width:min(' + opts.width + 'px,100%)' : ''}">${html}</div>`;
  m.addEventListener('mousedown', e => { if (e.target === m) closeModal(); });
  document.body.appendChild(m);
  const f = m.querySelector('[autofocus]'); if (f) setTimeout(() => f.focus(), 30);
  return new Promise(res => { modalResolve = res; });
}
function closeModal(v) { const m = $('#modal'); if (m) m.remove(); if (modalResolve) { const r = modalResolve; modalResolve = null; r(v); } }
function confirmDlg(title, text, ok = 'Confirm', danger = false) {
  return modal(`<header><h3>${esc(title)}</h3></header><div class="body"><div class="dim">${text}</div></div>
    <footer><button class="btn ghost" data-act="modal-close">Cancel</button><button class="btn ${danger ? 'danger' : 'pri'}" data-act="modal-ok" autofocus>${esc(ok)}</button></footer>`);
}
function menu(anchor, items) {
  closeMenu();
  const m = document.createElement('div');
  m.id = 'menu';
  m.style.cssText = 'position:fixed;z-index:85;background:var(--s2);border:1px solid var(--line3);box-shadow:0 12px 36px #0009;padding:4px;min-width:190px';
  m.innerHTML = items.map((it, i) => it === '-' ? '<div style="height:1px;background:var(--line);margin:4px"></div>' :
    `<div class="pi${it.danger ? ' crit' : ''}" data-mi="${i}" style="${it.danger ? 'color:var(--crit)' : ''}">${it.icon || ''}<span>${esc(it.label)}</span>${it.k ? `<span class="k">${esc(it.k)}</span>` : ''}</div>`).join('');
  document.body.appendChild(m);
  const r = anchor.getBoundingClientRect();
  const w = m.offsetWidth, h = m.offsetHeight;
  m.style.left = clamp(r.right - w, 8, innerWidth - w - 8) + 'px';
  m.style.top = (r.bottom + 4 + h > innerHeight ? r.top - h - 4 : r.bottom + 4) + 'px';
  m.addEventListener('click', e => { const el = e.target.closest('[data-mi]'); if (!el) return; const it = items[+el.dataset.mi]; closeMenu(); it.fn && it.fn(); });
  m.addEventListener('mouseover', e => { const el = e.target.closest('.pi'); $$('.pi', m).forEach(x => x.classList.toggle('on', x === el)); });
  setTimeout(() => document.addEventListener('mousedown', menuAway, true), 0);
}
function menuAway(e) { if (!e.target.closest('#menu')) closeMenu(); }
function closeMenu() { const m = $('#menu'); if (m) m.remove(); document.removeEventListener('mousedown', menuAway, true); }
document.addEventListener('mouseover', e => {
  const el = e.target.closest('[data-tip]'); const t = $('#tip');
  if (!el) { t.classList.add('hide'); return; }
  t.innerHTML = el.getAttribute('data-tip-html') ? el.getAttribute('data-tip') : esc(el.getAttribute('data-tip'));
  t.classList.remove('hide');
  const r = el.getBoundingClientRect(); const w = t.offsetWidth, h = t.offsetHeight;
  t.style.left = clamp(r.left + r.width / 2 - w / 2, 8, innerWidth - w - 8) + 'px';
  t.style.top = (r.top - h - 8 < 8 ? r.bottom + 8 : r.top - h - 8) + 'px';
});

/* ── charts (sparkline + hover line chart) ──────────────── */
function spark(vals, color = 'var(--acc)', h = 30, max) {
  const v = (vals || []).filter(x => x != null);
  if (v.length < 2) return `<svg viewBox="0 0 100 ${h}" preserveAspectRatio="none" style="height:${h}px"><line x1="0" y1="${h - 1}" x2="100" y2="${h - 1}" stroke="var(--line2)" stroke-width="1" vector-effect="non-scaling-stroke"/></svg>`;
  const m = Math.max(max || 0, ...v, 1e-9);
  const pts = v.map((x, i) => `${(i / (v.length - 1) * 100).toFixed(2)},${(h - 2 - (x / m) * (h - 4)).toFixed(2)}`);
  return `<svg viewBox="0 0 100 ${h}" preserveAspectRatio="none" style="height:${h}px"><path d="M0,${h} L${pts.join(' L')} L100,${h} Z" fill="${color}" opacity=".12"/><polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>`;
}
function lineChart(id, series, opts = {}) {
  // series: [{name,color,pts:[[t,v]...]}]; one y-axis only
  const H = opts.h || 150, W = 600, pl = 34, pr = 8, pt = 8, pb = 18;
  const all = series.flatMap(s => s.pts);
  if (all.length < 2) return `<div class="chart" data-key="${id}"><div class="empty" style="height:${H}px;display:grid;place-items:center">No data yet</div></div>`;
  const t0 = Math.min(...all.map(p => p[0])), t1 = Math.max(...all.map(p => p[0]));
  const vmax = opts.max ? Math.max(opts.max, ...all.map(p => p[1] || 0)) : Math.max(opts.min || 1, Math.max(...all.map(p => p[1] || 0)) * 1.12);
  const x = t => pl + (t - t0) / Math.max(1, t1 - t0) * (W - pl - pr);
  const y = v => pt + (1 - (v || 0) / vmax) * (H - pt - pb);
  const ticks = [0, .5, 1].map(f => vmax * f);
  let g = `<g class="grid">${ticks.map(v => `<line x1="${pl}" x2="${W - pr}" y1="${y(v)}" y2="${y(v)}"/>`).join('')}</g>`;
  g += `<g class="axis">${ticks.map(v => `<text x="${pl - 6}" y="${y(v) + 3}" text-anchor="end">${opts.fmt ? opts.fmt(v) : v < 10 ? v.toFixed(1) : fmtK(v)}</text>`).join('')}
    <text x="${pl}" y="${H - 3}">${fmtClock(t0)}</text><text x="${W - pr}" y="${H - 3}" text-anchor="end">${fmtClock(t1)}</text></g>`;
  for (const s of series) {
    if (s.pts.length < 2) continue;
    const d = s.pts.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ',' + y(p[1]).toFixed(1)).join(' ');
    g += `<path d="${d} L${x(s.pts[s.pts.length - 1][0]).toFixed(1)},${y(0)} L${x(s.pts[0][0]).toFixed(1)},${y(0)} Z" fill="${s.color}" opacity=".08"/>`;
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" stroke-linejoin="round"/>`;
  }
  const payload = esc(JSON.stringify({ t0, t1, vmax, pl, pr, pt, pb, W, H, unit: opts.unit || '', s: series.map(s => ({ n: s.name, c: s.color, p: s.pts })) }));
  return `<div class="chart" data-key="${id}" data-chart="${payload}"><svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="height:${H}px">${g}<line class="xh" x1="0" x2="0" y1="${pt}" y2="${H - pb}" stroke="var(--tx3)" stroke-dasharray="2 3" opacity="0"/></svg><div class="hov hide"></div></div>`;
}
document.addEventListener('mousemove', e => {
  const c = e.target.closest('.chart[data-chart]');
  $$('.chart .hov').forEach(h => { if (!c || !c.contains(h)) h.classList.add('hide'); });
  $$('.chart .xh').forEach(l => { if (!c || !c.contains(l)) l.setAttribute('opacity', '0'); });
  if (!c) return;
  const d = JSON.parse(c.dataset.chart); const svg = c.querySelector('svg'); const r = svg.getBoundingClientRect();
  const fx = (e.clientX - r.left) / r.width * d.W; if (fx < d.pl || fx > d.W - d.pr) return;
  const t = d.t0 + (fx - d.pl) / (d.W - d.pl - d.pr) * (d.t1 - d.t0);
  const rows = d.s.map(s => { let best = s.p[0]; for (const p of s.p) if (Math.abs(p[0] - t) < Math.abs(best[0] - t)) best = p; return { n: s.n, c: s.c, v: best[1], t: best[0] }; });
  const hv = c.querySelector('.hov'); const xh = c.querySelector('.xh');
  xh.setAttribute('x1', fx); xh.setAttribute('x2', fx); xh.setAttribute('opacity', '1');
  hv.innerHTML = `<div class="faint">${fmtClock(rows[0].t)}</div>` + rows.map(r => `<div><span style="display:inline-block;width:8px;height:2px;background:${r.c};vertical-align:3px;margin-right:6px"></span>${esc(r.n)} <b>${fmtN(r.v, r.v < 10 ? 1 : 0)}</b>${esc(d.unit)}</div>`).join('');
  hv.classList.remove('hide'); hv.style.left = (e.clientX - c.getBoundingClientRect().left) + 'px'; hv.style.top = '0px';
});

/* ── routing ─────────────────────────────────────────────── */
function parseRoute() {
  const h = location.hash.replace(/^#\/?/, '');
  const [path, qs] = h.split('?');
  const parts = (path || 'deck').split('/').map(decodeURIComponent);
  const q = {}; new URLSearchParams(qs || '').forEach((v, k) => { q[k] = v; });
  return { view: parts[0] || 'deck', arg: parts[1] || '', sub: parts[2] || '', q };
}
function go(hash) { if (location.hash === hash) render(true); else location.hash = hash; }

/* ── shared pieces ───────────────────────────────────────── */
function led(state) { return `<span class="led ${esc(state || 'offline')}"></span>`; }
function stateTag(e) {
  const st = e.state || 'offline';
  const cls = { ready: 'ok', booting: 'boot', crashed: 'crit', sleeping: 'sleep' }[st] || '';
  return `<span class="tag ${cls}">${esc(STATE_LABEL[st] || st)}</span>`;
}
function hostMem(hid) {
  const st = S.snap.hosts[hid] || {}; const m = st.mem || {};
  const total = m.gpu_total || m.total || 0;
  const avail = m.gpu_total ? total - (m.gpu_used || 0) : (m.available || 0);
  const cached = m.gpu_total ? 0 : Math.min(m.cached || 0, avail);
  const engines = Object.values(S.snap.engines).filter(e => e.host === hid && (e.state === 'ready' || e.state === 'booting') && e.reserve)
    .sort((a, b) => engineIdx(a.name) - engineIdx(b.name));
  const reserved = engines.reduce((s, e) => s + (e.reserve || 0), 0);
  const other = Math.max(0, total - avail - reserved);
  return { total, avail, cached, free: Math.max(0, avail - cached), engines, reserved, other, uma: m.uma, headroom: (S.settings.headroom_gib || 4) * GIB };
}
function engineIdx(name) { const names = Object.keys(S.specs).sort((a, b) => ((S.specs[a].created || 0) - (S.specs[b].created || 0)) || a.localeCompare(b)); return names.indexOf(name); }
function tank(hid, o = {}) {
  const M = hostMem(hid);
  if (!M.total) return `<div class="tank${o.small ? ' sm' : ''}"><div class="seg-free"><span>${S.snap.hosts[hid]?.online ? 'reading memory…' : 'offline'}</span></div></div>`;
  const pct = b => (b / M.total * 100).toFixed(3) + '%';
  const evict = new Set(o.evict || []);
  let h = '';
  let replaced = 0;
  for (const e of M.engines) {
    if (o.replace && e.name === o.replace) { replaced = e.reserve || 0; continue; }
    const c = engineColor(e.name);
    h += `<div class="seg-e ${e.state}${evict.has(e.name) ? ' evict' : ''}" data-key="t-${esc(e.name)}" data-go="#/engines/${esc(e.name)}" style="width:${pct(e.reserve)};background-color:${c}" data-tip="${esc(e.name)} · ${fmtB(e.reserve)} reserved (${Math.round((e.util || 0) * 100)}% share)\n${esc(e.model)}"><span>${esc(e.name)}<small>${fmtGiB(e.reserve)} GiB</small></span></div>`;
  }
  if (M.other > M.total * 0.004) h += `<div class="seg-os" data-key="os" style="width:${pct(M.other)}" data-tip="System and other processes · ${fmtB(M.other)}"><span>${M.other > M.total * 0.04 ? 'sys' : ''}</span></div>`;
  let freeLeft = replaced + M.free + M.cached + [...evict].reduce((s, n) => s + ((S.snap.engines[n] || {}).reserve || 0), 0);
  if (o.ghost && o.ghost.need) {
    const g = o.ghost; const over = g.need > freeLeft - M.headroom * 0.5;
    const w = Math.min(g.need, Math.max(freeLeft, 0) + (over ? g.need * 0 : 0));
    h += `<div class="seg-ghost${over ? ' over' : ''}" data-key="ghost" style="width:${pct(Math.max(w, M.total * 0.01))}" data-tip="${esc(g.label || 'new engine')} would reserve ${fmtB(g.need)}"><span>${esc(g.label || 'new')} ${fmtGiB(g.need)} GiB${over ? ' · short ' + fmtGiB(g.need - freeLeft) : ''}</span></div>`;
    freeLeft -= w;
  }
  if (M.cached > M.total * 0.004 && !(o.ghost && o.ghost.need > M.free)) h += `<div class="seg-cache" data-key="cache" style="width:${pct(M.cached)}" data-tip="Page cache · ${fmtB(M.cached)}\nReclaimable, but CUDA on unified memory may count it as used. vllm-lab flushes it before a start when needed."><span>${M.cached > M.total * 0.05 ? 'cache' : ''}</span></div>`;
  const freeB = Math.max(0, M.avail + replaced); const ff = Math.max(0, freeLeft) / M.total;
  h += `<div class="seg-free" data-key="free" data-tip="${fmtB(freeB)} available"><span>${ff > 0.1 ? fmtGiB(freeB) + ' GiB free' : ff > 0.035 ? fmtGiB(freeB) : ''}</span></div>`;
  return `<div class="tank${o.small ? ' sm' : ''}">${h}</div>` + (o.scale ? `<div class="tank-scale"><span>0</span><span>${fmtGiB(M.total / 2)}</span><span>${fmtGiB(M.total)} GiB</span></div>` : '');
}
const FIX = {
  'make-room': ['Make room', 'Put other engines to sleep, then start'],
  'fit-util': ['Fit to free memory', 'Lower the memory share so it fits what is free now'],
  'flush-cache': ['Flush page cache', 'Drop the page cache, then start again'],
  'set-context': ['Use suggested context', 'Lower max context to what fits'],
  'lower-context': ['Halve context', 'Lower max context'],
  'gemma-runtime': ['Build patched runtime', 'Build the Transformers 5.14.1 image and restart'],
  'trust-remote-code': ['Trust remote code', 'Allow the repo’s custom modeling code and restart'],
  'quant-auto': ['Auto quantization', 'Let vLLM read the quantization from the checkpoint'],
  'change-port': ['Move to a free port', ''], 'to-llamacpp': ['Switch to llama.cpp', ''], offline: ['Start offline', 'Set HF_HUB_OFFLINE=1 (weights must be cached)'],
  force: ['Start anyway', ''], recreate: ['Recreate', 'Apply the current config'], start: ['Start', ''],
  'set-token': ['Set HF token', ''], 'open-hf': ['Open on Hugging Face', ''], edit: ['Edit config', ''], logs: ['Open log', ''],
  policy: ['Origin policy', ''], library: ['Open library', ''], doctor: ['Run doctor', ''], 'host-test': ['Test host', ''], 'host-enable': ['Hosts', ''],
  'webui-login': ['Open WebUI login', ''], 'webui-sync': ['Sync Open WebUI', ''], 'docker-group': ['How to fix', ''], adopt: ['Adopt', ''],
  'chmod-config': ['Fix permissions', ''], 'create-network': ['Create network', ''], 'pull-image': ['Pull image', ''], 'webui-open': ['Open WebUI', ''],
};
function fixButtons(name, fixes, cls = 'xs') {
  return (fixes || []).filter(f => FIX[f]).map((f, i) => `<button class="btn ${cls}${i === 0 ? ' pri' : ''}" data-act="fix" data-name="${esc(name)}" data-fix="${esc(f)}" ${FIX[f][1] ? `data-tip="${esc(FIX[f][1])}"` : ''}>${esc(FIX[f][0])}</button>`).join('');
}
function crashCard(name, cr) {
  if (!cr) return '';
  return `<div class="crash"><b>${esc(cr.title)}</b>${cr.hint ? `<div class="h">${esc(cr.hint)}</div>` : ''}${cr.line ? `<div class="ln" data-tip="${esc(cr.line)}">${esc(cr.line)}</div>` : ''}<div class="fixes">${fixButtons(name, cr.fixes)}</div></div>`;
}
function bootPipe(e, big) {
  const b = e.boot || {}; const stages = S.stages.length ? S.stages : [['pull', 'Image'], ['download', 'Weights'], ['load', 'Load'], ['compile', 'Compile'], ['graphs', 'CUDA graphs'], ['serve', 'Serve']];
  const idx = Math.max(0, stages.findIndex(s => s[0] === b.stage));
  const cells = stages.map((s, i) => i < idx ? '<div class="st done"></div>' : i === idx ? `<div class="st cur${b.pct == null ? ' indet' : ''}"><i style="width:${b.pct ?? 0}%"></i></div>` : '<div class="st"></div>').join('');
  const labs = stages.map((s, i) => `<span class="${i < idx ? 'done' : i === idx ? 'cur' : ''}">${esc(s[1])}</span>`).join('');
  const job = e.job && e.job.kind !== 'bench' ? e.job : null;
  const detail = job && job.stage && !b.stage ? job.stage : [b.detail, b.pct != null ? b.pct + '%' : ''].filter(Boolean).join(' · ');
  return `<div style="display:flex;flex-direction:column;gap:8px"><div><div class="pipe">${cells}</div><div class="pipe-lab">${labs}</div></div>
    <div class="bootline"><span class="ellip">${esc(detail || 'starting')}</span><span class="faint nowrap">${b.elapsed != null ? fmtDur(b.elapsed) : ''}</span></div></div>`;
}
function histOf(name) { return S.hist.e[name] || []; }

/* ── engine module (deck card) ───────────────────────────── */
function engineCard(e) {
  const spec = S.specs[e.name] || {}; const st = e.state || 'offline';
  const color = engineColor(e.name);
  const m = e.metrics || {};
  let body = '';
  if (st === 'ready') {
    const h = histOf(e.name).slice(-90);
    body = `<div class="metrics">
      <div class="metric"><div class="v">${m.gen_tps != null ? fmtN(m.gen_tps, m.gen_tps < 10 ? 1 : 0) : '0'}<small>tok/s</small></div><div class="l">Output</div></div>
      <div class="metric"><div class="v">${m.ttft != null ? fmtN(m.ttft * 1000) + '<small>ms</small>' : '–'}</div><div class="l">TTFT</div></div>
      <div class="metric"><div class="v">${fmtN(m.running || 0)}<small>/ ${fmtN(m.waiting || 0)}</small></div><div class="l">Run / queue</div></div>
      <div class="metric"><div class="v">${m.kv != null ? fmtN(m.kv * 100) + '<small>%</small>' : '–'}</div><div class="l">KV cache</div></div></div>
      <div class="spark-row">${spark(h.map(p => p[1]), color, 30)}</div>
      <div class="kvbar" data-tip="KV cache ${m.kv != null ? fmtN(m.kv * 100, 1) + '%' : '–'}${m.kv_capacity ? ' of ' + fmtN(m.kv_capacity) + ' tokens' : ''}"><i style="width:${((m.kv || 0) * 100).toFixed(1)}%;background:${color}"></i></div>`;
  } else if (st === 'booting') {
    body = bootPipe(e);
  } else if (st === 'crashed') {
    body = crashCard(e.name, e.crash);
  } else if (st === 'offline') {
    body = `<div class="dim">${esc(hostOf(e.host)?.label || e.host)} is unreachable${e.why ? ' · ' + esc(e.why) : ''}</div>`;
  } else {
    const total = hostTotal(e.host);
    const need = spec.backend === 'vllm' && total ? spec.util * total : 0;
    body = `<div class="row wrap" style="gap:14px">
      ${need ? `<div class="metric"><div class="v">${fmtGiB(need)}<small>GiB</small></div><div class="l">Reserves</div></div>` : ''}
      <div class="metric"><div class="v">${fmtK(spec.max_len)}</div><div class="l">Context</div></div>
      <div class="metric"><div class="v" style="font-size:13px">${esc(spec.kv_dtype || 'auto')}</div><div class="l">KV dtype</div></div>
      ${st === 'sleeping' ? `<div class="grow"></div><span class="tag sleep" data-tip="Asleep. A gateway request for this model wakes it.">${I.moon.replace('<svg', '<svg width="11" height="11"')} wakes on request</span>` : ''}</div>`;
  }
  const tags = [];
  if (e.drift) tags.push(`<span class="tag warn" data-tip="Settings changed since this container was created. Recreate to apply.">config changed</span>`);
  if (e.legacy && st !== 'absent') tags.push(`<span class="tag" data-tip="Created before vllm-lab 2. Works as-is; recreate once to track config drift.">legacy</span>`);
  if (st === 'ready' && m.spec_accept != null) tags.push(`<span class="tag" data-tip="Speculative decoding acceptance rate">spec ${fmtN(m.spec_accept * 100)}%</span>`);
  if (st === 'ready' && e.idle_sleep_min) {
    const left = e.idle_sleep_min * 60 - (Date.now() / 1000 - (e.last_active || Date.now() / 1000));
    tags.push(`<span class="tag sleep" data-tip="Sleeps after ${e.idle_sleep_min} min without requests">${left > 0 ? 'sleeps in ' + fmtDur(left) : 'sleeping soon'}</span>`);
  }
  const job = e.job;
  let prim = '';
  if (job && st !== 'booting') prim = `<span class="faint mono ellip" style="font-size:11px">${esc(job.stage || job.kind)}</span>`;
  else if (st === 'ready') prim = `<a class="btn sm" href="#/play?e=${encodeURIComponent(e.name)}">${I.play}Chat</a>`;
  else if (st === 'sleeping') prim = `<button class="btn sm pri" data-act="start" data-name="${esc(e.name)}">${I.bolt}Wake</button>`;
  else if (['stopped', 'absent', 'crashed'].includes(st)) prim = `<button class="btn sm ${st === 'crashed' ? '' : 'pri'}" data-act="start" data-name="${esc(e.name)}">${I.start}Start</button>`;
  const stopBtn = (st === 'ready' || st === 'booting') && (!job || st === 'booting') ? `<button class="btn sm ghost icon" data-act="stop" data-name="${esc(e.name)}" data-tip="Stop — frees ${fmtB(e.reserve)}">${I.stop}</button>` : '';
  const url = e.human_url && e.port ? `<span class="mono faint ellip" style="font-size:11px;margin-left:auto" data-tip="Click to copy">${''}<span data-act="copy" data-text="${esc(e.human_url)}" style="cursor:copy">:${esc(e.port)}/v1</span></span>` : '<span class="grow"></span>';
  return `<div class="mod ${st}${['stopped', 'absent', 'sleeping', 'offline'].includes(st) ? ' off' : ''}" data-key="m-${esc(e.name)}">
    <div class="stripe" style="background:${color}"></div>
    <div class="mh">${led(st)}<a class="nm" href="#/engines/${encodeURIComponent(e.name)}">${esc(e.name)}</a>${stateTag(e)}<span class="grow"></span><span class="faint mono" style="font-size:11px">${esc(hostOf(e.host)?.label || e.host)}</span></div>
    <div class="mm" data-tip="${esc(e.model)}">${esc(e.model)}</div>
    <div class="mbody">${body}${tags.length ? `<div class="row wrap" style="gap:5px">${tags.join('')}</div>` : ''}</div>
    <div class="mf">${prim}${stopBtn}<a class="btn sm ghost icon" href="#/engines/${encodeURIComponent(e.name)}/logs" data-tip="Log">${I.logs}</a>${url}<button class="btn sm ghost icon" data-act="engine-menu" data-name="${esc(e.name)}">${I.more}</button></div>
  </div>`;
}
function sortedEngines(filter) {
  const hostsOrder = (S.settings.hosts || []).map(h => h.id);
  return Object.values(S.snap.engines).filter(e => S.specs[e.name]).filter(e => !filter || filter(e))
    .sort((a, b) => (hostsOrder.indexOf(a.host) - hostsOrder.indexOf(b.host)) || ((STATE_ORDER[a.state] ?? 9) - (STATE_ORDER[b.state] ?? 9)) || a.name.localeCompare(b.name));
}

/* ── deck ────────────────────────────────────────────────── */
function hostBand(h) {
  const st = h.st || {}; const g = (st.gpus || [])[0] || {};
  const hh = (S.hist.h[h.id] || []).slice(-90);
  const M = hostMem(h.id);
  const running = Object.values(S.snap.engines).filter(e => e.host === h.id && e.state === 'ready');
  const tps = running.reduce((s, e) => s + ((e.metrics || {}).gen_tps || 0), 0);
  if (!h.enabled) return `<div class="hostband" data-key="hb-${esc(h.id)}"><div class="hb-head"><a class="hb-name" href="#/hosts">${led('offline')}${esc(h.label)}</a><span class="tag">disabled</span><span class="faint">${esc(h.note || '')}</span><span class="grow"></span><a class="btn sm" href="#/hosts">Set up</a></div></div>`;
  if (!st.online) return `<div class="hostband" data-key="hb-${esc(h.id)}"><div class="hb-head"><a class="hb-name" href="#/hosts">${led('crashed')}${esc(h.label)}</a><span class="tag crit">unreachable</span><span class="dim mono" style="font-size:11.5px">${esc(st.why || 'connecting…')}</span><span class="grow"></span><button class="btn sm" data-act="host-test" data-host="${esc(h.id)}">Test connection</button></div></div>`;
  const gauges = [
    ['GPU', g.util != null ? fmtN(g.util) : '–', '%', spark(hh.map(p => p[2]), 'var(--tx2)', 16, 100)],
    ['Temp', g.temp != null ? fmtN(g.temp) : '–', '°C', spark(hh.map(p => p[3]), 'var(--tx2)', 16)],
    ['Power', g.power != null ? fmtN(g.power) : '–', 'W', spark(hh.map(p => p[4]), 'var(--tx2)', 16)],
    ['Memory', fmtGiB(M.total - M.avail), '/ ' + fmtGiB(M.total) + ' GiB', spark(hh.map(p => p[1]), 'var(--tx2)', 16, 100)],
    ['Output', fmtN(tps, tps < 10 ? 1 : 0), 'tok/s', ''],
  ];
  return `<div class="hostband" data-key="hb-${esc(h.id)}">
    <div class="hb-head"><a class="hb-name" href="#/hosts">${led('ready')}${esc(h.label)}</a>
      <span class="faint mono" style="font-size:11.5px">${esc(g.name || 'no GPU')}${M.uma ? ' · unified memory' : ''}${st.disk ? ' · ' + fmtB(st.disk.free) + ' disk free' : ''}</span>
      <div class="hb-meta">${gauges.map(([l, v, u, sp]) => `<div class="gauge"><span class="cap">${l}</span><span class="v">${v}<small>${u}</small></span>${sp ? `<div class="spk">${sp}</div>` : ''}</div>`).join('')}</div></div>
    ${tank(h.id, { scale: true })}
  </div>`;
}
function feedHTML(n = 40, filter) {
  const evs = S.events.filter(e => !filter || filter(e)).slice(-n).reverse();
  if (!evs.length) return '<div class="faint" style="padding:10px 0">Quiet so far.</div>';
  const seen = new Set();
  return evs.map(e => { const first = !seen.has(e.source); seen.add(e.source); return `<div class="ev ${esc(e.level)}" data-key="ev-${esc(e.id)}"><span class="t">${fmtClock(e.t)}</span><span class="d"></span><span class="m">${linkify(e, first)}</span></div>`; }).join('');
}
function linkify(e, withFixes) {
  let m = esc(e.msg); const src = e.source || '';
  if (S.specs[src]) m = m.replace(new RegExp('\\b' + src.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b'), `<a href="#/engines/${encodeURIComponent(src)}"><b>${esc(src)}</b></a>`);
  if (withFixes && e.data && e.data.fixes && S.specs[src] && (S.snap.engines[src] || {}).state === 'crashed') m += `<div class="fixes" style="margin-top:5px">${fixButtons(src, e.data.fixes.slice(0, 2))}</div>`;
  return m;
}
const views = {};
views.deck = {
  title: () => 'Deck',
  render() {
    const hosts = hostList();
    const f = store.get('deckFilter', 'all');
    const list = sortedEngines(f === 'live' ? e => ['ready', 'booting', 'crashed'].includes(e.state) : null);
    const gw = S.settings.gateway || {};
    const gwReq = Object.values(S.snap.gateway || {}).reduce((s, g) => s + (g.requests || 0), 0);
    return `<div class="cols">
      <div>
        <div class="sec">${hosts.map(hostBand).join('')}</div>
        <div class="sec"><div class="sh"><h2>Engines</h2><div class="rule"></div>
          <div class="seg"><button class="${f === 'all' ? 'on' : ''}" data-act="deck-filter" data-v="all">All</button><button class="${f === 'live' ? 'on' : ''}" data-act="deck-filter" data-v="live">Live</button></div>
          <a class="btn sm pri" href="#/launch">${I.launch}New engine</a></div>
          ${list.length ? `<div class="mods">${list.map(engineCard).join('')}</div>` : `<div class="empty"><b>No engines ${f === 'live' ? 'running' : 'yet'}</b>${f === 'live' ? 'Everything is stopped or asleep.' : '<a class="btn pri" href="#/launch" style="margin-top:10px">Launch a model</a>'}</div>`}
        </div>
      </div>
      <aside class="panel">
        <div class="sec"><div class="sh"><h2>Gateway</h2>${tip('One OpenAI-compatible endpoint for every engine on every host. Ask for a sleeping model and it wakes up.')}<div class="rule"></div><span class="led ${gw.enabled ? 'ready' : 'offline'}"></span></div>
          <div class="urlrow" style="grid-template-columns:1fr auto;border:0;padding:0 0 6px"><code data-act="copy" data-text="${esc(location.origin + '/v1')}" style="cursor:copy">${esc(location.origin)}/v1</code><button class="btn xs" data-act="copy" data-text="${esc(location.origin + '/v1')}">${I.copy}</button></div>
          <div class="row faint mono" style="font-size:11.5px;gap:14px"><span>${fmtN(gwReq)} request${gwReq === 1 ? '' : 's'}</span><span>${gw.autowake ? 'auto-wake on' : 'auto-wake off'}</span>${gw.key_set ? '<span>key required</span>' : ''}</div>
        </div>
        <div class="sec"><div class="sh"><h2>Activity</h2><div class="rule"></div><a class="btn xs ghost" href="#/activity">All</a></div><div class="feed">${feedHTML(30)}</div></div>
      </aside></div>`;
  },
  live: true,
};

/* ── engines list ───────────────────────────────────────── */
views.engines = {
  title: r => r.arg ? `<span class="crumb">Engines /</span> ${esc(r.arg)}` : 'Engines',
  render(r) { return r.arg ? engineDetail(r.arg, r.sub || 'overview') : engineTable(); },
  live: true,
};
function engineTable() {
  const q = (store.get('engQ', '') || '').toLowerCase();
  const list = sortedEngines(e => !q || (e.name + ' ' + e.model + ' ' + e.host).toLowerCase().includes(q));
  const um = Object.values(S.snap.hosts).flatMap(h => (h.unmanaged || []).map(u => ({ ...u, host: h.id })));
  const rows = list.map(e => {
    const m = e.metrics || {}; const spec = S.specs[e.name] || {};
    return `<tr data-key="r-${esc(e.name)}"><td style="width:14px">${led(e.state)}</td>
      <td class="nowrap"><a href="#/engines/${encodeURIComponent(e.name)}" style="font-weight:650;text-decoration:none">${esc(e.name)}</a>${e.drift ? ' <span class="tag warn">drift</span>' : ''}</td>
      <td>${stateTag(e)}${e.state === 'booting' ? ` <span class="faint mono" style="font-size:11px">${esc(((e.boot || {}).detail || ''))}</span>` : ''}${e.state === 'crashed' ? ` <span class="crit" style="font-size:12px">${esc((e.crash || {}).title || '')}</span>` : ''}</td>
      <td class="faint">${esc(hostOf(e.host)?.label || e.host)}</td>
      <td class="mono ellip" style="max-width:320px;font-size:12px" data-tip="${esc(e.model)}">${esc(e.model)}</td>
      <td class="mono">${esc(spec.backend === 'llamacpp' ? 'llama.cpp' : 'vLLM')}</td>
      <td class="r num">${e.state === 'ready' && m.gen_tps != null ? fmtN(m.gen_tps, 1) : '–'}</td>
      <td class="r num">${e.state === 'ready' && m.kv != null ? fmtN(m.kv * 100) + '%' : '–'}</td>
      <td class="r num">${e.reserve ? fmtB(e.reserve) : '–'}</td>
      <td class="mono faint" style="font-size:11.5px">${e.port ? ':' + e.port : '–'}</td>
      <td><div class="acts">${e.state === 'offline' ? '' : e.state === 'ready' || e.state === 'booting' ? `<button class="btn xs" data-act="stop" data-name="${esc(e.name)}">${I.stop}Stop</button>` : `<button class="btn xs" data-act="start" data-name="${esc(e.name)}">${I.start}Start</button>`}<button class="btn xs ghost icon" data-act="engine-menu" data-name="${esc(e.name)}">${I.more}</button></div></td></tr>`;
  }).join('');
  return `<div class="row" style="margin-bottom:14px"><input type="search" placeholder="Filter engines" value="${esc(store.get('engQ', ''))}" data-input="engQ" style="max-width:300px" data-keep><span class="grow"></span><a class="btn pri" href="#/launch">${I.launch}New engine</a></div>
    <table class="tbl"><thead><tr><th></th><th>Engine</th><th>State</th><th>Host</th><th>Model</th><th>Backend</th><th class="r">tok/s</th><th class="r">KV</th><th class="r">Reserved</th><th>Port</th><th></th></tr></thead>
    <tbody>${rows || '<tr><td colspan="11"><div class="empty"><b>No engines match</b></div></td></tr>'}</tbody></table>
    ${um.length ? `<div class="sp2"></div><div class="sh"><h2>Unmanaged containers</h2>${tip('Containers with the engine prefix that vllm-lab did not create. Adopt reads their command line and turns them into managed engines.')}<div class="rule"></div></div>
      <table class="tbl"><tbody>${um.map(u => `<tr data-key="u-${esc(u.name)}"><td>${led(u.state === 'running' ? 'ready' : 'stopped')}</td><td class="mono">${esc(u.name)}</td><td class="faint mono">${esc(u.image)}</td><td class="faint">${esc(u.status)}</td><td><div class="acts"><button class="btn xs" data-act="adopt" data-host="${esc(u.host)}" data-container="${esc(u.name)}">Adopt</button></div></td></tr>`).join('')}</tbody></table>` : ''}`;
}

/* ── engine detail ──────────────────────────────────────── */
function engineDetail(name, tab) {
  const e = S.snap.engines[name]; const spec = S.specs[name];
  if (!spec) return `<div class="empty"><b>No engine named ${esc(name)}</b><a class="btn" href="#/engines" style="margin-top:10px">All engines</a></div>`;
  const st = e ? e.state : 'offline';
  const tabs = [['overview', 'Overview'], ['logs', 'Log'], ['connect', 'Connect'], ['config', 'Config'], ['bench', 'Bench'], ['inspect', 'Inspect']];
  const act = [];
  if (st === 'ready' || st === 'booting') { act.push(`<a class="btn" href="#/play?e=${encodeURIComponent(name)}">${I.play}Chat</a>`); act.push(`<button class="btn" data-act="stop" data-name="${esc(name)}">${I.stop}Stop</button>`); }
  else if (st !== 'offline') act.push(`<button class="btn pri" data-act="start" data-name="${esc(name)}">${st === 'sleeping' ? I.bolt + 'Wake' : I.start + 'Start'}</button>`);
  if (e && e.drift) act.push(`<button class="btn" data-act="recreate" data-name="${esc(name)}" data-tip="Apply the saved config">${I.restart}Recreate</button>`);
  act.push(`<button class="btn icon" data-act="engine-menu" data-name="${esc(name)}">${I.more}</button>`);
  let body = '';
  if (tab === 'overview') body = engineOverview(name, e || {}, spec);
  else if (tab === 'logs') body = `<div id="logpane" data-keep></div>`;
  else if (tab === 'connect') body = engineConnect(name, e || {}, spec);
  else if (tab === 'config') body = `<div id="cfgpane" data-keep></div>`;
  else if (tab === 'bench') body = `<div id="benchpane" data-keep></div>`;
  else if (tab === 'inspect') body = `<div id="inspane" data-keep></div>`;
  return `<div class="row top" style="margin-bottom:16px;gap:14px"><div class="stripe" style="width:4px;align-self:stretch;background:${engineColor(name)}"></div>
      <div class="grow"><div class="row" style="gap:10px">${led(st)}<span style="font:700 22px/1 var(--sans)">${esc(name)}</span>${e ? stateTag(e) : ''}${e && e.drift ? '<span class="tag warn">config changed</span>' : ''}</div>
      <div class="row wrap faint mono" style="font-size:12px;margin-top:7px;gap:14px"><a href="${hfUrl(spec.model)}" target="_blank" rel="noopener" style="color:var(--tx2)">${esc(spec.model)} ${I.ext.replace('<svg', '<svg width="10" height="10"')}</a><span>${esc(hostOf(spec.host)?.label || spec.host)}</span><span>${spec.backend === 'llamacpp' ? 'llama.cpp' : 'vLLM'}</span><span>:${esc(spec.port)}</span></div></div>
      <div class="row">${act.join('')}</div></div>
    <div class="tabs" data-keep-tabs>${tabs.map(([k, l]) => `<button class="${tab === k ? 'on' : ''}" data-go="#/engines/${encodeURIComponent(name)}/${k}">${l}</button>`).join('')}</div>${body}`;
}
function engineOverview(name, e, spec) {
  const m = e.metrics || {}; const c = e.container || {}; const b = e.boot || {};
  const h = histOf(name);
  const color = engineColor(name);
  let status = '';
  if (e.state === 'booting') status = `<div class="sec">${bootPipe(e, true)}</div>`;
  if (e.state === 'crashed') status = `<div class="sec">${crashCard(name, e.crash)}</div>`;
  if (e.job && e.state !== 'booting') status += `<div class="sec faint mono">${esc(e.job.kind)} · ${esc(e.job.stage || '')}</div>`;
  const total = hostTotal(spec.host);
  const gw = (S.snap.gateway || {})[name];
  return `${status}<div class="cols">
    <div>
      <div class="metrics" style="grid-template-columns:repeat(6,minmax(0,1fr));margin-bottom:18px">
        <div class="metric"><div class="v">${m.gen_tps != null ? fmtN(m.gen_tps, 1) : '–'}<small>tok/s</small></div><div class="l">Output</div></div>
        <div class="metric"><div class="v">${m.prompt_tps != null ? fmtN(m.prompt_tps) : '–'}<small>tok/s</small></div><div class="l">Prefill</div></div>
        <div class="metric"><div class="v">${m.ttft != null ? fmtN(m.ttft * 1000) : '–'}<small>ms</small></div><div class="l">TTFT</div></div>
        <div class="metric"><div class="v">${fmtN(m.running || 0)}<small>/ ${fmtN(m.waiting || 0)}</small></div><div class="l">Run / queue</div></div>
        <div class="metric"><div class="v">${m.prefix_hit != null ? fmtN(m.prefix_hit * 100) + '<small>%</small>' : '–'}</div><div class="l">Prefix hits</div></div>
        <div class="metric"><div class="v">${m.spec_accept != null ? fmtN(m.spec_accept * 100) + '<small>%</small>' : '–'}</div><div class="l">Spec accept</div></div>
      </div>
      <div class="sh"><h2>Throughput</h2><div class="rule"></div><span class="faint mono" style="font-size:11px">last ${fmtDur(h.length ? h[h.length - 1][0] - h[0][0] : 0)}</span></div>
      ${lineChart('c-tps-' + name, [{ name: 'output', color, pts: h.map(p => [p[0], p[1]]) }], { unit: ' tok/s', h: 150, min: 10 })}
      <div class="sp"></div>
      <div class="sh"><h2>KV cache &amp; queue</h2><div class="rule"></div></div>
      ${lineChart('c-kv-' + name, [{ name: 'KV %', color: 'var(--tx2)', pts: h.map(p => [p[0], p[2]]) }], { unit: '%', h: 110, max: 100, fmt: v => Math.round(v) + '%' })}
    </div>
    <aside class="panel">
      <div class="sh"><h2>Memory</h2><div class="rule"></div></div>
      ${tank(spec.host, { small: true })}
      <div class="sp"></div>
      <dl class="kvl">
        <dt>Share</dt><dd>${spec.backend === 'vllm' ? `${fmtN(spec.util * 100)}% · ${fmtB(spec.util * total)}` : 'llama.cpp (dynamic)'}</dd>
        <dt>Context</dt><dd>${fmtN(spec.max_len)} tokens · KV ${esc(spec.kv_dtype)}</dd>
        ${b.kv_tokens ? `<dt>KV capacity</dt><dd>${fmtN(b.kv_tokens)} tokens${b.max_conc ? ` · ${fmtN(b.max_conc, 1)}× at full context` : ''}</dd>` : ''}
        ${b.weights_gib ? `<dt>Weights</dt><dd>${fmtN(b.weights_gib, 1)} GiB</dd>` : ''}
        ${b.took ? `<dt>Last boot</dt><dd>${fmtDur(b.took)}</dd>` : ''}
        <dt>Uptime</dt><dd>${e.uptime ? fmtDur(e.uptime) : '–'}</dd>
        <dt>Idle sleep</dt><dd>${spec.idle_sleep_min ? spec.idle_sleep_min + ' min' : 'off'} · wake ${spec.wake ? 'on' : 'off'}</dd>
        ${gw ? `<dt>Gateway</dt><dd>${fmtN(gw.requests)} req · ${fmtN(gw.errors)} err · ${fmtAgo(gw.last)} ago</dd>` : ''}
      </dl>
      <div class="sp2"></div><div class="sh"><h2>Container</h2><div class="rule"></div></div>
      <dl class="kvl">
        <dt>Name</dt><dd>${esc((S.settings.container_prefix || '') + name)}</dd>
        <dt>Image</dt><dd>${esc(c.image || spec.image || S.settings.vllm_image)}</dd>
        <dt>ID</dt><dd>${esc(c.id || '–')}</dd>
        <dt>Status</dt><dd>${esc(c.status || 'none')}${c.exit != null && c.status !== 'running' ? ' · exit ' + esc(c.exit) : ''}${c.oom ? ' · OOM' : ''}</dd>
        <dt>Restarts</dt><dd>${esc(c.restarts ?? 0)} · policy ${esc(c.policy || '–')}</dd>
        <dt>Started</dt><dd>${esc((c.started || '').replace('T', ' ').slice(0, 19) || '–')}</dd>
      </dl>
    </aside></div>`;
}
function engineConnect(name, e, spec) {
  const human = e.human_url || '';
  const inner = `http://${(S.settings.container_prefix || '')}${name}:8000/v1`;
  const gw = location.origin + '/v1';
  const model = spec.served_name || spec.model;
  const key = (S.settings.gateway || {}).key_set ? '$VLLM_LAB_KEY' : 'local';
  const snippets = {
    curl: `curl ${human}/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello"}]}'`,
    python: `from openai import OpenAI\n\nclient = OpenAI(base_url="${human}", api_key="local")\nr = client.chat.completions.create(\n    model="${model}",\n    messages=[{"role": "user", "content": "Hello"}],\n)\nprint(r.choices[0].message.content)`,
    gateway: `curl ${gw}/chat/completions \\\n  -H "Authorization: Bearer ${key}" -H "Content-Type: application/json" \\\n  -d '{"model": "${name}", "stream": true, "messages": [{"role": "user", "content": "Hello"}]}'\n\n# the gateway wakes ${name} if it is asleep`,
    env: `export OPENAI_BASE_URL=${human}\nexport OPENAI_API_KEY=local\nexport OPENAI_MODEL=${model}`,
  };
  const which = store.get('snip', 'curl');
  return `<div class="diagram"><div class="n"><b>You, scripts, apps</b><code>${esc(human)}</code></div><div class="a"></div>
      <div class="n"><b>${esc(hostOf(spec.host)?.label || spec.host)} :${esc(spec.port)}</b><code>published on ${esc((S.settings.hosts || []).find(h => h.id === spec.host)?.bind || '127.0.0.1')}</code></div><div class="a"></div>
      <div class="n"><b>Container :8000</b><code>${esc(inner)}</code></div></div>
    <div class="urlrow"><span class="cap">Human URL ${tip('What you, scripts and other apps on the host use. Never port 8000 — that one only exists inside Docker.')}</span><code>${esc(human)}</code><button class="btn xs" data-act="copy" data-text="${esc(human)}">${I.copy}Copy</button></div>
    <div class="urlrow"><span class="cap">Open WebUI ${tip('Containers on the same Docker network reach the engine by name. vllm-lab registers this automatically.')}</span><code>${esc(inner)}</code><button class="btn xs" data-act="copy" data-text="${esc(inner)}">${I.copy}Copy</button></div>
    <div class="urlrow"><span class="cap">Gateway ${tip('One URL for every engine. Pick the engine with the model field; sleeping engines wake on demand.')}</span><code>${esc(gw)} · model "${esc(name)}"</code><button class="btn xs" data-act="copy" data-text="${esc(gw)}">${I.copy}Copy</button></div>
    <div class="urlrow"><span class="cap">Model id</span><code>${esc(model)}</code><button class="btn xs" data-act="copy" data-text="${esc(model)}">${I.copy}Copy</button></div>
    <div class="sp2"></div>
    <div class="seg" style="margin-bottom:10px">${Object.keys(snippets).map(k => `<button class="${k === which ? 'on' : ''}" data-act="snip" data-v="${k}">${k}</button>`).join('')}</div>
    <div class="code"><button class="btn xs cp" data-act="copy" data-text="${esc(snippets[which])}">${I.copy}</button><pre>${esc(snippets[which])}</pre></div>`;
}

/* ── planner math (mirrors plan_numbers in vllm_lab.py) ─── */
function kvBytesPerEl(kvd, dtype) { const k = String(kvd || 'auto').toLowerCase(); if (k.includes('fp8') || k.includes('int8')) return 1; if (k.includes('fp4')) return .5; if (k === 'auto' && String(dtype || '').includes('float32')) return 4; return 2; }
function planNumbers(p, F) {
  const meta = (p && p.meta) || {}; const kv = meta.kv || {};
  const bpe = kvBytesPerEl(F.kv_dtype, kv.dtype);
  let weights = (meta.weights_bytes || 0);
  if (F.backend === 'llamacpp' || !weights) { const g = (meta.gguf || []).find(x => x.quant === F.gguf) || null; weights = g ? g.bytes : (meta.gguf_bytes || weights); }
  weights += (p.drafts || []).reduce((s, d) => s + (d.bytes || 0), 0);
  const ctx = F.max_len, seqs = Math.max(1, F.seqs || 1);
  const sw = kv.sliding_window || ctx;
  const tokFull = (kv.kv_elems_full || 0) * bpe, tokSl = (kv.kv_elems_sliding || 0) * bpe;
  const perSeq = tokFull * ctx + tokSl * Math.min(ctx, sw) + (kv.mamba_bytes_per_seq || 0);
  const overhead = p.overhead || 2.5 * GIB;
  const need = weights + overhead + perSeq * seqs;
  const total = p.total || hostTotal(F.host) || 0;
  const utilNeed = total ? need / total : null;
  const utilRec = utilNeed ? Math.min(0.95, Math.ceil(utilNeed * 1.06 * 100) / 100) : null;
  const util = F.share === 'manual' ? F.util : (utilRec || F.util);
  const pool = util * total - weights - overhead;
  const maxCtx = tokFull ? Math.max(0, (pool / seqs - (kv.mamba_bytes_per_seq || 0) - tokSl * Math.min(ctx, sw)) / tokFull) : null;
  return { weights, perSeq, kvTotal: perSeq * seqs, overhead, need, total, utilNeed, utilRec, util, reserve: util * total, kvKnown: !!kv.known, maxCtx, tokBytes: tokFull, pool };
}

/* ── form model shared by Launch and engine Config ─────── */
let F = null;          // the spec being edited
let FP = null;         // plan for F.model
let FMODE = 'launch';  // 'launch' | 'edit'
const FORM_DEFAULTS = { name: '', host: 'titan', backend: 'vllm', model: '', util: 0.5, max_len: 32768, kv_dtype: 'fp8', quant: '', trust_remote_code: false,
  max_num_seqs: '', served_name: '', extra: '', env: {}, autostart: true, idle_sleep_min: 0, wake: true, note: '', image: '', port: '', gguf: '', seqs: 1, share: 'auto', blueprint: '' };
function formFrom(spec, mode) {
  FMODE = mode;
  F = { ...FORM_DEFAULTS, ...JSON.parse(JSON.stringify(spec || {})) };
  F.share = mode === 'edit' ? 'manual' : 'auto';
  F.seqs = F.seqs || 1;
  if (!F.host || !(S.settings.hosts || []).some(h => h.id === F.host)) F.host = firstOnlineHost();
  FP = null;
}
function firstOnlineHost() { const hs = hostList(); return (hs.find(h => h.enabled && h.st.online) || hs[0] || { id: 'titan' }).id; }
function slugify(model) {
  let b = String(model || '').split('/').pop().split(':')[0].toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
  for (const j of ['-instruct', '-it', '-chat', '-hf', '-nvfp4', '-fp8', '-awq', '-gptq', '-gguf', '-bf16']) if (b.endsWith(j) && b.length > j.length + 2) b = b.slice(0, -j.length);
  b = b.slice(0, 28).replace(/-$/, '') || 'engine';
  let n = b, i = 2; while (S.specs[n] && FMODE === 'launch') n = b + '-' + (i++);
  return n;
}
async function loadPlan() {
  if (!F || !F.model) return;
  const key = F.model + '|' + F.host + '|' + F.extra;
  if (FP && FP._key === key) return;
  FP = { _key: key, loading: true, meta: {} };
  updateFit();
  try {
    const p = await api(`/api/plan?model=${encodeURIComponent(F.model)}&host=${encodeURIComponent(F.host)}&extra=${encodeURIComponent(F.extra || '')}`);
    if (!F || F.model + '|' + F.host + '|' + F.extra !== key) return;
    FP = { ...p, _key: key };
    const kv = (p.meta || {}).kv || {};
    if (FMODE === 'launch' && !F._ctxTouched && kv.ctx_max) F.max_len = Math.min(kv.ctx_max, F.max_len > 0 && F._fromBp ? F.max_len : Math.min(kv.ctx_max, 32768));
    if (p.meta && p.meta.gguf && !F.gguf) F.gguf = p.meta.gguf_default || '';
    if (p.meta && p.meta.format === 'gguf' && F.backend !== 'llamacpp' && FMODE === 'launch') F.backend = 'llamacpp';
  } catch (e) { FP = { _key: key, meta: { error: e.message } }; }
  const mh = $('#modelhead'); if (mh) mh.innerHTML = modelHead();
  const ctl = $('#ctlbox'); if (ctl) morph(ctl, ctlBox());
  updateFit();
}
function modelHead() {
  if (!F || !F.model) return '';
  const m = (FP && FP.meta) || {}; const [org, rest] = splitId(F.model); const kv = m.kv || {};
  const tags = [];
  if (m.params) tags.push(`<span class="tag solid">${fmtParams(m.params)} params</span>`);
  if (m.weights_bytes) tags.push(`<span class="tag solid">${fmtB(m.weights_bytes)} weights</span>`);
  if (kv.arch) tags.push(`<span class="tag">${esc(kv.arch.replace(/ForCausalLM|ForConditionalGeneration/, ''))}</span>`);
  if (kv.quant_method) tags.push(`<span class="tag">${esc(kv.quant_method)}</span>`);
  if (kv.ctx_max) tags.push(`<span class="tag">${fmtK(kv.ctx_max)} native ctx</span>`);
  if (m.gated) tags.push(`<span class="tag warn" data-tip="Accept the terms on Hugging Face with the account that owns your token.">gated</span>`);
  if (m.license) tags.push(`<span class="tag">${esc(m.license)}</span>`);
  if (FP && FP.on_disk) tags.push(`<span class="tag ok">on disk</span>`);
  if (m.source === 'local') tags.push(`<span class="tag" data-tip="Hub unreachable — read from the local cache">local metadata</span>`);
  const pol = m.policy || {};
  if (pol.restricted) tags.push(`<span class="tag crit" data-tip="Origin policy: ${esc(pol.via)}">${pol.blocked ? 'blocked' : 'restricted'} origin</span>`);
  return `<div class="row top"><div class="grow"><div style="font:600 17px/1.25 var(--mono);overflow-wrap:anywhere"><span class="faint">${esc(org)}</span>${esc(rest)}</div>
    <div class="row wrap" style="gap:5px;margin-top:8px">${FP && FP.loading ? '<span class="faint">reading model…</span>' : tags.join('')}</div>
    ${m.error ? `<div class="warn" style="margin-top:8px;font-size:12.5px">${esc(m.error)}${m.hint ? ' — ' + esc(m.hint) : ''}</div>` : ''}</div>
    <a class="btn sm ghost" href="${hfUrl(F.model)}" target="_blank" rel="noopener">${I.ext}Hub</a></div>`;
}
const CTX_STEPS = [2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144, 524288, 1048576];
function ctxSteps() { const mx = ((FP && FP.meta && FP.meta.kv) || {}).ctx_max || 131072; const s = CTX_STEPS.filter(x => x <= mx); if (!s.includes(mx)) s.push(mx); if (F.max_len && !s.includes(F.max_len)) { s.push(F.max_len); s.sort((a, b) => a - b); } return s; }
function ctlBox() {
  const steps = ctxSteps(); const ci = Math.max(0, steps.indexOf(F.max_len));
  const n = FP && !FP.loading ? planNumbers(FP, F) : null;
  const vllm = F.backend === 'vllm';
  const gg = ((FP && FP.meta) || {}).gguf || [];
  return `<div class="fgrid" style="grid-template-columns:1fr 1fr">
    <label class="f"><span>Max context ${tip('Longest prompt + reply one request can use. Longer context costs KV-cache memory per sequence.')}</span>
      <div class="row"><input type="range" min="0" max="${steps.length - 1}" step="1" value="${ci}" data-f="ctx_i" data-steps="${steps.join(',')}"><span class="num nowrap" style="min-width:64px;text-align:right">${fmtK(F.max_len)}</span></div></label>
    <label class="f"><span>Parallel sequences ${tip('How many full-length conversations should fit in the KV cache at once. Short chats share the cache far better than this worst case.')}</span>
      <div class="row"><input type="range" min="1" max="32" step="1" value="${F.seqs}" data-f="seqs"><span class="num" style="min-width:28px;text-align:right">${F.seqs}</span></div></label>
    ${vllm ? `<label class="f"><span>KV cache dtype ${tip('fp8 halves KV memory versus bf16 with negligible quality loss on Blackwell.')}</span>
      <div class="seg">${['fp8', 'auto'].map(v => `<button type="button" class="${F.kv_dtype === v ? 'on' : ''}" data-f="kv_dtype" data-v="${v}">${v === 'auto' ? 'model dtype' : v}</button>`).join('')}</div></label>
    <label class="f"><span>Memory share ${tip('--gpu-memory-utilization: the slice of unified memory this vLLM engine reserves for weights, activations and KV cache. Auto sizes it from the plan.')}</span>
      <div class="row"><div class="seg">${['auto', 'manual'].map(v => `<button type="button" class="${F.share === v ? 'on' : ''}" data-f="share" data-v="${v}">${v}</button>`).join('')}</div>
      ${F.share === 'manual' ? `<input type="range" min="0.03" max="0.95" step="0.01" value="${F.util}" data-f="util" style="flex:1"><span class="num" style="min-width:36px;text-align:right">${Math.round(F.util * 100)}%</span>` : `<span class="num dim">${n && n.utilRec ? Math.round(n.utilRec * 100) + '%' : '–'}</span>`}</div></label>`
    : `<label class="f"><span>GGUF quant</span><select data-f="gguf">${gg.length ? gg.map(g => `<option value="${esc(g.quant)}" ${g.quant === F.gguf ? 'selected' : ''}>${esc(g.quant)} · ${fmtB(g.bytes)}</option>`).join('') : '<option value="">default</option>'}</select></label>`}
  </div>`;
}
function fitVerdict() {
  if (!F || !F.model) return '<div class="faint">Pick a model to see how it fits.</div>';
  if (!FP || FP.loading) return '<div class="faint">Planning…</div>';
  const n = planNumbers(FP, F); const M = hostMem(F.host);
  const own = (S.snap.engines[F.name] || {}); const ownBack = FMODE === 'edit' && ['ready', 'booting'].includes(own.state) ? own.reserve || 0 : 0;
  const reserve = F.backend === 'vllm' ? n.reserve : n.need;
  const free = M.avail - M.headroom + ownBack;
  const others = Object.values(S.snap.engines).filter(e => e.host === F.host && e.name !== F.name && ['ready', 'booting'].includes(e.state)).sort((a, b) => (b.reserve || 0) - (a.reserve || 0));
  let cls = 'good', head = '', sub = '', evict = [];
  if (!M.total) { cls = 'meh'; head = 'Host memory unknown'; sub = 'The host is offline or still being read.'; }
  else if (reserve > M.total - M.headroom - M.other) { cls = 'bad'; head = 'Too big for ' + esc(hostOf(F.host)?.label || F.host); sub = `Needs ${fmtB(reserve)} — the box has ${fmtB(M.total)}. Try a smaller quant, shorter context or fewer sequences.`; }
  else if (reserve > free) {
    let gain = 0; for (const o of others) { if (reserve <= free + gain) break; evict.push(o.name); gain += o.reserve || 0; }
    cls = 'meh'; head = `Short ${fmtB(reserve - free)}`; sub = evict.length ? `Launching will offer to put <b>${evict.map(esc).join(', ')}</b> to sleep (${fmtB(gain)}).` : 'Free memory elsewhere first.';
  } else { head = 'Fits'; sub = `Reserves ${fmtB(reserve)} of ${fmtB(Math.max(0, M.avail + ownBack))} free`; }
  if (F.backend === 'vllm' && n.kvKnown && F.share === 'manual' && n.utilNeed && F.util < n.utilNeed) {
    sub += `<div class="warn" style="margin-top:4px">At ${Math.round(F.util * 100)}% the KV cache holds ~${fmtK(Math.max(0, n.maxCtx))} tokens per sequence — below the ${fmtK(F.max_len)} context. vLLM will refuse to start.</div>`;
    if (cls === 'good') cls = 'meh';
  }
  if (!n.kvKnown && F.backend === 'vllm') sub += '<div class="faint" style="margin-top:4px">No config.json — KV cache not estimated.</div>';
  const parts = [['Weights', n.weights, 'var(--tx2)'], ['KV cache', n.kvTotal, engineColor(F.name) || 'var(--acc)'], ['Runtime', n.overhead, 'var(--tx4)']];
  if (F.backend === 'vllm' && reserve > n.need) parts.push(['Spare KV', reserve - n.need, 'var(--line3)']);
  const mx = Math.max(...parts.map(p => p[1]), 1);
  return `<div class="verdict ${cls}"><span class="big">${fmtGiB(reserve)}<small style="font-size:12px;color:var(--tx3)"> GiB</small></span><div class="grow"><b>${head}</b><div class="dim" style="font-size:12.5px">${sub}</div></div></div>
    <div class="sp"></div>${tank(F.host, { ghost: { need: reserve, label: F.name || 'new' }, evict, scale: true, replace: FMODE === 'edit' ? (F._orig || F.name) : '' })}
    <div class="sp"></div><div class="breakdown">${parts.map(([l, v, c]) => `<span class="dim">${l}</span><div class="bar"><i style="width:${(v / mx * 100).toFixed(1)}%;background:${c}"></i></div><span class="num">${fmtB(v)}</span>`).join('')}
    ${n.tokBytes ? `<span class="faint">Per token</span><span class="faint mono" style="font-size:11px">${fmtB(n.tokBytes)} × ${fmtK(F.max_len)} ctx × ${F.seqs} seq</span><span></span>` : ''}</div>`;
}
function updateFit() { const el = $('#fit'); if (el) morph(el, fitVerdict()); const c = $('#ctlbox'); if (c && FP && !FP.loading) morph(c, ctlBox()); refreshPreview(); }
const refreshPreview = debounce(async () => {
  const el = $('#cmdprev'); if (!el || !F || !F.model) return;
  try {
    const d = await post('/api/preview', { spec: specFromForm() });
    el.innerHTML = `${(d.errors || []).map(e => `<div class="crit" style="margin-bottom:6px">${esc(e)}</div>`).join('')}${(d.dropped || []).length ? `<div class="warn" style="margin-bottom:6px">Ignored in extra args (set by fields): ${esc(d.dropped.join(' '))}</div>` : ''}
      <div class="code"><button class="btn xs cp" data-act="copy" data-text="${esc(d.shell)}">${I.copy}</button><pre style="white-space:pre-wrap">${esc(d.shell)}</pre></div>`;
  } catch (e) { el.innerHTML = `<div class="crit">${esc(e.message)}</div>`; }
}, 350);
function specFromForm() {
  const n = FP && !FP.loading ? planNumbers(FP, F) : null;
  const util = F.backend === 'vllm' ? (F.share === 'auto' && n && n.utilRec ? n.utilRec : F.util) : F.util;
  const env = typeof F.env === 'string' ? Object.fromEntries(F.env.split('\n').map(l => l.trim()).filter(l => l && l.includes('=')).map(l => [l.slice(0, l.indexOf('=')).trim(), l.slice(l.indexOf('=') + 1).trim()])) : (F.env || {});
  const out = { name: F.name, host: F.host, backend: F.backend, model: F.model, util: Number(util), max_len: Number(F.max_len), kv_dtype: F.kv_dtype, quant: F.quant,
    trust_remote_code: !!F.trust_remote_code, max_num_seqs: F.max_num_seqs ? Number(F.max_num_seqs) : null, served_name: F.served_name, extra: F.extra, env,
    autostart: !!F.autostart, idle_sleep_min: Number(F.idle_sleep_min || 0), wake: !!F.wake, note: F.note, image: F.image, gguf: F.gguf, blueprint: F.blueprint };
  if (F.port) out.port = Number(F.port);
  return out;
}
function envText(env) { return typeof env === 'string' ? env : Object.entries(env || {}).map(([k, v]) => k + '=' + v).join('\n'); }
function formHTML() {
  const hosts = (S.settings.hosts || []);
  const edit = FMODE === 'edit';
  return `<div id="modelhead">${modelHead()}</div>
  <div class="sp"></div>
  <div class="fgrid" style="grid-template-columns:1.2fr 1fr 1fr">
    <label class="f"><span>Engine name</span><input data-f="name" value="${esc(F.name)}" class="mono" spellcheck="false" ${edit ? 'data-tip="Renaming needs the container removed first"' : ''}></label>
    <label class="f"><span>Host</span><select data-f="host">${hosts.map(h => `<option value="${esc(h.id)}" ${h.id === F.host ? 'selected' : ''} ${h.enabled ? '' : 'disabled'}>${esc(h.label)}${h.enabled ? '' : ' (disabled)'}</option>`).join('')}</select></label>
    <label class="f"><span>Backend</span><div class="seg"><button type="button" class="${F.backend === 'vllm' ? 'on' : ''}" data-f="backend" data-v="vllm">vLLM</button><button type="button" class="${F.backend === 'llamacpp' ? 'on' : ''}" data-f="backend" data-v="llamacpp">llama.cpp</button></div></label>
    ${edit ? `<label class="f wall"><span>Model</span><input data-f="model" value="${esc(F.model)}" class="mono" spellcheck="false"></label>` : ''}
  </div>
  <div class="sp2"></div>
  <div class="sh"><h2>Fit</h2>${tip('vllm-lab reads the model config from the Hub (or the local cache) and estimates weights + KV cache, then sizes the memory share so it fits beside what is already running.')}<div class="rule"></div></div>
  <div id="ctlbox">${ctlBox()}</div>
  <div class="sp"></div><div id="fit">${fitVerdict()}</div>
  <div class="sp2"></div>
  <details ${edit ? 'open' : ''}><summary class="cap" style="cursor:pointer;padding:6px 0">Advanced</summary><div class="sp"></div>
  <div class="fgrid">
    <label class="f"><span>Image ${tip('Leave empty for the default from Settings.')}</span><input data-f="image" value="${esc(F.image)}" class="mono" placeholder="${esc(F.backend === 'llamacpp' ? S.settings.llamacpp_image : S.settings.vllm_image)}"></label>
    <label class="f"><span>Host port ${tip('Empty picks a free port from the pool.')}</span><input data-f="port" value="${esc(F.port || '')}" class="mono" placeholder="auto"></label>
    <label class="f"><span>Served name ${tip('The model id clients send. Defaults to the Hugging Face id.')}</span><input data-f="served_name" value="${esc(F.served_name)}" class="mono" placeholder="${esc(F.model)}"></label>
    <label class="f"><span>Max sequences ${tip('--max-num-seqs: cap concurrent requests. Empty = vLLM default.')}</span><input data-f="max_num_seqs" value="${esc(F.max_num_seqs || '')}" class="mono" placeholder="default"></label>
    ${F.backend === 'vllm' ? `<label class="f"><span>Quantization ${tip('Leave empty: vLLM reads it from the checkpoint (NVFP4, FP8, AWQ…).')}</span><input data-f="quant" value="${esc(F.quant)}" class="mono" placeholder="auto"></label>` : ''}
    <label class="f"><span>Idle sleep (min) ${tip('Stop after this many minutes with no requests, freeing memory. The gateway wakes it again. 0 = never.')}</span><input data-f="idle_sleep_min" value="${esc(F.idle_sleep_min || 0)}" class="mono"></label>
    <label class="f wall"><span>Extra arguments ${tip('Passed straight to vllm serve / llama-server. Flags that have a field above are ignored here.')}</span><textarea data-f="extra" rows="3" spellcheck="false">${esc(F.extra)}</textarea></label>
    <label class="f w2"><span>Environment</span><textarea data-f="env" rows="2" spellcheck="false" placeholder="KEY=value">${esc(envText(F.env))}</textarea></label>
    <label class="f w2"><span>Note</span><input data-f="note" value="${esc(F.note)}"></label>
    <div class="row wrap wall" style="gap:22px">
      ${F.backend === 'vllm' ? `<label class="chk"><input type="checkbox" data-f="trust_remote_code" ${F.trust_remote_code ? 'checked' : ''}>Trust remote code</label>` : ''}
      <label class="chk"><input type="checkbox" data-f="autostart" ${F.autostart ? 'checked' : ''}>Start with the host ${tip('Docker restart policy unless-stopped: comes back after a reboot.')}</label>
      <label class="chk"><input type="checkbox" data-f="wake" ${F.wake ? 'checked' : ''}>Wake on gateway request</label>
    </div>
  </div>
  <div class="sp"></div><div class="cap" style="margin-bottom:8px">Command</div><div id="cmdprev"><div class="faint">…</div></div>
  </details>
  <div class="sp2"></div>
  <div class="row">${edit
    ? `<button class="btn pri" data-act="form-save" data-recreate="1">${I.restart}Save &amp; apply</button><button class="btn" data-act="form-save">Save</button><button class="btn ghost" data-act="form-bp">Save as blueprint</button><span class="grow"></span><button class="btn danger" data-act="engine-delete" data-name="${esc(F._orig || F.name)}">${I.trash}Delete engine</button>`
    : `<button class="btn pri" data-act="form-launch" ${((FP && FP.meta && FP.meta.policy) || {}).blocked ? 'disabled' : ''}>${I.start}Launch</button><button class="btn" data-act="form-create">Save without starting</button><button class="btn ghost" data-act="form-bp">Save as blueprint</button>`}</div>`;
}
function onFormInput(el) {
  const k = el.dataset.f; if (!F || !k) return;
  let v = el.type === 'checkbox' ? el.checked : (el.dataset.v !== undefined ? el.dataset.v : el.value);
  if (k === 'ctx_i') { const steps = el.dataset.steps.split(',').map(Number); F.max_len = steps[+v] || F.max_len; F._ctxTouched = true; morph($('#ctlbox'), ctlBox()); updateFit(); return; }
  if (k === 'seqs') { F.seqs = +v; morph($('#ctlbox'), ctlBox()); updateFit(); return; }
  if (k === 'util') { F.util = +v; morph($('#ctlbox'), ctlBox()); updateFit(); return; }
  if (k === 'share') { if (v === 'manual' && FP && !FP.loading) { const n = planNumbers(FP, F); if (n.utilRec && F.share === 'auto') F.util = n.utilRec; } F.share = v; morph($('#ctlbox'), ctlBox()); updateFit(); return; }
  if (k === 'env') { F.env = v; refreshPreview(); return; }
  F[k] = v;
  if (k === 'backend') { const f = $('#formpane'); if (f) { f.innerHTML = formHTML(); } updateFit(); return; }
  if (k === 'host' || k === 'model' || k === 'extra') { if (k === 'model') { FP = null; } loadPlan(); }
  if (['kv_dtype', 'gguf', 'name'].includes(k)) { if (k !== 'name') morph($('#ctlbox'), ctlBox()); updateFit(); return; }
  refreshPreview();
}

/* ── launch view ────────────────────────────────────────── */
let LQ = { tab: store.get('lqTab', 'hub'), q: '', kind: '', rows: null, err: '', loading: false, disk: null };
views.launch = {
  title: () => 'Launch',
  render(r) {
    return `<div class="cols launch">
      <div>
        <div class="tabs">${[['hub', 'Hugging Face'], ['bp', 'Blueprints'], ['disk', 'On disk']].map(([k, l]) => `<button class="${LQ.tab === k ? 'on' : ''}" data-act="lq-tab" data-v="${k}">${l}</button>`).join('')}</div>
        <div id="picker">${pickerHTML()}</div>
      </div>
      <div class="panel" id="formpane">${F && F.model ? formHTML() : `<div class="empty" style="margin-top:40px"><b>Pick a model</b>Search the Hub, start from a blueprint, or reuse weights already on disk.</div>`}</div>
    </div>`;
  },
  mount(r) {
    if (!F || FMODE !== 'launch' || r.q.model || r.q.bp) formFrom({}, 'launch');
    if (r.q.bp) pickBlueprint(r.q.bp);
    else if (r.q.model) pickModel(r.q.model);
    const inp = $('#hubq'); if (inp && LQ.tab === 'hub') { inp.focus(); if (!LQ.rows) runSearch(); }
    if (LQ.tab === 'disk') loadDisk();
    if (F && F.model) { loadPlan(); refreshPreview(); }
  },
  tick() { if (F && F.model && FMODE === 'launch') updateFitOnly(); },
};
function updateFitOnly() { const el = $('#fit'); if (el) morph(el, fitVerdict()); }
function pickerHTML() {
  if (LQ.tab === 'bp') {
    const bps = Object.values(S.bps).sort((a, b) => (a.builtin === b.builtin ? 0 : a.builtin ? 1 : -1) || String(a.title).localeCompare(b.title));
    return `<div class="bp">${bps.map(b => { const pol = b.spec && b.spec.model ? '' : ''; return `<div class="bpc${F && F.blueprint === b.id ? ' on' : ''}" data-act="pick-bp" data-v="${esc(b.id)}" data-key="bp-${esc(b.id)}">
      <div class="row"><b class="grow">${esc(b.title)}</b>${b.builtin ? '' : `<button class="btn xs ghost icon" data-act="bp-del" data-v="${esc(b.id)}" data-tip="Delete blueprint">${I.x}</button>`}</div>
      <span class="mk">${esc(b.maker || '')}${b.spec && b.spec.backend === 'llamacpp' ? ' · llama.cpp' : ''}</span>
      <span class="mono faint ellip" style="font-size:11px">${esc((b.spec || {}).model || '')}</span>
      <div class="row wrap" style="gap:4px">${(b.tags || []).map(t => `<span class="tag">${esc(t)}</span>`).join('')}</div></div>`; }).join('')}</div>
      <div class="sp"></div><div class="row"><label class="btn sm">${I.down}Import blueprints<input type="file" accept=".json" data-act-change="bp-import" class="hide"></label><button class="btn sm ghost" data-act="bp-export">Export mine</button></div>`;
  }
  if (LQ.tab === 'disk') {
    const d = LQ.disk;
    if (!d) return '<div class="faint">Reading cache…</div>';
    if (d.error) return `<div class="crit">${esc(d.error)}</div>`;
    if (!d.weights.length) return `<div class="empty"><b>No weights cached on ${esc(hostOf(d.host)?.label || d.host)}</b>Downloads land in ${esc(d.cache)}/hub</div>`;
    return `<div class="hits">${d.weights.map(w => `<div class="hit${F && F.model === w.model ? ' on' : ''}${w.policy.blocked ? ' blocked' : ''}" data-act="pick-model" data-v="${esc(w.model)}" data-key="w-${esc(w.model)}">
      <span class="id"><span class="org">${esc(splitId(w.model)[0])}</span>${esc(splitId(w.model)[1])}</span><span class="num">${fmtB(w.bytes)}</span>
      <div class="meta">${w.used_by.length ? `<span class="tag">${esc(w.used_by.join(', '))}</span>` : '<span class="tag">unused</span>'}${w.incomplete ? '<span class="tag warn">incomplete</span>' : ''}${w.policy.restricted ? '<span class="tag crit">restricted</span>' : ''}</div></div>`).join('')}</div>`;
  }
  return `<div class="row" style="margin-bottom:12px"><input id="hubq" type="search" placeholder="Search Hugging Face — nemotron, gemma, llama, gpt-oss…" value="${esc(LQ.q)}" data-input="hubq" autocomplete="off" spellcheck="false">
      <div class="seg">${[['', 'All'], ['gguf', 'GGUF']].map(([k, l]) => `<button class="${LQ.kind === k ? 'on' : ''}" data-act="lq-kind" data-v="${k}">${l}</button>`).join('')}</div></div>
    <div id="hits">${hitsHTML()}</div>`;
}
function hitsHTML() {
  if (LQ.loading && !LQ.rows) return '<div class="faint">Searching…</div>';
  if (LQ.err) return `<div class="empty"><b>Search failed</b>${esc(LQ.err)}</div>`;
  const rows = LQ.rows || [];
  if (!rows.length) return `<div class="empty"><b>Nothing found</b>Try a family name, an org, or paste a full org/model id.</div>`;
  const pasted = /^[\w.-]+\/[\w.-]+$/.test(LQ.q.trim()) && !rows.some(r => r.id === LQ.q.trim());
  return (pasted ? `<div class="hit" data-act="pick-model" data-v="${esc(LQ.q.trim())}"><span class="id">Use <b>${esc(LQ.q.trim())}</b></span><span></span></div>` : '') +
    `<div class="hits">${rows.map(r => {
      const [org, rest] = splitId(r.id); const pol = r.policy || {};
      return `<div class="hit${F && F.model === r.id ? ' on' : ''}${pol.blocked ? ' blocked' : ''}" data-act="pick-model" data-v="${esc(r.id)}" data-key="h-${esc(r.id)}">
        <span class="id"><span class="org">${esc(org)}</span>${esc(rest)}</span><span class="num">${fmtK(r.downloads)} ↓</span>
        <div class="meta">${r.params ? `<span class="tag solid">${fmtParams(r.params)}</span>` : ''}${(r.quant || []).slice(0, 3).map(q => `<span class="tag">${esc(q)}</span>`).join('')}
          ${r.gated ? '<span class="tag warn">gated</span>' : ''}${r.cached ? '<span class="tag ok">on disk</span>' : ''}${pol.restricted ? `<span class="tag crit" data-tip="Origin policy: ${esc(pol.via)}">${pol.blocked ? 'blocked' : 'restricted'}</span>` : ''}
          <span class="faint mono" style="font-size:11px;margin-left:auto">${esc((r.modified || '').slice(0, 10))}</span></div></div>`;
    }).join('')}</div>`;
}
const runSearch = debounce(async () => {
  LQ.loading = true; LQ.err = '';
  const q = LQ.q;
  try {
    const d = await api(`/api/hf/search?q=${encodeURIComponent(q)}&kind=${LQ.kind}&host=${encodeURIComponent(F ? F.host : '')}`);
    if (q !== LQ.q) return;
    LQ.rows = d.rows;
  } catch (e) { LQ.err = e.message + (e.data && e.data.hint ? ' — ' + e.data.hint : ''); LQ.rows = null; }
  LQ.loading = false;
  const h = $('#hits'); if (h) morph(h, hitsHTML());
}, 280);
async function loadDisk() {
  const hid = F ? F.host : firstOnlineHost();
  try { LQ.disk = await api('/api/library?host=' + encodeURIComponent(hid)); } catch (e) { LQ.disk = { error: e.message }; }
  const p = $('#picker'); if (p && LQ.tab === 'disk') morph(p, pickerHTML());
}
function pickModel(id) {
  if (!F || FMODE !== 'launch') formFrom({}, 'launch');
  const keepHost = F.host;
  formFrom({ ...F, model: id, name: '', blueprint: '', host: keepHost }, 'launch');
  F.name = slugify(id);
  if (/gguf/i.test(id)) F.backend = 'llamacpp';
  $('#formpane').innerHTML = formHTML();
  const p = $('#picker'); if (p) morph(p, pickerHTML());
  loadPlan();
}
function pickBlueprint(id) {
  const b = S.bps[id]; if (!b) return toast('Unknown blueprint ' + id, true);
  const keepHost = F ? F.host : firstOnlineHost();
  formFrom({ ...(b.spec || {}), host: keepHost, blueprint: id }, 'launch');
  F._fromBp = true; F._ctxTouched = true; F.name = slugify(F.model);
  F.share = 'auto';
  $('#formpane').innerHTML = formHTML();
  const p = $('#picker'); if (p) morph(p, pickerHTML());
  loadPlan();
}
async function formSubmit(mode, btn) {
  const spec = specFromForm();
  if (!spec.model) return toast('Pick a model first', true);
  btn && btn.classList.add('busy');
  try {
    if (mode === 'launch') {
      const n = FP && !FP.loading ? planNumbers(FP, F) : null;
      let on_conflict = 'ask', evict = null;
      if (n && n.total) {
        const M = hostMem(F.host); const need = F.backend === 'vllm' ? n.reserve : n.need;
        if (need > M.avail - M.headroom) {
          const fit = await post('/api/fit', { spec });
          if (fit.known && !fit.fits_after_flush) {
            const choice = await conflictDialog(spec.name || 'new engine', fit, spec);
            if (!choice) return;
            on_conflict = choice.mode; evict = choice.evict || null;
            if (choice.mode === 'shrink') { spec.util = fit.suggest.util; on_conflict = 'ask'; }
          }
        }
      }
      const d = await post('/api/engines', { spec, create: true, launch: true, on_conflict, evict });
      if (d.job) S.mine.add(d.job);
      toast(`Launching ${d.spec.name}`);
      F = null;
      go('#/engines/' + encodeURIComponent(d.spec.name));
    } else if (mode === 'create') {
      const d = await post('/api/engines', { spec, create: true });
      toast(`Saved ${d.spec.name}`); F = null; go('#/engines/' + encodeURIComponent(d.spec.name));
    } else if (mode === 'save' || mode === 'apply') {
      const orig = F._orig || F.name;
      const d = await post('/api/engines', { spec, old_name: orig });
      toast(`Saved ${d.spec.name}`);
      if (mode === 'apply') { const r = await post(`/api/engines/${encodeURIComponent(d.spec.name)}/recreate`, {}); if (r.job) S.mine.add(r.job); }
      S.specs[d.spec.name] = d.spec;
      formFrom(d.spec, 'edit'); F._orig = d.spec.name;
      if (orig !== d.spec.name) go('#/engines/' + encodeURIComponent(d.spec.name) + '/config');
    } else if (mode === 'bp') {
      const title = spec.model.split('/').pop();
      const d = await post('/api/blueprints', { id: slugify(spec.model), title, spec });
      S.bps[d.id] = d; toast('Saved blueprint ' + d.title);
    }
  } catch (e) { showError(e); } finally { btn && btn.classList.remove('busy'); }
}
function conflictDialog(name, fit, spec) {
  const others = fit.others || [];
  const ev = (fit.suggest || {}).evict || [];
  const shrink = (fit.suggest || {}).util;
  const opts = [];
  if (ev.length) opts.push({ mode: 'evict', evict: ev, t: `Put ${ev.join(', ')} to sleep`, d: `Frees ${fmtB(others.filter(o => ev.includes(o.name)).reduce((s, o) => s + o.reserve, 0))}. They wake again on the next gateway request.` });
  if (shrink && (spec || S.specs[name] || {}).backend !== 'llamacpp') opts.push({ mode: 'shrink', t: `Shrink to ${Math.round(shrink * 100)}% share`, d: `Fits in what is free now (${fmtB(shrink * fit.total)}). Less room for KV cache — shorter context or fewer parallel requests.` });
  if (others.length > ev.length) opts.push({ mode: 'solo', t: 'Run it alone', d: `Put every other engine on this host to sleep (${others.map(o => o.name).join(', ')}).` });
  opts.push({ mode: 'force', t: 'Start anyway', d: 'Let vLLM try. It will likely refuse to start or be killed for memory.' });
  window._conf = opts; window._confSel = 0;
  const hid = (spec || S.specs[name] || {}).host;
  const html = () => `<header><div class="grow"><h3>${esc(name)} needs ${fmtB(fit.need)}</h3><div class="dim" style="margin-top:4px">${fmtB(Math.max(0, fit.available - fit.headroom))} is free on ${esc(hostOf(hid)?.label || hid)}.</div></div></header>
    <div class="body">${tank(hid, { ghost: { need: fit.need, label: name }, evict: opts[window._confSel].mode === 'evict' ? opts[window._confSel].evict : opts[window._confSel].mode === 'solo' ? others.map(o => o.name) : [] })}<div class="sp"></div>
    ${opts.map((o, i) => `<div class="choice${i === window._confSel ? ' on' : ''}" data-act="conf-pick" data-i="${i}"><input type="radio" ${i === window._confSel ? 'checked' : ''}><div><b>${esc(o.t)}</b><div class="d">${esc(o.d)}</div></div></div>`).join('')}</div>
    <footer><button class="btn ghost" data-act="modal-close">Cancel</button><button class="btn pri" data-act="conf-go" autofocus>Continue</button></footer>`;
  window._confRender = () => { const d = $('#modal .dlg'); if (d) d.innerHTML = html(); };
  return modal(html(), { width: 620 });
}
async function startEngine(name, body = {}) {
  const spec = S.specs[name];
  if (!body.on_conflict && spec && spec.backend === 'vllm') {
    try {
      const fit = await post(`/api/engines/${encodeURIComponent(name)}/fit`, {});
      if (fit.known && !fit.fits_after_flush) {
        const c = await conflictDialog(name, fit, spec);
        if (!c) return;
        if (c.mode === 'shrink') { body.util = fit.suggest.util; } else { body.on_conflict = c.mode; if (c.evict) body.evict = c.evict; }
      }
    } catch (e) { /* host may be busy; let the job report */ }
  }
  await startJob(`/api/engines/${encodeURIComponent(name)}/start`, body);
}

/* ── engine config tab ──────────────────────────────────── */
function mountConfig(name) {
  const spec = S.specs[name]; if (!spec) return;
  formFrom(spec, 'edit'); F._orig = name; F._ctxTouched = true;
  const pane = $('#cfgpane');
  pane.innerHTML = `<div class="cols"><div id="formpane">${formHTML()}</div><aside class="panel"><div class="sh"><h2>Blueprint</h2><div class="rule"></div></div>
    <div class="dim" style="font-size:12.5px">${spec.blueprint ? 'Started from <b>' + esc((S.bps[spec.blueprint] || {}).title || spec.blueprint) + '</b>' : 'Custom engine'}</div>
    <div class="sp2"></div><div class="sh"><h2>Container</h2><div class="rule"></div></div>
    <div class="row wrap"><button class="btn" data-act="remove" data-name="${esc(name)}" data-tip="Deletes the container; config and weights stay">${I.trash}Remove container</button></div></aside></div>`;
  loadPlan(); refreshPreview();
}

/* ── playground ─────────────────────────────────────────── */
const PG = { cols: [], catalog: [], sys: store.get('pgSys', ''), temp: store.get('pgTemp', 0.7), max: store.get('pgMax', 1024), compare: false, busy: false };
function pgCol(engine) { return { engine, msgs: [], ctrl: null }; }
views.play = {
  title: () => 'Playground',
  render() {
    const n = PG.cols.length || 1;
    const opts = sel => PG.catalog.map(m => `<option value="${esc(m.engine.startsWith('ollama@') ? m.id : m.engine)}" ${sel === (m.engine.startsWith('ollama@') ? m.id : m.engine) ? 'selected' : ''}>${esc(m.engine.startsWith('ollama@') ? m.id + ' · ollama' : m.engine)}${m.state !== 'ready' ? ' · ' + (m.state === 'booting' ? 'booting' : 'asleep — wakes') : ''}</option>`).join('');
    return `<div class="row wrap" style="margin-bottom:12px;gap:14px">
        <label class="chk"><input type="checkbox" data-act-change="pg-compare" ${PG.compare ? 'checked' : ''}>Compare side by side</label>
        <label class="row" style="gap:8px"><span class="cap">Temp</span><input type="number" step="0.1" min="0" max="2" value="${PG.temp}" data-pg="temp" style="width:70px"></label>
        <label class="row" style="gap:8px"><span class="cap">Max tokens</span><input type="number" step="64" min="16" max="32768" value="${PG.max}" data-pg="max" style="width:90px"></label>
        <details class="grow" style="min-width:260px"><summary class="cap" style="cursor:pointer">System prompt${PG.sys ? ' ·' : ''}</summary><textarea data-pg="sys" rows="2" style="margin-top:8px" placeholder="You are…">${esc(PG.sys)}</textarea></details>
        <button class="btn ghost" data-act="pg-clear">${I.trash}Clear</button></div>
      <div class="chatwrap" style="--n:${n}">${PG.cols.map((c, i) => `<div class="chatcol" data-key="col-${i}">
        <header><span class="led ${esc((S.snap.engines[c.engine] || {}).state || 'ready')}"></span><select data-pg-engine="${i}" style="height:27px">${opts(c.engine)}</select>
          <span class="faint mono ellip" style="font-size:11px" id="pgstat-${i}"></span></header>
        <div class="msgs" id="msgs-${i}" data-keep>${c.msgs.map(msgHTML).join('') || `<div class="faint" style="margin:auto;text-align:center">${PG.catalog.length ? 'Say something.' : 'No engines are running or wakeable.<br><a href="#/launch">Launch one</a>'}</div>`}</div></div>`).join('')}</div>
      <div class="composer"><textarea id="pgin" rows="2" placeholder="Message — Enter to send, Shift+Enter for a new line" data-keep></textarea>
        <button class="btn pri" data-act="pg-send" id="pgsend" style="height:44px">${I.send}Send</button><button class="btn hide" data-act="pg-stop" id="pgstop" style="height:44px">${I.stop}Stop</button></div>`;
  },
  async mount(r) {
    try { PG.catalog = (await api('/api/catalog')).models; } catch { PG.catalog = []; }
    const want = r.q.e;
    const first = want || (PG.cols[0] && PG.cols[0].engine) || (PG.catalog.find(m => m.state === 'ready') || PG.catalog[0] || {}).engine || '';
    if (!PG.cols.length || (want && PG.cols[0].engine !== want)) PG.cols = [pgCol(first)];
    if (PG.compare && PG.cols.length < 2) PG.cols.push(pgCol((PG.catalog.find(m => m.engine !== first && m.state === 'ready') || PG.catalog.find(m => m.engine !== first) || {}).engine || first));
    $('#view .page').innerHTML = views.play.render(r);
    PG.cols.forEach((c, i) => { const m = $('#msgs-' + i); if (m) m.scrollTop = m.scrollHeight; });
    const inp = $('#pgin'); if (inp) inp.focus();
  },
};
function msgHTML(m) {
  if (m.role === 'user') return `<div class="msg user">${esc(m.content)}</div>`;
  const stat = m.stat ? `<div class="stat">${m.stat}</div>` : '';
  return `<div class="msg assistant${m.err ? ' err' : ''}">${m.think ? `<div class="think">${esc(m.think)}</div>` : ''}${esc(m.content)}${m.live ? '<span class="caret"></span>' : ''}${stat}</div>`;
}
async function pgSend() {
  const inp = $('#pgin'); const text = inp.value.trim(); if (!text || PG.busy) return;
  inp.value = ''; PG.busy = true; $('#pgsend').classList.add('hide'); $('#pgstop').classList.remove('hide');
  await Promise.all(PG.cols.map((c, i) => pgRun(c, i, text)));
  PG.busy = false; const s = $('#pgsend'); if (s) { s.classList.remove('hide'); $('#pgstop').classList.add('hide'); }
}
async function pgRun(c, i, text) {
  if (!c.engine) return;
  c.msgs.push({ role: 'user', content: text });
  const out = { role: 'assistant', content: '', think: '', live: true, stat: '' };
  c.msgs.push(out);
  const box = $('#msgs-' + i);
  const paint = () => { if (!box) return; box.innerHTML = c.msgs.map(msgHTML).join(''); box.scrollTop = box.scrollHeight; };
  paint();
  const messages = [...(PG.sys ? [{ role: 'system', content: PG.sys }] : []), ...c.msgs.filter(m => m !== out && !m.err).map(m => ({ role: m.role, content: m.content }))];
  c.ctrl = new AbortController();
  const t0 = performance.now(); let ttft = null, n = 0, usage = null;
  const st = $('#pgstat-' + i);
  const eng = S.snap.engines[c.engine];
  if (eng && eng.state !== 'ready' && st) st.textContent = 'waking ' + c.engine + '…';
  try {
    const r = await fetch('/v1/chat/completions', { method: 'POST', signal: c.ctrl.signal, headers: { 'Content-Type': 'application/json', 'X-Lab-Client': '1' },
      body: JSON.stringify({ model: c.engine, messages, stream: true, temperature: Number(PG.temp), max_tokens: Number(PG.max), stream_options: { include_usage: true } }) });
    if (!r.ok) { const t = await r.text(); let m = t; try { const j = JSON.parse(t); m = (j.error && (j.error.message || j.error)) || t; } catch { } throw new Error(m); }
    const rd = r.body.getReader(); const dec = new TextDecoder(); let buf = '';
    let last = 0;
    for (;;) {
      const { value, done } = await rd.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, k).trim(); buf = buf.slice(k + 1);
        if (line.startsWith(': waking')) { if (st) st.textContent = line.slice(2); continue; }
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim(); if (data === '[DONE]') continue;
        let ev; try { ev = JSON.parse(data); } catch { continue; }
        if (ev.error) throw new Error(ev.error.message || ev.error);
        const d = ev.choices && ev.choices[0] && ev.choices[0].delta || {};
        const piece = d.content || ''; const think = d.reasoning_content || d.reasoning || '';
        if ((piece || think) && ttft == null) ttft = (performance.now() - t0) / 1000;
        if (piece || think) n++;
        out.content += piece; out.think += think;
        if (ev.usage) usage = ev.usage;
        const now = performance.now();
        if (now - last > 50) { last = now; paint(); if (st && ttft != null) st.textContent = `${fmtN(n / Math.max(0.05, (now - t0) / 1000 - ttft), 1)} tok/s`; }
      }
    }
    const dt = (performance.now() - t0) / 1000; const toks = (usage && usage.completion_tokens) || n;
    out.stat = `${toks} tokens · ${fmtN(toks / Math.max(0.05, dt - (ttft || 0)), 1)} tok/s · first token ${ttft != null ? fmtN(ttft * 1000) + ' ms' : '–'}`;
    if (st) st.textContent = out.stat;
  } catch (e) {
    if (e.name === 'AbortError') out.stat = 'stopped';
    else { out.err = true; out.content = out.content || e.message; }
  }
  out.live = false; c.ctrl = null; paint();
}

/* ── engine: log pane ───────────────────────────────────── */
let LOG = { es: null, lines: [], name: '', follow: true, q: '', level: 'all', paused: false };
function classify(l) {
  if (/\b(ERROR|Traceback|Error:|Exception|CRITICAL|OutOfMemory|Killed)\b/.test(l)) return 'err';
  if (/\bWARN(ING)?\b/.test(l)) return 'warn';
  if (/%\|| Completed \|/.test(l)) return 'prog';
  if (/Application startup complete|Starting vLLM API server|server is listening|model loaded/.test(l)) return 'ok';
  return '';
}
function mountLogs(name) {
  stopLogs();
  LOG = { ...LOG, es: null, lines: [], name, paused: false };
  const pane = $('#logpane');
  pane.innerHTML = `<div class="logbar"><input type="search" placeholder="Filter" data-log="q" value="${esc(LOG.q)}" style="max-width:280px">
    <div class="seg">${[['all', 'All'], ['err', 'Errors'], ['warn', 'Warnings']].map(([k, l]) => `<button class="${LOG.level === k ? 'on' : ''}" data-act="log-level" data-v="${k}">${l}</button>`).join('')}</div>
    <label class="chk"><input type="checkbox" data-log="follow" ${LOG.follow ? 'checked' : ''}>Follow</label>
    <span class="grow"></span><span class="faint mono" id="logstat" style="font-size:11px"></span>
    <button class="btn sm" data-act="log-pause">Pause</button><button class="btn sm" data-act="log-copy">${I.copy}Copy</button><button class="btn sm" data-act="log-dl">${I.down}Save</button></div>
    <div class="logview" id="logview"></div>`;
  const es = new EventSource(`/api/engines/${encodeURIComponent(name)}/logs/stream?tail=800`);
  LOG.es = es;
  es.onmessage = ev => { if (LOG.paused) return; const batch = JSON.parse(ev.data); LOG.lines.push(...batch); if (LOG.lines.length > 20000) LOG.lines.splice(0, LOG.lines.length - 20000); paintLogs(batch); };
  es.addEventListener('end', () => { const s = $('#logstat'); if (s) s.textContent = 'container stopped — stream ended'; es.close(); });
  es.onerror = () => { const s = $('#logstat'); if (s) s.textContent = 'reconnecting…'; };
  es.onopen = () => { const s = $('#logstat'); if (s) s.textContent = 'live'; };
}
function stopLogs() { if (LOG.es) { LOG.es.close(); LOG.es = null; } }
function logLineHTML(l) {
  const c = classify(l);
  if (LOG.level === 'err' && c !== 'err') return '';
  if (LOG.level === 'warn' && c !== 'warn' && c !== 'err') return '';
  const q = LOG.q.toLowerCase();
  if (q && !l.toLowerCase().includes(q)) return '';
  let h = esc(l);
  const m = h.match(/^(\d{4}-\d\d-\d\dT[\d:.]+Z) /);
  if (m) h = `<span class="ts">${m[1].slice(11, 19)}</span> ` + h.slice(m[0].length);
  if (q) h = h.replace(new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi'), s => `<mark>${s}</mark>`);
  return `<div class="l ${c}">${h}</div>`;
}
function paintLogs(batch) {
  const v = $('#logview'); if (!v) return;
  if (batch) v.insertAdjacentHTML('beforeend', batch.map(logLineHTML).join(''));
  else v.innerHTML = LOG.lines.map(logLineHTML).join('') || '<div class="faint" style="padding:10px 14px">No lines match.</div>';
  while (v.childElementCount > 6000) v.firstElementChild.remove();
  if (LOG.follow) v.scrollTop = v.scrollHeight;
  const s = $('#logstat'); if (s && LOG.es && LOG.es.readyState === 1) s.textContent = `${fmtN(LOG.lines.length)} lines · live`;
}

/* ── engine: bench pane ─────────────────────────────────── */
async function mountBench(name) {
  const pane = $('#benchpane');
  const B = store.get('bench', { concurrency: 4, requests: 16, max_tokens: 256, prompt_tokens: 0 });
  let runs = [];
  try { runs = (await api(`/api/engines/${encodeURIComponent(name)}/bench`)).runs; } catch { }
  const last = runs[runs.length - 1];
  pane.innerHTML = `<div class="cols"><div>
    <div class="fgrid" style="grid-template-columns:repeat(4,1fr)">
      <label class="f"><span>Concurrent ${tip('Requests in flight at once — like that many users typing.')}</span><input type="number" min="1" max="128" data-b="concurrency" value="${B.concurrency}"></label>
      <label class="f"><span>Requests</span><input type="number" min="1" max="1000" data-b="requests" value="${B.requests}"></label>
      <label class="f"><span>Output tokens</span><input type="number" min="8" max="8192" data-b="max_tokens" value="${B.max_tokens}"></label>
      <label class="f"><span>Prompt tokens ${tip('Pad the prompt to roughly this many tokens to measure prefill. 0 = short prompt.')}</span><input type="number" min="0" max="200000" data-b="prompt_tokens" value="${B.prompt_tokens}"></label></div>
    <div class="sp"></div><div class="row"><button class="btn pri" data-act="bench-run" data-name="${esc(name)}">${I.bench}Run benchmark</button><span class="faint" style="font-size:12px">Runs against the live engine. Other traffic skews results.</span></div>
    <div class="sp2"></div>
    ${last ? `<div class="sh"><h2>Latest</h2><div class="rule"></div><span class="faint mono" style="font-size:11px">${new Date(last.t * 1000).toLocaleString('en-US')}</span></div>
    <div class="metrics" style="grid-template-columns:repeat(5,1fr)">
      <div class="metric"><div class="v">${fmtN(last.agg_tps)}<small>tok/s</small></div><div class="l">Aggregate</div></div>
      <div class="metric"><div class="v">${fmtN(last.tps_p50, 1)}<small>tok/s</small></div><div class="l">Per stream p50</div></div>
      <div class="metric"><div class="v">${fmtN((last.ttft_p50 || 0) * 1000)}<small>ms</small></div><div class="l">TTFT p50</div></div>
      <div class="metric"><div class="v">${fmtN((last.ttft_p95 || 0) * 1000)}<small>ms</small></div><div class="l">TTFT p95</div></div>
      <div class="metric"><div class="v">${fmtN(last.errors)}</div><div class="l">Errors</div></div></div>` : '<div class="empty"><b>No runs yet</b>Measure real throughput on this box before you size for users.</div>'}
    </div><aside class="panel"><div class="sh"><h2>History</h2><div class="rule"></div></div>
    ${runs.length ? `<table class="tbl"><thead><tr><th>When</th><th class="r">Conc</th><th class="r">tok/s</th><th class="r">TTFT</th></tr></thead><tbody>${runs.slice().reverse().slice(0, 30).map(r => `<tr><td class="faint mono" style="font-size:11px">${fmtAgo(r.t)} ago</td><td class="r num">${r.concurrency}</td><td class="r num">${fmtN(r.agg_tps)}</td><td class="r num">${fmtN((r.ttft_p50 || 0) * 1000)}ms</td></tr>`).join('')}</tbody></table>` : '<div class="faint">–</div>'}
    </aside></div>`;
}

/* ── engine: inspect pane ───────────────────────────────── */
async function mountInspect(name) {
  const pane = $('#inspane');
  pane.innerHTML = '<div class="faint">Loading…</div>';
  let cmd = null, ins = null;
  try { cmd = await api(`/api/engines/${encodeURIComponent(name)}/command`); } catch (e) { cmd = { shell: e.message }; }
  try { ins = (await api(`/api/engines/${encodeURIComponent(name)}/inspect`)).json; } catch (e) { ins = { error: e.message }; }
  pane.innerHTML = `<div class="sh"><h2>docker run</h2>${tip('Exactly what vllm-lab runs. The HF token comes from an env file (mode 600), never the command line.')}<div class="rule"></div></div>
    <div class="code"><button class="btn xs cp" data-act="copy" data-text="${esc(cmd.shell)}">${I.copy}</button><pre style="white-space:pre-wrap">${esc(cmd.shell)}</pre></div>
    <div class="sp2"></div><div class="sh"><h2>docker inspect</h2><div class="rule"></div><button class="btn xs" data-act="copy" data-text="${esc(JSON.stringify(ins, null, 2))}">${I.copy}Copy</button></div>
    <div class="code" style="max-height:60vh"><pre>${ins ? esc(JSON.stringify(ins, null, 2)) : 'No container.'}</pre></div>`;
}

/* ── host picker used by several pages ───────────────────── */
function hostSeg(cur, act) {
  return `<div class="seg">${hostList().map(h => `<button class="${h.id === cur ? 'on' : ''}" data-act="${act}" data-v="${esc(h.id)}" ${h.enabled ? '' : 'disabled style="opacity:.4"'}>${esc(h.label)}</button>`).join('')}</div>`;
}
function curHost(key) { const h = store.get(key, ''); const hs = hostList(); return (hs.find(x => x.id === h && x.enabled) || hs.find(x => x.enabled && x.st.online) || hs[0] || {}).id || 'titan'; }

/* ── library ─────────────────────────────────────────────── */
const LIB = { data: null, host: '', err: '' };
views.library = {
  title: () => 'Library',
  render() {
    const hid = curHost('libHost'); const hs = S.snap.hosts[hid] || {};
    const d = LIB.data && LIB.host === hid ? LIB.data : null;
    const disk = hs.disk || (d && d.disk) || null;
    const weights = d ? d.weights : [];
    const wsum = weights.reduce((s, w) => s + w.bytes, 0);
    const dls = hs.downloads || []; const sizes = hs.dl_sizes || {};
    const ol = (hs.ollama || {});
    const maxb = Math.max(1, ...weights.map(w => w.bytes));
    return `<div class="row wrap" style="margin-bottom:18px">${hostSeg(hid, 'lib-host')}<span class="grow"></span>
        <input type="text" id="dlid" placeholder="org/model to pre-download" class="mono" style="max-width:340px" data-keep><button class="btn pri" data-act="lib-dl">${I.down}Download</button></div>
      ${disk ? `<div class="sec"><div class="sh"><h2>Disk</h2><div class="rule"></div><span class="faint mono" style="font-size:11px">${esc(disk.path || '')}</span></div>
        <div class="tank sm"><div class="seg-e" style="width:${(wsum / disk.total * 100).toFixed(2)}%;background:var(--tx3)"><span>weights ${fmtB(wsum)}</span></div><div class="seg-os" style="width:${(Math.max(0, disk.used - wsum) / disk.total * 100).toFixed(2)}%"><span>other</span></div><div class="seg-free"><span>${fmtB(disk.free)} free</span></div></div></div>` : ''}
      ${dls.length ? `<div class="sec"><div class="sh"><h2>Downloading</h2><div class="rule"></div></div>${dls.map(x => { const got = sizes['models--' + String(x.model).replace('/', '--')] || 0; const p = x.total ? got / x.total : null; return `<div class="row" style="padding:8px 0;border-bottom:1px solid var(--line)" data-key="dl-${esc(x.model)}"><span class="mono grow">${esc(x.model)}</span><span class="num faint">${fmtB(got)}${x.total ? ' / ' + fmtB(x.total) : ''}</span><div class="kvbar" style="width:160px"><i style="width:${p != null ? (p * 100).toFixed(1) : 30}%;background:var(--acc)"></i></div><button class="btn xs danger" data-act="docker" data-host="${esc(hid)}" data-op="rm" data-target="${esc(x.container)}">Cancel</button></div>`; }).join('')}</div>` : ''}
      <div class="sec"><div class="sh"><h2>Hugging Face weights</h2><div class="rule"></div><span class="faint mono" style="font-size:11px">${d ? weights.length + ' models' : ''}</span><button class="btn xs ghost" data-act="lib-refresh">${I.restart}</button></div>
        ${LIB.err ? `<div class="crit">${esc(LIB.err)}</div>` : !d ? '<div class="faint">Reading cache…</div>' : weights.length ? `<table class="tbl"><thead><tr><th>Model</th><th>Size</th><th>Used by</th><th>Modified</th><th></th></tr></thead><tbody>
        ${weights.map(w => `<tr data-key="w-${esc(w.model)}"><td><span class="mono" style="font-weight:600">${esc(w.model)}</span>${w.incomplete ? ` <span class="tag warn" data-tip="${w.incomplete} partial file(s) — a download was interrupted">incomplete</span>` : ''}${w.policy.restricted ? ' <span class="tag crit">restricted origin</span>' : ''}</td>
          <td class="num">${fmtB(w.bytes)}<div class="sizebar"><i style="width:${(w.bytes / maxb * 100).toFixed(1)}%"></i></div></td>
          <td>${w.used_by.length ? w.used_by.map(n => `<a class="tag ${w.running.includes(n) ? 'ok' : ''}" href="#/engines/${encodeURIComponent(n)}">${esc(n)}</a>`).join(' ') : '<span class="faint">unused</span>'}</td>
          <td class="faint mono" style="font-size:11.5px">${w.modified ? fmtAgo(w.modified) + ' ago' : '–'}</td>
          <td><div class="acts"><a class="btn xs" href="#/launch?model=${encodeURIComponent(w.model)}">${I.start}Launch</a><button class="btn xs danger" data-act="lib-del" data-model="${esc(w.model)}" ${w.running.length ? 'disabled' : ''}>${I.trash}</button></div></td></tr>`).join('')}</tbody></table>`
          : `<div class="empty"><b>Nothing cached yet</b>Weights download the first time an engine starts, or pre-download them above.</div>`}</div>
      ${d && d.gguf && d.gguf.length ? `<div class="sec"><div class="sh"><h2>GGUF (llama.cpp)</h2><div class="rule"></div></div><table class="tbl"><tbody>${d.gguf.map(g => `<tr><td class="mono">${esc(g.file)}</td><td class="num r">${fmtB(g.bytes)}</td></tr>`).join('')}</tbody></table></div>` : ''}
      <div class="sec"><div class="sh"><h2>Ollama</h2>${tip('Ollama is detected on each host at :11434. Its models appear in the gateway and playground too.')}<div class="rule"></div>${ol.ok ? `<span class="faint mono" style="font-size:11px">v${esc(ol.version || '?')}</span>` : ''}</div>
        ${!ol.ok ? '<div class="faint">Not running on this host.</div>' : `<div class="row" style="margin-bottom:10px"><input type="text" id="olid" placeholder="llama3.2:3b" class="mono" style="max-width:260px" data-keep><button class="btn sm" data-act="ol-pull" data-host="${esc(hid)}">${I.down}Pull</button></div>
        <table class="tbl"><thead><tr><th>Model</th><th>Params</th><th>Quant</th><th>Size</th><th>State</th><th></th></tr></thead><tbody>${(ol.models || []).map(m => { const ld = (ol.loaded || []).find(x => x.name === m.name); return `<tr data-key="ol-${esc(m.name)}"><td class="mono">${esc(m.name)}${m.policy && m.policy.restricted ? ` <span class="tag crit" data-tip="Origin policy: ${esc(m.policy.via)}">restricted origin</span>` : ''}</td><td>${esc(m.params || '')}</td><td class="faint">${esc(m.quant || '')}</td><td class="num">${fmtB(m.size)}</td><td>${ld ? `<span class="tag ok">loaded · ${fmtB(ld.vram)}</span>` : '<span class="faint">on disk</span>'}</td>
          <td><div class="acts">${ld ? `<button class="btn xs" data-act="ol" data-op="unload" data-host="${esc(hid)}" data-model="${esc(m.name)}">Unload</button>` : `<button class="btn xs" data-act="ol" data-op="load" data-host="${esc(hid)}" data-model="${esc(m.name)}">Load</button>`}<a class="btn xs" href="#/play?e=${encodeURIComponent(m.name)}">${I.play}</a><button class="btn xs danger" data-act="ol" data-op="delete" data-host="${esc(hid)}" data-model="${esc(m.name)}">${I.trash}</button></div></td></tr>`; }).join('') || '<tr><td colspan="6" class="faint">No models pulled.</td></tr>'}</tbody></table>`}</div>`;
  },
  mount() { loadLib(); },
  live: true,
};
async function loadLib() {
  const hid = curHost('libHost'); LIB.err = '';
  try { LIB.data = await api('/api/library?host=' + encodeURIComponent(hid)); LIB.host = hid; } catch (e) { LIB.err = e.message; LIB.data = null; LIB.host = hid; }
  repaint('library');
}

/* ── hosts ───────────────────────────────────────────────── */
views.hosts = {
  title: () => 'Hosts',
  render() {
    const hs = hostList();
    return `<div class="row" style="margin-bottom:16px"><span class="grow"></span><button class="btn pri" data-act="host-edit" data-host="">${I.launch}Add host</button></div>
      <div class="cols half">${hs.map(h => hostCard(h)).join('')}</div>`;
  },
  live: true,
};
function hostCard(h) {
  const st = h.st || {}; const v = st.versions || {}; const g = (st.gpus || [])[0] || {}; const m = st.mem || {};
  const on = h.enabled && st.online;
  const engines = Object.keys(S.specs).filter(n => S.specs[n].host === h.id);
  return `<div class="hostcard" data-key="hc-${esc(h.id)}">
    <div class="row">${led(!h.enabled ? 'offline' : on ? 'ready' : 'crashed')}<span style="font:700 17px var(--sans)">${esc(h.label)}</span><span class="faint mono">${esc(h.id)}</span>
      ${!h.enabled ? '<span class="tag">disabled</span>' : on ? '' : '<span class="tag crit">unreachable</span>'}<span class="grow"></span>
      <button class="btn sm" data-act="host-test" data-host="${esc(h.id)}">${I.plug}Test</button><button class="btn sm ghost icon" data-act="host-edit" data-host="${esc(h.id)}" data-tip="Edit">${I.edit}</button></div>
    ${!on && h.enabled ? `<div class="crit mono" style="font-size:12px">${esc(st.why || 'connecting…')}</div>` : ''}
    ${on ? tank(h.id, { small: true }) : ''}
    <dl class="kvl">
      <dt>Access</dt><dd>${h.ssh ? `ssh ${esc(h.ssh)}${h.ssh_port ? ':' + esc(h.ssh_port) : ''} · ${esc(st.reach || h.reach)}` : 'local'}</dd>
      <dt>Human address</dt><dd>${esc(h.human_host)} · binds ${esc(h.bind)}</dd>
      <dt>Network</dt><dd>${esc(h.network)}</dd>
      <dt>Weights cache</dt><dd>${esc(h.cache || (st.disk || {}).path || 'default')}</dd>
      ${on ? `<dt>GPU</dt><dd>${esc(g.name || 'none')}${v.driver ? ' · driver ' + esc(v.driver) : ''}${v.cuda ? ' · CUDA ' + esc(v.cuda) : ''}</dd>
      <dt>Memory</dt><dd>${fmtB(m.total)}${m.uma ? ' unified' : ''} · ${fmtB(m.available)} available</dd>
      <dt>System</dt><dd>${esc(v.os || '')}${v.arch ? ' · ' + esc(v.arch) : ''}</dd>
      <dt>Docker</dt><dd>${esc(v.docker || '?')}${v.nvidia_runtime ? ' · nvidia runtime' : ''}</dd>
      <dt>Load</dt><dd>${(st.load || []).map(x => x.toFixed(2)).join(' ')} · ${esc(st.cpus || '?')} CPUs · up ${fmtDur(st.uptime)}</dd>` : ''}
      <dt>Engines</dt><dd>${engines.map(n => `<a href="#/engines/${encodeURIComponent(n)}">${esc(n)}</a>`).join(', ') || '–'}</dd>
    </dl>
    <div class="row wrap">${h.enabled ? `<button class="btn sm" data-act="host-toggle" data-host="${esc(h.id)}" data-v="0">Disable</button>` : `<button class="btn sm pri" data-act="host-toggle" data-host="${esc(h.id)}" data-v="1">Enable</button>`}
      ${on && m.uma ? `<button class="btn sm" data-act="host-flush" data-host="${esc(h.id)}" data-tip="Drop the page cache so CUDA sees the memory as free (needs passwordless sudo for one command)">Flush page cache</button>` : ''}
      <a class="btn sm ghost" href="#/containers?host=${encodeURIComponent(h.id)}">${I.containers}Containers</a><span class="grow"></span>
      ${engines.length ? '' : `<button class="btn sm danger" data-act="host-remove" data-host="${esc(h.id)}">${I.trash}</button>`}</div>
  </div>`;
}
function hostForm(h) {
  const x = h || { id: '', label: '', ssh: '', ssh_port: '', human_host: '', bind: '127.0.0.1', network: 'titan-ai', cache: '', reach: 'auto', ports: '', note: '', enabled: true };
  return modal(`<header><h3>${h ? 'Edit ' + esc(h.label) : 'Add a host'}</h3></header><div class="body"><form id="hostf" class="fgrid" style="grid-template-columns:1fr 1fr" onsubmit="return false">
    <label class="f"><span>Id</span><input name="id" value="${esc(x.id)}" class="mono" placeholder="atlas" autofocus></label>
    <label class="f"><span>Label</span><input name="label" value="${esc(x.label)}" placeholder="Atlas"></label>
    <label class="f"><span>SSH target ${tip('user@host. Leave empty for this machine. Key auth only — run ssh-copy-id first.')}</span><input name="ssh" value="${esc(x.ssh)}" class="mono" placeholder="titan@10.20.0.12"></label>
    <label class="f"><span>SSH port</span><input name="ssh_port" value="${esc(x.ssh_port || '')}" class="mono" placeholder="22"></label>
    <label class="f"><span>Human address ${tip('How people and apps reach engines on this host: an IP or DNS name.')}</span><input name="human_host" value="${esc(x.human_host)}" class="mono" placeholder="10.20.0.12"></label>
    <label class="f"><span>Bind ${tip('Address engine ports publish on. 127.0.0.1 keeps them off the LAN; the manager reaches them through an SSH tunnel.')}</span><input name="bind" value="${esc(x.bind)}" class="mono"></label>
    <label class="f"><span>Docker network</span><input name="network" value="${esc(x.network)}" class="mono"></label>
    <label class="f"><span>Reach ${tip('auto: SSH tunnel when bound to loopback, direct otherwise.')}</span><select name="reach">${['auto', 'ssh', 'direct'].map(v => `<option ${x.reach === v ? 'selected' : ''}>${v}</option>`).join('')}</select></label>
    <label class="f"><span>Weights cache</span><input name="cache" value="${esc(x.cache)}" class="mono" placeholder="~/.cache/huggingface"></label>
    <label class="f"><span>Port pool ${tip('Optional override, e.g. 58100-58119')}</span><input name="ports" value="${esc(x.ports)}" class="mono" placeholder="global pool"></label>
    <label class="f wall"><span>Note</span><input name="note" value="${esc(x.note)}"></label>
    <label class="chk wall"><input type="checkbox" name="enabled" ${x.enabled ? 'checked' : ''}>Enabled</label></form></div>
    <footer><button class="btn ghost" data-act="modal-close">Cancel</button><button class="btn pri" data-act="host-save" data-old="${esc(h ? h.id : '')}">Save</button></footer>`);
}

/* ── containers (raw docker) ────────────────────────────── */
const CT = { data: null, host: '', q: '' };
views.containers = {
  title: () => 'Containers',
  render(r) {
    if (r.q.host) store.set('ctHost', r.q.host);
    const hid = curHost('ctHost'); const d = CT.host === hid ? CT.data : null;
    const q = (CT.q || '').toLowerCase();
    const live = S.snap.hosts[hid] ? null : null;
    const rows = d ? d.containers.filter(c => !q || (c.name + c.image).toLowerCase().includes(q)) : [];
    return `<div class="row wrap" style="margin-bottom:16px">${hostSeg(hid, 'ct-host')}<input type="search" placeholder="Filter" value="${esc(CT.q)}" data-input="ctq" style="max-width:240px" data-keep><span class="grow"></span>
      <button class="btn" data-act="ct-pull" data-host="${esc(hid)}">${I.down}Pull image</button><button class="btn" data-act="docker" data-host="${esc(hid)}" data-op="prune-images" data-target="-">Prune dangling</button><button class="btn ghost icon" data-act="ct-refresh">${I.restart}</button></div>
      ${!d ? '<div class="faint">Loading…</div>' : d.error ? `<div class="empty"><b>Docker is not usable on this host</b>${esc(d.error)}</div>` : `
      <div class="sec"><div class="sh"><h2>Containers</h2><div class="rule"></div><span class="faint mono" style="font-size:11px">${rows.length}</span></div>
      <table class="tbl"><thead><tr><th></th><th>Name</th><th>Image</th><th>Status</th><th>Ports</th><th class="r">CPU</th><th class="r">Memory</th><th></th></tr></thead><tbody>
      ${rows.map(c => `<tr data-key="c-${esc(c.name)}"><td style="width:14px">${led(c.state === 'running' ? 'ready' : c.state === 'restarting' ? 'crashed' : 'stopped')}</td>
        <td><span class="mono" style="font-weight:600">${esc(c.name)}</span>${c.engine ? ` <a class="tag acc" href="#/engines/${encodeURIComponent(c.engine)}">engine</a>` : ''}${c.download ? ' <span class="tag">download</span>' : ''}</td>
        <td class="mono faint ellip" style="font-size:11.5px;max-width:260px" data-tip="${esc(c.image)}">${esc(c.image)}</td><td class="faint" style="font-size:12px">${esc(c.status)}</td>
        <td class="mono" style="font-size:11.5px">${esc(c.ports || '')}</td><td class="r num">${esc(c.cpu || '–')}</td><td class="r num" style="font-size:11.5px">${esc((c.mem || '–').split(' / ')[0])}</td>
        <td><div class="acts"><button class="btn xs" data-act="ct-logs" data-host="${esc(hid)}" data-target="${esc(c.name)}">${I.logs}</button>
          ${c.state === 'running' ? `<button class="btn xs" data-act="docker" data-host="${esc(hid)}" data-op="restart" data-target="${esc(c.name)}" data-tip="Restart">${I.restart}</button><button class="btn xs" data-act="docker" data-host="${esc(hid)}" data-op="stop" data-target="${esc(c.name)}" data-tip="Stop">${I.stop}</button>` : `<button class="btn xs" data-act="docker" data-host="${esc(hid)}" data-op="start" data-target="${esc(c.name)}" data-tip="Start">${I.start}</button>`}
          <button class="btn xs" data-act="ct-inspect" data-host="${esc(hid)}" data-target="${esc(c.name)}" data-tip="Inspect">{ }</button>
          <button class="btn xs danger" data-act="docker" data-host="${esc(hid)}" data-op="rm" data-target="${esc(c.name)}" data-confirm="Remove container ${esc(c.name)}?">${I.trash}</button></div></td></tr>`).join('') || '<tr><td colspan="8" class="faint">No containers.</td></tr>'}</tbody></table></div>
      <div class="cols half"><div class="sec"><div class="sh"><h2>Images</h2><div class="rule"></div></div>
        <table class="tbl"><tbody>${d.images.map(i => `<tr><td class="mono" style="font-size:12px">${esc(i.ref)}${i.used ? ' <span class="tag ok">in use</span>' : ''}${i.dangling ? ' <span class="tag">dangling</span>' : ''}</td><td class="r num">${esc(i.size)}</td><td class="faint" style="font-size:12px">${esc(i.created || '')}</td><td><div class="acts">${i.used ? '' : `<button class="btn xs danger" data-act="docker" data-host="${esc(hid)}" data-op="rmi" data-target="${esc(i.ref)}" data-confirm="Remove image ${esc(i.ref)}?">${I.trash}</button>`}</div></td></tr>`).join('')}</tbody></table></div>
      <div class="sec"><div class="sh"><h2>Networks</h2><div class="rule"></div></div><table class="tbl"><tbody>${d.networks.map(n => `<tr><td class="mono">${esc(n.name)}</td><td class="faint">${esc(n.driver)}</td><td class="faint">${esc(n.scope)}</td></tr>`).join('')}</tbody></table></div></div>`}`;
  },
  mount() { loadCT(); },
  tick() { if (Date.now() - (CT.t || 0) > 5000) loadCT(); },
};
async function loadCT() {
  const hid = curHost('ctHost'); CT.t = Date.now();
  try { CT.data = await api('/api/hosts/' + encodeURIComponent(hid) + '/containers'); } catch (e) { CT.data = { error: e.message, containers: [], images: [], networks: [] }; }
  CT.host = hid;
  if (S.route.view === 'containers') morph($('#view .page'), views.containers.render(S.route));
}

/* ── open webui ─────────────────────────────────────────── */
const WU = { data: null, loading: false };
views.webui = {
  title: () => 'Open WebUI',
  render() {
    const w = S.settings.webui || {}; const d = WU.data;
    const loginMode = store.get('wuLogin', w.api_key_set ? 'key' : 'password');
    return `<div class="cols"><div>
      <div class="row" style="margin-bottom:14px">${led(d ? (d.up ? 'ready' : 'crashed') : 'offline')}<a href="${esc(w.url)}" target="_blank" rel="noopener" style="font:700 17px var(--sans);text-decoration:none">${esc(w.url || 'not set')} ${I.ext.replace('<svg', '<svg width="11" height="11"')}</a><span class="grow"></span>
        <button class="btn" data-act="wu-check">${I.restart}Check</button><button class="btn pri" data-act="wu-sync">${I.link}Sync now</button></div>
      ${!d ? `<div class="faint">${WU.loading ? 'Checking Open WebUI and every saved connection…' : ''}</div>` : `
      <div class="sec">${d.checks.map(c => `<div class="check"><span class="ic ${c.ok ? 'ok' : 'crit'}">${c.ok ? '✓' : '✗'}</span><div><div class="t">${esc(c.title)}</div><div class="dd">${esc(c.detail || '')}</div></div><div>${c.fix === 'webui-sync' ? '<button class="btn xs pri" data-act="wu-sync">Sync</button>' : ''}</div></div>`).join('')}</div>
      <div class="sec"><div class="sh"><h2>Connections</h2>${tip('vllm-lab adds a connection when an engine becomes ready and removes it when the engine stops, so the model picker never waits on dead URLs. Connections you added yourself are left alone.')}<div class="rule"></div></div>
        <table class="tbl"><thead><tr><th></th><th>URL</th><th>Owner</th><th>From the WebUI container</th></tr></thead><tbody>
        ${d.connections.map(c => `<tr><td style="width:14px">${led(c.ok == null ? 'offline' : c.ok ? 'ready' : 'crashed')}</td><td class="mono" style="font-size:12px">${esc(c.url)}</td><td>${c.owned ? `<span class="tag acc">vllm-lab${c.engine ? ' · ' + esc(c.engine) : ''}</span>` : '<span class="tag">yours</span>'}</td><td class="faint mono" style="font-size:11.5px">${esc(c.detail || '')}</td></tr>`).join('') || '<tr><td colspan="4" class="faint">None</td></tr>'}
        ${(d.missing || []).map(u => `<tr><td>${led('booting')}</td><td class="mono" style="font-size:12px">${esc(u)}</td><td><span class="tag warn">missing</span></td><td class="faint">Sync adds it</td></tr>`).join('')}</tbody></table></div>
      <div class="sec"><div class="sh"><h2>In the model picker</h2><div class="rule"></div></div><div class="chips">${(d.models || []).map(m => `<span class="chip" style="padding-right:9px">${esc(m)}</span>`).join('') || '<span class="faint">Nothing yet.</span>'}</div></div>`}
    </div>
    <aside class="panel">
      <div class="sh"><h2>Login</h2><div class="rule"></div></div>
      <div class="seg" style="margin-bottom:12px"><button class="${loginMode === 'key' ? 'on' : ''}" data-act="wu-login-mode" data-v="key">API key</button><button class="${loginMode === 'password' ? 'on' : ''}" data-act="wu-login-mode" data-v="password">Email + password</button></div>
      <form id="wuf" class="fgrid" style="grid-template-columns:1fr" onsubmit="return false">
        <label class="f"><span>URL</span><input name="url" value="${esc(w.url)}" class="mono"></label>
        ${loginMode === 'key' ? `<label class="f"><span>API key ${tip('Open WebUI → Settings → Account → API keys. Needs an admin account.')}</span><input name="api_key" type="password" placeholder="${w.api_key_set ? '•••••••• saved' : 'sk-…'}"></label>`
        : `<label class="f"><span>Admin email</span><input name="email" value="${esc(w.email || '')}"></label><label class="f"><span>Password</span><input name="password" type="password" placeholder="${w.password_set ? '•••••••• saved' : ''}"></label>`}
        <label class="f"><span>Container ${tip('Used to test connections from inside Open WebUI, exactly as it sees them.')}</span><input name="container" value="${esc(w.container || 'titan-webui')}" class="mono"></label>
        <label class="f"><span>Runs on host</span><select name="host">${(S.settings.hosts || []).map(h => `<option value="${esc(h.id)}" ${h.id === w.host ? 'selected' : ''}>${esc(h.label)}</option>`).join('')}</select></label>
        <label class="f"><span>Mode ${tip('direct: one connection per running engine (Docker DNS). gateway: a single connection to the vllm-lab gateway, so sleeping engines show up and wake when chosen. Gateway mode needs the gateway listening on the Docker network (Settings → Gateway → listen: docker).')}</span>
          <select name="mode"><option value="direct" ${w.mode !== 'gateway' ? 'selected' : ''}>direct — one connection per engine</option><option value="gateway" ${w.mode === 'gateway' ? 'selected' : ''}>gateway — one connection, wake on demand</option></select></label>
        <label class="chk"><input type="checkbox" name="auto_sync" ${w.auto_sync !== false ? 'checked' : ''}>Sync automatically</label>
        <label class="chk"><input type="checkbox" name="prune_stopped" ${w.prune_stopped !== false ? 'checked' : ''}>Remove stopped engines</label>
        <div class="row"><button class="btn pri" data-act="wu-save">Save</button>${w.api_key_set || w.password_set ? `<button class="btn ghost" data-act="wu-forget">Forget login</button>` : ''}</div></form>
    </aside></div>`;
  },
  mount() { if (!WU.data && !WU.loading) wuCheck(); },
};
async function wuCheck() {
  WU.loading = true; repaint('webui');
  try { WU.data = await api('/api/webui'); } catch (e) { WU.data = { checks: [{ ok: false, title: 'Check failed', detail: e.message }], connections: [], models: [] }; }
  WU.loading = false; repaint('webui');
}

/* ── doctor ─────────────────────────────────────────────── */
const DOC = { data: null, loading: false };
views.doctor = {
  title: () => 'Doctor',
  render() {
    const d = DOC.data;
    const icon = { ok: ['✓', 'ok'], warn: ['!', 'warn'], fail: ['✗', 'crit'], info: ['i', 'boot'] };
    let groups = {};
    if (d) for (const c of d.checks) (groups[c.group] = groups[c.group] || []).push(c);
    const s = d ? d.summary : {};
    return `<div class="row" style="margin-bottom:18px;gap:22px">
      ${d ? `<div class="metric"><div class="v crit">${s.fail}</div><div class="l">Failing</div></div><div class="metric"><div class="v warn">${s.warn}</div><div class="l">Warnings</div></div><div class="metric"><div class="v boot">${s.info}</div><div class="l">Notes</div></div><div class="metric"><div class="v ok">${s.ok}</div><div class="l">Passing</div></div>` : ''}
      <span class="grow"></span><button class="btn pri ${DOC.loading ? 'busy' : ''}" data-act="doc-run">${I.doctor}Run checks</button></div>
      ${!d ? `<div class="faint">${DOC.loading ? 'Checking every host, engine and integration…' : ''}</div>` : Object.entries(groups).map(([g, cs]) => `<div class="sec"><div class="sh"><h2>${esc(g)}</h2><div class="rule"></div></div>
        ${cs.map(c => `<div class="check"><span class="ic ${icon[c.level][1]}">${icon[c.level][0]}</span><div><div class="t">${esc(c.title)}</div>${c.detail ? `<div class="dd">${esc(c.detail)}</div>` : ''}</div>
          <div>${c.fix && FIX[c.fix] ? `<button class="btn xs ${c.level === 'fail' ? 'pri' : ''}" data-act="doc-fix" data-fix="${esc(c.fix)}" data-host="${esc(c.host || '')}" data-target="${esc(c.target || '')}">${esc(FIX[c.fix][0])}</button>` : ''}</div></div>`).join('')}</div>`).join('')}`;
  },
  mount() { if (!DOC.data && !DOC.loading) docRun(); },
};
async function docRun() {
  if (DOC.loading) return; DOC.loading = true; repaint('doctor');
  try { DOC.data = await api('/api/doctor'); } catch (e) { toast(e.message, true); }
  DOC.loading = false; repaint('doctor');
  updateRail();
}

/* ── settings ───────────────────────────────────────────── */
views.settings = {
  title: () => 'Settings',
  render() {
    const s = S.settings; const g = s.gateway || {}; const p = s.policy || {}; const u = s.ui || {};
    const theme = store.get('theme', 'dark'); const acc = store.get('accent', 'signal');
    return `<div style="max-width:980px">
    <div class="sec" id="hf"><div class="sh"><h2>Hugging Face</h2><div class="rule"></div></div>
      <div class="fgrid" style="grid-template-columns:2fr 1fr">
        <label class="f"><span>Token ${tip('Read token from huggingface.co/settings/tokens. Stored in config.json (mode 600) and handed to containers through an env file, never on the command line.')}</span><input type="password" data-s="hf_token" placeholder="${s.hf_token_set ? '•••••••• saved' + (s.hf_token_env ? ' (from HF_TOKEN env)' : '') : 'hf_…'}" autocomplete="off"></label>
        <label class="f"><span>&nbsp;</span><div class="row"><button class="btn" data-act="set-save" data-keys="hf_token">Save token</button><button class="btn ghost" data-act="hf-who">Test</button></div></label>
        <label class="f"><span>Hub endpoint ${tip('Leave empty for huggingface.co. Set for a mirror.')}</span><input data-s="hf_endpoint" value="${esc(s.hf_endpoint)}" class="mono" placeholder="https://huggingface.co"></label></div></div>
    <div class="sec"><div class="sh"><h2>Runtime</h2><div class="rule"></div></div>
      <div class="fgrid" style="grid-template-columns:1fr 1fr">
        <label class="f"><span>vLLM image</span><input data-s="vllm_image" value="${esc(s.vllm_image)}" class="mono"></label>
        <label class="f"><span>llama.cpp image ${tip('Must publish linux/arm64 for GB10.')}</span><input data-s="llamacpp_image" value="${esc(s.llamacpp_image)}" class="mono"></label>
        <label class="f"><span>Container prefix ${tip('Engine containers are named prefix + engine. Existing titan-* containers keep working.')}</span><input data-s="container_prefix" value="${esc(s.container_prefix)}" class="mono"></label>
        <label class="f"><span>Port pool</span><input data-s="port_pool" value="${esc(s.port_pool)}" class="mono"></label>
        <label class="f"><span>Reserved ports</span><input data-s="port_reserved" value="${esc(s.port_reserved)}" class="mono"></label></div></div>
    <div class="sec"><div class="sh"><h2>Memory</h2><div class="rule"></div></div>
      <div class="fgrid">
        <label class="f"><span>Headroom GiB ${tip('Kept free for the OS and desktop when fitting engines.')}</span><input type="number" step="0.5" data-s="headroom_gib" value="${esc(s.headroom_gib)}"></label>
        <label class="f"><span>Crash guard ${tip('After this many restarts in a crash loop, stop the container so it stops grabbing memory. 0 = off.')}</span><input type="number" data-s="crash_guard" value="${esc(s.crash_guard)}"></label>
        <label class="chk" style="align-self:end"><input type="checkbox" data-s="auto_flush_cache" ${s.auto_flush_cache ? 'checked' : ''}>Flush page cache before a start when needed</label></div></div>
    <div class="sec"><div class="sh"><h2>Origin policy</h2>${tip('Weights from these orgs — or fine-tunes whose Hub metadata names them as base_model — are flagged or blocked in search, launch and pulls.')}<div class="rule"></div></div>
      <div class="row" style="margin-bottom:12px"><div class="seg">${['block', 'warn', 'off'].map(m => `<button class="${p.mode === m ? 'on' : ''}" data-act="pol-mode" data-v="${m}">${m}</button>`).join('')}</div>
        <input id="orgadd" placeholder="add org" class="mono" style="max-width:200px" data-keep><button class="btn sm" data-act="pol-add">Add</button></div>
      <div class="chips">${(p.orgs || []).map(o => `<span class="chip">${esc(o)}<button data-act="pol-del" data-v="${esc(o)}">×</button></span>`).join('')}</div></div>
    <div class="sec"><div class="sh"><h2>Gateway</h2><div class="rule"></div></div>
      <div class="fgrid">
        <label class="chk"><input type="checkbox" data-s="gateway.enabled" ${g.enabled ? 'checked' : ''}>Enabled</label>
        <label class="chk"><input type="checkbox" data-s="gateway.autowake" ${g.autowake ? 'checked' : ''}>Wake sleeping engines on request</label>
        <label class="f"><span>Wake timeout (s)</span><input type="number" data-s="gateway.wake_timeout" value="${esc(g.wake_timeout)}"></label>
        <label class="f"><span>Extra listener ${tip('Also serve /v1 on another address: an IP, or "docker" for the Docker network gateway so containers like Open WebUI can reach it. Restart the service after changing.')}</span><input data-s="gateway.listen" value="${esc(g.listen || '')}" class="mono" placeholder="off"></label>
        <label class="f"><span>Listener port</span><input type="number" data-s="gateway.port" value="${esc(g.port)}"></label>
        <label class="f"><span>API key</span><div class="row">${g.key_set ? '<span class="tag ok">set</span>' : '<span class="tag">none</span>'}<button class="btn sm" data-act="gw-key">Generate</button>${g.key_set ? '<button class="btn sm ghost" data-act="gw-key-clear">Clear</button>' : ''}</div></label></div></div>
    <div class="sec"><div class="sh"><h2>Console access</h2><div class="rule"></div></div>
      <div class="fgrid" style="grid-template-columns:1fr 1fr">
        <label class="f"><span>Access token ${tip('Required when the console listens beyond loopback. Browsers log in once; scripts send Authorization: Bearer.')}</span><div class="row"><input type="password" id="uitok" placeholder="${u.token_set ? '•••••••• set' : 'none'}" data-keep><button class="btn sm" data-act="ui-tok">Set</button>${u.token_set ? '<button class="btn sm ghost" data-act="ui-tok-clear">Clear</button>' : ''}</div></label>
        <label class="f"><span>Allowed Host headers ${tip('Extra hostnames this console answers to, comma separated (e.g. titan.lab). Blocks DNS-rebinding attacks.')}</span><input data-s="ui.allowed_hosts" value="${esc((u.allowed_hosts || []).join(', '))}" class="mono"></label></div></div>
    <div class="sec"><div class="sh"><h2>Appearance</h2><div class="rule"></div></div>
      <div class="row" style="gap:24px"><div class="seg">${['dark', 'light'].map(t => `<button class="${theme === t ? 'on' : ''}" data-act="theme" data-v="${t}">${t}</button>`).join('')}</div>
        <div class="swatches">${[['signal', '#ff8a3d'], ['ion', '#39c6f0'], ['nominal', '#8bd11a'], ['violet', '#b18cff']].map(([k, c]) => `<button class="sw${acc === k ? ' on' : ''}" style="background:${c}" data-act="accent" data-v="${k}" data-tip="${k}"></button>`).join('')}</div></div></div>
    <div class="sec"><div class="sh"><h2>Backup</h2><div class="rule"></div></div>
      <div class="row"><button class="btn" data-act="export">${I.down}Export engines &amp; blueprints</button><span class="faint mono" style="font-size:11.5px">config lives in ${esc(s.conf_dir)}</span></div></div>
    <div class="row"><button class="btn pri" data-act="set-save">Save settings</button><span class="faint" style="font-size:12px">v${esc(S.version)}</span></div>
    </div>`;
  },
};

/* ── activity ───────────────────────────────────────────── */
views.activity = {
  title: () => 'Activity',
  render() {
    const lv = store.get('actLv', 'all');
    const f = lv === 'all' ? null : e => lv === 'problems' ? (e.level === 'error' || e.level === 'warn') : e.level === lv;
    return `<div class="row" style="margin-bottom:14px"><div class="seg">${[['all', 'All'], ['problems', 'Problems'], ['ok', 'Good news']].map(([k, l]) => `<button class="${lv === k ? 'on' : ''}" data-act="act-lv" data-v="${k}">${l}</button>`).join('')}</div></div>
      <div class="feed" style="max-width:900px">${feedHTML(400, f)}</div>`;
  },
  live: true,
};

function repaint(view) { if (S.route.view === view) { const v = views[view]; morph($('#view .page'), v.render(S.route)); updateRail(); } }

/* ── shell ──────────────────────────────────────────────── */
const NAV = [['deck', 'Deck', 'deck'], ['engines', 'Engines', 'engines'], ['launch', 'Launch', 'launch'], ['play', 'Playground', 'play'], '-',
  ['library', 'Library', 'library'], ['hosts', 'Hosts', 'hosts'], ['containers', 'Containers', 'containers'], ['webui', 'Open WebUI', 'webui'], '-',
  ['doctor', 'Doctor', 'doctor'], ['activity', 'Activity', 'activity'], ['settings', 'Settings', 'settings']];
function shell() {
  $('#app').innerHTML = `<aside id="rail"><div class="brand">${BRAND}<div><b>VLLM-LAB</b><small>local AI control plane</small></div></div>
    <nav class="main" id="nav"></nav><div class="railhosts" id="railhosts"></div><div class="railfoot" id="railfoot"></div></aside>
    <main id="main"><div id="top"><button class="btn ghost icon" id="navbtn" data-act="navtoggle">${I.activity}</button><h1 id="title"></h1><div class="fleet" id="fleet"></div><div class="search" data-act="palette">${I.search}<span class="grow">Jump to…</span><kbd>Ctrl K</kbd></div></div>
    <div id="view"><div class="page"></div></div></main>`;
}
function updateRail() {
  const crashed = Object.values(S.snap.engines).filter(e => e.state === 'crashed').length;
  const docFail = DOC.data ? DOC.data.summary.fail : 0;
  morph($('#nav'), NAV.map(n => n === '-' ? '<div class="sep"></div>' : `<a href="#/${n[0]}" class="${S.route.view === n[0] ? 'on' : ''}" data-key="n-${n[0]}">${I[n[2]]}<span>${n[1]}</span>${n[0] === 'engines' && crashed ? `<span class="badge">${crashed}</span>` : ''}${n[0] === 'doctor' && docFail ? `<span class="badge">${docFail}</span>` : ''}</a>`).join(''));
  morph($('#railhosts'), hostList().map(h => {
    const M = hostMem(h.id); const on = h.enabled && h.st.online;
    const bars = on && M.total ? M.engines.map(e => `<i style="width:${(e.reserve / M.total * 100).toFixed(2)}%;background:${engineColor(e.name)}"></i>`).join('') + `<i style="width:${(M.other / M.total * 100).toFixed(2)}%;background:var(--tx4)"></i>` : '';
    return `<a class="rh" href="#/hosts" data-key="rh-${esc(h.id)}">${led(!h.enabled ? 'offline' : on ? 'ready' : 'crashed')}<span class="nm">${esc(h.label)}</span><span class="v">${on && M.total ? fmtGiB(M.avail) + ' free' : !h.enabled ? 'off' : 'down'}</span><div class="mbar">${bars}</div></a>`;
  }).join(''));
  const tps = Object.values(S.snap.engines).reduce((s, e) => s + (e.state === 'ready' ? ((e.metrics || {}).gen_tps || 0) : 0), 0);
  const ready = Object.values(S.snap.engines).filter(e => e.state === 'ready').length;
  const booting = Object.values(S.snap.engines).filter(e => e.state === 'booting').length;
  morph($('#fleet'), `<span class="pill">${led(ready ? 'ready' : 'offline')}${ready} ready</span>${booting ? `<span class="pill">${led('booting')}${booting} booting</span>` : ''}${crashed ? `<span class="pill crit">${led('crashed')}${crashed} crashed</span>` : ''}<span class="pill" data-tip="Output tokens per second across every engine">${fmtN(tps, tps < 10 ? 1 : 0)} tok/s</span>`);
  morph($('#railfoot'), `<span class="led ${S.connected ? 'ready' : 'crashed'}"></span><span class="grow">${S.connected ? 'live' : 'reconnecting'}</span><span>v${esc(S.version)}</span><button class="btn xs ghost icon" data-act="theme-toggle" data-tip="Theme">${I.sun}</button>`);
}
function updateTray() {
  let el = $('#jobs');
  const t = Date.now() / 1000;
  const list = Object.values(S.jobs).filter(j => j.state === 'running' || (!S.dismissed.has(j.id) && ((j.state === 'error' && S.mine.has(j.id)) || (j.t1 && t - j.t1 < (j.state === 'error' ? 12 : 5))))).sort((a, b) => a.t0 - b.t0).slice(-5);
  morph(el, list.map(j => {
    const err = j.error || {};
    const indet = j.state === 'running' && j.progress == null;
    const last = (j.lines || []).slice(-1)[0];
    const eng = S.specs[j.target] ? j.target : '';
    return `<div class="job ${j.state}" data-key="j-${esc(j.id)}"><div class="jt">${j.state === 'running' ? led('booting') : j.state === 'ok' ? led('ready') : led('crashed')}<span class="grow ellip">${esc(j.title)}</span>
        <span class="faint mono" style="font-size:10.5px">${fmtDur((j.t1 || t) - j.t0)}</span>${j.state === 'running' ? `<button class="btn xs ghost" data-act="job-cancel" data-id="${esc(j.id)}" data-tip="Cancel waiting (the engine keeps booting)">${I.x}</button>` : `<button class="btn xs ghost" data-act="job-dismiss" data-id="${esc(j.id)}">${I.x}</button>`}</div>
      ${j.state === 'running' ? `<div class="js">${esc(j.stage || (last && last[1]) || '…')}</div><div class="pb${indet ? ' indet' : ''}"><i style="width:${((j.progress || 0) * 100).toFixed(1)}%"></i></div>`
        : j.state === 'ok' ? `<div class="js ok">${esc(j.result)}</div>`
        : `<div class="js">${esc(err.error || j.result)}</div>${err.hint ? `<div class="hint">${esc(err.hint)}</div>` : ''}${eng ? `<div class="fixes">${fixButtons(eng, (err.fixes || []).filter(f => f !== 'force' || !(err.data && err.data.fit)))}${err.data && err.data.fit ? `<button class="btn xs" data-act="conflict" data-name="${esc(eng)}" data-id="${esc(j.id)}">Options…</button>` : ''}<button class="btn xs ghost" data-act="job-log" data-id="${esc(j.id)}">Details</button></div>` : `<div class="fixes"><button class="btn xs ghost" data-act="job-log" data-id="${esc(j.id)}">Details</button></div>`}`}
    </div>`;
  }).join(''));
}

/* ── render loop ────────────────────────────────────────── */
let renderQueued = false;
function render(force) {
  const r = S.route; const v = views[r.view] || views.deck;
  $('#title').innerHTML = typeof v.title === 'function' ? v.title(r) : v.title;
  const page = $('#view .page');
  if (force) { page.innerHTML = v.render(r); v.mount && v.mount(r); }
  else if (v.live) morph(page, v.render(r));
  else if (v.tick) v.tick(r);
  updateRail();
}
function queueRender() { if (renderQueued) return; renderQueued = true; requestAnimationFrame(() => { renderQueued = false; render(false); }); }
function onRoute() {
  document.body.classList.remove('navopen');
  const prev = S.route;
  S.route = parseRoute();
  if (prev.view === 'engines' && prev.sub === 'logs') stopLogs();
  render(true);
  const r = S.route;
  if (r.view === 'engines' && r.arg) {
    if (r.sub === 'logs') mountLogs(r.arg);
    if (r.sub === 'config') mountConfig(r.arg);
    if (r.sub === 'bench') mountBench(r.arg);
    if (r.sub === 'inspect') mountInspect(r.arg);
  }
  $('#view').scrollTop = 0;
}
window.addEventListener('hashchange', onRoute);

/* ── live stream ────────────────────────────────────────── */
function connect() {
  const es = new EventSource('/api/stream');
  es.addEventListener('snapshot', ev => { S.connected = true; onSnapshot(JSON.parse(ev.data)); });
  es.addEventListener('job', ev => onJob(JSON.parse(ev.data)));
  es.addEventListener('event', ev => { const e = JSON.parse(ev.data); if (!S.events.some(x => x.id === e.id)) { S.events.push(e); S.events = S.events.slice(-500); } if (e.level === 'error' && !S.mine.size) { /* surfaced in feed */ } queueRender(); });
  es.onopen = () => { S.connected = true; const b = $('.conn'); if (b) b.remove(); updateRail(); };
  es.onerror = () => { S.connected = false; updateRail(); if (!$('.conn')) { const b = document.createElement('div'); b.className = 'conn'; b.textContent = 'Lost the console server — reconnecting'; document.body.appendChild(b); } };
}
async function onSnapshot(snap) {
  S.snap = snap;
  if (snap.specs) S.specs = snap.specs;
  if (snap.conf_m !== S.confM) { const first = S.confM === null; S.confM = snap.conf_m; if (!first) { try { S.settings = await api('/api/settings'); } catch { } } }
  for (const [n, e] of Object.entries(snap.engines)) {
    if (e.state !== 'ready' && e.state !== 'booting') continue;
    const m = e.metrics || {}; const arr = S.hist.e[n] = S.hist.e[n] || [];
    const pt = [snap.t, m.gen_tps || 0, (m.kv || 0) * 100, m.running || 0, m.waiting || 0];
    if (!arr.length || snap.t - arr[arr.length - 1][0] > 1.5) { arr.push(pt); if (arr.length > 400) arr.splice(0, arr.length - 400); }
  }
  for (const [hid, h] of Object.entries(snap.hosts)) {
    if (!h.online || !h.mem) continue; const g = (h.gpus || [])[0] || {}; const arr = S.hist.h[hid] = S.hist.h[hid] || [];
    const pt = [snap.t, h.mem.total ? 100 * (1 - h.mem.available / h.mem.total) : null, g.util, g.temp, g.power];
    if (!arr.length || snap.t - arr[arr.length - 1][0] > 1.5) { arr.push(pt); if (arr.length > 400) arr.splice(0, arr.length - 400); }
  }
  queueRender();
}
function onJob(j) {
  const prev = S.jobs[j.id];
  S.jobs[j.id] = j;
  if (prev && prev.state === 'running' && j.state !== 'running') {
    const hook = S.jobHooks[j.id]; if (hook) { delete S.jobHooks[j.id]; hook(j); }
    if (j.state === 'ok' && S.mine.has(j.id)) toast(j.result);
    if (S.route.view === 'engines' && S.route.sub === 'bench' && j.kind === 'bench') mountBench(S.route.arg);
    if (j.kind === 'download' && S.route.view === 'library') loadLib();
    if (S.route.view === 'hosts' && j.kind === 'host-test' && j.state === 'ok') toast(j.result);
  }
  if (!prev && S.mine.has(j.id) === false && j.state === 'running') { /* another client's job — still shown */ }
  updateTray();
}
setInterval(updateTray, 1000);

/* ── actions (one delegated handler) ────────────────────── */
const A = {
  'modal-close': () => closeModal(false),
  'modal-ok': () => closeModal(true),
  palette: () => openPalette(),
  navtoggle: () => document.body.classList.toggle('navopen'),
  copy: el => copyText(el.dataset.text || ''),
  'deck-filter': el => { store.set('deckFilter', el.dataset.v); render(false); },
  start: el => startEngine(el.dataset.name),
  stop: el => startJob(`/api/engines/${encodeURIComponent(el.dataset.name)}/stop`),
  recreate: el => startJob(`/api/engines/${encodeURIComponent(el.dataset.name)}/recreate`),
  restart: el => startJob(`/api/engines/${encodeURIComponent(el.dataset.name)}/restart`),
  remove: async el => { const n = el.dataset.name; if (!await confirmDlg('Remove container?', `The <b>${esc(n)}</b> container is deleted. Its config and downloaded weights stay; Start recreates it.`, 'Remove', true)) return; try { toast((await post(`/api/engines/${encodeURIComponent(n)}/remove`)).msg); } catch (e) { showError(e); } },
  'engine-delete': async el => { const n = el.dataset.name; if (!await confirmDlg('Delete engine?', `Removes the <b>${esc(n)}</b> container and its config. Downloaded weights stay in the library.`, 'Delete engine', true)) return; try { await post(`/api/engines/${encodeURIComponent(n)}/remove`, { delete: true }); toast('Deleted ' + n); F = null; go('#/engines'); } catch (e) { showError(e); } },
  fix: el => doFix(el.dataset.name, el.dataset.fix),
  'engine-menu': el => engineMenu(el, el.dataset.name),
  adopt: async el => { try { const d = await post(`/api/hosts/${encodeURIComponent(el.dataset.host)}/adopt`, { container: el.dataset.container }); toast('Adopted as ' + d.spec.name); } catch (e) { showError(e); } },
  snip: el => { store.set('snip', el.dataset.v); render(false); },
  'host-test': el => startJob(`/api/hosts/${encodeURIComponent(el.dataset.host)}/test`),
  'host-edit': async el => { const h = (S.settings.hosts || []).find(x => x.id === el.dataset.host); await hostForm(h); },
  'host-save': async el => {
    const f = $('#hostf'); const d = Object.fromEntries(new FormData(f).entries()); d.enabled = f.enabled.checked; if (d.ssh_port === '') d.ssh_port = null;
    try { await post('/api/hosts', { host: d, old_id: el.dataset.old }); S.settings = await api('/api/settings'); closeModal(true); toast('Saved ' + d.id); render(true); if (d.ssh) startJob(`/api/hosts/${encodeURIComponent(d.id)}/test`); } catch (e) { showError(e); }
  },
  'host-toggle': async el => { const h = (S.settings.hosts || []).find(x => x.id === el.dataset.host); try { await post('/api/hosts', { host: { ...h, enabled: el.dataset.v === '1' }, old_id: h.id }); S.settings = await api('/api/settings'); render(true); } catch (e) { showError(e); } },
  'host-remove': async el => { if (!await confirmDlg('Remove host?', `Forget <b>${esc(el.dataset.host)}</b>. Nothing on the machine is touched.`, 'Remove', true)) return; try { await post(`/api/hosts/${encodeURIComponent(el.dataset.host)}/remove`); S.settings = await api('/api/settings'); render(true); } catch (e) { showError(e); } },
  'host-flush': async el => { try { toast((await post(`/api/hosts/${encodeURIComponent(el.dataset.host)}/flush`)).msg); } catch (e) { showError(e); } },
  'lib-host': el => { store.set('libHost', el.dataset.v); LIB.data = null; render(true); },
  'lib-refresh': () => loadLib(),
  'lib-dl': () => { const id = $('#dlid').value.trim(); if (!id) return; startJob('/api/library/download', { model: id, host: curHost('libHost') }); $('#dlid').value = ''; },
  'lib-del': async el => { const m = el.dataset.model; if (!await confirmDlg('Delete weights?', `Deletes <b>${esc(m)}</b> from the cache on ${esc(curHost('libHost'))}. It downloads again the next time an engine needs it.`, 'Delete', true)) return; try { toast((await post('/api/library/delete', { model: m, host: curHost('libHost') })).msg); loadLib(); } catch (e) { showError(e); } },
  ol: async el => { if (el.dataset.op === 'delete' && !await confirmDlg('Delete from Ollama?', esc(el.dataset.model), 'Delete', true)) return; el.classList.add('busy'); try { toast((await post('/api/ollama', { host: el.dataset.host, op: el.dataset.op, model: el.dataset.model })).msg); } catch (e) { showError(e); } el.classList.remove('busy'); },
  'ol-pull': el => { const m = $('#olid').value.trim(); if (m) startJob('/api/ollama', { host: el.dataset.host, op: 'pull', model: m }); },
  'ct-host': el => { store.set('ctHost', el.dataset.v); CT.data = null; render(true); },
  'ct-refresh': () => loadCT(),
  'ct-logs': async el => { try { const d = await post(`/api/hosts/${encodeURIComponent(el.dataset.host)}/docker`, { op: 'logs', target: el.dataset.target }); textModal('logs · ' + el.dataset.target, d.text); } catch (e) { showError(e); } },
  'ct-inspect': async el => { try { const d = await post(`/api/hosts/${encodeURIComponent(el.dataset.host)}/docker`, { op: 'inspect', target: el.dataset.target }); textModal('inspect · ' + el.dataset.target, JSON.stringify(d.json, null, 2)); } catch (e) { showError(e); } },
  'ct-pull': async el => { const img = prompt('Image to pull (e.g. vllm/vllm-openai:latest)'); if (img) startJob(`/api/hosts/${encodeURIComponent(el.dataset.host)}/pull`, { image: img.trim() }); },
  docker: async el => { if (el.dataset.confirm && !await confirmDlg('Are you sure?', esc(el.dataset.confirm), 'Yes', true)) return; el.classList.add('busy'); try { toast((await post(`/api/hosts/${encodeURIComponent(el.dataset.host)}/docker`, { op: el.dataset.op, target: el.dataset.target })).msg); loadCT(); } catch (e) { showError(e); } el.classList.remove('busy'); },
  'wu-check': () => wuCheck(),
  'wu-sync': async el => { el.classList.add('busy'); try { const d = await post('/api/webui/sync'); toast(`Open WebUI: +${d.added.length} −${d.removed.length}`); await wuCheck(); } catch (e) { showError(e); } el.classList.remove('busy'); },
  'wu-login-mode': el => { store.set('wuLogin', el.dataset.v); render(true); },
  'wu-save': async () => { const f = $('#wuf'); const d = Object.fromEntries(new FormData(f).entries()); d.auto_sync = f.auto_sync.checked; d.prune_stopped = f.prune_stopped.checked; try { const r = await post('/api/webui/save', d); S.settings = r.settings; toast('Saved'); WU.data = null; wuCheck(); } catch (e) { showError(e); } },
  'wu-forget': async () => { try { const r = await post('/api/webui/save', { clear_password: true, clear_api_key: true }); S.settings = r.settings; render(true); } catch (e) { showError(e); } },
  'doc-run': () => docRun(),
  'doc-fix': async el => { const { fix, host, target } = el.dataset; const nav = { 'set-token': '#/settings', 'webui-login': '#/webui', 'host-test': '#/hosts', library: '#/library', policy: '#/settings', edit: target ? `#/engines/${target}/config` : '#/engines', logs: `#/engines/${target}/logs`, 'docker-group': null };
    if (fix in nav) { if (nav[fix]) go(nav[fix]); else textModal('Docker permissions', 'sudo usermod -aG docker $USER\n# log out and back in, or for the service add to labui.service:\n[Service]\nSupplementaryGroups=docker'); return; }
    el.classList.add('busy'); try { const d = await post('/api/doctor', { fix, host, target }); if (d.job) S.mine.add(d.job); else toast(d.msg); setTimeout(docRun, 800); } catch (e) { showError(e); } el.classList.remove('busy'); },
  'set-save': async () => { try { S.settings = await post('/api/settings', collectSettings()); toast('Settings saved'); render(true); } catch (e) { showError(e); } },
  'hf-who': async () => { try { const d = await api('/api/hf/whoami'); d.ok ? toast('Token works — signed in as ' + d.name) : toast('Token check: ' + d.why, true); } catch (e) { showError(e); } },
  'pol-mode': async el => { try { S.settings = await post('/api/settings', { policy: { mode: el.dataset.v } }); render(true); } catch (e) { showError(e); } },
  'pol-add': async () => { const v = $('#orgadd').value.trim(); if (!v) return; try { S.settings = await post('/api/settings', { policy: { orgs: [...(S.settings.policy.orgs || []), v] } }); render(true); } catch (e) { showError(e); } },
  'pol-del': async el => { try { S.settings = await post('/api/settings', { policy: { orgs: (S.settings.policy.orgs || []).filter(o => o !== el.dataset.v) } }); render(true); } catch (e) { showError(e); } },
  'gw-key': async () => { try { const d = await post('/api/settings', { gateway: { generate_key: true } }); S.settings = d; textModal('Gateway API key', d.new_gateway_key + '\n\nShown once. Clients send it as: Authorization: Bearer <key>'); render(true); } catch (e) { showError(e); } },
  'gw-key-clear': async () => { try { S.settings = await post('/api/settings', { gateway: { clear_key: true } }); render(true); } catch (e) { showError(e); } },
  'ui-tok': async () => { const v = $('#uitok').value; if (v.length < 12) return toast('Use at least 12 characters', true); try { S.settings = await post('/api/settings', { ui: { token: v } }); toast('Access token set'); render(true); } catch (e) { showError(e); } },
  'ui-tok-clear': async () => { try { S.settings = await post('/api/settings', { ui: { clear_token: true } }); render(true); } catch (e) { showError(e); } },
  theme: el => { setTheme(el.dataset.v); render(true); },
  'theme-toggle': () => { setTheme(store.get('theme', 'dark') === 'dark' ? 'light' : 'dark'); render(false); },
  accent: el => { store.set('accent', el.dataset.v); document.documentElement.dataset.accent = el.dataset.v; render(true); },
  export: async () => { download('vllm-lab-export.json', JSON.stringify({ vllm_lab: S.version, engines: S.specs, blueprints: Object.fromEntries(Object.entries(S.bps).filter(([, b]) => !b.builtin)) }, null, 2)); },
  'act-lv': el => { store.set('actLv', el.dataset.v); render(false); },
  'lq-tab': el => { LQ.tab = el.dataset.v; store.set('lqTab', LQ.tab); $$('.tabs button', $('#view')).forEach(b => b.classList.toggle('on', b.dataset.v === LQ.tab)); $('#picker').innerHTML = pickerHTML(); if (LQ.tab === 'disk') loadDisk(); if (LQ.tab === 'hub') { const i = $('#hubq'); i && i.focus(); if (!LQ.rows) runSearch(); } },
  'lq-kind': el => { LQ.kind = el.dataset.v; $('#picker').innerHTML = pickerHTML(); runSearch(); },
  'pick-model': el => pickModel(el.dataset.v),
  'pick-bp': (el, ev) => { if (ev.target.closest('[data-act="bp-del"]')) return; pickBlueprint(el.dataset.v); },
  'bp-del': async (el, ev) => { ev.stopPropagation(); try { S.bps = (await post('/api/blueprints/delete', { id: el.dataset.v })).blueprints; $('#picker').innerHTML = pickerHTML(); } catch (e) { showError(e); } },
  'bp-export': () => download('vllm-lab-blueprints.json', JSON.stringify({ blueprints: Object.fromEntries(Object.entries(S.bps).filter(([, b]) => !b.builtin)) }, null, 2)),
  'form-launch': el => formSubmit('launch', el),
  'form-create': el => formSubmit('create', el),
  'form-save': el => formSubmit(el.dataset.recreate ? 'apply' : 'save', el),
  'form-bp': el => formSubmit('bp', el),
  'conf-pick': el => { window._confSel = +el.dataset.i; window._confRender(); },
  'conf-go': () => closeModal(window._conf[window._confSel]),
  conflict: async el => { const j = S.jobs[el.dataset.id]; const fit = j && j.error && j.error.data && j.error.data.fit; if (!fit) return; S.dismissed.add(j.id); updateTray(); const c = await conflictDialog(el.dataset.name, fit); if (!c) return; const body = c.mode === 'shrink' ? { util: fit.suggest.util } : { on_conflict: c.mode, evict: c.evict }; startJob(`/api/engines/${encodeURIComponent(el.dataset.name)}/start`, body); },
  'job-cancel': el => post(`/api/jobs/${el.dataset.id}/cancel`).catch(showError),
  'job-dismiss': el => { S.dismissed.add(el.dataset.id); updateTray(); },
  'job-log': el => { const j = S.jobs[el.dataset.id]; if (j) textModal(j.title, (j.lines || []).map(l => `${String(l[0]).padStart(6)}s  ${l[1]}`).join('\n') + (j.error ? `\n\n${j.error.error}\n${j.error.hint || ''}${j.error.data && j.error.data.line ? '\n\n' + j.error.data.line : ''}${j.error.trace ? '\n\n' + j.error.trace : ''}` : '')); },
  'pg-send': () => pgSend(),
  'pg-stop': () => PG.cols.forEach(c => c.ctrl && c.ctrl.abort()),
  'pg-clear': () => { PG.cols.forEach(c => { c.msgs = []; }); views.play.mount(S.route); },
  'bench-run': el => { const b = {}; $$('[data-b]').forEach(i => { b[i.dataset.b] = Number(i.value); }); store.set('bench', b); startJob(`/api/engines/${encodeURIComponent(el.dataset.name)}/bench`, b); },
  'log-level': el => { LOG.level = el.dataset.v; $$('.logbar .seg button').forEach(b => b.classList.toggle('on', b.dataset.v === LOG.level)); paintLogs(); },
  'log-pause': el => { LOG.paused = !LOG.paused; el.textContent = LOG.paused ? 'Resume' : 'Pause'; },
  'log-copy': () => copyText(LOG.lines.join('\n')),
  'log-dl': () => download(`${LOG.name}.log`, LOG.lines.join('\n'), 'text/plain'),
};
document.addEventListener('click', e => {
  const go_ = e.target.closest('[data-go]');
  if (go_ && !e.target.closest('[data-act]')) { go(go_.dataset.go); return; }
  const el = e.target.closest('[data-act]'); if (!el) return;
  const fn = A[el.dataset.act]; if (!fn) return;
  if (el.tagName === 'A' && !el.getAttribute('href')) e.preventDefault();
  if (el.tagName === 'BUTTON' || el.dataset.act === 'palette' || el.classList.contains('hit') || el.classList.contains('choice') || el.classList.contains('bpc') || el.tagName === 'SPAN' || el.tagName === 'CODE') e.preventDefault();
  const r = fn(el, e);
  if (r && r.catch) r.catch(err => showError(err));
});
document.addEventListener('input', e => {
  const el = e.target;
  if (el.dataset.f) { onFormInput(el); return; }
  const k = el.dataset.input;
  if (k === 'hubq') { LQ.q = el.value; runSearch(); }
  if (k === 'engQ') { store.set('engQ', el.value); render(false); }
  if (k === 'ctq') { CT.q = el.value; morph($('#view .page'), views.containers.render(S.route)); }
  if (el.dataset.log === 'q') { LOG.q = el.value; paintLogs(); }
  if (el.dataset.pg) { PG[el.dataset.pg] = el.value; store.set('pg' + el.dataset.pg[0].toUpperCase() + el.dataset.pg.slice(1), el.value); }
});
document.addEventListener('change', e => {
  const el = e.target;
  if (el.dataset.f && (el.tagName === 'SELECT' || el.type === 'checkbox')) { onFormInput(el); return; }
  if (el.dataset.log === 'follow') { LOG.follow = el.checked; }
  if (el.dataset.pgEngine !== undefined) { PG.cols[+el.dataset.pgEngine].engine = el.value; PG.cols[+el.dataset.pgEngine].msgs = []; views.play.mount(S.route); }
  const ac = el.dataset.actChange;
  if (ac === 'pg-compare') { PG.compare = el.checked; if (!PG.compare) PG.cols = PG.cols.slice(0, 1); views.play.mount(S.route); }
  if (ac === 'bp-import') { const f = el.files[0]; if (!f) return; f.text().then(t => post('/api/blueprints/import', JSON.parse(t))).then(d => { S.bps = d.blueprints; toast(`Imported ${d.imported} blueprint(s)`); $('#picker').innerHTML = pickerHTML(); }).catch(showError); }
});
document.addEventListener('click', e => { if (e.target.closest('[data-f][data-v]') && e.target.closest('[data-f]').tagName === 'BUTTON') onFormInput(e.target.closest('[data-f]')); });
document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
  if (e.key === 'Escape') { if ($('#pal')) closePalette(); else if ($('#menu')) closeMenu(); else if ($('#modal')) closeModal(false); return; }
  if (e.target.id === 'pgin' && e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); pgSend(); }
  if (e.key === 'Enter' && e.target.closest('#modal') && e.target.tagName === 'INPUT') { const ok = $('#modal [data-act="host-save"], #modal [data-act="modal-ok"]'); if (ok) ok.click(); }
});
function setTheme(t) { store.set('theme', t); document.documentElement.dataset.theme = t; }
function collectSettings() {
  const out = {};
  $$('[data-s]').forEach(el => {
    const k = el.dataset.s; let v = el.type === 'checkbox' ? el.checked : el.value;
    if (k === 'hf_token' && !v) return;
    if (k === 'ui.allowed_hosts') v = v.split(',').map(x => x.trim()).filter(Boolean);
    const [a, b] = k.split('.');
    if (b) { out[a] = out[a] || {}; out[a][b] = v; } else out[a] = v;
  });
  return out;
}
function textModal(title, text) {
  return modal(`<header><h3 class="grow">${esc(title)}</h3><button class="btn sm" data-act="copy" data-text="${esc(text)}">${I.copy}Copy</button></header><div class="body"><div class="code" style="max-height:62vh"><pre style="white-space:pre-wrap">${esc(text)}</pre></div></div><footer><button class="btn" data-act="modal-close">Close</button></footer>`, { width: 860 });
}
async function doFix(name, fix) {
  const nav = { 'set-token': '#/settings', 'open-hf': null, edit: `#/engines/${encodeURIComponent(name)}/config`, logs: `#/engines/${encodeURIComponent(name)}/logs`, policy: '#/settings', library: '#/library', doctor: '#/doctor', 'host-test': '#/hosts', 'host-enable': '#/hosts', 'webui-login': '#/webui' };
  if (fix === 'open-hf') { window.open(hfUrl((S.specs[name] || {}).model), '_blank', 'noopener'); return; }
  if (fix in nav) { go(nav[fix]); return; }
  if (fix === 'make-room') {
    const spec = S.specs[name];
    try { const fit = await post(`/api/engines/${encodeURIComponent(name)}/fit`, {}); const c = await conflictDialog(name, fit, spec); if (!c) return; const body = c.mode === 'shrink' ? { util: fit.suggest.util, recreate: true } : { on_conflict: c.mode, evict: c.evict, recreate: true }; return startJob(`/api/engines/${encodeURIComponent(name)}/start`, body); } catch (e) { return showError(e); }
  }
  if (fix === 'start') return startEngine(name);
  return startJob(`/api/engines/${encodeURIComponent(name)}/fix`, { fix });
}
function engineMenu(anchor, name) {
  const e = S.snap.engines[name] || {}; const st = e.state; const n = encodeURIComponent(name);
  const items = [];
  if (st === 'ready' || st === 'booting') { items.push({ label: 'Stop', icon: I.stop, fn: () => A.stop({ dataset: { name } }) }); items.push({ label: 'Put to sleep', icon: I.moon, fn: () => startJob(`/api/engines/${n}/stop`, { sleep: true }) }); items.push({ label: 'Restart', icon: I.restart, fn: () => A.restart({ dataset: { name } }) }); }
  else items.push({ label: st === 'sleeping' ? 'Wake' : 'Start', icon: I.start, fn: () => startEngine(name) });
  items.push({ label: 'Recreate with saved config', icon: I.restart, fn: () => A.recreate({ dataset: { name } }) });
  if (st === 'ready') { items.push({ label: 'Quick probe', icon: I.bolt, fn: async () => { try { const d = await post(`/api/engines/${n}/probe`); d.ok ? toast(`${name}: "${d.text}" · first token ${fmtN((d.ttft || 0) * 1000)} ms`) : toast(`${name}: ${d.error}`, true); } catch (er) { showError(er); } } }); items.push({ label: 'Benchmark', icon: I.bench, fn: () => go(`#/engines/${n}/bench`) }); }
  items.push('-');
  items.push({ label: 'Log', icon: I.logs, fn: () => go(`#/engines/${n}/logs`) });
  items.push({ label: 'Connect', icon: I.link, fn: () => go(`#/engines/${n}/connect`) });
  items.push({ label: 'Edit config', icon: I.edit, fn: () => go(`#/engines/${n}/config`) });
  items.push({ label: 'Duplicate', icon: I.copy, fn: () => { formFrom({ ...S.specs[name], name: '', port: '' }, 'launch'); F.name = slugify(F.model); F.share = 'manual'; F._ctxTouched = true; go('#/launch'); setTimeout(() => { $('#formpane').innerHTML = formHTML(); loadPlan(); }, 30); } });
  items.push('-');
  items.push({ label: 'Remove container', icon: I.trash, fn: () => A.remove({ dataset: { name } }) });
  items.push({ label: 'Delete engine', icon: I.trash, danger: true, fn: () => A['engine-delete']({ dataset: { name } }) });
  menu(anchor, items);
}

/* ── command palette ────────────────────────────────────── */
let PAL = { items: [], sel: 0 };
function paletteItems() {
  const it = [];
  NAV.filter(n => n !== '-').forEach(n => it.push({ g: 'Go', t: n[1], icon: I[n[2]], fn: () => go('#/' + n[0]) }));
  for (const name of Object.keys(S.specs).sort()) {
    const e = S.snap.engines[name] || {}; const n = encodeURIComponent(name);
    if (e.state === 'ready' || e.state === 'booting') it.push({ g: name, t: `Stop ${name}`, icon: I.stop, fn: () => A.stop({ dataset: { name } }) });
    else it.push({ g: name, t: `${e.state === 'sleeping' ? 'Wake' : 'Start'} ${name}`, icon: I.start, fn: () => startEngine(name) });
    it.push({ g: name, t: `Chat with ${name}`, icon: I.play, fn: () => go('#/play?e=' + n) });
    it.push({ g: name, t: `Log · ${name}`, icon: I.logs, fn: () => go(`#/engines/${n}/logs`) });
    it.push({ g: name, t: `Open ${name}`, icon: I.engines, fn: () => go(`#/engines/${n}`) });
  }
  Object.values(S.bps).forEach(b => it.push({ g: 'Launch', t: `Launch ${b.title}`, icon: I.launch, fn: () => go('#/launch?bp=' + encodeURIComponent(b.id)) }));
  it.push({ g: 'Do', t: 'Run doctor', icon: I.doctor, fn: () => { go('#/doctor'); docRun(); } });
  it.push({ g: 'Do', t: 'Sync Open WebUI', icon: I.link, fn: () => post('/api/webui/sync').then(d => toast(`Open WebUI: +${d.added.length} −${d.removed.length}`)).catch(showError) });
  it.push({ g: 'Do', t: 'Toggle theme', icon: I.sun, fn: () => A['theme-toggle']() });
  return it;
}
function openPalette() {
  closePalette(); PAL = { items: paletteItems(), sel: 0, list: [] };
  const p = document.createElement('div'); p.id = 'pal';
  p.innerHTML = `<div class="palbox"><input placeholder="Type a command, engine or page…" id="palin" autocomplete="off" spellcheck="false"><div class="pallist" id="pallist"></div></div>`;
  p.addEventListener('mousedown', e => { if (e.target === p) closePalette(); });
  document.body.appendChild(p);
  const inp = $('#palin'); inp.focus();
  const draw = () => {
    const q = inp.value.toLowerCase().trim();
    const score = t => { t = t.toLowerCase(); if (!q) return 1; if (t.includes(q)) return 3 - t.indexOf(q) / 100; let i = 0; for (const c of t) if (c === q[i]) i++; return i === q.length ? 1 : 0; };
    PAL.list = PAL.items.map(x => ({ ...x, s: score(x.t + ' ' + x.g) })).filter(x => x.s > 0).sort((a, b) => b.s - a.s).slice(0, 40);
    PAL.sel = clamp(PAL.sel, 0, Math.max(0, PAL.list.length - 1));
    $('#pallist').innerHTML = PAL.list.map((x, i) => `<div class="pi${i === PAL.sel ? ' on' : ''}" data-pi="${i}">${x.icon}<span class="g">${esc(x.g)}</span><span>${esc(x.t)}</span></div>`).join('') || '<div class="faint" style="padding:12px">No match</div>';
    const on = $('#pallist .on'); if (on) on.scrollIntoView({ block: 'nearest' });
  };
  inp.addEventListener('input', () => { PAL.sel = 0; draw(); });
  inp.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { PAL.sel++; draw(); e.preventDefault(); }
    if (e.key === 'ArrowUp') { PAL.sel--; draw(); e.preventDefault(); }
    if (e.key === 'Enter') { const x = PAL.list[PAL.sel]; closePalette(); if (x) x.fn(); }
  });
  $('#pallist').addEventListener('click', e => { const el = e.target.closest('[data-pi]'); if (!el) return; const x = PAL.list[+el.dataset.pi]; closePalette(); x.fn(); });
  draw();
}
function closePalette() { const p = $('#pal'); if (p) p.remove(); }

/* ── boot ───────────────────────────────────────────────── */
(async function init() {
  document.documentElement.dataset.theme = store.get('theme', 'dark');
  document.documentElement.dataset.accent = store.get('accent', 'signal');
  shell();
  try {
    const b = await api('/api/bootstrap');
    S.version = b.version; S.settings = b.settings; S.specs = b.specs; S.bps = b.blueprints; S.stages = b.stages || [];
    S.snap = b.snapshot; S.confM = b.snapshot.conf_m; S.events = b.events || [];
    for (const j of b.jobs || []) S.jobs[j.id] = j;
    for (const [n, arr] of Object.entries((b.history || {}).engines || {})) S.hist.e[n] = arr;
    for (const [n, arr] of Object.entries((b.history || {}).hosts || {})) S.hist.h[n] = arr;
  } catch (e) { $('#view .page').innerHTML = `<div class="empty"><b>Could not load the console</b>${esc(e.message)}</div>`; return; }
  onRoute();
  updateTray();
  connect();
})();

</script></body></html>
'''


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
