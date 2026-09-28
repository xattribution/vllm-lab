"""Shared state for the vllm-lab simulator (fake docker / nvidia-smi / procfs)."""
import contextlib
import fcntl
import json
import os
import time
from pathlib import Path

SIM = Path(os.environ.get("FAKE_SIM_HOME", str(Path.home() / ".fake-lab")))
STATE = SIM / "docker.json"
PROC = Path(os.environ.get("VLLM_LAB_PROC", str(SIM / "proc")))
LOGS = SIM / "logs"
TOTAL_KB = 124_000_000          # ~118 GiB visible, like a GB10
BASE_USED_KB = 7_500_000        # OS + desktop
GIB = 1024 ** 3


def ensure():
    SIM.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    (PROC / "sys/vm").mkdir(parents=True, exist_ok=True)
    dc = PROC / "sys/vm/drop_caches"
    if not dc.exists():
        dc.write_text("0\n")
    for name, text in (("loadavg", "0.42 0.51 0.60 1/812 4242\n"), ("uptime", "86400.00 1000000.00\n")):
        p = PROC / name
        if not p.exists():
            p.write_text(text)


@contextlib.contextmanager
def locked():
    ensure()
    fd = os.open(str(SIM / ".lock"), os.O_RDWR | os.O_CREAT)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        st = json.loads(STATE.read_text()) if STATE.exists() else {}
        st.setdefault("containers", {})
        st.setdefault("images", {
            "vllm/vllm-openai:v0.27.1": {"id": "sha256:1a2b3c4d5e6f", "size": "18.2GB"},
            "ghcr.io/open-webui/open-webui:main": {"id": "sha256:9f8e7d6c5b4a", "size": "4.1GB"},
        })
        st.setdefault("networks", {"bridge": {"driver": "bridge", "gateway": "172.17.0.1"}, "host": {"driver": "host"}, "none": {"driver": "null"}})
        st.setdefault("pagecache_kb", 9_000_000)
        st.setdefault("drop_mtime", 0)
        yield st
        write_meminfo(st)
        tmp = STATE.with_suffix(".tmp")
        tmp.write_text(json.dumps(st, indent=1))
        os.replace(tmp, STATE)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return False


def reservations_kb(st, exclude: str = "") -> int:
    total = 0
    for c in st["containers"].values():
        if c["name"] == exclude or c["state"] != "running" or not alive(c.get("pid")):
            continue
        total += int(c.get("reserve_kb") or 0)
    return total


def write_meminfo(st):
    dc = PROC / "sys/vm/drop_caches"
    with contextlib.suppress(OSError):
        m = dc.stat().st_mtime
        if m > st.get("drop_mtime", 0) and dc.read_text().strip() == "3":
            st["drop_mtime"] = m
            st["pagecache_kb"] = 400_000
    res = reservations_kb(st)
    cache = int(st.get("pagecache_kb") or 0)
    free = max(200_000, TOTAL_KB - BASE_USED_KB - res - cache)
    avail = free + int(cache * 0.95)
    PROC.mkdir(parents=True, exist_ok=True)
    (PROC / "meminfo").write_text(
        f"MemTotal:       {TOTAL_KB} kB\nMemFree:        {free} kB\nMemAvailable:   {avail} kB\n"
        f"Buffers:          120000 kB\nCached:         {cache} kB\nSwapCached:            0 kB\n"
        f"SwapTotal:      16000000 kB\nSwapFree:       16000000 kB\nSReclaimable:     300000 kB\n")


def free_kb(st, exclude: str = "") -> int:
    return max(0, TOTAL_KB - BASE_USED_KB - reservations_kb(st, exclude) - int(st.get("pagecache_kb") or 0))


def now_iso(t=None) -> str:
    t = t or time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1e6):06d}Z"
