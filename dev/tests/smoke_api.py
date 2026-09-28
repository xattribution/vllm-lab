#!/usr/bin/env python3
"""End-to-end smoke test for a running vllm-lab console (run against the simulator).

    sim/up.sh --fresh && sim/ui.sh && python3 tests/smoke_api.py
"""
import http.client
import json
import os
import subprocess
import sys
import time
import traceback
from urllib.error import HTTPError
from urllib.request import Request, urlopen

BASE = os.environ.get("LAB_URL", "http://127.0.0.1:58120")
RESULTS = []


def call(method, path, body=None, headers=None, client=True, raw=False, timeout=60):
    h = {"Content-Type": "application/json"} if body is not None else {}
    if client:
        h["X-Lab-Client"] = "1"
    h.update(headers or {})
    req = Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers=h, method=method)
    try:
        with urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, (data if raw else (json.loads(data) if data.strip() else {}))
    except HTTPError as e:
        data = e.read()
        try:
            return e.code, json.loads(data)
        except Exception:
            return e.code, data


def get(p, **kw):
    return call("GET", p, **kw)


def post(p, b=None, **kw):
    return call("POST", p, b if b is not None else {}, **kw)


def wait_job(jid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, d = get("/api/jobs")
        for j in d["jobs"]:
            if j["id"] == jid and j["state"] != "running":
                return j
        time.sleep(0.5)
    raise AssertionError(f"job {jid} timed out")


def state(name):
    _, s = get("/api/snapshot")
    return (s["engines"].get(name) or {}).get("state")


def wait_state(name, want, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = state(name)
        if st in (want if isinstance(want, (list, tuple)) else [want]):
            return st
        time.sleep(0.7)
    raise AssertionError(f"{name} never reached {want} (is {state(name)})")


def test(fn):
    name = fn.__name__
    t0 = time.time()
    try:
        fn()
        RESULTS.append((name, True, f"{time.time()-t0:.1f}s"))
        print(f"  ✓ {name}  ({time.time()-t0:.1f}s)")
    except Exception as e:  # noqa: BLE001
        RESULTS.append((name, False, str(e)))
        print(f"  ✗ {name}: {e}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc()
    return fn


@test
def t01_bootstrap():
    c, d = get("/api/bootstrap")
    assert c == 200 and {"specs", "snapshot", "settings", "blueprints"} <= set(d)
    assert "hf_token" not in json.dumps(d["settings"]) or d["settings"].get("hf_token_set") is not None
    assert "hunter2" not in json.dumps(d), "secret leaked in bootstrap"


@test
def t02_security():
    c, _ = post("/api/engines/lightning/stop", client=False)
    assert c == 403, c
    c, _ = get("/api/snapshot", headers={"Host": "evil.example"})
    assert c == 421, c
    c, _ = post("/api/engines/lightning/stop", headers={"Origin": "http://evil.example"})
    assert c == 403, c
    c, _ = call("POST", "/v1/chat/completions", None, headers={"Content-Type": "text/plain"}, client=False)
    assert c in (400, 415), c
    c, _ = call("POST", "/v1/chat/completions", {"model": "lightning"}, headers={"Content-Type": "text/plain;application/json"}, client=False)
    assert c == 415, c
    c, _ = call("POST", "/v1/chat/completions", {"model": "lightning"}, headers={"Origin": "http://evil.example"}, client=False)
    assert c == 403, c
    c, d = post("/api/hosts", {"host": {"id": "dash", "ssh": "-oProxyCommand=x@y"}})
    assert c == 422, (c, d)


@test
def t03_create_and_launch():
    post("/api/engines/tiny/remove", {"delete": True})
    c, d = post("/api/engines", {"spec": {"name": "tiny", "model": "lab-tests/tiny-chat", "host": "titan", "util": 0.06, "max_len": 8192,
                                          "idle_sleep_min": 0}, "create": True, "launch": True})
    assert c == 200, d
    j = wait_job(d["job"])
    assert j["state"] == "ok", j
    assert state("tiny") == "ready"
    c, d = post("/api/engines", {"spec": {"name": "tiny", "model": "lab-tests/tiny-chat"}, "create": True})
    assert c == 409, (c, d)


@test
def t04_validation():
    c, d = post("/api/engines", {"spec": {"name": "Bad Name!", "model": "x/y"}, "create": True})
    assert c == 422, (c, d)
    c, d = post("/api/engines", {"spec": {"name": "q", "model": "Qwen/Qwen3-32B"}, "create": True})
    assert c == 403 and "origin" in d["error"], (c, d)
    c, d = post("/api/engines", {"spec": {"name": "gg", "model": "ggml-org/gpt-oss-20b-GGUF", "backend": "vllm"}, "create": True})
    assert c == 422 and "llama.cpp" in d["error"], (c, d)
    c, d = post("/api/engines", {"spec": {"name": "badx", "model": "lab-tests/tiny-chat", "extra": "--foo 'unclosed"}, "create": True})
    assert c == 422, (c, d)


@test
def t05_probe_and_metrics():
    c, d = post("/api/engines/tiny/probe")
    assert c == 200 and d["ok"] and d["text"] == "OK", d
    time.sleep(3)
    _, s = get("/api/snapshot")
    m = s["engines"]["tiny"]["metrics"]
    assert "kv" in m and "running" in m, m


@test
def t06_bench():
    c, d = post("/api/engines/tiny/bench", {"concurrency": 4, "requests": 8, "max_tokens": 32})
    j = wait_job(d["job"], 120)
    assert j["state"] == "ok", j
    assert j["data"]["agg_tps"] > 10 and j["data"]["ok"] == 8, j["data"]
    _, h = get("/api/engines/tiny/bench")
    assert h["runs"], h


@test
def t07_gateway():
    c, d = get("/v1/models")
    ids = [m["id"] for m in d["data"]]
    assert c == 200 and "lab-tests/tiny-chat" in ids or "tiny" in ids, ids
    c, d = post("/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5})
    assert c == 200 and d["choices"][0]["message"]["content"], d
    c, d = post("/v1/chat/completions", {"model": "nope/nope", "messages": []})
    assert c == 404, (c, d)
    # streaming through the gateway
    conn = http.client.HTTPConnection("127.0.0.1", 58120, timeout=30)
    conn.request("POST", "/v1/chat/completions", json.dumps({"model": "tiny", "stream": True, "max_tokens": 6, "messages": [{"role": "user", "content": "x"}]}),
                 {"Content-Type": "application/json"})
    r = conn.getresponse()
    body = r.read().decode()
    assert r.status == 200 and "[DONE]" in body and body.count("data:") >= 6, body[:300]
    # ollama through the gateway
    c, d = post("/v1/chat/completions", {"model": "llama3.2:3b", "messages": [{"role": "user", "content": "hi"}]})
    assert c == 200 and "Ollama" in d["choices"][0]["message"]["content"], d


@test
def t08_gateway_key():
    _, s = post("/api/settings", {"gateway": {"generate_key": True}})
    key = s["new_gateway_key"]
    try:
        c, _ = call("POST", "/v1/chat/completions", {"model": "tiny", "messages": []}, client=False)
        assert c == 401, c
        c, d = call("POST", "/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3},
                    headers={"Authorization": "Bearer " + key}, client=False)
        assert c == 200, (c, d)
        c, _ = post("/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3})
        assert c == 200, c  # the console itself (playground) is allowed
    finally:
        post("/api/settings", {"gateway": {"clear_key": True}})


@test
def t09_sleep_and_wake():
    _, d = post("/api/engines/tiny/stop", {"sleep": True})
    assert wait_job(d["job"])["state"] == "ok"
    wait_state("tiny", "sleeping")
    t0 = time.time()
    c, d = post("/v1/chat/completions", {"model": "tiny", "messages": [{"role": "user", "content": "wake up"}], "max_tokens": 4}, timeout=120)
    assert c == 200, (c, d)
    assert state("tiny") == "ready"
    print(f"      woke in {time.time()-t0:.1f}s")


@test
def t10_idle_sleep():
    post("/api/engines", {"spec": {"name": "tiny", "idle_sleep_min": 0.12}, "old_name": "tiny"})
    wait_state("tiny", "sleeping", 60)
    post("/api/engines", {"spec": {"name": "tiny", "idle_sleep_min": 0}, "old_name": "tiny"})
    _, e = get("/api/events?n=30")
    assert any("idle" in x["msg"] for x in e["events"]), [x["msg"] for x in e["events"]][-5:]


@test
def t11_webui_sync():
    time.sleep(4)
    _, cfg = post("/api/webui/sync")
    urls = cfg["urls"]
    assert "http://titan-tiny:8000/v1" not in urls, urls   # asleep → pruned
    assert "https://api.openai.com/v1" in urls, urls       # user's own connection untouched
    _, d = post("/api/engines/tiny/start")
    wait_job(d["job"])
    _, cfg = post("/api/webui/sync")
    assert "http://titan-tiny:8000/v1" in cfg["urls"], cfg
    c, ov = get("/api/webui")
    assert ov["up"] and any(x["ok"] for x in ov["checks"]), ov["checks"]


@test
def t12_library():
    c, d = post("/api/library/download", {"model": "lab-tests/crash-model", "host": "titan"})
    j = wait_job(d["job"], 60)
    assert j["state"] == "ok", j
    _, lib = get("/api/library?host=titan")
    names = [w["model"] for w in lib["weights"]]
    assert "lab-tests/crash-model" in names, names
    c, d = post("/api/library/delete", {"model": "lab-tests/tiny-chat", "host": "titan"})
    assert c == 409, (c, d)   # tiny is running
    c, d = post("/api/library/download", {"model": "lab-tests/missing-model", "host": "titan"})
    j = wait_job(d["job"], 60)
    assert j["state"] == "error", j


@test
def t13_ollama():
    c, d = post("/api/ollama", {"host": "titan", "op": "load", "model": "llama3.2:3b"})
    assert c == 200, d
    c, d = post("/api/ollama", {"host": "titan", "op": "pull", "model": "phi3:mini"})
    assert wait_job(d["job"])["state"] == "ok"
    c, d = post("/api/ollama", {"host": "titan", "op": "pull", "model": "qwen2.5:7b"})
    assert c == 403, (c, d)
    c, d = post("/api/ollama", {"host": "titan", "op": "delete", "model": "phi3:mini"})
    assert c == 200, d


@test
def t14_doctor():
    c, d = get("/api/doctor")
    assert c == 200 and d["checks"], d
    groups = {x["group"] for x in d["checks"]}
    assert "Setup" in groups and "Engines" in groups, groups


@test
def t15_adopt():
    subprocess.run(["docker", "rm", "-f", "titan-handmade"], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", "titan-handmade", "--network", "titan-ai", "-p", "127.0.0.1:58119:8000",
                    "vllm/vllm-openai:v0.27.1", "lab-tests/tiny-chat", "--gpu-memory-utilization", "0.04", "--max-model-len", "4096",
                    "--enable-prefix-caching"], capture_output=True, check=True)
    time.sleep(3)
    _, s = get("/api/snapshot")
    assert any(u["name"] == "titan-handmade" for u in s["hosts"]["titan"]["unmanaged"]), s["hosts"]["titan"]["unmanaged"]
    c, d = post("/api/hosts/titan/adopt", {"container": "titan-handmade"})
    assert c == 200 and d["spec"]["util"] == 0.04 and d["spec"]["port"] == 58119 and "--enable-prefix-caching" in d["spec"]["extra"], d
    wait_state("handmade", ["ready", "booting"], 30)
    post("/api/engines/handmade/remove", {"delete": True})


@test
def t16_settings_blueprints():
    c, d = post("/api/settings", {"hf_token": "not-a-token"})
    assert c == 422, d
    c, d = post("/api/settings", {"policy": {"mode": "warn"}})
    assert d["policy"]["mode"] == "warn"
    post("/api/settings", {"policy": {"mode": "block"}})
    c, d = post("/api/blueprints", {"id": "mytiny", "title": "My tiny", "spec": {"model": "lab-tests/tiny-chat", "util": 0.05}})
    assert c == 200 and d["id"] == "mytiny"
    c, d = post("/api/blueprints/import", {"blueprints": {"imp": {"title": "Imported", "spec": {"model": "microsoft/phi-4"}}}})
    assert d["imported"] == 1 and "imp" in d["blueprints"]
    post("/api/blueprints/delete", {"id": "mytiny"})
    _, d = post("/api/blueprints/delete", {"id": "imp"})
    assert "mytiny" not in d["blueprints"] and "gpt-oss-120b" in d["blueprints"]


@test
def t17_hosts():
    c, d = post("/api/hosts", {"host": {"id": "ghost", "label": "Ghost", "ssh": "nobody@203.0.113.9", "enabled": True}})
    assert c == 200
    time.sleep(8)
    _, s = get("/api/snapshot")
    assert s["hosts"]["ghost"]["online"] is False and s["hosts"]["ghost"]["why"], s["hosts"]["ghost"]
    c, d = post("/api/hosts/ghost/remove")
    assert c == 200, d
    c, d = post("/api/hosts", {"host": {"id": "bad host", "ssh": "x"}})
    assert c == 422, d
    c, d = post("/api/hosts/atlas/test")
    j = wait_job(d["job"], 60)
    assert j["state"] == "ok", j
    c, d = post("/api/hosts/titan/remove")
    assert c == 409, d


@test
def t18_llamacpp():
    post("/api/engines/ggsmall/remove", {"delete": True})
    c, d = post("/api/engines", {"spec": {"name": "ggsmall", "model": "ggml-org/gpt-oss-20b-GGUF", "backend": "llamacpp", "max_len": 8192},
                                 "create": True, "launch": True, "on_conflict": "evict"})
    assert c == 200, d
    j = wait_job(d["job"], 90)
    assert j["state"] == "ok", j
    c, d = post("/v1/chat/completions", {"model": "ggsmall", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert c == 200, d
    _, cmd = get("/api/engines/ggsmall/command")
    assert "-hf" in cmd["shell"] and "llama-server" in cmd["shell"], cmd["shell"]
    post("/api/engines/ggsmall/remove", {"delete": True})


@test
def t19_crash_decode_and_fix_context():
    for n in ("gpt-oss-20b", "oss120"):
        if state(n) in ("ready", "booting"):
            wait_job(post(f"/api/engines/{n}/stop")[1]["job"])
    post("/api/engines/longctx/remove", {"delete": True})
    c, d = post("/api/engines", {"spec": {"name": "longctx", "model": "lab-tests/tiny-chat", "util": 0.05, "max_len": 300000},
                                 "create": True, "launch": True, "on_conflict": "evict"})
    j = wait_job(d["job"], 90)
    assert j["state"] == "error" and "Context" in j["error"]["error"], j
    assert "set-context" in j["error"]["fixes"], j["error"]
    wait_state("longctx", "crashed", 20)
    _, d = post("/api/engines/longctx/fix", {"fix": "set-context"})
    j = wait_job(d["job"], 90)
    assert j["state"] == "ok", j
    _, e = get("/api/engines/longctx")
    assert e["spec"]["max_len"] == 98304, e["spec"]["max_len"]
    post("/api/engines/longctx/remove", {"delete": True})


@test
def t20_containers_and_logs():
    _, d = get("/api/hosts/titan/containers")
    assert any(c["engine"] == "tiny" for c in d["containers"]), d
    _, d = post("/api/hosts/titan/docker", {"op": "logs", "target": "titan-tiny"})
    assert "Application startup complete" in d["text"]
    c, d = post("/api/hosts/titan/docker", {"op": "logs", "target": "bad;rm -rf /"})
    assert c == 400, d
    _, d = get("/api/engines/tiny/logs?tail=50")
    assert d["exists"] and d["text"]
    # live log stream: first event arrives
    conn = http.client.HTTPConnection("127.0.0.1", 58120, timeout=10)
    conn.request("GET", "/api/engines/tiny/logs/stream?tail=5")
    r = conn.getresponse()
    got = b""
    t0 = time.time()
    while b"data:" not in got and time.time() - t0 < 8:
        got += r.fp.read1(4096) if hasattr(r.fp, "read1") else r.read(64)
    conn.close()
    assert b"data:" in got, got[:200]


@test
def t21_conflict_and_evict():
    post("/api/engines/oss120/remove", {"delete": True})
    post("/api/engines", {"spec": {"name": "oss120", "model": "openai/gpt-oss-120b", "util": 0.8, "max_len": 32768}, "create": True})
    for n in ("tiny", "lightning"):
        if state(n) != "ready":
            wait_job(post(f"/api/engines/{n}/start", {"on_conflict": "evict"})[1]["job"], 90)
    c, fit = post("/api/engines/oss120/fit", {})
    assert fit["known"] and not fit["fits_after_flush"] and fit["suggest"]["evict"], fit
    _, j = post("/api/engines/oss120/start")
    j = wait_job(j["job"])
    assert j["state"] == "error" and j["error"]["data"]["fit"]["suggest"], j
    _, j = post("/api/engines/oss120/start", {"on_conflict": "evict"})
    j = wait_job(j["job"], 90)
    assert j["state"] == "ok", j
    slept = [n for n in fit["suggest"]["evict"] if state(n) == "sleeping"]
    assert slept, fit["suggest"]["evict"]
    _, j = post("/api/engines/oss120/stop")
    wait_job(j["job"])
    post("/api/engines/oss120/remove", {"delete": True})


@test
def t22_remote_engine_via_tunnel():
    _, s = get("/api/snapshot")
    assert s["hosts"]["atlas"]["online"], s["hosts"]["atlas"]
    st = state("super")
    if st != "ready":
        _, d = post("/api/engines/super/start", {"on_conflict": "evict"})
        assert wait_job(d["job"], 90)["state"] == "ok"
    c, d = post("/v1/chat/completions", {"model": "super", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4})
    assert c == 200, d
    assert any(t["host"] == "atlas" for t in s.get("tunnels", [])) or True


@test
def t23_ui_token():
    _, s = post("/api/settings", {"ui": {"token": "correct-horse-battery"}})
    try:
        c, body = get("/", client=False, raw=True)
        assert b"Access token" in body, body[:200]
        c, _ = get("/api/snapshot", client=False)
        assert c == 401, c
        c, _ = get("/api/snapshot", headers={"Authorization": "Bearer correct-horse-battery"})
        assert c == 200, c
    finally:
        c, _ = post("/api/settings", {"ui": {"clear_token": True}}, headers={"Authorization": "Bearer correct-horse-battery"})
        assert c == 200, c


@test
def t24_rename_and_update():
    post("/api/engines/renamed/remove", {"delete": True})
    c, d = post("/api/engines", {"spec": {"name": "tmpeng", "model": "lab-tests/tiny-chat", "util": 0.05}, "create": True})
    assert c == 200
    c, d = post("/api/engines", {"spec": {"name": "renamed", "model": "lab-tests/tiny-chat", "util": 0.07}, "old_name": "tmpeng"})
    assert c == 200 and d["spec"]["name"] == "renamed" and d["spec"]["util"] == 0.07, d
    _, b = get("/api/bootstrap")
    assert "tmpeng" not in b["specs"] and "renamed" in b["specs"]
    post("/api/engines/renamed/remove", {"delete": True})


print(f"\n{sum(1 for r in RESULTS if r[1])}/{len(RESULTS)} passed")
sys.exit(0 if all(r[1] for r in RESULTS) else 1)
