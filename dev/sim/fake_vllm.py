"""Simulated vLLM / llama.cpp OpenAI server with a realistic boot log, crash scenarios, metrics and streaming."""
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

args = sys.argv[1:]
listen = int(args[args.index("--listen") + 1])
rest = args[args.index("--listen") + 2:]
image = os.environ.get("FAKE_IMAGE", "")
llama = "llama" in image and "vllm" not in image
SCALE = float(os.environ.get("FAKE_BOOT_SCALE", "1"))


def opt(flag, default=None):
    if flag in rest:
        i = rest.index(flag)
        return rest[i + 1] if i + 1 < len(rest) else default
    return default


if llama:
    model = opt("-hf", "unknown/gguf")
    served = opt("--alias") or model
    max_len = int(opt("-c", "4096"))
    util = 0.2
else:
    model = rest[0] if rest and not rest[0].startswith("-") else opt("--model", "unknown/model")
    served = opt("--served-model-name") or model
    max_len = int(opt("--max-model-len", "8192"))
    util = float(opt("--gpu-memory-utilization", "0.9"))

need_kb = int(os.environ.get("FAKE_NEED_KB", "0"))
free_kb = int(os.environ.get("FAKE_FREE_KB", "999999999"))
total_kb = int(os.environ.get("FAKE_TOTAL_KB", "124000000"))
state = {"ready": False, "gen": 0.0, "prompt": 0.0, "ok": 0, "ttft_sum": 0.0, "ttft_n": 0, "e2e_sum": 0.0, "e2e_n": 0,
         "running": 0, "waiting": 0, "prefix_hits": 0, "prefix_q": 0, "acc": 0, "draft": 0}
lock = threading.Lock()


def log(s):
    ts = time.strftime("%m-%d %H:%M:%S")
    print(f"INFO {ts} [api_server.py:1] {s}", flush=True)


def warn(s):
    print(f"WARNING {time.strftime('%m-%d %H:%M:%S')} [core.py:1] {s}", flush=True)


def crash(lines, code=1):
    for ln in lines:
        print(ln, flush=True)
    sys.exit(code)


def sleep(s):
    time.sleep(s * SCALE)


def traceback_head():
    print("Traceback (most recent call last):", flush=True)
    print('  File "/usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/api_server.py", line 1971, in <module>', flush=True)


def boot():
    if llama:
        print(f"common_download_file_single: downloading from https://huggingface.co/{model} ...", flush=True)
        for p in (10, 40, 80, 100):
            print(f"downloading {model.split('/')[-1]}.gguf: {p}%|", flush=True)
            sleep(0.3)
        print("llama_model_load: loading model tensors, this can take a while...", flush=True)
        print("load_tensors: offloaded 37/37 layers to GPU", flush=True)
        sleep(0.8)
        print("main: model loaded", flush=True)
        print("main: server is listening on http://0.0.0.0:8000 - starting the main loop", flush=True)
        state["ready"] = True
        return
    log(f"vLLM API server version 0.27.1")
    log(f"non-default args: {{'model': '{model}', 'max_model_len': {max_len}, 'gpu_memory_utilization': {util}}}")
    if "--foo" in rest or "badarg" in model:
        crash(["usage: vllm serve [model_tag] [options]", "vllm serve: error: unrecognized arguments: --foo"], 2)
    if "gated" in model:
        traceback_head()
        crash([f"huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: Root=1-abc)",
               f"Cannot access gated repo for url https://huggingface.co/{model}/resolve/main/config.json.",
               f"Access to model {model} is restricted and you are not in the authorized list. Visit https://huggingface.co/{model} to ask for access."])
    if "missing" in model:
        traceback_head()
        crash([f"OSError: {model} is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'"])
    if "gemma-4" in model and "tf5141" not in image:
        traceback_head()
        crash(["transformers.configuration_utils.AmbiguousGlobalPerLayerAttributeError: head_dim is defined per-layer and globally; ambiguous per-layer attribute"])
    if "remote-code" in model and "--trust-remote-code" not in rest:
        traceback_head()
        crash([f"ValueError: The repository {model} contains custom code which must be executed to correctly load the model. Please pass the argument `trust_remote_code=True` to allow custom code to be run."])
    # download (first run only)
    cache = Path(os.environ.get("FAKE_CACHE_MOUNT") or "/nonexistent")
    folder = cache / "hub" / ("models--" + model.replace("/", "--"))
    if cache.exists() and not folder.exists():
        (folder / "blobs").mkdir(parents=True, exist_ok=True)
        snap = folder / "snapshots" / "main"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text(json.dumps({"architectures": ["FakeForCausalLM"], "num_hidden_layers": 40, "num_attention_heads": 32,
                                                     "num_key_value_heads": 8, "hidden_size": 4096, "max_position_embeddings": 131072}))
        shards = 4
        for s in range(1, shards + 1):
            with open(folder / "blobs" / f"shard{s}", "wb") as fh:
                fh.truncate(int(util * total_kb * 1024 * 0.4 / shards))
            for p in (0, 35, 70, 100):
                gb = 4.6 * p / 100
                print(f"model-0000{s}-of-0000{shards}.safetensors: {p}%|{'█' * (p // 10):<10}| {gb:.2f}G/4.60G [00:0{s}<00:03, 812MB/s]", flush=True)
                sleep(0.15)
    # memory check, like vLLM's worker does
    if need_kb and need_kb > free_kb:
        traceback_head()
        crash([f"ValueError: Free memory on device ({free_kb/1024/1024:.2f}/{total_kb/1024/1024:.2f} GiB) on startup is less than desired GPU memory utilization ({util}, {need_kb/1024/1024:.2f} GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes."])
    if "crash" in model:
        sleep(0.5)
        traceback_head()
        crash(["torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB. GPU 0 has a total capacity of 118.26 GiB"])
    n = 11
    log("Starting to load model " + model + "...")
    for i in range(1, n + 1):
        print(f"Loading safetensors checkpoint shards: {int(i/n*100):3d}% Completed | {i}/{n} [00:0{i%10}<00:00,  1.9it/s]", flush=True)
        sleep(0.18)
    weights = util * total_kb / 1024 / 1024 * 0.45
    log(f"Model loading took {weights:.2f} GiB and 4.21 seconds")
    kv_tokens = int((util * total_kb / 1024 / 1024 - weights - 2.5) * 1024 * 1024 * 1024 / (48 * 1024))
    if max_len > 262144:
        traceback_head()
        crash([f"ValueError: To serve at least one request with the models's max seq len ({max_len}), (40.00 GiB KV cache is needed, which is larger than the available KV cache memory (12.00 GiB). Based on the available memory, the estimated maximum model length is 98304. Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine."])
    log("torch.compile takes 6.12 s in total")
    sleep(0.6)
    log(f"GPU KV cache size: {kv_tokens:,} tokens")
    log(f"Maximum concurrency for {max_len:,} tokens per request: {kv_tokens/max_len:.2f}x")
    for p in (0, 25, 50, 75, 100):
        print(f"Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): {p:3d}%|{'█'*(p//10):<10}| {p//2}/51 [00:01<00:00]", flush=True)
        sleep(0.2)
    log("Graph capturing finished in 3 secs, took 0.61 GiB")
    log(f"Starting vLLM API server 0 on http://0.0.0.0:8000")
    print("INFO:     Started server process [1]", flush=True)
    print("INFO:     Application startup complete.", flush=True)
    state["kv_tokens"] = kv_tokens
    state["ready"] = True


TPS = float(os.environ.get("FAKE_TPS", "55"))
WORDS = ("The engine streams tokens from unified memory while the scheduler batches requests and the KV cache keeps "
         "earlier context close at hand so each new token costs only one decode step on the GPU").split()


def gen_tokens(n):
    for i in range(n):
        yield WORDS[i % len(WORDS)] + " "


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
        if self.path == "/health":
            if state["ready"]:
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            else:
                self.js({"error": "starting"}, 503)
            return
        if not state["ready"]:
            self.js({"error": "starting"}, 503)
            return
        if self.path.startswith("/v1/models"):
            self.js({"object": "list", "data": [{"id": served, "object": "model", "owned_by": "vllm", "max_model_len": max_len}]})
            return
        if self.path == "/metrics":
            with lock:
                s = dict(state)
            if llama:
                text = (f"llamacpp:prompt_tokens_total {s['prompt']}\nllamacpp:tokens_predicted_total {s['gen']}\n"
                        f"llamacpp:requests_processing {s['running']}\nllamacpp:requests_deferred {s['waiting']}\n")
            else:
                lab = f'model_name="{served}"'
                kvp = min(0.99, s["running"] * 0.07 + random.random() * 0.02)
                text = "\n".join([
                    "# HELP vllm:num_requests_running Number of requests in model execution batches.",
                    f"vllm:num_requests_running{{{lab}}} {s['running']}",
                    f"vllm:num_requests_waiting{{{lab}}} {s['waiting']}",
                    f"vllm:kv_cache_usage_perc{{{lab}}} {kvp}",
                    f"vllm:prompt_tokens_total{{{lab}}} {s['prompt']}",
                    f"vllm:generation_tokens_total{{{lab}}} {s['gen']}",
                    f'vllm:request_success_total{{finished_reason="stop",{lab}}} {s["ok"]}',
                    f'vllm:request_success_total{{finished_reason="length",{lab}}} 0',
                    f"vllm:time_to_first_token_seconds_sum{{{lab}}} {s['ttft_sum']}",
                    f"vllm:time_to_first_token_seconds_count{{{lab}}} {s['ttft_n']}",
                    f'vllm:time_to_first_token_seconds_bucket{{le="0.1",{lab}}} 0',
                    f"vllm:e2e_request_latency_seconds_sum{{{lab}}} {s['e2e_sum']}",
                    f"vllm:e2e_request_latency_seconds_count{{{lab}}} {s['e2e_n']}",
                    f"vllm:prefix_cache_hits_total{{{lab}}} {s['prefix_hits']}",
                    f"vllm:prefix_cache_queries_total{{{lab}}} {s['prefix_q']}",
                    f"vllm:spec_decode_num_accepted_tokens_total{{{lab}}} {s['acc']}",
                    f"vllm:spec_decode_num_draft_tokens_total{{{lab}}} {s['draft']}",
                    f'vllm:cache_config_info{{block_size="16",num_gpu_blocks="{int(s.get("kv_tokens", 160000))//16}",{lab}}} 1.0',
                ]) + "\n"
            raw = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.js({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not state["ready"]:
            self.js({"error": "starting"}, 503)
            return
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
            self.js({"error": "not found"}, 404)
            return
        if body.get("model") != served:
            self.js({"error": {"message": f"The model `{body.get('model')}` does not exist.", "type": "NotFoundError", "code": 404}}, 404)
            return
        max_tokens = int(body.get("max_tokens") or 64)
        prompt_text = json.dumps(body.get("messages") or body.get("prompt") or "")
        ptoks = max(4, len(prompt_text) // 4)
        with lock:
            state["running"] += 1
            conc = state["running"]
            state["prefix_q"] += ptoks
            state["prefix_hits"] += int(ptoks * 0.6)
        tps = TPS / (1 + 0.12 * (conc - 1))
        t0 = time.time()
        ttft = 0.08 + 0.00002 * ptoks + random.random() * 0.04
        n_out = max_tokens if body.get("ignore_eos") else random.randint(max(1, min(4, max_tokens), max_tokens // 2), max(1, max_tokens))
        if "single word" in prompt_text:
            n_out = 1
        try:
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def send(obj):
                    data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                    self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()
                time.sleep(ttft)
                i = 0
                for tok in gen_tokens(n_out):
                    piece = "OK" if n_out == 1 else tok
                    send({"id": "chatcmpl-x", "object": "chat.completion.chunk", "model": served,
                          "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
                    with lock:
                        state["gen"] += 1
                        state["draft"] += 3
                        state["acc"] += 2
                    i += 1
                    time.sleep(1 / tps)
                send({"id": "chatcmpl-x", "object": "chat.completion.chunk", "model": served, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
                if (body.get("stream_options") or {}).get("include_usage"):
                    send({"id": "chatcmpl-x", "object": "chat.completion.chunk", "model": served, "choices": [],
                          "usage": {"prompt_tokens": ptoks, "completion_tokens": n_out, "total_tokens": ptoks + n_out}})
                send("[DONE]")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                time.sleep(ttft + n_out / tps)
                text = "OK" if n_out == 1 else "".join(gen_tokens(n_out))
                with lock:
                    state["gen"] += n_out
                self.js({"id": "chatcmpl-x", "object": "chat.completion", "model": served,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                         "usage": {"prompt_tokens": ptoks, "completion_tokens": n_out, "total_tokens": ptoks + n_out}})
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with lock:
                state["running"] -= 1
                state["prompt"] += ptoks
                state["ok"] += 1
                state["ttft_sum"] += ttft
                state["ttft_n"] += 1
                state["e2e_sum"] += time.time() - t0
                state["e2e_n"] += 1


srv = ThreadingHTTPServer(("127.0.0.1", listen), H)
srv.daemon_threads = True
threading.Thread(target=srv.serve_forever, daemon=True).start()
try:
    boot()
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    pass
