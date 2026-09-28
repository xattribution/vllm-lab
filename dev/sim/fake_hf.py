"""A tiny Hugging Face Hub stand-in: search, model info, file tree, config.json, whoami."""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

G = 1024 ** 3


def st(total, shards, size):
    return total, [{"type": "file", "path": f"model-{i:05d}-of-{shards:05d}.safetensors", "size": 1000, "lfs": {"size": size // shards}} for i in range(1, shards + 1)]


NEMOTRON_H = {"architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "num_hidden_layers": 52, "num_attention_heads": 32,
              "num_key_value_heads": 2, "head_dim": 128, "hidden_size": 2688, "max_position_embeddings": 262144,
              "hybrid_override_pattern": "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME", "mamba_num_heads": 64,
              "mamba_head_dim": 64, "ssm_state_size": 128, "n_groups": 8, "conv_kernel": 4, "torch_dtype": "bfloat16",
              "quantization_config": {"quant_method": "modelopt"}}
MODELS = {
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4": dict(params=31_600_000_000, bytes=19.4 * G, shards=4, config=NEMOTRON_H, downloads=182000, likes=640, tags=["nvfp4", "modelopt", "text-generation", "license:other"]),
    "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DSpark": dict(params=900_000_000, bytes=1.1 * G, shards=1, config={**NEMOTRON_H, "num_hidden_layers": 2, "hybrid_override_pattern": "M*"}, downloads=9000, likes=40, tags=["nvfp4"]),
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4": dict(params=123_000_000_000, bytes=71.5 * G, shards=15, config={**NEMOTRON_H, "num_hidden_layers": 88, "hybrid_override_pattern": ("MEM*" * 22), "hidden_size": 4096}, downloads=64000, likes=410, tags=["nvfp4", "modelopt"]),
    "google/gemma-4-E4B-it": dict(params=8_000_000_000, bytes=15.9 * G, shards=3, gated=True,
                                  config={"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4",
                                          "text_config": {"num_hidden_layers": 42, "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 256,
                                                          "hidden_size": 2560, "sliding_window": 512, "num_kv_shared_layers": 18,
                                                          "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 7,
                                                          "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}}, downloads=540000, likes=1900, tags=["gemma4", "image-text-to-text", "license:gemma"]),
    "openai/gpt-oss-120b": dict(params=120_400_000_000, bytes=65.3 * G, shards=15,
                                config={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "num_hidden_layers": 36, "num_attention_heads": 64,
                                        "num_key_value_heads": 8, "head_dim": 64, "hidden_size": 2880, "sliding_window": 128,
                                        "layer_types": ["sliding_attention", "full_attention"] * 18, "max_position_embeddings": 131072,
                                        "quantization_config": {"quant_method": "mxfp4"}}, downloads=3_100_000, likes=4200, tags=["mxfp4", "vllm", "license:apache-2.0"]),
    "openai/gpt-oss-20b": dict(params=21_500_000_000, bytes=13.8 * G, shards=3,
                               config={"architectures": ["GptOssForCausalLM"], "model_type": "gpt_oss", "num_hidden_layers": 24, "num_attention_heads": 64,
                                       "num_key_value_heads": 8, "head_dim": 64, "hidden_size": 2880, "sliding_window": 128,
                                       "layer_types": ["sliding_attention", "full_attention"] * 12, "max_position_embeddings": 131072}, downloads=5_900_000, likes=3800, tags=["mxfp4"]),
    "meta-llama/Llama-3.1-8B-Instruct": dict(params=8_030_000_000, bytes=16.1 * G, shards=4, gated=True,
                                            config={"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 32, "num_attention_heads": 32,
                                                    "num_key_value_heads": 8, "hidden_size": 4096, "max_position_embeddings": 131072, "torch_dtype": "bfloat16"}, downloads=8_700_000, likes=4800, tags=["llama", "license:llama3.1"]),
    "nvidia/Llama-3.3-70B-Instruct-FP4": dict(params=40_000_000_000, bytes=39.8 * G, shards=9,
                                             config={"architectures": ["LlamaForCausalLM"], "model_type": "llama", "num_hidden_layers": 80, "num_attention_heads": 64,
                                                     "num_key_value_heads": 8, "hidden_size": 8192, "max_position_embeddings": 131072}, downloads=41000, likes=120, tags=["nvfp4", "modelopt", "base_model:meta-llama/Llama-3.3-70B-Instruct"]),
    "mistralai/Mistral-Small-3.2-24B-Instruct-2506": dict(params=24_000_000_000, bytes=48.0 * G, shards=10,
                                                         config={"architectures": ["Mistral3ForConditionalGeneration"], "text_config": {"num_hidden_layers": 40, "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 5120, "max_position_embeddings": 131072}}, downloads=900000, likes=600, tags=["mistral"]),
    "Qwen/Qwen3-32B": dict(params=32_800_000_000, bytes=65.5 * G, shards=17, config={"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 5120, "max_position_embeddings": 40960}, downloads=2_000_000, likes=900, tags=["qwen3"]),
    "someuser/Qwen3-32B-uncensored-NVFP4": dict(params=32_800_000_000, bytes=20.1 * G, shards=5, config={"architectures": ["Qwen3ForCausalLM"], "num_hidden_layers": 64, "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 5120}, downloads=4000, likes=12, tags=["nvfp4", "base_model:Qwen/Qwen3-32B", "base_model:quantized:Qwen/Qwen3-32B"]),
    "ggml-org/gpt-oss-20b-GGUF": dict(params=None, bytes=0, shards=0, gguf=[("gpt-oss-20b-mxfp4.gguf", 12.1 * G), ("gpt-oss-20b-Q4_K_M.gguf", 11.6 * G), ("gpt-oss-20b-Q8_0.gguf", 12.9 * G)], config=None, downloads=700000, likes=300, tags=["gguf"]),
    "microsoft/phi-4": dict(params=14_700_000_000, bytes=29.3 * G, shards=6, config={"architectures": ["Phi3ForCausalLM"], "num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 10, "hidden_size": 5120, "max_position_embeddings": 16384}, downloads=800000, likes=2000, tags=["phi3"]),
    "lab-tests/crash-model": dict(params=1_000_000_000, bytes=2 * G, shards=1, config={"num_hidden_layers": 16, "num_attention_heads": 16, "num_key_value_heads": 4, "hidden_size": 2048}, downloads=1, likes=0, tags=[]),
    "lab-tests/tiny-chat": dict(params=1_200_000_000, bytes=2.4 * G, shards=1, config={"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 16, "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 2048, "max_position_embeddings": 32768}, downloads=12, likes=1, tags=["text-generation"]),
}


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

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        p = u.path
        auth = self.headers.get("Authorization") or ""
        if p == "/api/whoami-v2":
            if auth.startswith("Bearer hf_") and "bad" not in auth:
                self.js({"name": "jared-lab", "orgs": [{"name": "lab"}], "auth": {"accessToken": {"role": "read"}}})
            else:
                self.js({"error": "Invalid credentials in Authorization header"}, 401)
            return
        if p == "/api/models":
            s = (q.get("search") or [""])[0].lower()
            filt = (q.get("filter") or [""])[0]
            rows = []
            for mid, m in MODELS.items():
                if s and s not in mid.lower():
                    continue
                if filt == "gguf" and not m.get("gguf"):
                    continue
                rows.append({"id": mid, "downloads": m["downloads"], "likes": m["likes"], "gated": "auto" if m.get("gated") else False,
                             "pipeline_tag": "text-generation", "tags": m["tags"], "lastModified": "2026-08-01T00:00:00.000Z",
                             "safetensors": {"total": m["params"]} if m.get("params") else None, "library_name": "transformers"})
            rows.sort(key=lambda r: -r["downloads"])
            self.js(rows[: int((q.get("limit") or ["30"])[0])])
            return
        if p.startswith("/api/models/"):
            rest = p[len("/api/models/"):]
            tree = rest.endswith("/tree/main")
            mid = rest[: -len("/tree/main")] if tree else rest
            m = MODELS.get(mid)
            if not m:
                self.js({"error": "Repository not found"}, 404)
                return
            if tree:
                files = [{"type": "file", "path": "config.json", "size": 1500}, {"type": "file", "path": "README.md", "size": 9000}]
                if m.get("shards"):
                    files += st(m["params"], m["shards"], int(m["bytes"]))[1]
                for name, size in m.get("gguf") or []:
                    files.append({"type": "file", "path": name, "size": 100, "lfs": {"size": int(size)}})
                self.js(files)
                return
            self.js({"id": mid, "author": mid.split("/")[0], "gated": "manual" if m.get("gated") else False, "downloads": m["downloads"],
                     "likes": m["likes"], "tags": m["tags"], "pipeline_tag": "text-generation", "lastModified": "2026-08-01T00:00:00.000Z",
                     "safetensors": {"total": m["params"], "parameters": {"BF16": m["params"]}} if m.get("params") else None,
                     "cardData": {"license": "other"}, "config": {"architectures": (m.get("config") or {}).get("architectures", [])}})
            return
        if "/resolve/main/config.json" in p:
            mid = p.strip("/").split("/resolve/")[0]
            m = MODELS.get(mid)
            if not m or not m.get("config"):
                self.js({"error": "Entry not found"}, 404)
                return
            if m.get("gated") and not auth.startswith("Bearer hf_"):
                self.js({"error": "Access to model is restricted"}, 401)
                return
            self.js(m["config"])
            return
        self.js({"error": "not found"}, 404)


ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1] if len(sys.argv) > 1 else 58990)), H).serve_forever()
