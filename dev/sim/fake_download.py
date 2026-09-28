"""Stands in for `python3 -c 'snapshot_download(...)'` inside the download container."""
import os
import sys
import time
from pathlib import Path

model = os.environ.get("VL_MODEL", "x/y")
cache = Path(os.environ.get("FAKE_CACHE_MOUNT") or "/tmp/fake-hf")
if "missing" in model:
    print("huggingface_hub.errors.RepositoryNotFoundError: 404 Client Error. Repository Not Found for url: https://huggingface.co/api/models/" + model, flush=True)
    sys.exit(1)
folder = cache / "hub" / ("models--" + model.replace("/", "--"))
blobs = folder / "blobs"
blobs.mkdir(parents=True, exist_ok=True)
(folder / "snapshots" / "abc123").mkdir(parents=True, exist_ok=True)
(folder / "snapshots" / "abc123" / "config.json").write_text('{"architectures":["LlamaForCausalLM"],"num_hidden_layers":32,"num_attention_heads":32,"num_key_value_heads":8,"hidden_size":4096,"max_position_embeddings":131072,"torch_dtype":"bfloat16"}')
total = int(os.environ.get("FAKE_DL_BYTES", str(6 * 1024**3)))
steps = 8
for i in range(1, steps + 1):
    f = blobs / f"blob{i}"
    with open(f, "wb") as fh:
        fh.truncate(total // steps)
    print(f"Fetching {steps} files: {int(i/steps*100)}%|", flush=True)
    time.sleep(0.4)
print("DONE", flush=True)
