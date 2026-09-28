# Developing vllm-lab without a DGX

The simulator stands in for everything the console talks to. It exercises vllm-lab's real code paths and runs on any Linux box with Python 3.10+.

## What the simulator provides

| Piece | Behaves like |
|---|---|
| `sim/fakebin/docker` | The Docker CLI subset vllm-lab uses: ps/inspect/run/stop/logs -f/stats/images/networks/exec/pull/build. State lives in `$FAKE_SIM_HOME`. |
| `sim/fakebin/nvidia-smi` | GB10 with unified memory (memory fields read `[N/A]`) |
| `sim/fake_vllm.py` | vLLM or llama.cpp server with a realistic boot log, `/health`, `/metrics` (V1 names) and streaming chat. It checks free memory the way vLLM does. |
| `sim/fake_hf.py` | Hugging Face Hub: search, model info, file tree, config.json, whoami |
| `sim/fake_webui.py` | Open WebUI admin API: sign-in, OpenAI connections, model list |
| `sim/fake_ollama.py` | Ollama API on :11434 |
| fake procfs | `meminfo` follows the running engines' reservations and page cache; writing `drop_caches` clears the cache |

## Crash scenarios

The model id picks the scenario:

| Model id contains | Result |
|---|---|
| `crash` | CUDA out of memory |
| `gated` | GatedRepoError |
| `missing` | Repository not found |
| `remote-code` | Needs trust_remote_code |
| `gemma-4` (with the stock image) | The Transformers per-layer error |
| an `--foo` extra argument | Argparse rejection |
| `--max-model-len` over 262144 | KV cache too small, with a suggested length |

## Run it

```bash
dev/tests/setup_sim.sh          # fresh simulator, configured console on :58120, Lightning started
python3 dev/tests/smoke_api.py  # 24 end-to-end API tests
# optional UI screenshots (needs playwright + chromium):
python3 dev/tests/ui_scenarios.py pages
```

- **Remote host.** To test a remote host too, run an sshd with a second user whose `~/.ssh/environment` sets `PATH` (fakebin first), `FAKE_SIM_HOME` and `VLLM_LAB_PROC`. Then run:
  `ATLAS_SSH=atlas@127.0.0.1 ATLAS_PORT=2222 dev/tests/setup_sim.sh`
- **Restarting the console.** `dev/sim/ui.sh` restarts it after you edit `vllm_lab.py`.
- **Boot speed.** `FAKE_BOOT_SCALE=3` slows boots down, so the boot pipeline is easier to watch.

## Test hooks

Two environment variables exist only for the simulator; both are harmless in production.

| Variable | Default | Effect |
|---|---|---|
| `VLLM_LAB_PROC` | `/proc` | Where procfs is read. On remote hosts, `${VLLM_LAB_PROC:-/proc}` is expanded remotely. |
| `VLLM_LAB_HOME` | `~/.config/vllm-lab` | Config directory. Isolates simulator state from real config. |
