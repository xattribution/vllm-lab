# vllm-lab 2.0.0

## Fixed from v1

### Security

| Issue | Fix |
|---|---|
| **XSS.** The console's `esc()` was a no-op: its entities had been decoded, so it mapped `<` to `<`. Container names, logs and Hub results went into `innerHTML` raw. | Everything is escaped now. |
| **CSRF.** Any website could POST to `/stop`, `/api/wipe` or `/api/fix`, because form posts were accepted with no origin check. | Mutations now need a custom header plus a same-origin `Origin`. The Host header is checked against DNS rebinding. |
| **SSH command injection.** Remote arguments were joined by the remote shell without quoting. | Every argument is shell-quoted. Names, model ids, images and hosts are validated. |
| **HF token exposure.** The token was passed as `-e HF_TOKEN=…`, so it showed up in `ps`. The v1 server-rendered page also echoed the Open WebUI password into the HTML. | The token now goes through an env file (600). Secrets never reach the browser. |

### Engines and memory

| Issue | Fix |
|---|---|
| **Port collisions.** Every custom engine defaulted to port 58100, which is Lightning's. | Ports come from a pool and are checked against every container on the host. |
| **GB10 memory math.** `nvidia-smi` reports memory as N/A on unified memory, so the "fit" logic never ran. | Fit now uses `/proc/meminfo` and accounts for the page cache. |
| **Dead metrics.** `avg_generation_throughput_toks_per_s` was removed in vLLM V1, so tok/s was always blank. | Rates are now computed from the V1 counters. |
| **Wipe weights on Atlas engines** deleted the *local* cache. | Weights are deleted on the host that owns them. |
| **Atlas probes** hit `atlas:port` while Atlas binds loopback. | They go through managed SSH tunnels. |
| **Recipe lock.** It overwrote any edit to lightning, gemma4 and super. | They are regular engines now. |
| **Crash loops.** `--restart unless-stopped` let a crashing engine loop forever while holding memory. | A crash guard stops it, and a failed boot halts instead of looping. |

### Performance and integrations

| Issue | Fix |
|---|---|
| **Slow refresh.** Each refresh ran dozens of serial `ssh`, `docker info` and `docker logs` calls. | One collector thread per host makes one shell round trip per tick and pushes results over SSE. |
| **Open WebUI clutter.** Dead connections piled up, and each one slowed the model picker. | Connections are synced both ways; your own connections are never touched. |
| **Hub sizes** were always 0: the siblings list has no sizes without `blobs=true`. | Sizes now come from the tree API. Search is one request instead of N+1. |
| **Ollama.** `ollama run model ""` was used as a loader. | Ollama is driven through its HTTP API. |
| **Hardcoded host.** The Docker and Health tabs only ever showed Titan. | Every page works for every host. |

## New

- **Live console.**
  - Server-sent events drive the whole UI, and a DOM morph keeps focus and scroll while the data changes.
  - Includes a command palette (Ctrl K), light and dark themes, and swappable accent colors.
- **Memory tank.** A unified-memory bar per host shows each engine's share. It previews a new engine as a ghost and shows which engines would be put to sleep to make room.
- **Fit planner.**
  - Reads `config.json` from the Hub, or from the local cache when offline.
  - Handles hybrid Mamba (Nemotron-H), sliding-window (gpt-oss, Gemma) and shared-KV layers.
  - Estimates KV bytes per token and sizes `--gpu-memory-utilization` for your context length and parallel sequences.
  - Counts speculative-decoding draft models too.
- **Boot pipeline.** A live stage bar (image, weights, load, compile, CUDA graphs, serve) with percentages. After boot it shows KV capacity and max concurrency.
- **Crash decoder.** About 20 known failure signatures map to a plain cause and one-click fixes.
- **Gateway.**
  - One OpenAI endpoint across hosts and backends, including Ollama.
  - Auto-wake, idle sleep, per-engine request stats and an optional API key.
- **Jobs.** Long operations run in the background with progress, cancel and fix buttons. The CLI shows the same live progress.
- **Playground.** Streaming chat with compare mode, TTFT and tok/s. **Bench** runs concurrent load with p50/p95 and keeps a history.
- **Library.** Per-host weights with usage, pre-download with progress, safe delete, and Ollama pull, load and unload.
- **Hosts.** Add a host, test it (SSH, Docker, GPU, cache), enable or disable it, flush the page cache. Includes versions, load and uptime.
- **Containers.** A Portainer-style view of containers, images and networks per host.
- **Doctor.** Fleet-wide checks with fixes. Among them: token validity, Docker group, NVIDIA runtime, disk, page cache, auto-start memory budget, port clashes, config drift, crashed engines, Open WebUI sync and gateway.
- **Adopt.** Turns a container started by hand into a managed engine by reading its command line.
- **Blueprints.** Built-ins for Nemotron, Gemma, Llama, gpt-oss, Mistral and Phi. Save your own, then import or export them.
- **Origin policy.** Blocks or flags restricted-origin weights, including `base_model` derivatives and Ollama families.
- **llama.cpp backend.** GGUF from Hugging Face, using the ARM64 CUDA 13 image.

## Compatibility

- **Kept as-is.** Existing `titan-*` containers, ports, `~/.config/vllm-lab`, and the `vllm-lab` commands `up`, `solo`, `down`, `stop`, `search`, `config`, `host` and `webui-add`.
- **Migrated automatically.** The config and `engines.json`. The v1 engines file is backed up.
- **Removed.** The v1 HTML routes (`/start`, `/stop`, `/save-webui`, `/logs`, `/test`). `/api/status` still answers for scripts.
