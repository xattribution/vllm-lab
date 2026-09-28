# Titan lab architecture (vllm-lab 2)

## Machines

| Name | Role |
|---|---|
| **Titan** (ASUS GX10 / GB10, 128 GB UMA) | Inference. Fast models. Lightning. Runs the vllm-lab console. |
| **Atlas** (second GX10) | Long-context / Super. Managed from Titan over SSH. |
| **Proxmox** (separate box) | Sites, Cloudflare Tunnel, dashboards. Never the public front for models. |

Titan and Atlas stay off the public internet. People on Titan use loopback. Later, Proxmox will call Titan over a private VLAN.

## What runs on Titan

```
Firefox on Titan
   │
   ├─ http://127.0.0.1:58110       Open WebUI (chat)
   ├─ http://127.0.0.1:58120       vllm-lab console
   ├─ http://127.0.0.1:58120/v1    vllm-lab gateway   ← one URL for every engine
   └─ http://127.0.0.1:58100/v1    Lightning, direct

docker network titan-ai
   titan-webui        :8080 inside  → 127.0.0.1:58110
   titan-lightning    :8000 inside  → 127.0.0.1:58100
   titan-<engine>     :8000 inside  → 127.0.0.1:581xx
```

- **Port 8000.** vLLM always listens on 8000 *inside* its container. That port is never for people.
- **Host port.** The host port (`58100`, `58102`, …) is what people and apps use.
- **Open WebUI.** Open WebUI is itself a container on `titan-ai`, so it reaches engines by Docker DNS, for example `http://titan-lightning:8000/v1`. vllm-lab keeps those connections in sync.

## Port map (Titan loopback)

| Port | What |
|---|---|
| 58100 | Lightning |
| 58101 | Reserved (Atlas Super uses it on Atlas) |
| 58102–58109, 58112–58119 | Engine pool (allocated automatically, no collisions) |
| 58110 | Open WebUI |
| 58120 | vllm-lab console and gateway |
| 58130 | Optional gateway listener on the Docker network (needs a key) |

All engine ports publish on `127.0.0.1` by default, so nothing on the LAN can reach them. Change a host's **bind** to publish elsewhere.

## Control plane

```
            ┌──────────── vllm_lab.py ui (systemd: labui) ────────────┐
 browser ◄──┤ console SPA ◄─ SSE /api/stream (snapshot, jobs, events)  │
 apps    ◄──┤ /v1 gateway ─ route by model ─ wake on demand ─ proxy     │
            │                                                          │
            │ collector, one thread per host, every 2 s:               │
            │   one shell round-trip: meminfo, nvidia-smi, df,         │
            │   docker ps, docker inspect of engine containers         │
            │   HTTP /health + /metrics per running engine             │
            │   docker logs tail while booting or crashed              │
            │ autopilot: crash guard, idle sleep, Open WebUI sync      │
            │ jobs: start / stop / fix / download / bench              │
            └──────┬───────────────────────────┬───────────────────────┘
                   │ subprocess                 │ ssh (ControlMaster) + ssh -L tunnels
                 Titan                        Atlas
```

- **Engine state** is derived, not stored. It comes from the container state, the `/health` answer, the log tail and the *desired* state the operator set. The possible states:

  | State | Meaning |
  |---|---|
  | ready | Container running and `/health` answers |
  | booting | Container running; the log tail shows the boot stage |
  | crashed | Exited on its own, or halted by the crash guard |
  | sleeping | Stopped by the operator or idle sleep; the gateway can wake it |
  | stopped | Stopped by the operator |
  | not started | No container yet |
  | unreachable | The host cannot be reached |
- **Boot stage** is read from the log tail: image, weights, load, compile, CUDA graphs, serve.
- **Config drift.** Each container carries a `vllm-lab.spec` label, a fingerprint of the engine config. When the saved config differs, the engine shows *config changed* and Recreate applies it.
- **Memory model (GB10).**
  - Each vLLM engine reserves `util × MemTotal`.
  - A start is checked against `MemFree`, which is what CUDA sees, and against `MemAvailable`, which includes reclaimable page cache.
  - If it fits only after reclaim, the page cache is flushed first. If it does not fit at all, the operator chooses: sleep idle engines, shrink the share, or force.
- **Remote hosts.**
  - Commands run through one persistent SSH control connection per host.
  - Engines that bind loopback on Atlas are reached through managed `ssh -L` tunnels, so the gateway, probes and playground all work before the AI VLAN exists.

## Why one engine at a time is no longer the rule

v1 stopped the other engine by default ("solo"). v2 fits engines side by side while the memory math allows it.

- **Prefer sleep over stop.** When a new engine needs room, idle engines are put to sleep, not deleted. The weights stay cached and a gateway request brings them back.
- **Scale to zero.** With idle sleep on each engine, the box runs only what is being used.
- **Why not `docker pause`.** Paused containers still hold UMA, so pausing is never used.

## Software stack

- DGX OS (Ubuntu ARM64 + NVIDIA drivers), Docker + NVIDIA Container Toolkit
- vLLM `vllm/vllm-openai:v0.27.1`, plus the patched `vllm-lab/vllm-openai:0.27.1-tf5141` for Gemma 4, which is built on demand
- llama.cpp `ghcr.io/ggml-org/llama.cpp:server-cuda13` (arm64) for GGUF
- Ollama, detected on any host at `:11434`
- Open WebUI `ghcr.io/open-webui/open-webui:main`
- Weights cache `~/.cache/huggingface` on each host
- Manager: `~/bin/vllm_lab.py` + `vllm-lab` wrapper + `labui.service`

## Data flow for chat

1. A person opens `http://127.0.0.1:58110`.
2. WebUI calls `http://titan-<engine>:8000/v1/chat/completions` (direct mode) or the vllm-lab gateway (gateway mode).
3. vLLM answers. Tokens never leave the box.

Scripts and the future Proxmox apps call `http://127.0.0.1:58120/v1` (the gateway) or an engine's own port. They never call port 8000.

## Credentials (local only)

`~/.config/vllm-lab/config.json` (600):

- Hugging Face token. It reaches containers through `engine.env` (600), not `docker run -e`, so it never appears in `ps`.
- Open WebUI admin API key, or email and password, used only to manage connections.
- Optional console access token and gateway API key.

This is an unclassified R&D enclave. Still, do not commit the config directory.

## What this lab is not

- Not classified processing.
- Not a substitute for a cleared linguist.
- Not Qwen or other restricted-origin weights. The origin policy blocks them in search, launch and pulls, including fine-tunes derived from them.
- Not a Cloudflare Tunnel on the GX10 itself.

## Enabling Atlas

```bash
ssh-copy-id titan@<atlas-ip>
vllm-lab host atlas --ssh titan@<atlas-ip> --human-host <atlas-ip> --enabled true
vllm-lab doctor
vllm-lab up super
```

Or use Hosts → Atlas → Edit → Test in the console. Until the AI VLAN exists, Atlas engines stay on Atlas loopback and the console reaches them through SSH tunnels. Once the VLAN exists, set Atlas's **bind** to its VLAN IP and **reach** to `direct`.

## Next

- Atlas: Super 120B on 58101.
- Proxmox: Caddy + Cloudflare Access calling the Titan gateway over the AI VLAN, with a gateway key.
- Speech, parse and safety as small sidecar engines once the translation booth is built.
