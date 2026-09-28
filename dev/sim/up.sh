#!/usr/bin/env bash
# Start the simulator. --fresh wipes all simulated state first.
source "$(dirname "$0")/env.sh"
if [[ "$1" == "--fresh" ]]; then
  pkill -f "[f]ake_vllm.py" 2>/dev/null; pkill -f "[f]ake_webui.py" 2>/dev/null; pkill -f "[f]ake_hf.py" 2>/dev/null; pkill -f "[f]ake_ollama.py" 2>/dev/null
  pkill -f "[v]llm_lab.py ui" 2>/dev/null; sleep 0.5
  rm -rf /tmp/labsim
fi
mkdir -p /tmp/labsim/hf/hub /tmp/labsim/conf
pgrep -f "[f]ake_hf.py" >/dev/null || (nohup python3 $SIMROOT/fake_hf.py 58990 >/tmp/labsim/hf.log 2>&1 &)
pgrep -f "[f]ake_ollama.py" >/dev/null || (nohup python3 $SIMROOT/fake_ollama.py 11434 >/tmp/labsim/ollama.log 2>&1 &)
docker network inspect titan-ai >/dev/null 2>&1 || docker network create titan-ai >/dev/null
docker inspect titan-webui >/dev/null 2>&1 || docker run -d --name titan-webui --restart unless-stopped --network titan-ai -p 127.0.0.1:58110:8080 ghcr.io/open-webui/open-webui:main >/dev/null
sleep 0.5
echo "simulator up"
