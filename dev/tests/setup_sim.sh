#!/usr/bin/env bash
# Fresh simulator + console configured for the smoke suite.
# Optional: ATLAS_SSH=user@host ATLAS_PORT=2222 to test a remote host over SSH.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="$HERE/../sim"
"$SIM/up.sh" --fresh
source "$SIM/env.sh"
lab config --hf-token hf_testtoken123456 --hf-endpoint http://127.0.0.1:58990 \
  --webui-email admin@lab.local --webui-password hunter2 >/dev/null
if [ -n "$ATLAS_SSH" ]; then
  lab host atlas --ssh "$ATLAS_SSH" --ssh-port "${ATLAS_PORT:-22}" --human-host 127.0.0.1 --enabled true \
     --cache /tmp/atlassim/hf >/dev/null
  python3 - <<'PY'
import json, os
p = os.environ["VLLM_LAB_HOME"] + "/config.json"
c = json.load(open(p))
for h in c["hosts"]:
    if h["id"] == "atlas":
        h["ports"] = "58140-58149"
json.dump(c, open(p, "w"), indent=2)
PY
fi
lab up lightning >/dev/null
"$SIM/ui.sh"
