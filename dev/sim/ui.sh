#!/usr/bin/env bash
# restart the console under the simulator
source "$(dirname "$0")/env.sh"
[ -f /tmp/labsim/ui.pid ] && kill "$(cat /tmp/labsim/ui.pid)" 2>/dev/null; pkill -f "[v]llm_lab.py ui" 2>/dev/null
sleep 0.7
nohup python3 "$LAB_PY" ui --no-browser > /tmp/labsim/ui.log 2>&1 &
echo $! > /tmp/labsim/ui.pid
sleep 1.5
head -3 /tmp/labsim/ui.log
