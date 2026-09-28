export SIMROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LAB_PY="${LAB_PY:-$SIMROOT/../../vllm_lab.py}"
export FAKE_SIM_HOME=/tmp/labsim
export VLLM_LAB_HOME=/tmp/labsim/conf
export VLLM_LAB_PROC=/tmp/labsim/proc
export HF_HOME=/tmp/labsim/hf
export PATH=$SIMROOT/fakebin:$PATH
export FAKE_BOOT_SCALE=${FAKE_BOOT_SCALE:-1}
lab() { python3 "$LAB_PY" "$@"; }
