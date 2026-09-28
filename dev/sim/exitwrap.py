"""Run a child, record its exit code next to the log (the simulator's container 'State.ExitCode')."""
import signal
import subprocess
import sys

exitfile, argv = sys.argv[1], sys.argv[2:]
p = subprocess.Popen(argv)


def fwd(sig, _):
    try:
        p.send_signal(sig)
    except Exception:
        pass


signal.signal(signal.SIGTERM, fwd)
code = p.wait()
if code < 0:
    code = 128 + (-code)
if code == 128 + signal.SIGTERM:
    code = 0
with open(exitfile, "w") as fh:
    fh.write(str(code))
sys.exit(code)
