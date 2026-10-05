# Single timing job (not an A/B): wait for a free GPU and low load, hold LP_HOLD lanes for the run only.
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import holdlib as h
h.hold_others()
try:
    rc = subprocess.run(sys.argv[1:]).returncode
finally:
    h.release_others()
sys.exit(rc)
