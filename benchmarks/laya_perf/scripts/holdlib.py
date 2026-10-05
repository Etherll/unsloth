# Timing-lane gates: free GPU (gpufree), 1-min load <= LP_LOAD_MAX, and the LP_HOLD lanes idle for the run only.
import os, time
import gpufree

LOAD_MAX = float(os.environ.get("LP_LOAD_MAX", "40"))
HOLD = os.environ.get("LP_HOLD", "").split()
LOGS = f"{os.environ['LP_ROOT']}/logs"
LANE = int(os.environ["LANE"])


def wait_load(log = print):
    said = False
    while os.getloadavg()[0] > LOAD_MAX:
        if not said:
            log(f"load {os.getloadavg()[0]:.0f} > {LOAD_MAX:.0f}: waiting")
            said = True
        time.sleep(60)


def release_others():
    for n in HOLD:
        try:
            os.remove(f"{LOGS}/hold_{n}")
        except FileNotFoundError:
            pass


def hold_others(log = print):
    while True:
        gpufree.wait(LANE, log)
        wait_load(log)
        for n in HOLD:
            open(f"{LOGS}/hold_{n}", "w").close()
        while any(os.path.exists(f"{LOGS}/running_gpu{n}") for n in HOLD):
            time.sleep(10)
        ok, why = gpufree.free(LANE)
        if ok and os.getloadavg()[0] <= LOAD_MAX:
            return
        release_others()
