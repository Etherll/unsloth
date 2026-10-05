# A GPU is free only with no compute process, < 1 GiB used and 0% util on 3 checks 10 s apart.
# `gpufree.py N` waits (re-checking every 2 min) and flags $LP_ROOT/logs/gpuN_busy after 30 min.
import os, subprocess, sys, time


def _q(n):
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader", "-i", str(n)],
                          capture_output = True, text = True, timeout = 30).stdout.strip()
    util, mem = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", str(n)],
                               capture_output = True, text = True, timeout = 30).stdout.strip().split(", ")
    return apps, int(util), int(mem)


def free(n):
    for i in range(3):
        apps, util, mem = _q(n)
        if apps or util != 0 or mem >= 1024:
            return False, f"apps={apps!r} util={util} mem={mem}"
        if i < 2:
            time.sleep(10)
    return True, "free"


def wait(n, log = print):
    t0 = time.time()
    flag = f"{os.environ['LP_ROOT']}/logs/gpu{n}_busy"
    while True:
        ok, why = free(n)
        if ok:
            if os.path.exists(flag):
                os.rename(flag, flag + f".cleared.{int(time.time())}")
            return
        log(f"gpu{n} busy: {why}; waited {int(time.time() - t0)} s")
        if time.time() - t0 > 1800 and not os.path.exists(flag):
            open(flag, "w").write(f"{time.ctime()} gpu{n} busy for 30 min: {why}\n")
        time.sleep(120)


if __name__ == "__main__":
    wait(int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ["LANE"]))
