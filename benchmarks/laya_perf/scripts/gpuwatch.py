# Shared box: record load average and every GPU's state at start/end, and poll our GPU every few
# seconds for compute processes that are not ours. A run with a foreign process is discarded.
import os, subprocess, threading


def _smi(args):
    try:
        return subprocess.run(["nvidia-smi"] + args, capture_output = True, text = True, timeout = 20).stdout.strip()
    except Exception as exc:
        return f"err {exc}"


def gpu_index():
    return os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]


def state():
    gpus = _smi(["--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"])
    apps = _smi(["--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader"])
    return {"loadavg": [round(x, 2) for x in os.getloadavg()], "gpus": gpus.splitlines(), "apps": apps.splitlines()}


def _mine(pid, me):
    # Our own children (compile workers) count as ours.
    while pid > 1:
        if pid == me:
            return True
        try:
            pid = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[1])
        except Exception:
            return False
    return False


def _cmd(pid):
    try:
        return open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors = "replace")[:200]
    except Exception:
        return "?"


class Watch:
    def __init__(self, every = 5.0):
        self.me, self.foreign, self.stop = os.getpid(), {}, threading.Event()
        self.start = state()
        self.thread = threading.Thread(target = self._loop, args = (every,), daemon = True)
        self.thread.start()

    def _poll(self):
        out = _smi(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits", "-i", gpu_index()])
        for line in out.splitlines():
            pid = line.split(",")[0].strip()
            if pid.isdigit() and not _mine(int(pid), self.me):
                self.foreign[pid] = f"{line} | {_cmd(pid)}"

    def _loop(self, every):
        while not self.stop.wait(every):
            self._poll()

    def done(self):
        self.stop.set()
        self._poll()
        return {"start": self.start, "end": state(), "foreign": self.foreign, "gpu": gpu_index()}
