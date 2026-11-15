"""Two things the page shows as measured facts, checked every few seconds while a call runs:
  1. network: every socket of our own processes (backend, Bonsai servers, speech bridge) must be loopback only;
  2. memory: how close macOS is to refusing new audio streams, and which other app is holding the most."""
import os
import re
import subprocess
import threading
import time

LOCAL = ("127.0.0.1", "[::1]", "localhost", "::1")


def _run(cmd, timeout=4):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def llama_pids() -> list:
    return [int(p) for p in _run(["pgrep", "-f", "bin/llama/llama-server"]).split()]


def external_sockets(pids, device_ip: str | None = None) -> list:
    """Sockets of these processes that are not loopback: [(pid, description)].
    The cable to the Olingo device, when there is one, is a known local link and is reported separately."""
    out = []
    for pid in pids:
        for ln in _run(["lsof", "-nP", "-i", "-a", "-p", str(pid)]).splitlines()[1:]:
            parts = ln.split()
            if len(parts) < 9:
                continue
            name = " ".join(parts[8:])
            ends = re.split(r"->", name.split(" ")[0])
            hosts = [e.rsplit(":", 1)[0] for e in ends]
            if device_ip and len(hosts) == 2 and hosts[1] == device_ip:
                continue                                   # the link to the Olingo device: reported by name instead
            if not all(h in LOCAL for h in hosts):
                out.append((pid, name))
    return out


def memory() -> dict:
    """wired / limit decides whether macOS will still start an audio stream; level 1 = normal, 2 = warning, 4 = critical."""
    def sysctl(k):
        try:
            return int(_run(["sysctl", "-n", k]).strip())
        except ValueError:
            return 0
    wired = 0
    for ln in _run(["vm_stat"]).splitlines():
        if ln.startswith("Pages wired down"):
            wired = int(ln.split(":")[1].strip().rstrip(".")) * 16384
    limit = sysctl("vm.global_user_wire_limit")
    return {"level": sysctl("kern.memorystatus_vm_pressure_level"), "wired_gb": round(wired / 2**30, 1),
            "limit_gb": round(limit / 2**30, 1), "ratio": round(wired / limit, 2) if limit else 0}


def footprints(bridge_pid=None) -> dict:
    """Memory each of our model processes really holds right now (macOS phys_footprint), in MB:
    {"Bonsai-2 9B": n, "Bonsai-2 4B": n, "Argmax speech bridge": n}. Both Bonsai models stay loaded side by side."""
    out = {}
    procs = [(int(p), " ".join(c)) for p, *c in (ln.split() for ln in _run(["pgrep", "-fl", "bin/llama/llama-server"]).splitlines()) if p.isdigit()]
    for pid, cmd in procs:
        m = re.search(r"bonsai2-(\d+B)", cmd)
        if m:
            out["Bonsai-2 " + m.group(1)] = pid
    if bridge_pid:
        out["Argmax speech bridge"] = bridge_pid
    res = {}
    for name, pid in out.items():
        m = re.search(r"phys_footprint:\s*([\d.]+)\s*(KB|MB|GB)", _run(["footprint", "-p", str(pid)]))
        if m:
            res[name] = round(float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)])
    return res


def hogs(ours, min_gb=1.5) -> list:
    """Other processes holding a lot of memory: [{name, gb}], biggest first."""
    out = []
    for ln in _run(["ps", "-axo", "pid=,rss=,comm="]).splitlines():
        p = ln.split(None, 2)
        if len(p) < 3 or int(p[0]) in ours:
            continue
        gb = int(p[1]) / 2**20
        if gb >= min_gb:
            path = p[2]
            m = re.search(r"/([^/]+)\.app/", path)
            label = m.group(1) if m else os.path.basename(path)
            if "lmstudio" in path.lower():
                label = "LM Studio (a loaded model)"
            out.append({"name": label, "gb": round(gb, 1)})
    return sorted(out, key=lambda x: -x["gb"])[:3]


class Monitor:
    """Runs while the app is up. push(kind, **data) gets 'privacy' and 'memory' updates."""
    def __init__(self, push, bridge_pid=lambda: None, every=5.0, device=lambda: None):
        self.push, self.bridge_pid, self.every, self.device = push, bridge_pid, every, device
        self.checks = 0
        self.external = 0
        self.last = {}
        self.mem = {}
        self.models = {}

    def pids(self):
        return [p for p in [os.getpid(), self.bridge_pid(), *llama_pids()] if p]

    def once(self):
        pids = self.pids()
        dev = self.device()                                # the Olingo device, if one is connected
        ext = external_sockets(pids, dev.peer_ip if dev else None)
        self.checks += 1
        self.external += len(ext)
        self.last = {"checks": self.checks, "external": self.external, "processes": len(pids),
                     "detail": [d for _, d in ext][:3], "device": dev.summary() if dev else None}
        self.mem = {**memory(), "hogs": hogs(set(pids))}
        self.push("privacy", **self.last)
        self.push("memory", **self.mem)
        if self.checks % 3 == 1:                           # every 15 s: what the models hold, measured
            self.models = footprints(self.bridge_pid())
            self.push("models", mb=self.models)

    def reset(self):
        self.checks = self.external = 0

    def start(self):
        def loop():
            while True:
                try:
                    self.once()
                except Exception:
                    pass
                time.sleep(self.every)
        threading.Thread(target=loop, daemon=True).start()
