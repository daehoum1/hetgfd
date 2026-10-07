"""Tiny file-based work queue so that one worker process per GPU can share a list of work units.

- every launch has a LAUNCH_ID; a worker claims a unit by creating state/claims/<LAUNCH_ID>/<unit> with O_EXCL,
  so two workers of the same launch never run the same unit (claims of older, dead launches are ignored)
- what is *done* is never read from claims, only from the result files, so a job killed at any moment
  loses at most the run in progress
- every worker refreshes state/<worker>.json every 30 s (heartbeat) with its launch id and current run
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

ALIVE_SEC = 120


class Queue:
    def __init__(self, state_dir: Path, launch_id: str, worker: str):
        self.state, self.launch, self.worker = Path(state_dir), launch_id, worker
        self.claims = self.state / "claims" / launch_id
        self.claims.mkdir(parents=True, exist_ok=True)
        self.status = {"current": "starting"}
        self._lock = threading.Lock()  # the heartbeat thread and the main loop both write the state file
        threading.Thread(target=self._beat, daemon=True).start()

    # ---- heartbeat
    def _beat(self):
        while True:
            try:
                self.beat()
            except OSError:  # e.g. a short storage hiccup: try again at the next beat
                pass
            time.sleep(30)

    def beat(self):
        with self._lock:
            atomic_write(self.state / f"{self.worker}.json",
                         json.dumps({"worker": self.worker, "launch": self.launch, "pid": os.getpid(),
                                     "host": os.uname().nodename, "time": time.time(),
                                     "time_str": time.strftime("%Y-%m-%d %H:%M:%S"), **self.status}))

    def worker_alive(self, worker: str) -> bool:
        try:
            s = json.loads((self.state / f"{worker}.json").read_text())
        except Exception:
            return False
        return s.get("launch") == self.launch and s.get("current") != "finished" and time.time() - s["time"] < ALIVE_SEC

    # ---- claims
    def _cpath(self, unit: str) -> Path:
        return self.claims / unit.replace("/", "_")

    def try_claim(self, unit: str) -> bool:
        try:
            fd = os.open(self._cpath(unit), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"worker": self.worker, "status": "running", "time": time.time()}, f)
        return True

    def release(self, unit: str):
        """mark the unit as attempted (results decide whether it is done)."""
        p = self._cpath(unit)
        atomic_write(p, json.dumps({"worker": self.worker, "status": "attempted", "time": time.time()}))

    def being_worked_on(self, unit: str) -> bool:
        """claimed in this launch by a live worker that is still running it."""
        try:
            c = json.loads(self._cpath(unit).read_text())
        except Exception:
            return False
        return c.get("status") == "running" and self.worker_alive(c["worker"])


def atomic_write(path: Path, text: str):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write("\n" + json.dumps(row) + "\n")  # leading newline: never glue onto a half-written line
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(paths) -> list:
    rows = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # empty line or half-written line from a killed job
                pass
    return rows


def run_queue(q: Queue, units, is_done, is_ready, run_unit, deps, log, poll=60):
    """units: ordered unit ids. deps(u) -> unit ids that must finish first (for the 'waiting' decision)."""
    while True:
        progressed, waiting = False, False
        for u in units:
            if is_done(u):
                continue
            if not is_ready(u):
                if any(q.being_worked_on(d) for d in deps(u)):
                    waiting = True  # another GPU is still producing what this unit needs
                continue
            if not q.try_claim(u):
                continue
            q.status["current"] = u
            q.beat()
            try:
                run_unit(u)
            finally:
                q.release(u)
            progressed = True
            break
        if progressed:
            continue
        if waiting:
            q.status["current"] = "waiting for other GPUs (stage 1)"
            time.sleep(poll)
            continue
        break
    q.status["current"] = "finished"
    q.beat()
    log.info(f"==== worker {q.worker} finished (no more work)")
