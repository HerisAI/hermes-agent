#!/usr/bin/env python3
"""GPU training job poller for the standalone `hermes-ml` sidecar container.

Design (see docker-compose.yml `ml` service):
  * The `hermes-ml` container runs this script as its main process. It has the
    GPU (runtime: nvidia) and a CUDA-enabled torch baked into /opt/hermes/.venv.
  * The `ml` profile (running inside the always-on `hermes` gateway) submits
    work by dropping a job spec JSON into  $HERMES_HOME/ml-jobs/queue/  on the
    shared ~/.hermes mount. No docker socket, no network call — just a file.
  * This poller picks up queued specs FIFO, runs ONE at a time (single GPU),
    streams stdout+stderr to logs/<id>.log, and writes machine-readable state
    to status/<id>.json. The gateway-side `gpu-train` skill reads those back.

Reliability properties this script is built around:
  * It is the only thing with GPU access, so a broken nvidia/CUDA stack can
    only ever take down THIS container — never the gateway or the crons.
  * The main loop never dies on a bad job: every iteration is wrapped, and a
    crash mid-job is recovered on restart (orphaned `running/` specs -> failed).
  * Job hand-off uses os.rename within the single shared filesystem, which is
    atomic, so a half-written spec is never executed.

Job spec schema (queue/<id>.json), all fields optional except `script`:
  {
    "id":      "20260603_142500_train-resnet",   # auto if omitted
    "name":    "train-resnet",
    "created": "2026-06-03T14:25:00Z",
    "workdir": "/opt/data/profiles/ml/workspace",  # cwd for the job
    "script":  "#!/bin/bash\nset -euo pipefail\npython train.py\n",
    "env":     {"WANDB_PROJECT": "..."}            # extra env vars
  }
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/opt/data"))
# Fixed global queue root, shared with submit.py/status.py. Deliberately NOT
# derived from HERMES_HOME: submitting profiles run with a per-profile
# HERMES_HOME (e.g. /opt/data/profiles/ml), so a HERMES_HOME-relative queue
# would not be the one this poller watches. /opt/data is the shared mount root.
JOBS_ROOT = Path(os.environ.get("ML_JOBS_DIR") or "/opt/data/ml-jobs")
QUEUE_DIR = JOBS_ROOT / "queue"
RUNNING_DIR = JOBS_ROOT / "running"
DONE_DIR = JOBS_ROOT / "done"
LOGS_DIR = JOBS_ROOT / "logs"
STATUS_DIR = JOBS_ROOT / "status"
POLL_INTERVAL = float(os.environ.get("ML_POLL_INTERVAL", "5"))

# Set by the SIGTERM handler so a `docker stop` interrupts the loop and the
# currently running child cleanly instead of being SIGKILLed after the grace
# period.
_SHUTDOWN = False
_CURRENT_PROC: subprocess.Popen | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    """Poller-level log line (distinct from per-job logs) -> container stdout."""
    print(f"[ml-poller {_now()}] {msg}", flush=True)


def _ensure_dirs() -> None:
    for d in (QUEUE_DIR, RUNNING_DIR, DONE_DIR, LOGS_DIR, STATUS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _write_status(job_id: str, **fields) -> None:
    """Merge-and-write status/<id>.json atomically."""
    path = STATUS_DIR / f"{job_id}.json"
    data: dict = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            data = {}
    data.update(fields)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def _recover_orphans() -> None:
    """A spec left in running/ means the poller (or container) died mid-job.

    We cannot resume it, so mark it failed and move it out of the way. This
    keeps a crash from blocking the queue forever."""
    for spec in sorted(RUNNING_DIR.glob("*.json")):
        job_id = spec.stem
        _log(f"recovering orphaned job {job_id} (poller restarted mid-run)")
        _write_status(
            job_id,
            state="failed",
            error="poller restarted while job was running; not resumed",
            finished_at=_now(),
        )
        shutil.move(str(spec), str(DONE_DIR / spec.name))


def _claim_next() -> Path | None:
    """Atomically move the oldest queued spec into running/ and return it.

    FIFO by filename (specs are timestamp-prefixed). os.rename is atomic on the
    shared mount, so two pollers (there is only one, but defensively) can't both
    claim the same job."""
    for spec in sorted(QUEUE_DIR.glob("*.json")):
        target = RUNNING_DIR / spec.name
        try:
            os.rename(spec, target)
            return target
        except OSError:
            continue  # someone/something else moved it; try the next
    return None


def _load_spec(spec_path: Path) -> dict | None:
    try:
        return json.loads(spec_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        _log(f"bad spec {spec_path.name}: {exc}")
        return None


def _run_job(spec_path: Path) -> None:
    global _CURRENT_PROC
    spec = _load_spec(spec_path)
    job_id = spec_path.stem

    if spec is None:
        _write_status(job_id, state="failed", error="unparseable job spec",
                      finished_at=_now())
        shutil.move(str(spec_path), str(DONE_DIR / spec_path.name))
        return

    script = spec.get("script")
    if not script:
        _write_status(job_id, state="failed", error="spec has no 'script'",
                      finished_at=_now())
        shutil.move(str(spec_path), str(DONE_DIR / spec_path.name))
        return

    workdir = spec.get("workdir") or str(HERMES_HOME / "profiles" / "ml" / "workspace")
    Path(workdir).mkdir(parents=True, exist_ok=True)

    # Job environment: inherit the container env (venv already on PATH, CUDA
    # visible), then layer the spec's extra vars.
    env = os.environ.copy()
    env.update({k: str(v) for k, v in (spec.get("env") or {}).items()})

    log_path = LOGS_DIR / f"{job_id}.log"
    _write_status(
        job_id,
        state="running",
        name=spec.get("name", job_id),
        workdir=workdir,
        started_at=_now(),
        log=str(log_path),
    )
    _log(f"starting job {job_id} (cwd={workdir})")

    # Materialize the script to a temp file and run it shebang-aware: if the
    # script declares an interpreter (`#!/usr/bin/env python3`, `#!/bin/bash`,
    # …) honor it by exec'ing the file directly; otherwise default to bash.
    # This lets workers submit either a shell job or a standalone Python file
    # (submit.py ensures a shebang is present for `.py` scripts).
    has_shebang = script.lstrip().startswith("#!")
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(script)
        script_file = fh.name
    os.chmod(script_file, 0o755)
    cmd = [script_file] if has_shebang else ["/bin/bash", script_file]

    rc = None
    try:
        with open(log_path, "w", buffering=1) as logf:
            logf.write(f"=== job {job_id} started {_now()} ===\n")
            logf.flush()
            _CURRENT_PROC = subprocess.Popen(
                cmd,
                cwd=workdir,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            rc = _CURRENT_PROC.wait()
            logf.write(f"\n=== job {job_id} exited rc={rc} {_now()} ===\n")
    except Exception as exc:  # noqa: BLE001 - never let one job kill the loop
        _log(f"job {job_id} crashed in poller: {exc}")
        _write_status(job_id, state="failed", error=str(exc), finished_at=_now())
        shutil.move(str(spec_path), str(DONE_DIR / spec_path.name))
        return
    finally:
        _CURRENT_PROC = None
        try:
            os.unlink(script_file)
        except OSError:
            pass

    state = "done" if rc == 0 else "failed"
    _write_status(job_id, state=state, exit_code=rc, finished_at=_now())
    shutil.move(str(spec_path), str(DONE_DIR / spec_path.name))
    _log(f"job {job_id} finished state={state} rc={rc}")


def _handle_term(signum, frame):  # noqa: ARG001
    global _SHUTDOWN
    _SHUTDOWN = True
    _log(f"received signal {signum}; shutting down after current job step")
    if _CURRENT_PROC is not None:
        _log("forwarding SIGTERM to running job")
        try:
            _CURRENT_PROC.terminate()
        except Exception:  # noqa: BLE001
            pass


def main() -> int:
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    _ensure_dirs()
    _log(f"poller up. jobs root={JOBS_ROOT} interval={POLL_INTERVAL}s")

    # Best-effort: confirm GPU visibility once at startup for the logs.
    try:
        import torch  # noqa: PLC0415

        _log(f"torch {torch.__version__} cuda_available={torch.cuda.is_available()} "
             f"devices={torch.cuda.device_count()}")
    except Exception as exc:  # noqa: BLE001
        _log(f"torch import/check failed (jobs can still run their own envs): {exc}")

    _recover_orphans()

    while not _SHUTDOWN:
        try:
            spec_path = _claim_next()
            if spec_path is None:
                time.sleep(POLL_INTERVAL)
                continue
            _run_job(spec_path)
        except Exception as exc:  # noqa: BLE001 - loop must survive anything
            _log(f"unexpected loop error: {exc}")
            time.sleep(POLL_INTERVAL)

    _log("poller exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
