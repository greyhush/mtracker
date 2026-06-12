"""Python API for mtracker - use programmatically in training scripts."""

from typing import Dict, List, Optional
from mtracker.db import Database


class Run:
    """Context manager for a training run.

    Usage:
        with Run("my-experiment", project="physics", config={"lr": 5e-5}) as run:
            for step in range(100):
                loss = train_step()
                run.log("loss", loss, step=step)
                run.log("lr", get_lr(), step=step)
    """

    def __init__(self, name: str, project: str = "default",
                 config: Optional[Dict] = None, tags: Optional[List[str]] = None,
                 db_path: Optional[str] = None):
        self.name = name
        self.project = project
        self.config = config or {}
        self.tags = tags or []
        self.db = Database(db_path)
        self.run_id = self.db.create_run(name, project, config, tags)
        self._step = 0

    @property
    def id(self) -> int:
        return self.run_id

    def log(self, name: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if step is not None:
            self._step = step
        self.db.log_metric(self.run_id, name, value, step=step or self._step)

    def log_dict(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics at once."""
        if step is not None:
            self._step = step
        items = [(name, val, step or self._step) for name, val in metrics.items()]
        self.db.log_metrics_batch(self.run_id, items)

    def set_tags(self, tags: List[str]):
        """Update run tags."""
        import json
        self.db.conn.execute(
            "UPDATE runs SET tags=? WHERE id=?",
            (json.dumps(tags), self.run_id)
        )
        self.db.conn.commit()
        self.tags = tags

    def set_notes(self, notes: str):
        """Add notes to the run."""
        self.db.conn.execute(
            "UPDATE runs SET notes=? WHERE id=?",
            (notes, self.run_id)
        )
        self.db.conn.commit()

    def finish(self, status: str = "completed", notes: str = ""):
        """Mark run as finished."""
        self.db.end_run(self.run_id, status, notes)

    def fail(self, notes: str = ""):
        """Mark run as failed."""
        self.db.end_run(self.run_id, "failed", notes)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.fail(f"{exc_type.__name__}: {exc_val}")
        else:
            self.finish()
        self.db.close()
        return False


def start_run(name: str, project: str = "default", config: Optional[Dict] = None,
              tags: Optional[List[str]] = None, db_path: Optional[str] = None) -> Run:
    """Start a new run (non-context-manager style)."""
    return Run(name, project, config, tags, db_path)


def log_metric(run: Run, name: str, value: float, step: Optional[int] = None):
    """Log a metric to a run."""
    run.log(name, value, step)


def end_run(run: Run, status: str = "completed", notes: str = ""):
    """End a run."""
    run.finish(status, notes)


def compare_runs(run_ids: List[int], db_path: Optional[str] = None) -> List[Dict]:
    """Compare multiple runs by their IDs."""
    db = Database(db_path)
    result = []
    for rid in run_ids:
        run = db.get_run(rid)
        if run:
            run["metrics"] = db.get_latest_metrics(rid)
            result.append(run)
    db.close()
    return result


def import_log_file(run_id: int, log_path: str, db_path: Optional[str] = None):
    """Import a log file into an existing run."""
    from mtracker.parser import parse_log_file

    db = Database(db_path)
    entries = parse_log_file(log_path)

    for entry in entries:
        step = entry.get("step")
        for name, value in entry.get("metrics", {}).items():
            db.log_metric(run_id, name, value, step=step)

    db.close()
    return len(entries)


def import_trainer_state(run_id: int, state_path: str, db_path: Optional[str] = None):
    """Import a HuggingFace trainer_state.json into an existing run."""
    from mtracker.parser import parse_trainer_state

    db = Database(db_path)
    state = parse_trainer_state(state_path)

    count = 0
    for entry in state.get("log_history", []):
        step = entry.get("step")
        for name, value in entry.get("metrics", {}).items():
            db.log_metric(run_id, name, value, step=step)
            count += 1

    db.close()
    return count
