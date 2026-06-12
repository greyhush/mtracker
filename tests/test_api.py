"""Tests for mtracker Python API."""

import os
import tempfile
import pytest
from mtracker.api import Run, start_run, end_run, log_metric, compare_runs, import_log_file


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestRunContextManager:
    def test_basic_run(self, db_path):
        with Run("test_run", project="myproject", config={"lr": 5e-5},
                 tags=["tag1"], db_path=db_path) as run:
            run.log("loss", 0.5, step=1)
            run.log("loss", 0.3, step=2)
            run.log("acc", 0.9, step=2)

        from mtracker.db import Database
        db = Database(db_path)
        runs = db.list_runs()
        assert len(runs) == 1
        assert runs[0]["name"] == "test_run"
        assert runs[0]["status"] == "completed"
        assert runs[0]["project"] == "myproject"

        metrics = db.get_metrics(runs[0]["id"])
        assert len(metrics) == 3
        db.close()

    def test_auto_fail(self, db_path):
        try:
            with Run("failing_run", db_path=db_path) as run:
                run.log("loss", 0.5, step=1)
                raise ValueError("boom")
        except ValueError:
            pass

        from mtracker.db import Database
        db = Database(db_path)
        runs = db.list_runs()
        assert runs[0]["status"] == "failed"
        assert "boom" in runs[0]["notes"]
        db.close()

    def test_log_dict(self, db_path):
        with Run("test_run", db_path=db_path) as run:
            run.log_dict({"loss": 0.5, "acc": 0.8, "lr": 5e-5}, step=1)

        from mtracker.db import Database
        db = Database(db_path)
        runs = db.list_runs()
        metrics = db.get_metrics(runs[0]["id"])
        assert len(metrics) == 3
        db.close()


class TestStandaloneFunctions:
    def test_start_end(self, db_path):
        run = start_run("standalone", db_path=db_path)
        log_metric(run, "loss", 0.5, step=1)
        end_run(run)

        from mtracker.db import Database
        db = Database(db_path)
        runs = db.list_runs()
        assert len(runs) == 1
        assert runs[0]["status"] == "completed"
        db.close()

    def test_compare_runs(self, db_path):
        r1 = start_run("run1", project="p", db_path=db_path)
        log_metric(r1, "loss", 0.5, step=1)
        end_run(r1)

        r2 = start_run("run2", project="p", db_path=db_path)
        log_metric(r2, "loss", 0.3, step=1)
        end_run(r2)

        results = compare_runs([r1.id, r2.id], db_path=db_path)
        assert len(results) == 2
        assert results[0]["metrics"]["loss"]["value"] == 0.5
        assert results[1]["metrics"]["loss"]["value"] == 0.3


class TestImportLogFile:
    def test_import_jsonl(self, db_path):
        import json
        fd, log_path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"step": 1, "loss": 0.5}) + "\n")
            f.write(json.dumps({"step": 2, "loss": 0.3}) + "\n")

        run = start_run("imported", db_path=db_path)
        count = import_log_file(run.id, log_path, db_path=db_path)
        assert count == 2

        from mtracker.db import Database
        db = Database(db_path)
        latest = db.get_latest_metrics(run.id)
        assert latest["loss"]["value"] == 0.3
        db.close()

        os.unlink(log_path)
