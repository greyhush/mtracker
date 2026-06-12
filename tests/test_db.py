"""Tests for mtracker database layer."""

import os
import tempfile
import pytest
from mtracker.db import Database


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


class TestDatabase:
    def test_create_run(self, db):
        run_id = db.create_run("test_run", "myproject", {"lr": 5e-5}, ["tag1"])
        assert run_id > 0
        run = db.get_run(run_id)
        assert run["name"] == "test_run"
        assert run["project"] == "myproject"
        assert run["config"]["lr"] == 5e-5
        assert run["tags"] == ["tag1"]
        assert run["status"] == "running"

    def test_end_run(self, db):
        run_id = db.create_run("test_run")
        db.end_run(run_id, "completed", "all good")
        run = db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["notes"] == "all good"
        assert run["ended_at"] is not None

    def test_log_metric(self, db):
        run_id = db.create_run("test_run")
        db.log_metric(run_id, "loss", 0.5, step=1)
        db.log_metric(run_id, "loss", 0.3, step=2)
        db.log_metric(run_id, "acc", 0.8, step=1)

        metrics = db.get_metrics(run_id)
        assert len(metrics) == 3

        loss_metrics = db.get_metrics(run_id, "loss")
        assert len(loss_metrics) == 2

    def test_log_metrics_batch(self, db):
        run_id = db.create_run("test_run")
        db.log_metrics_batch(run_id, [
            ("loss", 0.5, 1),
            ("acc", 0.8, 1),
            ("loss", 0.3, 2),
        ])
        metrics = db.get_metrics(run_id)
        assert len(metrics) == 3

    def test_get_latest_metrics(self, db):
        run_id = db.create_run("test_run")
        db.log_metric(run_id, "loss", 0.5, step=1)
        db.log_metric(run_id, "loss", 0.3, step=2)
        db.log_metric(run_id, "acc", 0.9, step=5)

        latest = db.get_latest_metrics(run_id)
        assert latest["loss"]["value"] == 0.3
        assert latest["loss"]["step"] == 2
        assert latest["acc"]["value"] == 0.9

    def test_get_metric_summary(self, db):
        run_id = db.create_run("test_run")
        for i in range(10):
            db.log_metric(run_id, "loss", 1.0 - i * 0.1, step=i)

        summary = db.get_metric_summary(run_id, "loss")
        assert summary is not None
        assert summary["min_val"] == pytest.approx(0.1)
        assert summary["max_val"] == pytest.approx(1.0)
        assert summary["count"] == 10
        assert summary["last_val"] == pytest.approx(0.1)

    def test_list_runs(self, db):
        db.create_run("run1", project="proj_a")
        db.create_run("run2", project="proj_b")
        db.create_run("run3", project="proj_a")

        all_runs = db.list_runs()
        assert len(all_runs) == 3

        proj_a = db.list_runs(project="proj_a")
        assert len(proj_a) == 2

    def test_list_runs_by_status(self, db):
        r1 = db.create_run("run1")
        r2 = db.create_run("run2")
        db.end_run(r1, "completed")

        running = db.list_runs(status="running")
        assert len(running) == 1
        assert running[0]["name"] == "run2"

        completed = db.list_runs(status="completed")
        assert len(completed) == 1

    def test_delete_run(self, db):
        run_id = db.create_run("test_run")
        db.log_metric(run_id, "loss", 0.5, step=1)
        db.delete_run(run_id)
        assert db.get_run(run_id) is None
        assert db.get_metrics(run_id) == []

    def test_nonexistent_run(self, db):
        assert db.get_run(999) is None
