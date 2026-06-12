"""Tests for mtracker log parser."""

import json
import os
import tempfile
import pytest
from mtracker.parser import parse_log_file, parse_trainer_state


def _write_temp(content: str, suffix: str = ".jsonl") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


class TestParseJsonl:
    def test_basic(self):
        content = '\n'.join([
            json.dumps({"step": 100, "loss": 0.5, "learning_rate": 5e-5, "epoch": 0.1}),
            json.dumps({"step": 200, "loss": 0.3, "learning_rate": 4e-5, "epoch": 0.2}),
        ])
        path = _write_temp(content, ".jsonl")
        entries = parse_log_file(path)
        os.unlink(path)

        assert len(entries) == 2
        assert entries[0]["step"] == 100
        assert entries[0]["metrics"]["loss"] == 0.5
        assert entries[1]["metrics"]["loss"] == 0.3

    def test_skip_non_numeric(self):
        content = json.dumps({"step": 1, "loss": 0.5, "status": "ok"})
        path = _write_temp(content, ".jsonl")
        entries = parse_log_file(path)
        os.unlink(path)

        assert len(entries) == 1
        assert "status" not in entries[0]["metrics"]

    def test_empty(self):
        path = _write_temp("", ".jsonl")
        entries = parse_log_file(path)
        os.unlink(path)
        assert entries == []


class TestParseJson:
    def test_trainer_state(self):
        state = {
            "global_step": 1000,
            "max_steps": 3750,
            "log_history": [
                {"step": 100, "loss": 0.5},
                {"step": 200, "loss": 0.3},
            ]
        }
        path = _write_temp(json.dumps(state), ".json")
        entries = parse_log_file(path)
        os.unlink(path)

        assert len(entries) == 2
        assert entries[0]["step"] == 100

    def test_list_format(self):
        data = [
            {"step": 1, "loss": 0.8},
            {"step": 2, "loss": 0.6},
        ]
        path = _write_temp(json.dumps(data), ".json")
        entries = parse_log_file(path)
        os.unlink(path)
        assert len(entries) == 2


class TestParseCsv:
    def test_basic(self):
        content = "step,loss,accuracy\n1,0.5,0.8\n2,0.3,0.9\n3,0.2,0.95"
        path = _write_temp(content, ".csv")
        entries = parse_log_file(path)
        os.unlink(path)

        assert len(entries) == 3
        assert entries[0]["step"] == 1
        assert entries[0]["metrics"]["loss"] == 0.5


class TestParseTextLog:
    def test_hf_trainer_format(self):
        content = """
  0%|          | 1/1720 [00:05<2:37:02, 5.42s/it]
{'loss': 0.5, 'grad_norm': 0.025, 'learning_rate': 5e-05, 'epoch': 0.1}
  1%|          | 2/1720 [00:10<2:30:00, 5.23s/it]
{'loss': 0.35, 'grad_norm': 0.023, 'learning_rate': 4.9e-05, 'epoch': 0.2}
"""
        path = _write_temp(content, ".log")
        entries = parse_log_file(path)
        os.unlink(path)

        assert len(entries) == 2
        assert entries[0]["metrics"]["loss"] == 0.5
        assert entries[0]["epoch"] == 0.1


class TestParseTrainerState:
    def test_full_state(self):
        state = {
            "global_step": 3000,
            "max_steps": 3750,
            "num_train_epochs": 3,
            "epoch": 2.4,
            "best_metric": None,
            "log_history": [
                {"step": 100, "loss": 0.5, "learning_rate": 5e-5},
                {"step": 3000, "loss": 0.25, "learning_rate": 5.8e-6},
            ]
        }
        path = _write_temp(json.dumps(state), ".json")
        result = parse_trainer_state(path)
        os.unlink(path)

        assert result["global_step"] == 3000
        assert result["max_steps"] == 3750
        assert len(result["log_history"]) == 2


class TestFileNotFound:
    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_log_file("/nonexistent/path.json")
