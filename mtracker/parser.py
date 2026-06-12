"""Auto-detect and parse training log files from various frameworks."""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_log_file(path: str) -> List[Dict]:
    """Auto-detect format and parse a training log file.

    Returns list of dicts with: step, metrics (dict of name->value)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    content = path.read_text(encoding="utf-8", errors="replace")

    # Try formats in order of specificity
    if path.suffix == ".jsonl":
        return _parse_jsonl(content)
    elif path.suffix == ".json":
        return _parse_json(content)
    elif path.suffix == ".csv":
        return _parse_csv(content)
    else:
        # Text log - try known patterns
        return _parse_text_log(content)


def _parse_jsonl(content: str) -> List[Dict]:
    """Parse JSONL format (one JSON object per line) - used by HuggingFace trainer."""
    entries = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        step = obj.get("step") or obj.get("global_step")
        epoch = obj.get("epoch")

        metrics = {}
        for key, val in obj.items():
            if key in ("step", "global_step", "epoch", "learning_rate",
                       "grad_norm", "total_flos", "train_runtime",
                       "train_samples_per_second", "train_steps_per_second"):
                if isinstance(val, (int, float)):
                    metrics[key] = val
            elif key.endswith("_loss") or key == "loss":
                if isinstance(val, (int, float)):
                    metrics[key] = val

        if metrics:
            entries.append({"step": step, "epoch": epoch, "metrics": metrics})

    return entries


def _parse_json(content: str) -> List[Dict]:
    """Parse JSON format - could be trainer_state.json or similar."""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    # HuggingFace trainer_state.json
    if "log_history" in data:
        return _parse_jsonl("\n".join(json.dumps(e) for e in data["log_history"]))

    # Could be a list of entries
    if isinstance(data, list):
        return _parse_jsonl("\n".join(json.dumps(e) for e in data))

    return []


def _parse_csv(content: str) -> List[Dict]:
    """Parse CSV format."""
    lines = content.strip().split("\n")
    if len(lines) < 2:
        return []

    headers = [h.strip() for h in lines[0].split(",")]
    entries = []

    for line in lines[1:]:
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(headers):
            continue

        row = {}
        for h, v in zip(headers, vals):
            try:
                row[h] = float(v)
            except ValueError:
                row[h] = v

        step = row.get("step") or row.get("global_step")
        metrics = {}
        for k, v in row.items():
            if isinstance(v, float) and k not in ("step", "global_step"):
                metrics[k] = v

        if metrics:
            entries.append({"step": int(step) if step else None, "metrics": metrics})

    return entries


# Regex patterns for text log parsing
_LOSS_PATTERN = re.compile(
    r"[\'\"]loss[\'\"]:\s*([\d.]+)"
)
_STEP_PATTERN = re.compile(
    r"(\d+)%\|.*?\|\s*(\d+)/(\d+)"
)
_EPOCH_PATTERN = re.compile(
    r"[\'\"]epoch[\'\"]:\s*([\d.]+)"
)
_LR_PATTERN = re.compile(
    r"[\'\"]learning_rate[\'\"]:\s*([\d.e\-+]+)"
)
_GRAD_NORM_PATTERN = re.compile(
    r"[\'\"]grad_norm[\'\"]:\s*([\d.e\-+]+)"
)


def _parse_text_log(content: str) -> List[Dict]:
    """Parse text-based training logs (HuggingFace Trainer, LlamaFactory, etc.)."""
    entries = []

    # Look for metric lines like: {'loss': 0.257, 'grad_norm': 0.025, 'learning_rate': 1.3e-05, 'epoch': 2.08}
    metric_line_pattern = re.compile(r"\{[^}]+\}")

    for line in content.split("\n"):
        match = metric_line_pattern.search(line)
        if not match:
            continue

        try:
            obj = json.loads(match.group().replace("'", '"'))
        except json.JSONDecodeError:
            continue

        step = obj.get("step")
        epoch = obj.get("epoch")

        metrics = {}
        for key, val in obj.items():
            if isinstance(val, (int, float)):
                metrics[key] = val

        if not step:
            # Try to extract step from progress bar pattern
            step_match = _STEP_PATTERN.search(line)
            if step_match:
                step = int(step_match.group(2))

        if metrics:
            entries.append({"step": step, "epoch": epoch, "metrics": metrics})

    return entries


def parse_trainer_state(path: str) -> Dict:
    """Parse a HuggingFace trainer_state.json and return summary + metrics."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    result = {
        "global_step": data.get("global_step"),
        "max_steps": data.get("max_steps"),
        "num_train_epochs": data.get("num_train_epochs"),
        "epoch": data.get("epoch"),
        "best_metric": data.get("best_metric"),
        "log_history": [],
    }

    if "log_history" in data:
        for entry in data["log_history"]:
            parsed = {"step": entry.get("step"), "metrics": {}}
            for k, v in entry.items():
                if k != "step" and isinstance(v, (int, float)):
                    parsed["metrics"][k] = v
            if parsed["metrics"]:
                result["log_history"].append(parsed)

    return result
