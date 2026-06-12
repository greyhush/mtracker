# mtracker

Lightweight ML experiment tracker. Local SQLite storage. CLI-first. Auto-parse training logs.

No server. No account. No cloud. Just track your experiments.

## Install

```bash
pip install -e .

# With Rich terminal UI (recommended):
pip install -e ".[rich]"
```

## Quick Start

### CLI

```bash
# Start a run
mtracker start "qwen3-9b-sft" --project physics --tags "qlora,4bit" --config '{"lr": 5e-5, "batch_size": 16}'

# Log metrics
mtracker log 1 loss 0.5 --step 100
mtracker log 1 loss 0.3 --step 200

# End the run
mtracker end 1 --status completed

# List all runs
mtracker list

# Show run details
mtracker show 1

# Compare runs
mtracker compare 1,2,3

# Import from HuggingFace trainer_state.json
mtracker import trainer_state.json --create --project physics

# Import from any log file (auto-detects format: JSONL, JSON, CSV, text)
mtracker import train.log --create

# Check status
mtracker status
```

### Python API

```python
from mtracker import Run

# Context manager - auto-completes on success, auto-fails on exception
with Run("my-experiment", project="physics", config={"lr": 5e-5}) as run:
    for step in range(1000):
        loss = train_step()
        run.log("loss", loss, step=step)
        run.log("lr", get_lr(), step=step)

# Or use log_dict for multiple metrics at once
with Run("my-experiment") as run:
    run.log_dict({"loss": 0.5, "acc": 0.8, "lr": 5e-5}, step=1)
```

### Import Existing Logs

```python
from mtracker.api import import_log_file, import_trainer_state
from mtracker import Run

# Import HuggingFace trainer_state.json
run = Run("physics-sft", project="physics")
import_trainer_state(run.id, "saves/sft_lora/checkpoint-3000/trainer_state.json")

# Import any log file (auto-detects format)
import_log_file(run.id, "train.log")
```

## Supported Log Formats

| Format | Extension | Source |
|--------|-----------|--------|
| JSONL | `.jsonl` | HuggingFace Trainer logs |
| JSON | `.json` | `trainer_state.json`, custom |
| CSV | `.csv` | Any CSV with headers |
| Text | `.log`, `.txt` | HuggingFace progress output, LlamaFactory |

## Storage

All data stored locally in `~/.mtracker/mtracker.db` (SQLite). No accounts, no cloud, no telemetry.

## Commands

| Command | Description |
|---------|-------------|
| `mtracker init` | Initialize database |
| `mtracker start NAME` | Start a new run |
| `mtracker end ID` | End a run |
| `mtracker log ID NAME VALUE` | Log a metric |
| `mtracker list` | List all runs |
| `mtracker show ID` | Show run details |
| `mtracker import FILE` | Import log file |
| `mtracker compare IDS` | Compare runs |
| `mtracker delete ID` | Delete a run |
| `mtracker status` | Show current status |

## License

MIT
