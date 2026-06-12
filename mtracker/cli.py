#!/usr/bin/env python3
"""mtracker CLI - Track ML experiments from the terminal."""

import argparse
import sys
import json
from pathlib import Path


def cmd_init(args):
    """Initialize mtracker database."""
    from mtracker.db import Database
    db = Database(args.db)
    print(f"mtracker initialized at {db.db_path}")
    db.close()


def cmd_start(args):
    """Start a new run."""
    from mtracker.db import Database
    db = Database(args.db)
    config = json.loads(args.config) if args.config else {}
    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []
    run_id = db.create_run(args.name, args.project, config, tags)
    print(f"Started run '{args.name}' (ID: {run_id})")
    db.close()


def cmd_end(args):
    """End a run."""
    from mtracker.db import Database
    db = Database(args.db)
    db.end_run(args.run_id, args.status, args.notes or "")
    print(f"Run {args.run_id} ended with status '{args.status}'")
    db.close()


def cmd_log(args):
    """Log a metric to a run."""
    from mtracker.db import Database
    db = Database(args.db)
    db.log_metric(args.run_id, args.name, args.value, step=args.step)
    step_str = f" at step {args.step}" if args.step else ""
    print(f"Logged {args.name}={args.value}{step_str}")
    db.close()


def cmd_list(args):
    """List runs."""
    from mtracker.db import Database
    from mtracker.display import print_runs_table
    db = Database(args.db)
    runs = db.list_runs(project=args.project, status=args.status, limit=args.limit)
    print_runs_table(runs, metrics_db=db)
    db.close()


def cmd_show(args):
    """Show details of a run."""
    from mtracker.db import Database
    from mtracker.display import print_run_detail
    db = Database(args.db)
    run = db.get_run(args.run_id)
    if not run:
        print(f"Run {args.run_id} not found.")
        sys.exit(1)
    print_run_detail(run, metrics_db=db)
    db.close()


def cmd_import(args):
    """Import a log file or trainer_state.json into a run."""
    from mtracker.db import Database
    from mtracker.api import import_log_file, import_trainer_state

    if args.create:
        # Auto-create a run from the file
        db = Database(args.db)
        name = Path(args.file).stem
        run_id = db.create_run(
            name=name,
            project=args.project,
            config={"source_file": str(args.file)}
        )
        db.close()
        print(f"Created run '{name}' (ID: {run_id})")
    else:
        run_id = args.run_id
        if not run_id:
            print("Error: --run-id required (or use --create)")
            sys.exit(1)

    path = Path(args.file)
    if path.name == "trainer_state.json":
        count = import_trainer_state(run_id, str(args.file), args.db)
    else:
        count = import_log_file(run_id, str(args.file), args.db)

    print(f"Imported {count} metric entries into run {run_id}")


def cmd_compare(args):
    """Compare multiple runs."""
    from mtracker.api import compare_runs
    from mtracker.display import print_compare

    run_ids = [int(x) for x in args.run_ids.split(",")]
    runs_data = compare_runs(run_ids, args.db)
    print_compare(runs_data)


def cmd_delete(args):
    """Delete a run."""
    from mtracker.db import Database
    db = Database(args.db)
    if not args.force:
        run = db.get_run(args.run_id)
        if not run:
            print(f"Run {args.run_id} not found.")
            sys.exit(1)
        resp = input(f"Delete run {args.run_id} '{run['name']}'? [y/N] ")
        if resp.lower() != 'y':
            print("Cancelled.")
            return
    db.delete_run(args.run_id)
    print(f"Run {args.run_id} deleted.")
    db.close()


def cmd_status(args):
    """Show current status - active runs and latest metrics."""
    from mtracker.db import Database
    from mtracker.display import print_runs_table
    db = Database(args.db)
    running = db.list_runs(status="running", limit=10)
    if running:
        print("Active runs:")
        print_runs_table(running, metrics_db=db)
    else:
        print("No active runs.")

    recent = db.list_runs(limit=5)
    if recent:
        print("\nRecent runs:")
        print_runs_table(recent, metrics_db=db)
    db.close()


def main():
    parser = argparse.ArgumentParser(
        prog="mtracker",
        description="Lightweight ML experiment tracker"
    )
    parser.add_argument("--db", default=None, help="Database path (default: ~/.mtracker/mtracker.db)")

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize database")

    # start
    p = sub.add_parser("start", help="Start a new run")
    p.add_argument("name", help="Run name")
    p.add_argument("--project", "-p", default="default", help="Project name")
    p.add_argument("--config", "-c", default=None, help="Config JSON string")
    p.add_argument("--tags", "-t", default=None, help="Comma-separated tags")

    # end
    p = sub.add_parser("end", help="End a run")
    p.add_argument("run_id", type=int, help="Run ID")
    p.add_argument("--status", "-s", default="completed",
                   choices=["completed", "failed", "stopped"])
    p.add_argument("--notes", "-n", default=None, help="End notes")

    # log
    p = sub.add_parser("log", help="Log a metric")
    p.add_argument("run_id", type=int, help="Run ID")
    p.add_argument("name", help="Metric name")
    p.add_argument("value", type=float, help="Metric value")
    p.add_argument("--step", "-s", type=int, default=None, help="Step number")

    # list
    p = sub.add_parser("list", help="List runs")
    p.add_argument("--project", "-p", default=None)
    p.add_argument("--status", "-s", default=None)
    p.add_argument("--limit", "-l", type=int, default=20)

    # show
    p = sub.add_parser("show", help="Show run details")
    p.add_argument("run_id", type=int, help="Run ID")

    # import
    p = sub.add_parser("import", help="Import log file into run")
    p.add_argument("file", help="Log file path")
    p.add_argument("--run-id", "-r", type=int, default=None, help="Target run ID")
    p.add_argument("--create", action="store_true", help="Auto-create run from file")
    p.add_argument("--project", "-p", default="default", help="Project (with --create)")

    # compare
    p = sub.add_parser("compare", help="Compare runs")
    p.add_argument("run_ids", help="Comma-separated run IDs (e.g. 1,2,3)")

    # delete
    p = sub.add_parser("delete", help="Delete a run")
    p.add_argument("run_id", type=int, help="Run ID")
    p.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    # status
    sub.add_parser("status", help="Show current status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "init": cmd_init,
        "start": cmd_start,
        "end": cmd_end,
        "log": cmd_log,
        "list": cmd_list,
        "show": cmd_show,
        "import": cmd_import,
        "compare": cmd_compare,
        "delete": cmd_delete,
        "status": cmd_status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
