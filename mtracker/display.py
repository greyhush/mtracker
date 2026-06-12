"""Rich terminal display for mtracker."""

from typing import Dict, List, Optional


def _format_duration(seconds: Optional[float]) -> str:
    if not seconds:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def _format_value(val: Optional[float], precision: int = 4) -> str:
    if val is None:
        return "-"
    if abs(val) < 0.001 or abs(val) > 1e6:
        return f"{val:.2e}"
    return f"{val:.{precision}f}"


def _get_status_icon(status: str) -> str:
    icons = {
        "running": "🟢",
        "completed": "✅",
        "failed": "❌",
        "crashed": "💥",
        "stopped": "⏹️",
    }
    return icons.get(status, "❓")


def print_runs_table(runs: List[Dict], show_metrics: bool = False,
                     metrics_db=None):
    """Print a formatted table of runs."""
    try:
        from rich.console import Console
        from rich.table import Table
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    if not runs:
        print("No runs found.")
        return

    if HAS_RICH:
        _print_rich_table(runs, show_metrics, metrics_db)
    else:
        _print_plain_table(runs, show_metrics, metrics_db)


def _print_rich_table(runs, show_metrics, metrics_db):
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="mtracker runs", show_lines=True)

    table.add_column("ID", style="dim", width=4)
    table.add_column("Status", width=3)
    table.add_column("Name", style="bold")
    table.add_column("Project", style="cyan")
    table.add_column("Tags", style="magenta")
    table.add_column("Duration", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Best Loss", justify="right", style="green")

    for run in runs:
        run_id = str(run["id"])
        status = _get_status_icon(run.get("status", ""))
        name = run.get("name", "")
        project = run.get("project", "default")
        tags = ", ".join(run.get("tags", [])) or "-"

        duration = None
        if run.get("ended_at") and run.get("started_at"):
            duration = run["ended_at"] - run["started_at"]
        elif run.get("status") == "running":
            import time
            duration = time.time() - run.get("started_at", time.time())

        # Get metrics summary
        steps = "-"
        best_loss = "-"
        if metrics_db:
            latest = metrics_db.get_latest_metrics(run["id"])
            if "step" in latest:
                steps = str(int(latest["step"]["value"]))
            elif "global_step" in latest:
                steps = str(int(latest["global_step"]["value"]))

            loss_summary = metrics_db.get_metric_summary(run["id"], "loss")
            if loss_summary:
                best_loss = _format_value(loss_summary["min_val"])

        table.add_row(
            run_id, status, name, project, tags,
            _format_duration(duration), steps, best_loss
        )

    console.print(table)


def _print_plain_table(runs, show_metrics, metrics_db):
    print(f"\n{'ID':<5} {'St':<3} {'Name':<20} {'Project':<12} {'Duration':<10} {'Steps':<8} {'Loss':<10}")
    print("-" * 70)

    for run in runs:
        run_id = run["id"]
        status = _get_status_icon(run.get("status", ""))
        name = run.get("name", "")[:20]
        project = run.get("project", "default")[:12]

        duration = "-"
        if run.get("ended_at") and run.get("started_at"):
            duration = _format_duration(run["ended_at"] - run["started_at"])

        steps = "-"
        best_loss = "-"
        if metrics_db:
            latest = metrics_db.get_latest_metrics(run["id"])
            if "step" in latest:
                steps = str(int(latest["step"]["value"]))
            loss_summary = metrics_db.get_metric_summary(run["id"], "loss")
            if loss_summary:
                best_loss = _format_value(loss_summary["min_val"])

        print(f"{run_id:<5} {status:<3} {name:<20} {project:<12} {duration:<10} {steps:<8} {best_loss:<10}")

    print()


def print_run_detail(run: Dict, metrics_db=None):
    """Print detailed info for a single run."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    if HAS_RICH:
        _print_rich_detail(run, metrics_db)
    else:
        _print_plain_detail(run, metrics_db)


def _print_rich_detail(run, metrics_db):
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Run info
    status = _get_status_icon(run.get("status", ""))
    info_lines = [
        f"[bold]{run.get('name', 'unnamed')}[/bold]",
        f"ID: {run['id']}  |  Project: {run.get('project', 'default')}  |  Status: {status} {run.get('status', '')}",
        f"Tags: {', '.join(run.get('tags', [])) or 'none'}",
    ]

    if run.get("started_at"):
        import time
        started = run["started_at"]
        if run.get("ended_at"):
            dur = run["ended_at"] - started
        else:
            dur = time.time() - started
        info_lines.append(f"Duration: {_format_duration(dur)}")

    config = run.get("config", {})
    if config:
        info_lines.append(f"Config: {config}")

    if run.get("notes"):
        info_lines.append(f"Notes: {run['notes']}")

    console.print(Panel("\n".join(info_lines), title="Run Info"))

    # Metrics
    if metrics_db:
        all_metrics = metrics_db.get_latest_metrics(run["id"])
        if all_metrics:
            table = Table(title="Latest Metrics")
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")
            table.add_column("Step", justify="right")
            table.add_column("Min", justify="right", style="green")
            table.add_column("Max", justify="right", style="red")
            table.add_column("Avg", justify="right")

            for name, info in sorted(all_metrics.items()):
                summary = metrics_db.get_metric_summary(run["id"], name)
                min_v = _format_value(summary["min_val"]) if summary else "-"
                max_v = _format_value(summary["max_val"]) if summary else "-"
                avg_v = _format_value(summary["avg_val"]) if summary else "-"
                table.add_row(
                    name,
                    _format_value(info["value"]),
                    str(info["step"]) if info["step"] else "-",
                    min_v, max_v, avg_v
                )

            console.print(table)


def _print_plain_detail(run, metrics_db):
    status = _get_status_icon(run.get("status", ""))
    print(f"\n{'='*50}")
    print(f"  {run.get('name', 'unnamed')}  {status}")
    print(f"  ID: {run['id']}  Project: {run.get('project', 'default')}")
    print(f"  Tags: {', '.join(run.get('tags', [])) or 'none'}")

    if run.get("notes"):
        print(f"  Notes: {run['notes']}")
    print(f"{'='*50}")

    if metrics_db:
        all_metrics = metrics_db.get_latest_metrics(run["id"])
        if all_metrics:
            print(f"\n{'Metric':<15} {'Value':<12} {'Step':<8} {'Min':<12} {'Max':<12}")
            print("-" * 60)
            for name, info in sorted(all_metrics.items()):
                summary = metrics_db.get_metric_summary(run["id"], name)
                min_v = _format_value(summary["min_val"]) if summary else "-"
                max_v = _format_value(summary["max_val"]) if summary else "-"
                print(f"{name:<15} {_format_value(info['value']):<12} "
                      f"{str(info['step'] or '-'):<8} {min_v:<12} {max_v:<12}")
    print()


def print_compare(runs_data: List[Dict]):
    """Print comparison table for multiple runs."""
    try:
        from rich.console import Console
        from rich.table import Table
        HAS_RICH = True
    except ImportError:
        HAS_RICH = False

    if not runs_data:
        print("No runs to compare.")
        return

    # Collect all metric names
    all_metric_names = set()
    for rd in runs_data:
        all_metric_names.update(rd.get("metrics", {}).keys())

    metric_names = sorted(all_metric_names)

    if HAS_RICH:
        console = Console()
        table = Table(title="Run Comparison")

        table.add_column("Run", style="bold")
        for m in metric_names:
            table.add_column(m, justify="right")

        for rd in runs_data:
            row = [rd.get("name", str(rd.get("id", "?")))]
            for m in metric_names:
                val = rd.get("metrics", {}).get(m, {})
                if isinstance(val, dict):
                    row.append(_format_value(val.get("value")))
                else:
                    row.append(_format_value(val))
            table.add_row(*row)

        console.print(table)
    else:
        print(f"\n{'Run':<20}", end="")
        for m in metric_names:
            print(f"{m:<15}", end="")
        print()
        print("-" * (20 + 15 * len(metric_names)))

        for rd in runs_data:
            name = rd.get("name", str(rd.get("id", "?")))[:20]
            print(f"{name:<20}", end="")
            for m in metric_names:
                val = rd.get("metrics", {}).get(m, {})
                if isinstance(val, dict):
                    print(f"{_format_value(val.get('value')):<15}", end="")
                else:
                    print(f"{_format_value(val):<15}", end="")
            print()
        print()
