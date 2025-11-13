"""Command-line entry points for the thermal twin package."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .scenarios.runner import ScenarioRunner
from .scenarios.schema import ScenarioConfig

app = typer.Typer(add_completion=False, help="Thermal Twin CLI")
console = Console()


@app.command()
def show_config(path: Path) -> None:
    """Display a parsed scenario configuration."""
    config = ScenarioConfig.from_file(path)
    console.print_json(data=config.model_dump())


@app.command()
def simulate(path: Path, output: Path | None = None) -> None:
    """Run a simulation defined by a scenario YAML."""
    config = ScenarioConfig.from_file(path)
    runner = ScenarioRunner(config)
    result = runner.run()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2))
    table = Table(title="Simulation Summary")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in result.items():
        table.add_row(str(key), str(value))
    console.print(table)


def app_entry() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    app_entry()
