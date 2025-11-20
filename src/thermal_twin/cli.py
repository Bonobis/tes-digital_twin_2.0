"""Command-line entry points for the thermal twin package."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Callable, Dict

import meshio
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskID, TextColumn, TimeRemainingColumn
from rich.table import Table

from .scenarios.runner import ScenarioRunner
from .scenarios.schema import ScenarioConfig

app = typer.Typer(add_completion=False, help="Thermal Twin CLI")
mesh_app = typer.Typer(help="Geometry and mesh utilities")
app.add_typer(mesh_app, name="mesh")
console = Console()


def _prepare_output_path(path: Path, label: str) -> None:
    if path.exists() and path.is_dir():
        raise typer.BadParameter(f"{label} must be a file path, not an existing directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_dispatcher(sinks: list[Callable[[Dict[str, Any]], None]]):
    if not sinks:
        return None

    def _dispatch(event: Dict[str, Any]) -> None:
        for sink in sinks:
            sink(event)

    return _dispatch


def _file_sink(handle) -> Callable[[Dict[str, Any]], None]:
    def _sink(event: Dict[str, Any]) -> None:
        handle.write(json.dumps(event) + "\n")
        handle.flush()

    return _sink


def _progress_sink(progress: Progress, task_id: TaskID) -> Callable[[Dict[str, Any]], None]:
    def _sink(event: Dict[str, Any]) -> None:
        event_type = event.get("event")
        if event_type == "start":
            progress.update(task_id, completed=0.0, description="Simulating")
        elif event_type == "step":
            progress_value = float(event.get("progress", 0.0))
            description = "Simulating"
            time_s = event.get("time")
            if time_s is not None:
                description = f"Sim t={time_s:.0f}s"
            progress.update(task_id, completed=max(0.0, min(1.0, progress_value)), description=description)
        elif event_type == "finish":
            progress.update(task_id, completed=1.0, description="Simulation complete")

    return _sink


@app.command()
def show_config(path: Path) -> None:
    """Display a parsed scenario configuration."""
    config = ScenarioConfig.from_file(path)
    console.print_json(data=config.model_dump(mode="json"))


@app.command()
def simulate(
    path: Path,
    output: Path | None = typer.Option(None, help="Optional JSON file to store the final summary."),
    telemetry_log: Path | None = typer.Option(None, help="Write streamed telemetry (NDJSON) to this path."),
    show_progress: bool = typer.Option(True, "--progress/--no-progress", help="Display a live progress bar."),
) -> None:
    """Run a simulation defined by a scenario YAML."""
    config = ScenarioConfig.from_file(path)
    with ExitStack() as stack:
        sinks: list[Callable[[Dict[str, Any]], None]] = []
        if telemetry_log is not None:
            _prepare_output_path(telemetry_log, "Telemetry log")
            log_handle = stack.enter_context(telemetry_log.open("w"))
            sinks.append(_file_sink(log_handle))
        if show_progress:
            progress = stack.enter_context(
                Progress(
                    TextColumn("{task.description}"),
                    BarColumn(),
                    TimeRemainingColumn(),
                )
            )
            task_id = progress.add_task("Simulating", total=1.0)
            sinks.append(_progress_sink(progress, task_id))
        telemetry_cb = _build_dispatcher(sinks)
        runner = ScenarioRunner(config, telemetry=telemetry_cb)
        result = runner.run()
    if telemetry_log is not None:
        result["telemetry_log"] = str(telemetry_log)
    if output:
        _prepare_output_path(output, "Output")
        output.write_text(json.dumps(result, indent=2))
    _render_summary(result)


@mesh_app.command("build")
def mesh_build(
    path: Path,
    preview: Path | None = typer.Option(None, help="Optional path to export a preview mesh (format inferred from extension)."),
) -> None:
    """Construct the geometry and generate a mesh without running the solver."""
    config = ScenarioConfig.from_file(path)
    runner = ScenarioRunner(config)
    geometry, mesh_result = runner.build_mesh()
    console.print(f"Mesh generated at [bold]{mesh_result.mesh_path}[/bold]")
    quality = mesh_result.quality
    if quality:
        console.print(f"Quality: min={quality.get('min', 0):.3f}, mean={quality.get('mean', 0):.3f}, cells={quality.get('count', 0)}")
    console.print(f"Geometry elements: {len(geometry.elements)}")
    if preview:
        _prepare_output_path(preview, "Preview output")
        meshio.write(preview, mesh_result.mesh)
        console.print(f"Preview mesh exported to [bold]{preview}[/bold]")


@mesh_app.command("preview")
def mesh_preview(mesh_path: Path, output: Path) -> None:
    """Convert an existing mesh file to another format for visualization."""
    mesh = meshio.read(mesh_path)
    _prepare_output_path(output, "Preview output")
    meshio.write(output, mesh)
    console.print(f"Mesh preview written to [bold]{output}[/bold]")


def _render_summary(result: Dict[str, Any]) -> None:
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

