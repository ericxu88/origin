"""Origin command-line interface.

Commands are added as their subsystems land (SPEC §8); every registered
command is fully functional.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

import origin_ml
from origin_ml.features.pipeline import build_default_pipeline
from origin_ml.scoring.stub import StubScorer

app = typer.Typer(
    name="origin",
    help="Origin: interpretable detection and explanation of LLM-generated text.",
    no_args_is_help=True,
    add_completion=False,
)


def _read_text(path: Path | None, text: str | None) -> str:
    if (path is None) == (text is None):
        raise typer.BadParameter("provide exactly one of PATH or --text")
    if path is not None:
        if str(path) == "-":
            return sys.stdin.read()
        if not path.exists():
            raise typer.BadParameter(f"file not found: {path}")
        return path.read_text(encoding="utf-8")
    assert text is not None
    return text


@app.command()
def version() -> None:
    """Print the Origin version."""
    typer.echo(origin_ml.__version__)


@app.command()
def features(
    path: Annotated[
        Path | None,
        typer.Argument(help="Text file to analyze, or '-' for stdin."),
    ] = None,
    text: Annotated[str | None, typer.Option("--text", help="Analyze a literal string.")] = None,
    scorer: Annotated[
        str,
        typer.Option(
            help="LM scorer: 'stub' (deterministic, offline) or 'none' (skip LM features)."
        ),
    ] = "stub",
    json_out: Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table.")] = False,
) -> None:
    """Extract Origin's statistical feature vector from a document (SPEC CLI-2)."""
    document = _read_text(path, text)
    if scorer == "stub":
        pipeline = build_default_pipeline(scorer=StubScorer())
    elif scorer == "none":
        pipeline = build_default_pipeline(scorer=None)
    else:
        raise typer.BadParameter(f"unknown scorer '{scorer}' (expected 'stub' or 'none')")

    vector = pipeline.extract(document)
    if json_out:
        typer.echo(json.dumps(vector.as_dict(), indent=2))
    else:
        width = max(len(name) for name in vector.names)
        for name, value in zip(vector.names, vector.values, strict=True):
            typer.echo(f"{name:<{width}}  {value: .6f}")


def main() -> None:
    """Console-script entry point."""
    app()


if __name__ == "__main__":
    main()
