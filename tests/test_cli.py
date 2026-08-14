"""CLI tests (SPEC CLI-2 for now; more commands land in later phases)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from origin_ml.cli import app

runner = CliRunner()


class TestFeaturesCommand:
    def test_json_output_with_stub_scorer(self) -> None:
        result = runner.invoke(app, ["features", "--text", "Hello world. Goodbye moon.", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert payload["sent.count"] == 2.0
        assert "ppl.doc_perplexity" in payload

    def test_scorer_none_drops_lm_features(self) -> None:
        result = runner.invoke(
            app, ["features", "--text", "Hello world.", "--scorer", "none", "--json"]
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output)
        assert not any(key.startswith("ppl.") for key in payload)

    def test_reads_file(self, tmp_path: Path) -> None:
        doc = tmp_path / "doc.txt"
        doc.write_text("One sentence here.", encoding="utf-8")
        result = runner.invoke(app, ["features", str(doc), "--json"])
        assert result.exit_code == 0, result.output
        assert json.loads(result.output)["sent.count"] == 1.0

    def test_table_output(self) -> None:
        result = runner.invoke(app, ["features", "--text", "Hello world."])
        assert result.exit_code == 0
        assert "sent.count" in result.output

    def test_requires_exactly_one_input(self) -> None:
        assert runner.invoke(app, ["features"]).exit_code != 0
        both = runner.invoke(app, ["features", "somefile", "--text", "x"])
        assert both.exit_code != 0

    def test_unknown_scorer_rejected(self) -> None:
        result = runner.invoke(app, ["features", "--text", "x", "--scorer", "bogus"])
        assert result.exit_code != 0


class TestVersionCommand:
    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.output.strip()
