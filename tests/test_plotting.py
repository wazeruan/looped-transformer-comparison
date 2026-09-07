import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from looped_transformer_comparison.plotting import plot_runs


ROOT = Path(__file__).resolve().parents[1]


def _write_metrics(run: Path, architecture: str):
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(json.dumps({
        "architecture": architecture,
        "config": {"model": {"seq_len": 8},
                   "training": {"batch_size": 2, "grad_accum": 4}},
    }))
    rows = [
        {"step": 1, "train_loss": 3.5, "lr": 0.001},
        {"step": 2, "train_loss": 3.0, "lr": 0.0009,
         "validation": {"loss": 3.2, "perplexity": 24.5}},
        {"step": 3, "train_loss": 2.7, "lr": 0.0008},
    ]
    (run / "metrics.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n{"  # incomplete flush
    )


def test_plot_once_renders_both_architectures_and_ignores_partial_line(tmp_path):
    _write_metrics(tmp_path / "standard", "standard")
    _write_metrics(tmp_path / "looped", "looped")

    result = plot_runs(tmp_path, once=True, dpi=72)

    assert result == {"output": str(tmp_path), "updated": True}
    for architecture in ("standard", "looped"):
        image = tmp_path / architecture / "training.png"
        assert image.stat().st_size > 0
        assert image.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert not (tmp_path / architecture / "training.tmp.png").exists()


def test_plot_missing_logs_and_invalid_interval(tmp_path):
    assert plot_runs(tmp_path, once=True) == {"output": str(tmp_path), "updated": False}
    with pytest.raises(ValueError, match="interval must be positive"):
        plot_runs(tmp_path, interval=0)


def test_plot_cli_once_and_help_are_headless(tmp_path):
    _write_metrics(tmp_path / "standard", "standard")
    env = {**os.environ, "MPLBACKEND": "Agg"}
    command = [sys.executable, "-m", "looped_transformer_comparison.cli", "plot",
               "--output", str(tmp_path), "--once", "--dpi", "72"]
    result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["updated"] is True
    help_result = subprocess.run(
        [sys.executable, "-m", "looped_transformer_comparison.cli", "plot", "--help"],
        cwd=ROOT, env=env,
                                 capture_output=True, text=True, timeout=30)
    assert help_result.returncode == 0
    assert "--watch" in help_result.stdout


def test_plot_launchers_have_valid_shell_and_slurm_syntax():
    for script in (ROOT / "scripts/plot.sh", ROOT / "scripts/plot.sbatch"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    assert "--watch" in (ROOT / "scripts/plot.sbatch").read_text()
