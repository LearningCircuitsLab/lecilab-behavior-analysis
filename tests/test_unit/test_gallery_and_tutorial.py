"""The examples of the gallery (docs/source/gallery_examples.py) and the tutorial notebook draw without errors."""

import os
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest
from matplotlib import pyplot as plt

REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO / "docs" / "source"))

import gallery_examples  # noqa: E402


@pytest.fixture(scope="module")
def summary():
    return pd.read_csv(REPO / "tests" / "data" / "NUO003_sessions_summary.csv", sep=";")


@pytest.mark.parametrize("item", gallery_examples.EXAMPLES, ids=lambda item: item.function.__name__)
def test_gallery_example(item, prepared_trials, summary):
    day = prepared_trials[prepared_trials["year_month_day"] == gallery_examples.TRAINING_DAY]
    fig = item.function(prepared_trials, day, summary)
    if item.interactive:
        assert fig.to_html(include_plotlyjs=False)
    else:
        assert fig.axes
        plt.close(fig)


def test_gallery_code_is_shown():
    """The page shows the body of each example (make_examples.example_code)."""
    import make_examples

    code = make_examples.example_code(gallery_examples.training_matrix_example)
    assert code.startswith("from matplotlib import pyplot as plt")
    assert "training_matrix_plot(p_left, sessions, settings" in code
    assert "return" not in code.splitlines()[-1]
    assert make_examples.example_code(gallery_examples.cohort_example).splitlines()[-1].startswith("fig = ")


def test_tutorial_notebook_runs(tmp_path):
    pytest.importorskip("marimo")
    result = subprocess.run(
        [sys.executable, str(REPO / "notebooks" / "tutorial.py")],
        cwd=tmp_path,  # from another folder: the notebook finds the data next to itself
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert "lapses:" in result.stdout
