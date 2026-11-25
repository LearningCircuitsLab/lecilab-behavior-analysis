# Tutorial

A [marimo](https://marimo.io) notebook that goes through the main analyses,
from the data saved by the village to the figures of the daily reports:
loading and preparing the trials, a first look per day, the training history
and session figures, psychometric curves and their fit, the settings of each
session and how they changed, and your own figure made of panels.

**[Open the tutorial, already run](_static/examples/tutorial.html)** (it is run
again on the example data each time the documentation is built).

## Running it yourself

The notebook is `notebooks/tutorial.py` in the repository. It runs on the
example data of the package (`tests/data`) and, by changing two lines, on a
project of your data folder.

```bash
pip install marimo
marimo edit notebooks/tutorial.py
```

(from the root of the repository). marimo notebooks are Python files: the
cells re-run when what they depend on changes (e.g. the day chosen in the
list), and `python notebooks/tutorial.py` runs it as a script.

Every plot of the package, with the code that draws it, is in the
[gallery](gallery.md).
