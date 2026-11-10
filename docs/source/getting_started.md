# Getting started

## Install

```bash
pip install git+https://github.com/LearningCircuitsLab/lecilab-behavior-analysis
```

or, to change the code, clone the repository and `pip install -e .`.

## Where the data is

Every project has a folder in the data folder of each computer:

```
<data folder>/
  <project>/
    sessions/<subject>/<subject>.csv     one row per trial
    village01/events.csv                 one folder per setup (box)
    village01/old_events/*.csv
    village01/sessions_summary.csv       one row per session
    village01/temperatures.csv
    village02/...
```

`io.server.get_outpath()` returns the data folder of the computer it runs on
(`/storage/training_village` on the cluster). Set the `LECILAB_DATA_PATH`
environment variable to use another folder (e.g. a sample of the data).

## A first figure

```python
from lecilab_behavior_analysis.io.loading import load_project_trials, load_project_logs
from lecilab_behavior_analysis.figures import subject_progress, session_summary

# trials of some subjects, prepared for any analysis (missing subjects are skipped)
df = load_project_trials("COT_cannula_GAD2_data", ["NUO057", "NUO060"], max_date="2026-05-26")
# events and sessions summary of every setup of the project, together
events, sessions_summary, setups = load_project_logs("COT_cannula_GAD2_data", max_date="2026-05-26")

mouse = df[df.subject == "NUO057"]
fig, problems = subject_progress(mouse, perf_window=100, events_df=events)
fig, problems = session_summary(mouse[mouse.year_month_day == "2026-05-25"])
# problems: the panels that could not be drawn (empty if all worked)
```

From a csv file, prepare the trials first:

```python
import pandas as pd
from lecilab_behavior_analysis.preprocessing import prepare

df = prepare(pd.read_csv("NUO003.csv", sep=";"))
```

`prepare` only adds the columns that are missing, so calling it again (or on
a subset) is cheap.

## Old code

The modules used before the package was reorganized still work, with the same
function names: `utils`, `df_transforms` (`analyze_df` is `prepare`),
`plots` and `figure_maker` (`subject_progress_figure`,
`session_summary_figure`; the failed panels are in `fig.panel_problems`).
New code should import from the modules described in the
[API](api_index).

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests -o addopts=""
```

`tests/data/NUO003_excerpt.csv.gz` holds 8 real days of one mouse, covering
every training stage. The tests draw every figure on each of them and fail if
any panel cannot be drawn.
