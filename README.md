# lecilab-behavior-analysis

Tools to analyze and plot the behavior of mice trained in the Training Village
(tasks in `auditory-task-python`). Used by the daily report
(`village-report-generator`), the visualizer (`behavior-data-visualizer`) and
the analysis notebooks (`behavior-analysis-notebooks`).

## Quick start

```python
import pandas as pd
from lecilab_behavior_analysis.preprocessing import prepare
from lecilab_behavior_analysis.figures import subject_progress, session_summary

df = pd.read_csv("NUO003.csv", sep=";")
df = prepare(df)                       # once, before any analysis or figure

fig, problems = subject_progress(df, perf_window=100)
fig, problems = session_summary(df[df.year_month_day == "2026-04-20"])
# problems: the panels that could not be drawn (empty if all worked)
```

The old entry points still work: `df_transforms.analyze_df` (same as
`prepare`), `figure_maker.subject_progress_figure` and
`figure_maker.session_summary_figure` (the failed panels are in
`fig.panel_problems`).

## How the package is organised

| Module | What it does |
|---|---|
| `preprocessing` | `prepare(df)`: fills missing task information, converts old names, reads the event columns once and adds every derived column (first choice, reaction time, center holds, total trial...). Figures and analyses only read these columns. |
| `data_format_changes` | Every change in the format of the saved data (what, since when, why) and how to read old and new rows. Add new format changes here. |
| `metrics/` | Numbers and tables from the trials, no plotting: `trial_events` (one trial's events), `performance`, `choices`, `timing`, `engagement`, `occupancy`, `sound`. |
| `models/` | Psychometric fits and logistic regressions. |
| `plots/` | Drawing functions: each one draws on the axes it is given. |
| `figures/` | Figures made of panels: `render.py` (drawing a layout, error handling), `panels.py` (one function per plot), `layouts.py` (which panels make each figure). |
| `reports/` | Tables and figures of the daily report; they never raise. |
| `io/` | Paths on each computer and on the cluster, rsync, loading csv files. |
| `utils`, `df_transforms`, `figure_maker` | Kept so that old code keeps working; they re-export the functions above with their old names. |

### Figures and panels

A figure is a layout of panels. Each panel is a small function that draws one
plot and lists the columns it needs:

```python
from lecilab_behavior_analysis.figures import panel, NotApplicable

@panel("center hold plot", needs={"year_month_day", "port2_holds"})
def center_holds(data, ax):
    ...                                  # data.df is the prepared dataframe
```

`render` draws every panel of a layout. A panel that fails shows a red
"Could not generate ..." box, the error is logged and returned in the list of
problems, and the rest of the figure is drawn. A panel with nothing to show
for this data raises `NotApplicable("message")` and shows the message in grey.
To add a plot to a figure: write the panel in `figures/panels.py` and add it
to a row in `figures/layouts.py`.

## Tests

```bash
python -m pytest tests -o addopts=""
```

`tests/data/NUO003_excerpt.csv.gz` holds 8 real days of one mouse (every
training stage); the tests draw every figure on each of them and fail if a
panel cannot be drawn. `example_data/` (large local files) is not tracked.
