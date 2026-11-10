# Figures

The images on this page are drawn from `tests/data` by the current code every
time the documentation is built.

## Training history of a subject

`figures.subject_progress(df, perf_window=100, events_df=None)`: calendar of
trials, training times, performance, trials and water per day, performance
by trial type, bias, center port, reaction times and engagement (needs the
events).

![Subject progress](_static/examples/subject_progress.png)

## Summary of a session

`figures.session_summary(df_of_one_day, perf_window=50)`.

![Session summary](_static/examples/session_summary.png)

## Panels and layouts

A figure is a `Layout`: rows of panels, each with a relative width. A panel
is a small function that draws one plot on the axes it is given, and lists
the columns it needs:

```python
import pandas as pd
from lecilab_behavior_analysis.figures import FigureData, Layout, NotApplicable, Row, panel, render
from lecilab_behavior_analysis.figures import panels as p

@panel("trials per day", needs={"year_month_day"})
def trials_per_day(data, ax):
    counts = data.df.groupby("year_month_day").size()   # data.df: the prepared trials
    ax.bar(pd.to_datetime(counts.index), counts.values)
    ax.set_ylabel("Trials")

@panel("broken plot", needs={"year_month_day"})
def broken(data, ax):
    raise ValueError("this panel has a bug")

@panel("optogenetics", needs={"year_month_day"})
def optogenetics(data, ax):
    if "opto" not in data.df.columns:
        raise NotApplicable("No optogenetics in these sessions")
    ...

layout = Layout([Row([(p.subject_summary, 1), (trials_per_day, 2)]),
                 Row([(broken, 1), (optogenetics, 1), (p.subject_performance, 1)])],
                width=12, height=6)
fig, problems = render(layout, FigureData(df, perf_window=100))
```

![A layout with a panel that fails](_static/examples/custom_layout.png)

`render` never stops because of one panel:

- a panel that raises shows a red *Could not generate ...* box, the error is
  logged, and it is returned in `problems` (the daily report lists them);
- a panel that needs a column the data does not have does the same, without
  running;
- a panel with nothing to show for this data raises
  `NotApplicable("message")`, which shows the message in grey (not a problem).

To add a plot to the standard figures, write the panel in `figures/panels.py`
and add it to a row in `figures/layouts.py`. Panels that need the same table
can compute it once per figure with `data.cached(key, function)`. A panel can
use several axes with `split_cell(ax, nrows, ncols)`.
