# lecilab-behavior-analysis

**[Documentation and examples](https://learningcircuitslab.github.io/lecilab-behavior-analysis/)** ·
[Daily reports](https://learningcircuitslab.github.io/village-report-generator/)

Analysis and figures of the behavior of mice trained in the
[Training Village](https://braincircuitsbehaviorlab.github.io/village/), in
the tasks of `auditory-task-python` (two-alternative forced choice with
visual, auditory and multisensory stimuli). Used by the daily reports
([village-report-generator](https://github.com/LearningCircuitsLab/village-report-generator)),
the visualizer (`behavior-data-visualizer`) and the analysis notebooks
(`behavior-analysis-notebooks`).

## What it does

- **Load** the trials of a subject or a project, and the events and sessions
  summary of every setup (box) of a project, found automatically.
- **Prepare** the trials once (`prepare`): read old and new data formats and
  add every derived column (choices, reaction times, center holds, misses...).
- **Measure** performance, choices and biases, timing, engagement, box
  occupancy; fit psychometric curves and regressions.
- **Draw** the training history of a subject and the summary of a session,
  as figures made of panels: a panel that fails shows an error box instead of
  breaking the figure.
- **Report**: tables and figures of the daily report that never raise.

## Quick start

```bash
pip install git+https://github.com/LearningCircuitsLab/lecilab-behavior-analysis
```

```python
from lecilab_behavior_analysis.io.loading import load_project_trials, load_project_logs
from lecilab_behavior_analysis.figures import subject_progress, session_summary

df = load_project_trials("COT_cannula_GAD2_data", ["NUO057", "NUO060"], max_date="2026-05-26")
events, sessions_summary, setups = load_project_logs("COT_cannula_GAD2_data", max_date="2026-05-26")

mouse = df[df.subject == "NUO057"]
fig, problems = subject_progress(mouse, perf_window=100, events_df=events)
fig, problems = session_summary(mouse[mouse.year_month_day == "2026-05-25"])
# problems: the panels that could not be drawn (empty if all worked)
```

The data is read from the data folder of each computer (`io.server.get_outpath`),
or from the folder in the `LECILAB_DATA_PATH` environment variable. From a csv
file: `prepare(pd.read_csv("NUO003.csv", sep=";"))`.

## How the package is organised

| Module | What it does |
|---|---|
| `io/` | Where the data is on each computer and on the cluster (`server`), loading trials, events and sessions summaries (`loading`). |
| `preprocessing` | `prepare(df)`: fills missing task information, converts old names, reads the event columns once and adds every derived column. |
| `data_format_changes` | Every change in the format of the saved data (what, since when, why) and how to read old and new rows. Add new format changes here. |
| `metrics/` | Numbers and tables, no plotting: `trial_events`, `performance`, `choices`, `timing`, `engagement`, `occupancy`, `sound`. |
| `models/` | Psychometric fits and logistic regressions. |
| `plots/` | Drawing functions: each one draws on the axes it is given. |
| `figures/` | Figures made of panels: `render.py` (drawing a layout, error handling), `panels.py` (one function per plot), `layouts.py` (which panels make each figure). |
| `reports/` | Tables and figures of the daily report; they never raise. |
| `utils`, `df_transforms`, `plots`, `figure_maker` | Old entry points, kept so that old code keeps working (`analyze_df` is `prepare`; `subject_progress_figure` and `session_summary_figure` put the failed panels in `fig.panel_problems`). |

To add a plot to a figure, write a panel in `figures/panels.py` and add it to
a row in `figures/layouts.py` (see the
[documentation](https://learningcircuitslab.github.io/lecilab-behavior-analysis/figures.html)).

## Tests

```bash
pip install -e ".[dev]"
python -m pytest tests -o addopts=""
```

`tests/data/NUO003_excerpt.csv.gz` holds 8 real days of one mouse (every
training stage); the tests draw every figure on each of them and fail if a
panel cannot be drawn. `example_data/` (large local files) is not tracked.

## Documentation

The documentation (`docs/source`) is built by a GitHub action on every push
to main, with example figures drawn from `tests/data` by the current code,
and published on GitHub Pages. To build it locally:

```bash
pip install -r docs/requirements.txt
sphinx-build docs/source docs/build
```
