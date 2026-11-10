# Daily reports

`reports.overview` has one function per cell of the overview of the daily
report ([village-report-generator](https://learningcircuitslab.github.io/village-report-generator/documentation/)).
They never raise: if something fails, they return a figure with the error,
and the failure is recorded so that the report can list it
(`overview.failed_cells()`, `overview.problems_table(...)`).

| Function | Shows |
|---|---|
| `tv_report_table(events, sessions_summary, hours=24)` | Detections, sessions, water and weight of each subject in the last hours. |
| `trials_by_stage_table(df)` | Trials of each subject in each training stage. |
| `last_days_table(df, n_days=3)` | Trials and performance of each subject in the last days. |
| `performance_over_trials_figure(df)` | Performance of every subject through its trials (interactive). |
| `box_occupancy_figure(df)` | How much of each day the box was used, and at what time. |
| `cohort_trials_and_performance_figure(df)` | Trials and performance per day of the cohort, by training stage. |
| `problems_table(problems)` | The panels and cells that could not be generated. |

![Cohort](_static/examples/cohort.png)

![Box occupancy](_static/examples/box_occupancy.png)

Any function can be made safe in the same way:

```python
from lecilab_behavior_analysis.reports.overview import never_raises

@never_raises("My figure", "matplotlib")   # or "plotly"
def my_figure(df):
    ...
```
