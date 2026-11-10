# lecilab-behavior-analysis

Analysis and figures of the behavior of mice trained in the
[Training Village](https://braincircuitsbehaviorlab.github.io/village/), in
the tasks of `auditory-task-python` (two-alternative forced choice with
visual, auditory and multisensory stimuli).

It is used by:

- the **daily reports** of every project
  ([village-report-generator](https://learningcircuitslab.github.io/village-report-generator/documentation/)),
- the **visualizer** (`behavior-data-visualizer`),
- the **analysis notebooks** (`behavior-analysis-notebooks`).

![Training history of one mouse](_static/examples/subject_progress.png)

## What it does

| | |
|---|---|
| **Load** | Trials of a subject or of a project, and the events and sessions summary of every setup (box) of a project. |
| **Prepare** | `prepare(df)` reads old and new data formats and adds every derived column once (choices, reaction times, center holds, misses...). |
| **Measure** | Performance, choices and biases, timing, engagement, box occupancy, psychometric fits and regressions. |
| **Draw** | Ready-made figures (training history of a subject, summary of a session) made of panels that never break the whole figure. |
| **Report** | Tables and figures of the daily report that never raise. |

```{toctree}
:maxdepth: 2
:hidden:

getting_started
data
figures
reports
api_index
```
