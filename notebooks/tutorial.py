import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Analysing Training Village data with lecilab-behavior-analysis

    This notebook goes through the main analyses of the package, from the
    data saved by the village to the figures of the daily reports:

    1. load the trials of a mouse and prepare them,
    2. a first look: trials and performance per day,
    3. the ready-made figures (training history, one session),
    4. psychometric curves and their fit,
    5. the settings of each session, and how they changed through training,
    6. your own figure, made of panels.

    It runs on the example data of the package (`tests/data`: one mouse,
    NUO003, 8 days of training). To use your data, set `USE_EXAMPLE_DATA`
    to `False` and choose the project and mouse in the next cell.

    Open it with `marimo edit notebooks/tutorial.py` (from the root of the
    repository). Every plot of the package, with its code, is in the
    [gallery](https://learningcircuitslab.github.io/lecilab-behavior-analysis/gallery.html).
    """)
    return


@app.cell
def _():
    # where the data is: the example data of the package, or a project of the data folder
    USE_EXAMPLE_DATA = True
    PROJECT = "COT_cannula_data"  # a folder of the data folder (lecilab_behavior_analysis.io.server.get_outpath)
    MOUSE = "NUO003"
    return MOUSE, PROJECT, USE_EXAMPLE_DATA


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Load and prepare the trials

    The village saves one row per trial in
    `<data folder>/<project>/sessions/<mouse>/<mouse>.csv`.
    `load_project_trials` reads it and **prepares** it (`preprocessing.prepare`):
    it reads old and new data formats the same way (see
    `data_format_changes.py`) and adds every column the analyses use: the
    day (`year_month_day`), the trial number through training (`total_trial`),
    the first choice, misses, early pokeouts, reaction times, center holds...
    It also adds the performance through trials (`performance_w`).
    """)
    return


@app.cell
def _(MOUSE, PROJECT, USE_EXAMPLE_DATA, mo):
    import pandas as pd

    from lecilab_behavior_analysis.io.loading import load_project_logs, load_project_trials
    from lecilab_behavior_analysis.metrics.performance import get_performance_through_trials
    from lecilab_behavior_analysis.preprocessing import prepare

    if USE_EXAMPLE_DATA:
        _data = mo.notebook_dir().parent / "tests" / "data"
        _raw = pd.read_csv(_data / "NUO003_excerpt.csv.gz", sep=";", low_memory=False)
        _raw["date"] = pd.to_datetime(_raw["date"])
        trials = prepare(get_performance_through_trials(_raw, window=50))
        # the sessions summaries of the setups (with the settings of each session)
        summary = pd.read_csv(_data / "NUO003_sessions_summary.csv", sep=";")
    else:
        trials = load_project_trials(PROJECT, [MOUSE])
        _, summary, _ = load_project_logs(PROJECT)
    mo.md(
        f"**{len(trials)} trials** of {MOUSE}, from {trials['date'].min():%Y-%m-%d} to {trials['date'].max():%Y-%m-%d}"
    )
    return pd, summary, trials


@app.cell
def _(trials):
    trials[
        [
            "date",
            "session",
            "trial",
            "current_training_stage",
            "correct_side",
            "first_choice",
            "correct",
            "reaction_time",
        ]
    ].head()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. A first look: trials and performance per day

    Timeouts (`correct` empty) count as errors in the performance
    (`data_format_changes.correct_as_number`).
    """)
    return


@app.cell
def _(trials):
    import lecilab_behavior_analysis.data_format_changes as dfc

    _per_day = trials.assign(correct_number=trials["correct"].apply(dfc.correct_as_number))
    per_day = _per_day.groupby("year_month_day").agg(
        trials=("trial", "size"),
        sessions=("session", "nunique"),
        performance=("correct_number", lambda c: round(c.mean() * 100, 1)),
        stages=("current_training_stage", lambda s: ", ".join(sorted(s.unique()))),
    )
    per_day
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. The ready-made figures

    `figures.subject_progress` is the training history of the daily report:
    calendar, training times, performance, trials and water per day,
    performance by trial type, bias, center port, reaction times. Each
    panel that cannot be drawn shows why instead of stopping the figure;
    `problems` lists them.
    """)
    return


@app.cell
def _(trials):
    from lecilab_behavior_analysis.figures import session_summary, subject_progress

    history, problems = subject_progress(trials, perf_window=100)
    history
    return problems, session_summary


@app.cell
def _(problems):
    problems  # the panels that could not be drawn (e.g. engagement needs the events)
    return


@app.cell
def _(mo, trials):
    day_picker = mo.ui.dropdown(options=sorted(trials["year_month_day"].unique()), value="2025-10-15", label="Day")
    mo.md(f"""
    `figures.session_summary` is one day: performance, choices, center
    port, timing, and a psychometric panel per stimulus modality.

    {day_picker}
    """)
    return (day_picker,)


@app.cell
def _(day_picker, session_summary, trials):
    day = trials[trials["year_month_day"] == day_picker.value]
    _figure, _ = session_summary(day, perf_window=50)
    _figure
    return (day,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Psychometric curves

    `get_performance_by_difficulty_ratio` adds, for each trial, the value of
    its stimulus (`visual_stimulus_ratio`: brightness ratio of the ports,
    positive when the left one is brighter; `total_evidence_strength` for the
    cloud of tones) and the first choice as a number (1 left, 0 right).
    `psychometric_plot` draws the choices and a lapse logistic fit;
    `psychometric_curve_fitting_params` gives the parameters of that same fit
    (on the log of the brightness ratio, as the plot).
    """)
    return


@app.cell
def _(day):
    from matplotlib import pyplot as plt

    from lecilab_behavior_analysis.metrics.choices import get_performance_by_difficulty_ratio
    from lecilab_behavior_analysis.models.psychometric import psychometric_curve_fitting_params
    from lecilab_behavior_analysis.plots import psychometric_plot

    visual = get_performance_by_difficulty_ratio(
        day[(day["stimulus_modality"] == "visual") & (day["task"] != "Habituation")]
    )
    _fig, _ax = plt.subplots(figsize=(4, 3.5))
    if len(visual):
        psychometric_plot(visual, x="visual_stimulus_ratio", y="first_choice_numeric", ax=_ax, valueType="discrete")
        _, (lapse_left, lapse_right, slope, bias) = psychometric_curve_fitting_params(
            visual, x="visual_stimulus_ratio", y="first_choice_numeric", valueType="discrete"
        )
        print(f"lapses: left {lapse_left:.2f}, right {lapse_right:.2f}; slope {slope:.2f}; bias (log ratio) {bias:.2f}")
    _fig
    return (plt,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. The settings of each session

    The village saves the settings of every session (training stage,
    difficulties, punishment, reward...) in the `settings` column of the
    `sessions_summary.csv` of each setup. `io.settings.session_settings_table`
    makes a table with one row per session; `changing_settings` lists those
    that changed at some point.
    """)
    return


@app.cell
def _(summary):
    from lecilab_behavior_analysis.io import settings as session_settings

    settings_table = session_settings.session_settings_table(summary, "NUO003")
    changing = session_settings.changing_settings(settings_table)
    settings_table[["date", "task"] + changing]
    return changing, session_settings, settings_table


@app.cell
def _(mo, session_settings, settings_table):
    # the settings of one session, the changes since the previous session first
    _row = 3
    _text = session_settings.format_session_settings(
        session_settings.row_settings(settings_table.iloc[_row]),
        previous=session_settings.row_settings(settings_table.iloc[_row - 1]),
        title=f"NUO003, session of {settings_table['date'].iloc[_row]}",
    )
    mo.md(f"```text\n{_text}\n```")
    return


@app.cell
def _(changing, mo):
    setting_picker = mo.ui.multiselect(
        options=changing,
        value=[n for n in ("current_training_stage", "punishment", "hard_trials_on") if n in changing],
        label="Settings to show",
    )
    mo.md(f"""
    **How the settings and the choices changed, session by session.**
    `metrics.choices.get_choices_by_difficulty_matrix` gives the proportion
    of left choices by correct side and difficulty in each session (a
    psychometric curve per session); `plots.training_matrix_plot` draws it
    with the chosen settings as rows above it.

    {setting_picker}
    """)
    return (setting_picker,)


@app.cell
def _(day_picker, plt, session_settings, setting_picker, settings_table, trials):
    from lecilab_behavior_analysis.metrics.choices import get_choices_by_difficulty_matrix
    from lecilab_behavior_analysis.plots import training_matrix_plot

    p_left, n_trials, sessions = get_choices_by_difficulty_matrix(trials)
    _rows = session_settings.settings_for_sessions(settings_table, p_left.columns, list(setting_picker.value))
    _fig, _ax = plt.subplots(figsize=(10, 4.5))
    training_matrix_plot(p_left, sessions, _rows, highlight_day=day_picker.value, ax=_ax)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Your own figure, made of panels

    The ready-made figures are layouts of **panels**: small functions that
    draw one plot from the prepared trials (`data.df`). A panel lists the
    columns it needs, and raises `NotApplicable` when there is nothing to
    show. `render` draws a layout and never stops because of one panel.
    """)
    return


@app.cell
def _(pd, trials):
    from lecilab_behavior_analysis.figures import FigureData, Layout, Row, panel, render
    from lecilab_behavior_analysis.figures import panels as p

    @panel("trials per day", needs={"year_month_day"})
    def trials_per_day(data, ax):
        _counts = data.df.groupby("year_month_day").size()
        ax.bar(pd.to_datetime(_counts.index), _counts.values)
        ax.set_ylabel("Trials")

    _layout = Layout(
        [Row([(p.subject_summary, 1), (trials_per_day, 2)]), Row([(p.subject_performance, 1)])], width=12, height=6
    )
    own_figure, _ = render(_layout, FigureData(trials, perf_window=100))
    own_figure
    return


@app.cell
def _(mo):
    DOCS = "https://learningcircuitslab.github.io/lecilab-behavior-analysis"
    mo.md(rf"""
    ## Where next

    - The [gallery]({DOCS}/gallery.html): every plot, with its code.
    - The [API]({DOCS}/api_index.html): every function, by topic
      (`io`, `preprocessing`, `metrics`, `models`, `plots`, `figures`, `reports`).
    - `data_format_changes.py`: how old and new data formats are read.
    """)
    return


if __name__ == "__main__":
    app.run()
