"""Settings of each session (io.settings) and the training matrix (metrics.choices, plots.training)."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt

from lecilab_behavior_analysis.io import settings as st
from lecilab_behavior_analysis.metrics.choices import MATRIX_ROWS, get_choices_by_difficulty_matrix
from lecilab_behavior_analysis.plots import setting_change_labels, setting_values_to_colors, training_matrix_plot


def _summary(raw_trials) -> pd.DataFrame:
    """A sessions summary for the sessions of the test trials (settings change on the 3rd session)."""
    sessions = raw_trials.groupby("session").agg(date=("date", "first"), task=("task", "first")).reset_index()
    rows = []
    for i, session in sessions.iterrows():
        settings = {
            "current_training_stage": "TwoAFC_visual_easy" if i < 2 else "TwoAFC_visual_hard",
            "punishment": i >= 2,
            "reward_amount_ul": 3.0 if i < 2 else 2.0,
            "trial_sides": ["left", "right"],
            "run_mode": "Auto",
        }
        rows.append(
            {
                "date": session["date"].strftime("%Y-%m-%d %H:%M:%S"),
                "subject": raw_trials["subject"].iloc[0],
                "task": session["task"],
                "settings": json.dumps(settings),
            }
        )
    rows.append({"date": "2020-01-01 00:00:00", "subject": "other", "task": "TwoAFC", "settings": json.dumps({"a": 1})})
    rows.append({"date": "2020-01-02 00:00:00", "subject": "NUO003", "task": "TwoAFC", "settings": None})
    return pd.DataFrame(rows)


# settings ------------------------------------------------------------------------------
def test_session_settings_table(raw_trials):
    summary = _summary(raw_trials)
    table = st.session_settings_table(summary, "NUO003")
    assert len(table) == raw_trials["session"].nunique()  # the session without settings is left out
    assert table["date"].is_monotonic_increasing
    assert table["trial_sides"].iloc[0] == '["left", "right"]'  # lists as text
    assert set(st.changing_settings(table)) == {"current_training_stage", "punishment", "reward_amount_ul"}
    assert st.session_settings_table(None).empty
    assert st.session_settings_table(summary.drop(columns="settings")).empty


def test_get_session_settings_from_the_summary(raw_trials):
    summary = _summary(raw_trials)
    date = raw_trials.groupby("session")["date"].first().iloc[2]
    settings = st.get_session_settings("project", "NUO003", date, summary=summary, fetch=False)
    assert settings["current_training_stage"] == "TwoAFC_visual_hard" and settings["punishment"] is True


def test_get_session_settings_from_its_json_file(tmp_path):
    folder = tmp_path / "project" / "sessions" / "M1"
    folder.mkdir(parents=True)
    (folder / "M1_TwoAFC_20261125_143000.json").write_text(json.dumps({"punishment": False, "iti": 1.0}))
    settings = st.get_session_settings(
        "project", "M1", "2026-11-25 14:30:00", summary=pd.DataFrame(), root=tmp_path, fetch=False
    )
    assert settings == {"punishment": False, "iti": 1.0}
    assert (
        st.get_session_settings("project", "M1", "2026-11-26", summary=pd.DataFrame(), root=tmp_path, fetch=False)
        is None
    )


def test_get_session_settings_fetches_the_json_from_the_cluster(tmp_path, monkeypatch):
    """Not on this computer: copied from the cluster (rsync), then read."""
    from lecilab_behavior_analysis.io import server

    copied = {}

    def fake_rsync(project, file_path, credentials, local_path, exclude=None):
        copied.update(project=project, file_path=file_path, local_path=local_path)
        (tmp_path / "project" / "sessions" / "M1" / "M1_TwoAFC_20261125_143000.json").write_text('{"iti": 2.0}')
        return True

    monkeypatch.setattr(server, "get_idibaps_cluster_credentials", lambda: {"username": "u", "host": "h"})
    monkeypatch.setattr(server, "rsync_cluster_data", fake_rsync)
    settings = st.get_session_settings("project", "M1", "2026-11-25 14:30:00", summary=pd.DataFrame(), root=tmp_path)
    assert settings == {"iti": 2.0}
    assert copied["file_path"] == "sessions/M1/M1_TwoAFC_20261125_143000.json"


def test_no_cluster_access_is_not_an_error(tmp_path, monkeypatch):
    from lecilab_behavior_analysis.io import server

    def unknown():
        raise ValueError("Unknown host")

    monkeypatch.setattr(server, "get_idibaps_cluster_credentials", unknown)
    assert st.get_session_settings("project", "M1", "2026-11-25", summary=pd.DataFrame(), root=tmp_path) is None


def test_format_session_settings():
    previous = {"punishment": False, "iti": 1.0, "stage": "easy"}
    settings = {"punishment": True, "iti": 1.0, "stage": "hard", "new_one": 5}
    text = st.format_session_settings(settings, previous, title="M1, session of 2026-11-25")
    lines = text.splitlines()
    assert lines[0] == "M1, session of 2026-11-25"
    assert lines[1] == "3 changed since the previous session (*)"
    # changed first (alphabetically), marked, with the previous value
    assert lines[2].startswith("* new_one") and lines[2].endswith("(new)")
    assert lines[3].startswith("* punishment") and lines[3].endswith("(was False)")
    assert lines[4].startswith("* stage") and lines[4].endswith("(was easy)")
    assert lines[5].startswith("  iti")
    # the values start in the same column
    assert len({line.index(value) for line, value in zip(lines[2:6], ["5", "True", "hard", "1.0"])}) == 1
    assert "no changes" in st.format_session_settings({"a": 1}, {"a": 1})
    only = st.format_session_settings(settings, only=["iti", "stage"])
    assert [line.split()[0] for line in only.splitlines()] == ["iti", "stage"]


def test_settings_for_sessions(raw_trials):
    table = st.session_settings_table(_summary(raw_trials), "NUO003")
    dates = list(table["date"]) + [pd.Timestamp("2030-01-01")]
    values = st.settings_for_sessions(table, dates, ["punishment", "missing_setting"])
    assert values.loc["punishment"].tolist()[:3] == [False, False, True]
    assert pd.isna(values.loc["punishment"].iloc[-1])  # a session without settings
    assert values.loc["missing_setting"].isna().all()


# matrix -----------------------------------------------------------------------------------
def test_choices_by_difficulty_matrix(prepared_trials):
    p_left, n_trials, sessions = get_choices_by_difficulty_matrix(prepared_trials)
    assert list(p_left.index) == [f"{side} {difficulty}" for side, difficulty in MATRIX_ROWS]
    assert p_left.shape[1] == prepared_trials["session"].nunique() == len(sessions)
    values = p_left.to_numpy(dtype=float)
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 1
    # a cell is the proportion of left first choices of those trials
    session = sessions.iloc[-1]
    trials = prepared_trials[(prepared_trials["date"] == session["date"])]
    left_easy = trials[(trials["correct_side"] == "left") & (trials["difficulty"] == "easy")]
    choices = left_easy["first_choice"].isin(["left", "right"])
    if choices.any():
        expected = (left_easy.loc[choices, "first_choice"] == "left").mean()
        assert p_left.loc["left easy", session["date"]] == pytest.approx(expected)
        assert n_trials.loc["left easy", session["date"]] == choices.sum()
    assert sessions["n_trials"].sum() == len(prepared_trials)
    assert list(sessions.columns) == [
        "date",
        "day",
        "session",
        "current_training_stage",
        "stimulus_modality",
        "n_trials",
        "performance",
    ]


def test_matrix_of_habituation_only(prepared_trials):
    habituation = prepared_trials[prepared_trials["task"] == "Habituation"]
    p_left, n_trials, sessions = get_choices_by_difficulty_matrix(habituation)
    assert p_left.isna().all().all() and (n_trials == 0).all().all()
    assert len(sessions) == habituation["session"].nunique()


def test_setting_values_to_colors():
    codes, labels, kind = setting_values_to_colors(pd.Series([True, "False", None]))
    assert kind == "bool" and codes[:2].tolist() == [1, 0] and np.isnan(codes[2]) and labels[2] == ""
    codes, _, kind = setting_values_to_colors(pd.Series([1.0, 3.0, 2.0]))
    assert kind == "numeric" and codes.tolist() == [0, 1, 0.5]
    codes, labels, kind = setting_values_to_colors(pd.Series(["b", "a", "b"]))
    assert kind == "category" and codes.tolist() == [0, 1, 0]
    # a number that changes every session is not labelled at each change
    assert setting_change_labels([str(v) for v in range(20)], "numeric") == []
    assert setting_change_labels(["a", "a", "b", "", "b"], "category") == [0, 2, 4]
    # labels that would overlap are left out
    assert setting_change_labels(["a", "b", "c", "c", "c", "d"], "category", min_gap=3) == [0, 5]


def test_training_matrix_plot(raw_trials, prepared_trials):
    p_left, _, sessions = get_choices_by_difficulty_matrix(prepared_trials)
    table = st.session_settings_table(_summary(raw_trials), "NUO003")
    settings = st.settings_for_sessions(table, p_left.columns, ["current_training_stage", "punishment"])
    fig, ax = plt.subplots()
    training_matrix_plot(p_left, sessions, settings, highlight_day=sessions["day"].iloc[2], ax=ax)
    assert len(fig.axes) == 3  # matrix, settings rows, colour bar
    top = [a for a in fig.axes if a is not ax and a.get_yticklabels() and a.get_label() != "<colorbar>"][0]
    assert [t.get_text() for t in top.get_yticklabels()] == ["current_training_stage", "punishment"]
    plt.close(fig)
    fig, ax = plt.subplots()
    training_matrix_plot(p_left, sessions, ax=ax)  # without settings
    plt.close(fig)
