"""Loading the trials and logs of a project, as the daily report does."""

import logging

import pandas as pd

from lecilab_behavior_analysis.io.loading import load_project_trials, load_setup_logs
from lecilab_behavior_analysis.preprocessing import PREPARED_COLUMNS


def write_project(root, raw_trials):
    sessions = root / "proj" / "sessions" / "NUO003"
    sessions.mkdir(parents=True)
    raw_trials.to_csv(sessions / "NUO003.csv", sep=";", index=False)
    setup = root / "proj" / "village01"
    (setup / "old_events").mkdir(parents=True)
    header = "date;type;subject;description\n"
    (setup / "old_events" / "events_1.csv").write_text(
        header + "2025-09-01 10:00:00;INFO;NUO003;old; with ; separators\n"
    )
    (setup / "events.csv").write_text(
        header + "2025-09-05 10:00:00;START;NUO003;TwoAFC started\n" "2026-07-01 10:00:00;END;NUO003;TwoAFC ended\n"
    )
    (setup / "sessions_summary.csv").write_text(
        "date;subject;trials\n2025-09-05 10:00:00;NUO003;100\n" "2026-07-01 10:00:00;NUO003;50\n"
    )


def test_load_project_trials(tmp_path, raw_trials, caplog):
    write_project(tmp_path, raw_trials)
    with caplog.at_level(logging.WARNING):
        df = load_project_trials("proj", ["NUO003", "MISSING"], max_date="2025-12-31", root=tmp_path)
    assert set(df["subject"]) == {"NUO003"}
    assert df["date"].max() <= pd.Timestamp("2025-12-31")
    assert set(PREPARED_COLUMNS) | {"performance_w"} <= set(df.columns)
    assert "Skipping MISSING" in caplog.text


def test_load_project_trials_without_any_subject(tmp_path):
    assert load_project_trials("proj", ["MISSING"], root=tmp_path).empty


def test_load_setup_logs(tmp_path, raw_trials):
    write_project(tmp_path, raw_trials)
    events, summary = load_setup_logs("proj", "village01", max_date="2025-12-31", root=tmp_path)
    assert list(events["type"]) == ["INFO", "START"]  # old events first, after max_date dropped
    assert events.loc[0, "description"] == "old; with ; separators"
    assert pd.api.types.is_datetime64_any_dtype(events["date"])
    assert list(summary["trials"]) == [100]


def test_load_setup_logs_missing_folder(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        events, summary = load_setup_logs("proj", "village06", root=tmp_path)
    assert events is None and summary is None
    assert "No events for proj/village06" in caplog.text


def test_find_setups(tmp_path, raw_trials):
    from lecilab_behavior_analysis.io.loading import find_setups

    write_project(tmp_path, raw_trials)
    (tmp_path / "proj" / "videos").mkdir()  # not a setup: no logs
    (tmp_path / "proj" / "village02").mkdir()
    (tmp_path / "proj" / "village02" / "sessions_summary.csv").write_text("date;subject;trials\n")
    assert find_setups("proj", tmp_path) == ["village01", "village02"]
    (tmp_path / "proj" / "events.csv").write_text("date;type;subject;description\n")  # old layout
    assert find_setups("proj", tmp_path) == ["", "village01", "village02"]
    assert find_setups("missing", tmp_path) == []


def test_load_project_logs_combines_setups(tmp_path, raw_trials):
    import shutil

    from lecilab_behavior_analysis.io.loading import load_project_logs

    write_project(tmp_path, raw_trials)
    # a second box with the same old events (copied) and one more session
    shutil.copytree(tmp_path / "proj" / "village01", tmp_path / "proj" / "village02")
    (tmp_path / "proj" / "village02" / "events.csv").write_text(
        "date;type;subject;description\n2025-09-03 10:00:00;START;NUO003;TwoAFC started\n"
    )
    events, summary, setups = load_project_logs("proj", max_date="2025-12-31", root=tmp_path)
    assert setups == ["village01", "village02"]
    assert list(events["date"].astype(str)) == ["2025-09-01 10:00:00", "2025-09-03 10:00:00", "2025-09-05 10:00:00"]
    assert list(summary["trials"]) == [100]  # same row in both boxes, kept once


def test_data_path_variable(monkeypatch, tmp_path):
    from lecilab_behavior_analysis.io.server import get_outpath

    monkeypatch.setenv("LECILAB_DATA_PATH", str(tmp_path))
    assert get_outpath() == str(tmp_path)


def test_load_project_trials_compressed(tmp_path, raw_trials):
    folder = tmp_path / "proj" / "sessions" / "NUO003"
    folder.mkdir(parents=True)
    raw_trials.head(50).to_csv(folder / "NUO003.csv.gz", sep=";", index=False)
    assert len(load_project_trials("proj", ["NUO003"], root=tmp_path)) == 50


def test_load_session_dates(tmp_path, raw_trials):
    from lecilab_behavior_analysis.io.loading import load_session_dates

    write_project(tmp_path, raw_trials)
    # another project's mouse in the same box, and a mouse without a sessions summary
    (tmp_path / "proj" / "village01" / "sessions_summary.csv").write_text(
        "date;subject;trials\n2025-09-05 10:00:00;NUO003;100\n2025-09-06 10:00:00;OTHER;50\n"
    )
    other = tmp_path / "proj" / "sessions" / "NUO004"
    other.mkdir(parents=True)
    raw_trials.head(5).assign(subject="NUO004").to_csv(other / "NUO004.csv", sep=";", index=False)
    dates = load_session_dates("proj", ["NUO003", "NUO004", "MISSING"], root=tmp_path)
    assert set(dates["subject"]) == {"NUO003", "NUO004"}
    assert list(dates[dates.subject == "NUO003"]["date"].astype(str)) == ["2025-09-05 10:00:00"]
    assert list(dates[dates.subject == "NUO003"]["setup"]) == ["village01"]
    # from the trials file: the setup is its system_name
    assert set(dates[dates.subject == "NUO004"]["setup"]) == set(raw_trials.head(5)["system_name"])
    assert dates["date"].is_monotonic_increasing
