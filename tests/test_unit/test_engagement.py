"""Matching each session with the moment its subject returned home."""

import pandas as pd

from lecilab_behavior_analysis.metrics.engagement import get_box_usage_df, get_session_ends


def events(rows):
    return pd.DataFrame(rows, columns=["date", "type", "subject", "description"])


SESSIONS = pd.DataFrame(
    {
        "subject": ["A", "B", "A"],
        "date": pd.to_datetime(["2026-05-20 10:00:00", "2026-05-20 10:05:00", "2026-05-21 10:00:00"]),
    }
)


def test_each_session_ends_when_its_subject_returns_home():
    # B (another box) returns home before A, and the files are not in order
    ev = events(
        [
            ["2026-05-21 10:30:00", "INFO", "A", "Subject back home: 20.1 g"],
            ["2026-05-20 10:20:00", "INFO", "B", "Subject back home: 22.0 g"],
            ["2026-05-20 10:40:00", "INFO", "A", "The subject has returned home."],
            ["2026-05-20 10:10:00", "INFO", "A", "Something else"],
        ]
    )
    ends = get_session_ends(SESSIONS, ev).set_index(["subject", "date"])["session_duration"]
    assert ends[("A", pd.Timestamp("2026-05-20 10:00:00"))] == 40 * 60
    assert ends[("B", pd.Timestamp("2026-05-20 10:05:00"))] == 15 * 60
    assert ends[("A", pd.Timestamp("2026-05-21 10:00:00"))] == 30 * 60


def test_sessions_without_their_return_are_left_out():
    # the return of the first session of A is missing: the next one belongs to
    # the next session, and B returns more than max_hours later
    ev = events(
        [
            ["2026-05-21 10:30:00", "INFO", "A", "Subject back home: 20.1 g"],
            ["2026-05-20 20:05:00", "INFO", "B", "Subject back home: 22.0 g"],
        ]
    )
    ends = get_session_ends(SESSIONS, ev)
    assert list(zip(ends["subject"], ends["date"])) == [("A", pd.Timestamp("2026-05-21 10:00:00"))]


def test_box_usage_without_any_session_end_is_empty():
    usage = get_box_usage_df(SESSIONS, events([]), verbose=False)
    assert usage.empty
    assert "percentage_of_time" in usage.columns
