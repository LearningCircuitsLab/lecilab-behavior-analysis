"""prepare() and the readers of the event columns."""
import numpy as np
import pandas as pd
import pytest

import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
from lecilab_behavior_analysis.metrics.trial_events import parse_events
from lecilab_behavior_analysis.preprocessing import PREPARED_COLUMNS, prepare


@pytest.mark.parametrize(
    "value, expected",
    [
        ("[1.5, 2.25]", [1.5, 2.25]),
        ("[nan]", []),
        ("[]", []),
        ("", []),
        (np.nan, []),
        (None, []),
        (3.0, [3.0]),
        ([1.0, np.nan], [1.0]),
        ("[np.float64(1.5)]", [1.5]),
        ("[1e3]", [1000.0]),
    ],
)
def test_parse_events(value, expected):
    assert parse_events(value) == expected


def test_prepare_adds_every_column(raw_trials):
    df = prepare(raw_trials)
    assert set(PREPARED_COLUMNS) <= set(df.columns)
    assert len(df) == len(raw_trials)


def test_prepare_twice_changes_nothing(raw_trials):
    once = prepare(raw_trials)
    twice = prepare(once)
    pd.testing.assert_frame_equal(once.drop(columns="port2_holds"), twice.drop(columns="port2_holds"))


def test_prepare_on_a_day_keeps_the_trial_numbers_of_the_history(prepared_trials):
    day = prepared_trials[prepared_trials.year_month_day == "2025-12-18"]
    assert prepare(day)["total_trial"].tolist() == day["total_trial"].tolist()
    assert day["total_trial"].min() > 1


def test_columns_match_the_row_by_row_functions(raw_trials):
    """The fast column functions give the same values as the old row functions."""
    df = prepare(raw_trials)
    checks = {
        "first_choice": utils.first_poke_after_stimulus_state,
        "last_choice": utils.get_last_poke_of_trial,
        "last_choice_before_stimulus": utils.get_last_poke_before_stimulus_state,
        "miss_trial": utils.is_this_a_miss_trial,
        "early_pokeout": utils.is_this_an_early_pokeout_trial,
        "reaction_time": utils.trial_reaction_time,
    }
    for column, row_function in checks.items():
        expected = raw_trials.apply(row_function, axis=1)
        got = df[column]
        same = (got.astype(object) == expected.astype(object)) | (got.isna() & expected.isna())
        assert same.all(), column


def test_analyze_df_is_prepare(raw_trials):
    pd.testing.assert_frame_equal(
        dft.analyze_df(raw_trials).drop(columns="port2_holds"),
        prepare(raw_trials).drop(columns="port2_holds"),
    )


def test_prepare_without_task_columns():
    """Old or partial files without some columns can still be prepared."""
    df = pd.DataFrame({
        "subject": ["m"] * 3, "session": [1] * 3, "trial": [1, 2, 3],
        "date": ["2025-01-01 10:00:00"] * 3, "correct": [True, False, True],
        "TRIAL_START": [0.0, 10.0, 20.0], "TRIAL_END": [5.0, 15.0, 25.0],
    })
    out = prepare(df)
    assert out["total_trial"].tolist() == [1, 2, 3]
    assert out["early_pokeout"].isna().all()
