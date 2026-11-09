import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import pandas as pd
import pytest

EXCERPT = Path(__file__).parent / "data" / "NUO003_excerpt.csv.gz"


@pytest.fixture(scope="session")
def raw_trials() -> pd.DataFrame:
    """
    Real trials of one mouse (NUO003): 8 days covering every training stage
    (habituation, visual easy and hard, auditory easy and hard, multisensory,
    manual training), saved before the 2026-11-09 format changes. Includes
    days with very few trials.
    """
    df = pd.read_csv(EXCERPT, sep=";", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture(scope="session")
def prepared_trials(raw_trials) -> pd.DataFrame:
    """The trials as the daily report prepares them."""
    import lecilab_behavior_analysis.df_transforms as dft

    return dft.analyze_df(dft.get_performance_through_trials(raw_trials, window=50))
