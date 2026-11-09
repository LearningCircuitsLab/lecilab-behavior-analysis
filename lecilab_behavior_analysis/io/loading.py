"""Loading csv files: example data, events (with their old_events), trials of a subject or a project."""
import logging
import os
from pathlib import Path
from typing import Any, List, Optional, Union
import pandas as pd

from lecilab_behavior_analysis.io.server import get_outpath

logger = logging.getLogger(__name__)


def load_example_data(mouse_name) -> pd.DataFrame:
    outpath = get_outpath()
    df = pd.read_csv(outpath + "/example_data/" + mouse_name + "/" + mouse_name + "_fakedata.csv", sep=";")

    return df


def load_all_events(events_subfolder_location: str) -> pd.DataFrame:
    """
    Load all events from the local machine for a given project.
    """
    outpath = get_outpath()
    events_path = f"{outpath}/{events_subfolder_location}/events.csv"
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file for project {outpath}/{events_subfolder_location} does not exist.")
    
    events_df = read_csv_with_specific_number_of_columns(events_path, expected_columns=4, sep=";")

    # read events in old_events if they exist
    events_dfs_list = []
    old_events_path = f"{outpath}/{events_subfolder_location}/old_events"
    # list files in the old_events folder
    if os.path.exists(old_events_path):
        old_events_files = [f for f in os.listdir(old_events_path) if f.endswith('.csv')]
        # sort files by name to ensure they are in the correct order
        old_events_files.sort()
        # read each file and concatenate to events_df
        for old_file in old_events_files:
            old_file_path = os.path.join(old_events_path, old_file)
            old_events_df = read_csv_with_specific_number_of_columns(old_file_path, expected_columns=4, sep=";")
            events_dfs_list.append(old_events_df)
    events_df = pd.concat(events_dfs_list + [events_df], ignore_index=True)
    return events_df


def read_csv_with_specific_number_of_columns(file_path: str, expected_columns: int, sep: str = ";") -> pd.DataFrame:
    with open(file_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split(sep)
        rows = [line.rstrip("\n").split(sep, expected_columns-1) for line in f]

    return pd.DataFrame(rows, columns=header)


def load_subject_trials(csv_path: Union[str, Path], max_date: Optional[Any] = None) -> pd.DataFrame:
    """
    Trials of one subject from its .csv (sep ";"), with date as datetime,
    keeping only trials up to max_date if given.
    """
    df = pd.read_csv(csv_path, sep=";", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def load_project_trials(project: str, subjects: List[str], max_date: Optional[Any] = None,
                        root: Optional[Union[str, Path]] = None, window: int = 50) -> pd.DataFrame:
    """
    Trials of several subjects of a project, prepared for analysis
    (preprocessing.prepare), with the performance through trials
    (performance_w, rolling window of `window` trials) of each subject.

    Reads <root>/<project>/sessions/<subject>/<subject>.csv (root defaults to
    get_outpath()). A subject whose file is missing or unreadable is skipped
    with a warning, so that the other subjects can still be analyzed.
    """
    from lecilab_behavior_analysis.metrics.performance import get_performance_through_trials
    from lecilab_behavior_analysis.preprocessing import prepare

    root = Path(root) if root is not None else Path(get_outpath())
    frames = []
    for subject in subjects:
        path = root / project / "sessions" / subject / f"{subject}.csv"
        try:
            df = load_subject_trials(path, max_date)
        except Exception as error:
            logger.warning("Skipping %s: could not read %s (%s: %s)", subject, path, type(error).__name__, error)
            continue
        if df.empty:
            logger.warning("Skipping %s: no trials up to %s", subject, max_date)
            continue
        frames.append(get_performance_through_trials(df, window=window))
    if not frames:
        return pd.DataFrame()
    return prepare(pd.concat(frames, ignore_index=True))
