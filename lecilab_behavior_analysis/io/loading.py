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

    Reads <root>/<project>/sessions/<subject>/<subject>.csv, or
    <subject>.csv.gz if there is no .csv (root defaults to get_outpath()). A subject whose file is missing or unreadable is skipped
    with a warning, so that the other subjects can still be analyzed.
    """
    from lecilab_behavior_analysis.metrics.performance import get_performance_through_trials
    from lecilab_behavior_analysis.preprocessing import prepare

    root = Path(root) if root is not None else Path(get_outpath())
    frames = []
    for subject in subjects:
        path = root / project / "sessions" / subject / f"{subject}.csv"
        if not path.exists() and path.with_suffix(".csv.gz").exists():
            path = path.with_suffix(".csv.gz")  # compressed copy (e.g. sample data)
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


def load_setup_logs(project: str, setup_name: str, max_date: Optional[Any] = None,
                    root: Optional[Union[str, Path]] = None) -> tuple:
    """
    Events (events.csv and old_events) and sessions summary
    (sessions_summary.csv) of one setup of a project, with date as datetime
    and only rows up to max_date if given.

    Reads <root>/<project>/<setup_name>/ (root defaults to get_outpath()).
    Both are sorted by date (the old_events files are not always in order)
    and without repeated rows (events copied in more than one file).
    Returns (events_df, sessions_summary_df); each one is None, with a
    warning, if its file is missing or unreadable, so that a report can still
    show the trials.
    """
    root = Path(root) if root is not None else Path(get_outpath())
    folder = root / project / setup_name

    def _up_to_max_date(df: pd.DataFrame) -> pd.DataFrame:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if max_date is not None:
            df = df[df["date"] <= pd.Timestamp(max_date)]
        return df.drop_duplicates().sort_values("date", kind="stable").reset_index(drop=True)

    events_df = None
    events_path = folder / "events.csv"
    try:
        events_df = read_csv_with_specific_number_of_columns(str(events_path), expected_columns=4, sep=";")
        old_folder = folder / "old_events"
        old_files = sorted(old_folder.glob("*.csv")) if old_folder.exists() else []
        old = [read_csv_with_specific_number_of_columns(str(f), expected_columns=4, sep=";") for f in old_files]
        events_df = _up_to_max_date(pd.concat(old + [events_df], ignore_index=True))
    except Exception as error:
        logger.warning("No events for %s/%s: %s: %s", project, setup_name, type(error).__name__, error)
        events_df = None

    summary_df = None
    summary_path = folder / "sessions_summary.csv"
    try:
        summary_df = _up_to_max_date(pd.read_csv(summary_path, sep=";"))
    except Exception as error:
        logger.warning("No sessions summary for %s/%s: %s: %s", project, setup_name, type(error).__name__, error)
        summary_df = None

    return events_df, summary_df


LOG_FILES = ("events.csv", "sessions_summary.csv")


def find_setups(project: str, root: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Setups (boxes) where the subjects of a project trained: the folders of
    <root>/<project> (other than sessions) with an events.csv or a
    sessions_summary.csv, e.g. ["village01", "village02"]. Old projects keep
    those files directly in the project folder: that folder is the setup "".
    """
    folder = (Path(root) if root is not None else Path(get_outpath())) / project
    if not folder.is_dir():
        return []
    setups = sorted(
        child.name for child in folder.iterdir()
        if child.is_dir() and child.name != "sessions" and any((child / f).exists() for f in LOG_FILES)
    )
    if any((folder / f).exists() for f in LOG_FILES):
        setups.insert(0, "")
    return setups


def load_project_logs(project: str, max_date: Optional[Any] = None, root: Optional[Union[str, Path]] = None,
                      setups: Optional[List[str]] = None) -> tuple:
    """
    Events and sessions summary of every setup of a project (see find_setups
    and load_setup_logs), together, sorted by date and without repeated rows.
    Returns (events_df, sessions_summary_df, setups): each dataframe is None
    if no setup has it.
    """
    setups = find_setups(project, root) if setups is None else list(setups)
    logs = [load_setup_logs(project, setup, max_date, root) for setup in setups]

    def combine(frames):
        frames = [f for f in frames if f is not None]
        if not frames:
            return None
        return (pd.concat(frames, ignore_index=True).drop_duplicates()
                .sort_values("date", kind="stable").reset_index(drop=True))

    return combine([e for e, _ in logs]), combine([s for _, s in logs]), setups
