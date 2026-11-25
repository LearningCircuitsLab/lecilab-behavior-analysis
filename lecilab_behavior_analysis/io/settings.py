"""
The settings of each session: what the training protocol set for it
(training stage, difficulties, punishment, reward...).

The village saves them twice when a session ends:

- in the settings column of <project>/<setup>/sessions_summary.csv (as a
  JSON text), with the date (start) of the session and the subject,
- in <project>/sessions/<subject>/<subject>_<task>_<YYYYMMDD_HHMMSS>.json.

The date of a session is the same as the date column of its trials, so
settings and trials are matched by subject and date.

Functions:

- session_settings_table: one row per session, one column per setting,
  from the sessions summaries (e.g. those of io.loading.load_project_logs).
- get_session_settings: the settings of one session, from the summaries,
  or its .json file, fetched from the cluster if it is not on this computer.
- format_session_settings: the settings of a session as readable text,
  marking those that changed since the previous session.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from lecilab_behavior_analysis.io.server import get_outpath

logger = logging.getLogger(__name__)

# columns of session_settings_table that are not settings
SESSION_COLUMNS = ("date", "subject", "task", "setup")


def _parse_settings(text: Any) -> Optional[Dict[str, Any]]:
    """The settings saved as JSON text, or None if there are none or they cannot be read."""
    if isinstance(text, dict):
        return text
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        settings = json.loads(text)
    except (ValueError, TypeError):
        return None
    return settings if isinstance(settings, dict) else None


def _cell(value: Any) -> Any:
    """Lists and other containers as text, so that the table can be compared and shown."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value)
    return value


def session_settings_table(summary: Optional[pd.DataFrame], subject: Optional[str] = None) -> pd.DataFrame:
    """
    One row per session (of subject, if given) with its date, subject, task,
    setup (if the summary has it) and one column per setting, sorted by date.
    Sessions without settings are left out. Lists (e.g. trial_sides) are
    kept as text.

    Args:
        summary: sessions summaries, e.g. from io.loading.load_project_logs
            (columns date, subject, task, settings).
        subject: only this subject.
    """
    if summary is None or summary.empty or "settings" not in summary.columns:
        return pd.DataFrame(columns=list(SESSION_COLUMNS))
    rows = summary if subject is None else summary[summary["subject"] == subject]
    records = []
    for _, row in rows.iterrows():
        settings = _parse_settings(row["settings"])
        if settings is None:
            continue
        record = {column: row.get(column) for column in SESSION_COLUMNS if column in row.index}
        record.update({name: _cell(value) for name, value in settings.items() if name not in SESSION_COLUMNS})
        records.append(record)
    if not records:
        return pd.DataFrame(columns=list(SESSION_COLUMNS))
    table = pd.DataFrame(records)
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    return table.sort_values("date", kind="stable").drop_duplicates().reset_index(drop=True)


def row_settings(row: pd.Series) -> Dict[str, Any]:
    """The settings of one row of session_settings_table, as plain python values (missing ones left out)."""
    settings = {}
    for name, value in row.items():
        if name in SESSION_COLUMNS or (not isinstance(value, (list, dict)) and pd.isna(value)):
            continue
        settings[name] = value.item() if hasattr(value, "item") else value
    return settings


def session_file_stem(subject: str, task: str, date: Any) -> str:
    """<subject>_<task>_<YYYYMMDD_HHMMSS>: the name of the files of a session."""
    return f"{subject}_{task}_{pd.Timestamp(date).strftime('%Y%m%d_%H%M%S')}"


def fetch_session_settings_file(
    project: str, subject: str, task: str, date: Any, root: Optional[Union[str, Path]] = None
) -> Optional[Path]:
    """
    Copies the .json of a session from the cluster (rsync over ssh, see
    io.server) to the same place in the local data folder. Returns its path,
    or None if it could not be copied.
    """
    from lecilab_behavior_analysis.io.server import get_idibaps_cluster_credentials, rsync_cluster_data

    root = Path(root) if root is not None else Path(get_outpath())
    name = session_file_stem(subject, task, date) + ".json"
    folder = root / project / "sessions" / subject
    try:
        credentials = get_idibaps_cluster_credentials()
    except ValueError as error:  # this computer has no access to the cluster
        logger.warning("Cannot fetch %s from the cluster: %s", name, error)
        return None
    folder.mkdir(parents=True, exist_ok=True)
    if not rsync_cluster_data(project, f"sessions/{subject}/{name}", credentials, str(folder) + "/"):
        return None
    path = folder / name
    return path if path.exists() else None


def get_session_settings(
    project: str,
    subject: str,
    date: Any,
    task: str = "TwoAFC",
    summary: Optional[pd.DataFrame] = None,
    root: Optional[Union[str, Path]] = None,
    fetch: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    The settings of one session (subject, date of its start), looked for in:

    1. the sessions summaries (summary, or those of the project in the data
       folder: io.loading.load_project_logs),
    2. its .json file in <root>/<project>/sessions/<subject>/,
    3. the cluster (if fetch): the .json file is copied to that folder.

    Returns None if they are not found.
    """
    date = pd.Timestamp(date)
    if summary is None:
        from lecilab_behavior_analysis.io.loading import load_project_logs

        _, summary, _ = load_project_logs(project, root=root)
    table = session_settings_table(summary, subject)
    if not table.empty:
        found = table[table["date"] == date]
        if not found.empty:
            return row_settings(found.iloc[-1])
    root = Path(root) if root is not None else Path(get_outpath())
    path = root / project / "sessions" / subject / (session_file_stem(subject, task, date) + ".json")
    if not path.exists() and fetch:
        path = fetch_session_settings_file(project, subject, task, date, root) or path
    if path.exists():
        return _parse_settings(path.read_text())
    return None


def changed_settings(settings: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The settings whose value differs from the previous session: {name: previous value} (None if new)."""
    if previous is None:
        return {}
    changed: Dict[str, Any] = {}
    for name, value in settings.items():
        if name not in previous:
            changed[name] = None
        elif _cell(previous[name]) != _cell(value):
            changed[name] = previous[name]
    return changed


def format_session_settings(
    settings: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
    title: str = "",
    only: Optional[List[str]] = None,
) -> str:
    """
    The settings of a session as aligned text, one per line, the ones that
    changed since the previous session first and marked with * (with their
    previous value).

    Args:
        settings: the settings of the session (e.g. get_session_settings).
        previous: those of the previous session, to mark the changes.
        title: first line (e.g. subject and date).
        only: show only these settings (in this order).
    """
    names = [n for n in (only or sorted(settings)) if n in settings]
    changed = changed_settings(settings, previous)
    names = [n for n in names if n in changed] + [n for n in names if n not in changed]
    width = max((len(n) for n in names), default=0)
    lines = [title] if title else []
    if previous is not None:
        lines.append(
            f"{len(changed)} changed since the previous session (*)"
            if changed
            else "no changes since the previous session"
        )
    for name in names:
        value = _cell(settings[name])
        mark = "*" if name in changed else " "
        line = f"{mark} {name.ljust(width)}  {value}"
        if name in changed:
            line += "   (new)" if changed[name] is None else f"   (was {_cell(changed[name])})"
        lines.append(line)
    return "\n".join(lines)


def print_session_settings(project: str, subject: str, date: Any, **kwargs) -> None:
    """Prints the settings of one session (see get_session_settings), with the changes since the previous one."""
    from lecilab_behavior_analysis.io.loading import load_project_logs

    root = kwargs.get("root")
    _, summary, _ = load_project_logs(project, root=root)
    settings = get_session_settings(project, subject, date, summary=summary, **kwargs)
    if settings is None:
        print(f"No settings found for {subject} on {pd.Timestamp(date)}")
        return
    table = session_settings_table(summary, subject)
    earlier = table[table["date"] < pd.Timestamp(date)]
    previous = None
    if not earlier.empty:
        previous = row_settings(earlier.iloc[-1])
    print(format_session_settings(settings, previous, title=f"{subject}, session of {pd.Timestamp(date)}"))


def settings_for_sessions(table: pd.DataFrame, dates: Any, names: List[str]) -> pd.DataFrame:
    """
    The value of some settings in some sessions: one row per setting (names),
    one column per session date (dates, e.g. the columns of
    metrics.choices.get_choices_by_difficulty_matrix). NaN where a session
    has no settings saved or does not have that setting.
    """
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    out = pd.DataFrame(index=list(names), columns=dates, dtype=object)
    if table is None or table.empty:
        return out
    by_date = table.drop_duplicates("date", keep="last").set_index("date")
    for name in names:
        if name in by_date.columns:
            out.loc[name] = by_date[name].reindex(dates).values
    return out


def changing_settings(table: pd.DataFrame) -> List[str]:
    """The settings whose value is not the same in every session of the table (e.g. to offer them first)."""
    names = [c for c in table.columns if c not in SESSION_COLUMNS]
    return [n for n in names if table[n].astype(str).nunique(dropna=False) > 1]
