"""Tables for the daily report."""

import numpy as np
import pandas as pd


def generate_tv_report(events: pd.DataFrame, sessions_summary: pd.DataFrame, hours: int = 24):
    events["date"] = pd.to_datetime(events["date"])
    sessions_summary["date"] = pd.to_datetime(sessions_summary["date"])

    # get the highest date in the events and sessions_summary dataframes
    max_date = max(events["date"].max(), sessions_summary["date"].max())
    time_hours_ago = pd.Timestamp(max_date) - pd.Timedelta(hours=hours)

    detections = events[
        (
            events["description"].str.startswith(("Subject not", "Detection in", "Large", "Multiple"))
            | (events["type"] == "START")
        )
        & (events["date"] >= time_hours_ago)
    ]
    sessions = events[(events["type"] == "START") & (events["date"] >= time_hours_ago)]
    sessions_summary = sessions_summary[sessions_summary["date"] >= time_hours_ago]

    subject_detections = detections.groupby("subject").size().to_dict()
    subject_sessions = sessions.groupby("subject").size().to_dict()
    subject_water = sessions_summary.groupby("subject")["water"].sum().to_dict()
    subject_weight = sessions_summary.groupby("subject")["weight"].mean().to_dict()
    # reduce the weight to two decimal places
    subject_weight = {k: round(v, 2) if not np.isnan(v) else np.nan for k, v in subject_weight.items()}

    # select only subjects that have detections
    # sorted
    subjects = sorted(set(subject_detections.keys()))
    subject_detections = {subj: subject_detections.get(subj, 0) for subj in subjects}
    subject_sessions = {subj: subject_sessions.get(subj, 0) for subj in subjects}
    subject_water = {subj: subject_water.get(subj, 0) for subj in subjects}
    subject_weight = {subj: subject_weight.get(subj, np.nan) for subj in subjects}

    # generate a dataframe
    report_df = pd.DataFrame(
        {
            "Subject": subject_detections.keys(),
            "Detections": subject_detections.values(),
            "Sessions": subject_sessions.values(),
            "Water Consumed (μl)": [subject_water.get(subj, 0) for subj in subject_detections.keys()],
            "Average Weight (g)": [subject_weight.get(subj, np.nan) for subj in subject_detections.keys()],
        }
    )

    return report_df, max_date
