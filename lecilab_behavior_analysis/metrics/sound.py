"""Statistics of the cloud-of-tones stimulus."""

import numpy as np
import pandas as pd


def get_sound_stats(sound_dict: dict) -> dict:
    """
    This method returns a dictionary with the actual sound statistics
    of two sound matrices like a cloud of tones

    Args:
        sound_dict (dict): Dictionary with the entries for the two sound matrices

    Returns:
        sound_stats (dict): Dictionary with the sound statistics
    """
    # if the entries are dictionaries, convert them to dataframes
    if isinstance(sound_dict["high_tones"], dict):
        high_mat = pd.DataFrame(sound_dict["high_tones"])
    else:
        high_mat = sound_dict["high_tones"]
    if isinstance(sound_dict["low_tones"], dict):
        low_mat = pd.DataFrame(sound_dict["low_tones"])
    else:
        low_mat = sound_dict["low_tones"]
    # make the matrices binary
    high_mat = (high_mat > 0).astype(int)
    low_mat = (low_mat > 0).astype(int)
    # get the sound statistics
    high_mat_stats = analyze_sound_matrix(high_mat)
    low_mat_stats = analyze_sound_matrix(low_mat)
    # calculate the evidence strength for the high sound
    total_high_evidence_strength = sound_evidence_strength(
        high_mat_stats["percentage_of_timebins_with_evidence"], low_mat_stats["percentage_of_timebins_with_evidence"]
    )

    sound_stats = {
        "high_tones": high_mat_stats,
        "low_tones": low_mat_stats,
        "total_evidence_strength": total_high_evidence_strength,
    }

    return sound_stats


def analyze_sound_matrix(matrix: pd.DataFrame) -> dict:
    """
    This method analyzes a sound matrix and returns a dictionary with the sound statistics

    Args:
        matrix (pd.DataFrame): Sound matrix to analyze

    Returns:
        sound_stats (dict): Dictionary with the matrix statistics
    """
    # calculate the number of 1s
    evidences = np.sum(matrix.values)
    # calculate the percentage of 1s
    perc = evidences / matrix.size
    # calculate the real probability
    # of having at least one 1 in each column
    colsums = matrix.sum(axis=0)
    real_prob = np.sum(colsums > 0) / len(colsums)

    return {
        "number_of_tones": evidences,
        "total_percentage_of_tones": perc,
        "percentage_of_timebins_with_evidence": real_prob,
    }


def sound_evidence_strength(x, y):
    return (x - y) / (x + y)


def get_decibels_of_cloud_of_tones(sound_dict: dict) -> dict:
    # if the entries are dictionaries, convert them to dataframes
    if isinstance(sound_dict["high_tones"], dict):
        high_mat = pd.DataFrame(sound_dict["high_tones"])
    else:
        high_mat = sound_dict["high_tones"]
    if isinstance(sound_dict["low_tones"], dict):
        low_mat = pd.DataFrame(sound_dict["low_tones"])
    else:
        low_mat = sound_dict["low_tones"]
    # get the mean of the values that are different than zero
    high_db = high_mat[high_mat > 0].mean(axis=None)
    low_db = low_mat[low_mat > 0].mean(axis=None)
    # concatenate both dataframes
    combined_db = pd.concat([high_mat, low_mat], axis=1)
    overall_db = combined_db[combined_db > 0].mean(axis=None)

    return {"high_db": high_db, "low_db": low_db, "overall_db": overall_db}


def get_timebin_evidence(trial):
    evidence_timebin = np.zeros(len(trial["high_tones"]))
    for t, n in zip(trial["high_tones"], range(len(trial["high_tones"]))):
        for freq in trial["high_tones"][t]:
            if trial["high_tones"][t][freq] > 0:
                evidence_timebin[n] += 1
        for freq in trial["low_tones"][t]:
            if trial["low_tones"][t][freq] > 0:
                evidence_timebin[n] -= 1
    return evidence_timebin


def add_auditory_real_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df["number_of_tones_high"] = df["auditory_real_statistics"].apply(
        lambda x: eval(x)["high_tones"]["number_of_tones"]
    )
    df["number_of_tones_low"] = df["auditory_real_statistics"].apply(lambda x: eval(x)["low_tones"]["number_of_tones"])
    df["total_percentage_of_tones_high"] = df["auditory_real_statistics"].apply(
        lambda x: eval(x)["high_tones"]["total_percentage_of_tones"]
    )
    df["total_percentage_of_tones_low"] = df["auditory_real_statistics"].apply(
        lambda x: eval(x)["low_tones"]["total_percentage_of_tones"]
    )
    df["percentage_of_timebins_with_evidence_high"] = df["auditory_real_statistics"].apply(
        lambda x: eval(x)["high_tones"]["percentage_of_timebins_with_evidence"]
    )
    df["percentage_of_timebins_with_evidence_low"] = df["auditory_real_statistics"].apply(
        lambda x: eval(x)["low_tones"]["percentage_of_timebins_with_evidence"]
    )
    # df['total_evidence_strength'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['total_evidence_strength'])
    # the total evidence strength needs to be recalculated as in the task it was done with
    # suboptimal values (number of tones instead of percentage of timebins with evidence)
    df["total_evidence_strength"] = df.apply(
        lambda row: sound_evidence_strength(
            row["percentage_of_timebins_with_evidence_high"], row["percentage_of_timebins_with_evidence_low"]
        ),
        axis=1,
    )
    return df
