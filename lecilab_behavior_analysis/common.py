"""Small helpers shared by every module: column checks, colors, the modality of a dataframe."""
from matplotlib import pyplot as plt
from typing import Dict
from typing import List
from typing import Tuple
import numpy as np
import pandas as pd

import lecilab_behavior_analysis.data_format_changes as dfc


def column_checker(df: pd.DataFrame, required_columns: set):
    """
    This method checks if the required columns are present in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to check
        required_columns (set): Set of required columns
    """
    if not required_columns.issubset(df.columns):
        missing_columns = required_columns - set(df.columns)
        raise ValueError(
            "The dataframe must have the following columns: "
            + ", ".join(missing_columns)
        )


def list_to_colors(ids: np.array, cmap: str) -> Tuple[List[tuple], Dict]:
    unique_cts = pd.unique(ids)
    colormap = plt.get_cmap(cmap)
    colors = [colormap(i) for i in range(len(unique_cts))]
    # Create a color dictionary
    color_dict = {unique_cts[i]: colors[i] for i in range(len(unique_cts))}
    # Set the colors for each current_training_stage
    color_list = [color_dict[x] for x in list(ids)]
    return color_list, color_dict


def _get_single_modality(df: pd.DataFrame) -> str:
    """
    Returns the stimulus modality of the dataframe, with its current name
    (see data_format_changes.py). Raises if there is more than one.
    """
    modalities = df["stimulus_modality"].apply(dfc.normalize_modality).unique()
    if len(modalities) != 1:
        raise ValueError(
            f"The dataframe must contain one stimulus modality, got {list(modalities)}"
        )
    return modalities[0]
