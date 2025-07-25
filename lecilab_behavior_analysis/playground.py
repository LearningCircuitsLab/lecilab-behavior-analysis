
#%%
# from lecilab_behavior_analysis.figure_maker import subject_progress_figure
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

mouse = "ACV008"
project = "visual_and_COT_data"
local_path = Path(utils.get_outpath()) / Path(project) / Path("videos") / Path(mouse)
# create the directory if it doesn't exist
local_path.mkdir(parents=True, exist_ok=True)
# download the session data
utils.rsync_cluster_data(
    project_name=project,
    file_path="videos/ACV008/ACV008_TwoAFC_20250401_160725.mp4",
    local_path=str(local_path),
    credentials=utils.get_idibaps_cluster_credentials(),
)