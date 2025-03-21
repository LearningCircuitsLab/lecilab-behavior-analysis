
#%%
# from lecilab_behavior_analysis.figure_maker import subject_progress_figure
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
from pathlib import Path
import pandas as pd
import numpy as np
import ast

mouse = "ACV001"
local_path = Path(utils.get_outpath()) / Path("visual_and_COT_data") / Path("sessions") / Path(mouse)

df = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")

# fill information in df if it is missing
df = dft.fill_missing_data(df)

df = dft.add_day_column_to_df(df)
#%%# %%
