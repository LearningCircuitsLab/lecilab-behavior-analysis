
#%%
# from lecilab_behavior_analysis.figure_maker import subject_progress_figure
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
from pathlib import Path
import pandas as pd
import numpy as np
import ast

mouse = "ACV009"
local_path = Path(utils.get_outpath()) / Path("visual_and_COT_data") / Path("sessions") / Path(mouse)

df = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")

# fill information in df if it is missing
df = dft.analyze_df(df)
#%%
df = df[df.current_training_stage == "TwoAFC_visual_hard"]
#%%
df.dropna(subset=['visual_stimulus'], inplace=True)

#%%
# This cell will show you the actual values
df['visual_stimulus_devi'] = df['visual_stimulus'].apply(lambda x: abs(round(eval(x)[0] / eval(x)[1], 4)))
df['visual_stimulus_devi'].value_counts()
# %%
# This cell will show you the problem you had
df['visual_stimulus_devi'] = df['visual_stimulus'].apply(lambda x: abs(round(eval(x)[0] / eval(x)[1], 0)))
df['visual_stimulus_devi'].value_counts()

# rounding to 0 decimals (default if you dont specify), makes the values of 2.5 to sometimes be 2 and sometimes be 3