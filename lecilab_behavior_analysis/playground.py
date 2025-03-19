
#%%
# from lecilab_behavior_analysis.figure_maker import subject_progress_figure
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
from pathlib import Path
import pandas as pd
import numpy as np

mouse = "ACV001"
local_path = Path(utils.get_outpath()) / Path("visual_and_COT_data") / Path("sessions") / Path(mouse)

df = pd.read_csv(local_path / Path(f'{mouse}.csv'), sep=";")

df = dft.add_day_column_to_df(df)

dates_df = df.groupby(["year_month_day"]).count().reset_index()
print(dates_df)

# %%
