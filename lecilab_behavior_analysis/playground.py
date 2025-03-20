
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

# fill information in df if it is missing
df = dft.fill_missing_data(df)

df = dft.add_day_column_to_df(df)

dates_df = df.groupby(["year_month_day"]).count().reset_index()
print(dates_df)


# %%
import lecilab_behavior_analysis.plots as plots
import lecilab_behavior_analysis.utils as utils
bp_df = df.groupby(["year_month_day", "date", "current_training_stage"]).count()
# pivot the dataframe so that the individual "date" indexes are stacked
bp_df = bp_df.pivot_table(index="year_month_day", columns=["date", "current_training_stage"], values="trial")
# Plot the stacked barplot, coloring by current_training_stage
list_of_training_stages = [x[1] for x in bp_df.columns]
color_list, color_dict = utils.list_to_colors(ids=list_of_training_stages, cmap="tab20")
# %%

