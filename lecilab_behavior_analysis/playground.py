
#%%
# from lecilab_behavior_analysis.figure_maker import subject_progress_figure
import lecilab_behavior_analysis.df_transforms as dft
import lecilab_behavior_analysis.utils as utils
from pathlib import Path
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt

file_path = "/mnt/c/Users/HMARTINEZ/Desktop/weights.txt"

df = pd.read_csv(file_path, sep=";", header=None, names=["date","subject", "weights"])

#%%
nsubs = df["subject"].nunique()

median_values = df.groupby("subject")["weights"].median().reset_index()

fig, ax = plt.subplots(nrows=nsubs, ncols=1, figsize=(10, 2*nsubs))

for i, (sub, sub_df) in enumerate(df.groupby("subject")):

    xs = np.arange(len(sub_df["date"]))

    # get the median value of the weights
    median_weight = median_values.loc[median_values["subject"] == sub, "weights"].values[0]
    
    ax[i].plot(xs, sub_df["weights"], marker="o", linestyle="-")
    ax[i].set_title(f"Subject {sub}")
    ax[i].set_xlabel("Date")
    ax[i].set_ylabel("Weight (kg)")
    ax[i].tick_params(axis='x', rotation=45)
    ax[i].axhline(median_weight, color='red', linestyle='--', label='Median Weight')

plt.tight_layout()
plt.show()

median_values

#%%
def real_weight_inference(weight_array, threshold):
    """
    Conditions to call it a real weight:
     - minimum of 5 measurements
     - median larger than threshold
     - standard deviation of the last 3 measurements is
        smaller than 10% of the threshold
    """
    if len(weight_array) < 5:
        return False
    
    median_weight = np.median(weight_array[-5:])
    std_weight = np.std(weight_array[-3:])
    
    if median_weight > threshold and std_weight < 0.1 * threshold:
        return True
    else:
        return False

#%%
# Use a 5 measurement window to measure the median and the standard deviation
window_size_median = 5
window_size_std = 3
df["rolling_median"] = df.groupby("subject")["weights"].transform(lambda x: x.rolling(window=window_size_median, min_periods=1).median())
df["rolling_std"] = df.groupby("subject")["weights"].transform(lambda x: x.rolling(window=window_size_std, min_periods=1).std())
# Plot the rolling median and standard deviation
fig, ax = plt.subplots(nrows=nsubs, ncols=1, figsize=(10, 2*nsubs))
for i, (sub, sub_df) in enumerate(df.groupby("subject")):

    xs = np.arange(len(sub_df["date"]))

    # get the median value of the weights
    median_weight = median_values.loc[median_values["subject"] == sub, "weights"].values[0]
    
    ax[i].plot(xs, sub_df["weights"], marker="o", linestyle="-", label='Weight')
    ax[i].plot(xs, sub_df["rolling_median"], marker="o", linestyle="-", color='orange', label='Rolling Median')
    ax[i].fill_between(xs, sub_df["rolling_median"] - sub_df["rolling_std"], 
                       sub_df["rolling_median"] + sub_df["rolling_std"], color='orange', alpha=0.2, label='Rolling Std Dev')
    
    ax[i].set_title(f"Subject {sub}")
    ax[i].set_xlabel("Date")
    ax[i].set_ylabel("Weight (kg)")
    ax[i].tick_params(axis='x', rotation=45)
    ax[i].axhline(median_weight, color='red', linestyle='--', label='Median Weight')

    weight_array = sub_df["weights"].values
    for j in range(len(weight_array)):
        if real_weight_inference(weight_array[:j+1], 15):
            ax[i].axvline(x=j, color='green', linestyle='--', label='Real Weight Inference')
            # break for loop
            break

plt.tight_layout()
plt.legend()
plt.show()