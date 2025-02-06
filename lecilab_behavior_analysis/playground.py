from lecilab_behavior_analysis.figure_maker import subject_progress_figure
from lecilab_behavior_analysis.utils import load_example_data

mouse = "mouse1"
df = load_example_data(mouse)
fig = subject_progress_figure(df, mouse, perf_window=100)
