"""
Drawing functions. Each one draws on the axes it is given and does not
change the data.

Grouped by topic in plots.training, plots.performance, plots.choices,
plots.timing and plots.occupancy. Everything is re-exported here, so
``import lecilab_behavior_analysis.plots as plots`` keeps working.
"""

# flake8: noqa
from lecilab_behavior_analysis.plots.choices import (
    psychometric_plot,
    choice_by_difficulty_plot,
    bias_vs_trials_plot,
    triangle_polar_plot,
    plot_decision_evolution_triangle,
    plot_filter_model_variables,
    pointplot_stimulus_dual_dependence,
)
from lecilab_behavior_analysis.plots.occupancy import (
    plot_percentage_of_occupancy_per_day,
    plot_training_times_heatmap,
    plot_training_times_clock_heatmap,
    plot_transition_matrix,
    plot_transition_network_with_curved_edges,
)
from lecilab_behavior_analysis.plots.performance import (
    performance_vs_trials_plot,
    _correct_as_bool,
    number_of_correct_responses_plot,
    correct_left_and_right_plot,
    partially_stacked_barplot,
    repeat_or_alternate_performance_plot,
    side_correct_performance_plot,
    performance_by_decision_plot,
)
from lecilab_behavior_analysis.plots.timing import (
    mean_and_ci_by_day_plot,
    time_between_trials_and_reaction_time_plot,
    plot_time_between_trials_and_reaction_time,
    plot_number_of_pokes_histogram,
    plot_port_holding_time_histogram,
    plot_mean_and_cis_by_date,
    plot_trial_time_of_start,
    plot_box_usage_by_date,
)
from lecilab_behavior_analysis.plots.training import (
    calendar_plot,
    format_date_axis,
    trials_by_day_stacked_plot,
    water_by_day_plot,
    training_calendar_plot,
    rasterize_plot,
    trials_by_session_plot,
    trials_by_day_plot,
    trials_by_session_hist,
    water_by_date_plot,
    summary_text_plot,
    summary_matrix_plot,
    plot_table_from_df,
)
