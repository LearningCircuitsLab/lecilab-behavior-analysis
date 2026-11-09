"""
Kept so that old code keeps working:
``import lecilab_behavior_analysis.df_transforms as dft``.

The functions that used to live here were moved to modules grouped by topic:

- preprocessing: prepare() / analyze_df(), filling and normalizing the data
- metrics.performance: performance through trials, by difficulty, by decision
- metrics.choices: first / last choice, repeat or alternate, bias, psychometrics
- metrics.timing: reaction times, center holds, time between trials
- metrics.engagement, metrics.occupancy, metrics.sound

New code should import from those modules. Everything is re-exported here
with its old name.
"""

# flake8: noqa
from lecilab_behavior_analysis.common import (
    _get_single_modality,
)
from lecilab_behavior_analysis.metrics.choices import (
    add_repeat_or_alternate_numeric,
    add_port_where_animal_comes_from,
    get_left_choice,
    get_performance_by_difficulty_ratio,
    get_performance_by_difficulty_diff,
    add_mouse_first_choice,
    add_mouse_last_choice,
    get_triangle_polar_plot_df,
    get_bias_evolution_df,
    points_to_lines_for_bias_evolution,
    create_transition_matrix,
    _difference_correct_minus_incorrect,
    add_visual_stimulus_difference,
    get_left_auditory_stim,
    get_choice_before,
    parameters_for_fit,
    get_time_kernel_impact,
)
from lecilab_behavior_analysis.metrics.engagement import (
    add_engagement_column,
    get_box_usage_df,
)
from lecilab_behavior_analysis.metrics.occupancy import (
    get_dates_df,
    get_water_df,
    get_start_and_end_of_sessions_df,
    get_daily_occupancy_percentages,
    get_occupancy_heatmap,
    get_occupancy_matrix,
)
from lecilab_behavior_analysis.metrics.performance import (
    get_repeat_or_alternate_series,
    repeat_or_alternate_series_comparison,
    get_performance_through_trials,
    get_repeat_or_alternate_performance,
    get_evidence_ratio,
    get_performance_by_difficulty,
    get_training_summary_matrix,
    get_performance_by_decision_df,
)
from lecilab_behavior_analysis.metrics.sound import (
    add_auditory_real_statistics,
)
from lecilab_behavior_analysis.metrics.timing import (
    add_port_holds,
    add_reaction_time,
    add_time_between_trials,
    calculate_time_between_trials_and_reaction_time,
    add_inter_trial_interval_column_to_df,
    add_trial_initiation_time_to_df,
    add_trial_duration_column_to_df,
    add_trial_misses,
    get_center_hold_df,
    get_reaction_times_by_date_df,
    adjust_trials_and_time_of_start_to_first_session,
    add_early_pokeouts,
)
from lecilab_behavior_analysis.preprocessing import (
    PREPARED_COLUMNS,
    prepare,
    fill_missing_data,
    add_day_column_to_df,
    add_trial_of_day_column_to_df,
    reformat_df_columns,
    analyze_df,
)
