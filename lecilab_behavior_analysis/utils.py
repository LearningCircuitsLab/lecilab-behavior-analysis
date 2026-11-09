"""
Kept so that old code keeps working: ``import lecilab_behavior_analysis.utils as utils``.

The functions that used to live here were moved to modules grouped by topic:

- io.server / io.loading: cluster paths, ssh, rsync, loading csv files
- metrics.trial_events: reading the events of one trial (first choice, reaction time...)
- metrics.performance, metrics.sound, metrics.engagement: numbers computed from the trials
- models.psychometric, models.regression: model fitting
- reports.tables: tables for the daily report
- task_helpers: helpers used by the tasks (block sizes, bias)
- common: small checks shared by every module

New code should import from those modules. Everything is re-exported here
with its old name.
"""

# flake8: noqa
from lecilab_behavior_analysis.io.server import IDIBAPS_TV_PROJECTS
from lecilab_behavior_analysis.common import (
    column_checker,
    list_to_colors,
)
from lecilab_behavior_analysis.io.loading import (
    load_example_data,
    load_all_events,
    read_csv_with_specific_number_of_columns,
)
from lecilab_behavior_analysis.io.server import (
    get_outpath,
    get_idibaps_cluster_credentials,
    get_server_projects,
    get_folders_from_server,
    get_animals_in_project,
    get_files_for_animal,
    rsync_cluster_data,
    rsync_specific_file,
)
from lecilab_behavior_analysis.metrics.choices import (
    previous_impact_on_time_kernel,
)
from lecilab_behavior_analysis.metrics.engagement import (
    find_next_end_task_time_in_events,
    get_session_box_usage,
)
from lecilab_behavior_analysis.metrics.performance import (
    get_session_performance,
    get_day_performance,
    get_session_number_of_trials,
    get_day_number_of_trials,
    get_text_from_subset_df,
    get_text_from_subject_df,
)
from lecilab_behavior_analysis.metrics.sound import (
    get_sound_stats,
    analyze_sound_matrix,
    sound_evidence_strength,
    get_decibels_of_cloud_of_tones,
    get_timebin_evidence,
)
from lecilab_behavior_analysis.metrics.trial_events import (
    get_start_and_end_times,
    is_this_a_miss_trial,
    is_this_an_early_pokeout_trial,
    trial_reaction_time,
    trial_initiation_time,
    first_poke_after_stimulus_state,
    get_last_poke_of_trial,
    get_last_poke_before_stimulus_state,
    get_dictionary_event_as_list,
    get_repeat_or_alternate_to_numeric,
    get_trial_port_hold,
    side_and_difficulty_to_numeric,
    add_time_from_session_start,
    get_previous_row_index,
    transform_side_choice_to_numeric,
)
from lecilab_behavior_analysis.models.psychometric import (
    lapse_logistic_independent,
    fit_lapse_logistic_independent,
    psychometric_curve_fitting_params,
)
from lecilab_behavior_analysis.models.regression import (
    logi_model_fit_input,
    logi_model_fit,
    hierarchical_partitioning,
    drop_one_var_contribution,
    verify_params_time_kernel,
    filter_variables_for_model,
)
from lecilab_behavior_analysis.reports.tables import (
    generate_tv_report,
)
from lecilab_behavior_analysis.task_helpers import (
    get_block_size_truncexp_mean30,
    get_right_bias,
    get_block_size_uniform_pm30,
)
