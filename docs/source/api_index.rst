API
===

utils
-----

.. currentmodule:: lecilab_behavior_analysis.utils

.. autosummary::
    :toctree: api_generated
    :template: function.rst

    get_session_performance
    get_day_performance
    get_session_number_of_trials
    get_day_number_of_trials
    get_start_and_end_times
    get_block_size_truncexp_mean30
    get_right_bias
    get_block_size_uniform_pm30
    column_checker
    get_text_from_subset_df
    get_text_from_subject_df
    load_example_data
    get_sound_stats
    analyze_sound_matrix
    sound_evidence_strength
    get_outpath
    get_idibaps_cluster_credentials
    get_server_projects
    get_folders_from_server
    get_animals_in_project
    rsync_cluster_data
    list_to_colors
    is_this_a_miss_trial
    trial_reaction_time
    first_poke_after_stimulus_state
    get_last_poke_of_trial
    get_last_poke_before_stimulus_state
    get_dictionary_event_as_list
    get_repeat_or_alternate_to_numeric
    lapse_logistic_independent
    fit_lapse_logistic_independent
    get_trial_port_hold
    logi_model_fit_input
    logi_model_fit
    hierarchical_partitioning
    previous_impact_on_time_kernel
    verify_params_time_kernel
    generate_tv_report
    load_all_events
    filter_variables_for_model
    find_next_end_task_time_in_events
    get_session_box_usage

df_transforms
-------------

.. currentmodule:: lecilab_behavior_analysis.df_transforms

.. autosummary::
    :toctree: api_generated
    :template: function.rst

    fill_missing_data
    get_dates_df
    get_water_df
    get_repeat_or_alternate_series
    repeat_or_alternate_series_comparison
    add_port_where_animal_comes_from
    get_performance_through_trials
    get_repeat_or_alternate_performance
    get_evidence_ratio
    get_left_choice
    get_performance_by_difficulty
    add_auditory_real_statistics
    get_performance_by_difficulty_ratio
    get_performance_by_difficulty_diff
    side_and_difficulty_to_numeric
    get_training_summary_matrix
    calculate_time_between_trials_and_reaction_time
    add_inter_trial_interval_column_to_df
    add_trial_duration_column_to_df
    add_day_column_to_df
    add_trial_of_day_column_to_df
    add_trial_misses
    get_start_and_end_of_sessions_df
    get_daily_occupancy_percentages
    get_occupancy_heatmap
    get_occupancy_matrix
    reformat_df_columns
    analyze_df
    add_mouse_first_choice
    add_mouse_last_choice
    get_performance_by_decision_df
    get_triangle_polar_plot_df
    get_bias_evolution_df
    points_to_lines_for_bias_evolution
    create_transition_matrix
    add_visual_stimulus_difference
    get_center_hold_df
    get_reaction_times_by_date_df
    get_left_auditory_stim
    get_choice_before
    parameters_for_fit
    get_time_kernel_impact

plots
-----

.. currentmodule:: lecilab_behavior_analysis.plots

.. autosummary::
    :toctree: api_generated
    :template: function.rst

    training_calendar_plot
    rasterize_plot
    trials_by_session_plot
    trials_by_day_plot
    trials_by_session_hist
    performance_vs_trials_plot
    water_by_date_plot
    summary_text_plot
    number_of_correct_responses_plot
    correct_left_and_right_plot
    correct_left_and_right_plot_multisensory
    repeat_or_alternate_performance_plot
    psychometric_plot
    summary_matrix_plot
    side_correct_performance_plot
    plot_time_between_trials_and_reaction_time
    bias_vs_trials_plot
    performance_by_decision_plot
    triangle_polar_plot
    plot_decision_evolution_triangle
    plot_percentage_of_occupancy_per_day
    plot_training_times_heatmap
    plot_training_times_clock_heatmap
    plot_transition_matrix
    plot_transition_network_with_curved_edges
    plot_number_of_pokes_histogram
    plot_port_holding_time_histogram
    plot_mean_and_cis_by_date
    plot_table_from_df
    plot_filter_model_variables

figure_maker
------------

.. currentmodule:: lecilab_behavior_analysis.figure_maker

.. autosummary::
    :toctree: api_generated
    :template: function.rst

    subject_progress_figure
    session_summary_figure

generate_fake_dataset
---------------------

.. currentmodule:: lecilab_behavior_analysis.generate_fake_dataset

.. autosummary::
    :toctree: api_generated
    :template: function.rst

    generate_fake_dataset
