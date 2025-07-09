import pandas as pd
import numpy as np
import ast
from typing import Tuple, Union
import lecilab_behavior_analysis.utils as utils
import ast

def fill_missing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing data in the dataframe.
    """
    columns_to_fill = ["stimulus_modality", "current_training_stage", "difficulty", "correct_side"]
    for column in columns_to_fill:
        try:
            df[column] = df[column].fillna("not saved")
        except KeyError:
            df[column] = "not saved"
    # Fill missing values in the "correct" column
    try:
        df["correct"] = df["correct"].infer_objects(copy=False)
    except KeyError:
        df["correct"] = True

    return df


def get_dates_df(df: pd.DataFrame) -> pd.DataFrame:
    utils.column_checker(df, required_columns={"date", "current_training_stage"})
    dates_df = df.groupby(["date", "current_training_stage"]).count().reset_index()
    # set as index the date
    dates_df.set_index("date", inplace=True)
    dates_df.index = pd.to_datetime(dates_df.index)
    return dates_df


def get_water_df(df: pd.DataFrame, grouping_column: str = "date") -> pd.DataFrame:
    utils.column_checker(df, required_columns={grouping_column, "water"})
    water_df = df.groupby(grouping_column)["water"].sum()
    return water_df


def get_repeat_or_alternate_series(series: pd.Series) -> pd.Series:
    """
    Given a series of values, return a series of the same length
    with "repeat" or "alternate" depending on whether the value
    is the same as the previous one or not.

    Meant to be used for the trial side column (correct_side)
    Or the columns with the first choice of the mouse
    """
    prev_choices = series.shift(1, fill_value=np.nan)
    return repeat_or_alternate_series_comparison(series, prev_choices)


def repeat_or_alternate_series_comparison(
    series: pd.Series, comparison_series: pd.Series
) -> pd.Series:
    """
    Given a series of values, return a series of the same length
    with "repeat" or "alternate" depending on whether the value
    is the same between the two series or not.
    """
    repeat_or_alternate = np.where(
        series.isna() | comparison_series.isna(),
        None, # None instead of np.nan to keep dtype the same TODO: check!
        np.where(
            series == comparison_series,
            "repeat",
            "alternate"))

    return pd.Series(repeat_or_alternate, index=series.index)


def add_port_where_animal_comes_from(df_in: pd.DataFrame) -> pd.DataFrame:
    # TODO: the name of this function is not great, as it adds also extra information
    """
    The previous side port can come from two sources.
    One, from the previous trial, where the animal was drinking water
    (note, this should not necessarily be true as the animal can go to the
    other port afterwards without going through the middle, but if a trial has
    started before that happens, this function will take care of it)
    Second, from side port pokes previous to the initiation of the stimulus. This
    second condition can happen from two sources. Initial pokes before poking on
    the center, or poking in the center just brief enough to not trigger the
    stimulus state. In some cases this is not punished.
    """
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    # Get side port pokes before the stimulus state
    df["last_choice_before_stimulus"] = df.apply(utils.get_last_poke_before_stimulus_state, axis=1)
    # Get the last port in each trial
    if "last_choice" not in df.columns:
        df = add_mouse_last_choice(df)
    
    # Check if there are side ports pokes before the stimulus state,
    # and if not, return the previous trial data. If there are, return the
    # last side port pokes of current trial
    df['roa_choice'] = np.nan
    df['roa_choice'] = df['roa_choice'].astype(object)
    for mouse in df['subject'].unique():
        for session in df[df.subject == mouse]['session'].unique():
            df_mouse_session = df[np.logical_and(df['subject'] == mouse, df['session'] == session)]
            # compute which one is the last choice that they made
            last_choice_before_stimulus = df_mouse_session['last_choice_before_stimulus']
            last_choice_in_previous = df_mouse_session['last_choice'].shift(1, fill_value=np.nan)
            # define a lambda function that gets these two series and:
            # 1. if the last_choice_before_stimulus is not null, return it
            # 2. if the last_choice_before_stimulus is null, return the last_choice_in_previous
            # 3. if both are null, return np.nan
            lambda_func = lambda x, y: x if pd.notna(x) else (y if pd.notna(y) else np.nan)
            # apply the lambda function to the two series
            last_choice = last_choice_before_stimulus.combine(last_choice_in_previous, lambda_func)
            series_to_append = repeat_or_alternate_series_comparison(
                # first choice of each trial
                df_mouse_session['first_choice'],
                # last choice in the previous trial
                last_choice,
                )
            # add the new column to the original dataframe, as the series have the index
            # equal to the original dataframe
            df.loc[df_mouse_session.index, 'roa_choice'] = series_to_append
            # add also the port where the animal comes from
            df.loc[df_mouse_session.index, 'previous_port_before_stimulus'] = last_choice

    return df


def get_performance_through_trials(df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    utils.column_checker(df, required_columns={"session", "trial", "correct"})
    # make sure only one subject is present
    if df["subject"].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    # sort the df by "session" and "trial"
    df = df.sort_values(["session", "trial"])
    # add a column with the total number of trials if it doesn't exist
    if "total_trial" not in df.columns:
        df["total_trial"] = np.arange(1, df.shape[0] + 1)
    # calculate the performance as a mean of the last X trials
    df["performance_w"] = df.correct.rolling(window=window).mean() * 100

    return df


def get_repeat_or_alternate_performance(
    df_in: pd.DataFrame, window: int = 25
) -> pd.DataFrame:
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    if "repeat_or_alternate" not in df.columns:
        df["repeat_or_alternate"] = get_repeat_or_alternate_series(df.correct_side)

    df["repeat_or_alternate_performance"] = df.groupby("repeat_or_alternate")[
        "correct"
    ].transform(lambda x: x.rolling(window=window).mean() * 100)
    return df


def get_performance_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    utils.column_checker(df, required_columns={"difficulty", "correct", "correct_side"})
    pbd_df = df.groupby(["difficulty", "correct_side"]).correct.mean().unstack().reset_index()
    # melt the dataframe
    pbd_df = pbd_df.melt(id_vars=["difficulty"])
    # assing a numeric value to the side and the difficulty for plotting purposes
    pbd_df["leftward_evidence"] = pbd_df.apply(side_and_difficulty_to_numeric, axis=1)
    pbd_df["leftward_choices"] = np.where(pbd_df["correct_side"] == "left", pbd_df["value"], 1 - pbd_df["value"])
    return pbd_df

def add_auditory_real_statistics(df: pd.DataFrame) -> pd.DataFrame:
    df['number_of_tones_high'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['high_tones']['number_of_tones'])
    df['number_of_tones_low'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['low_tones']['number_of_tones'])
    df['total_percentage_of_tones_high'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['high_tones']['total_percentage_of_tones'])
    df['total_percentage_of_tones_low'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['low_tones']['total_percentage_of_tones'])
    df['percentage_of_timebins_with_evidence_high'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['high_tones']['percentage_of_timebins_with_evidence'])
    df['percentage_of_timebins_with_evidence_low'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['low_tones']['percentage_of_timebins_with_evidence'])
    df['total_evidence_strength'] = df['auditory_real_statistics'].apply(lambda x: eval(x)['total_evidence_strength'])
    return df

def get_performance_by_difficulty_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if df["stimulus_modality"].unique() == 'visual':
        stim_col = "visual_stimulus"
        ratio_col = "visual_stimulus_ratio"
        df["visual_stimulus_ratio"] = df["visual_stimulus"].apply(lambda x: abs(eval(x)[0] / eval(x)[1]))
        # df["visual_stimulus_ratio"] = df["visual_stimulus_ratio"].apply(np.log).round(4)
        df["visual_stimulus_ratio"] = df.apply(
            lambda row: row["visual_stimulus_ratio"] if row['correct_side'] == 'left' else -row["visual_stimulus_ratio"],
            axis=1
        )
    elif df["stimulus_modality"].unique() == "auditory":
        df = add_auditory_real_statistics(df)
    else:
        raise ValueError("modality must be either 'visual' or 'auditory'")
    df = add_mouse_first_choice(df)
    df['first_choice_numeric'] = df['first_choice'].apply(lambda x: 1 if x == 'left' else 0)

    return df


def get_performance_by_difficulty_diff(df: pd.DataFrame) -> pd.DataFrame:
    if df["current_training_stage"].str.contains("visual").any():
        stim_col = "visual_stimulus"
        diff_col = "visual_stimulus_diff"
    elif df["current_training_stage"].str.contains("auditory").any():
        stim_col = "auditory_stimulus"
        diff_col = "auditory_stimulus_diff"
    else:
        raise ValueError("modality must be one of 'visual' or 'auditory'")

    df[diff_col] = df[stim_col].apply(lambda x: abs(eval(x)[0] - eval(x)[1]))
    df[diff_col] = df.apply(
        lambda row: row[diff_col] if row['correct_side'] == 'left' else -row[diff_col],
        axis=1
    )
    df['first_choice_numeric'] = df['first_choice'].apply(lambda x: 1 if x == 'left' else 0)
    return df


def side_and_difficulty_to_numeric(row: pd.Series) -> float:
    match row.difficulty:
        case "easy":
            numval = 3
        case "medium":
            numval = 2
        case "hard":
            numval = 1
        case _:
            numval = 0
    match row.correct_side:
        case "left":
            pass
        case "right":
            numval *= -1
        case _:
            numval = 0
    
    return round(numval / 3, 3)


def get_training_summary_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    utils.column_checker(df, required_columns={"session"})
    # Initialize lists to save important data
    leftward_evidence_list = []
    leftward_choices_list = []
    # Initialize a dicctionary to save information about the sessions
    session_info = {}

    # process data from all sessions
    for session in df.session.unique():
        pbd_df = get_performance_by_difficulty(df[df.session == session])
        leftward_evidence_list.append(pbd_df.leftward_evidence.to_list())
        leftward_choices_list.append(pbd_df.leftward_choices.to_list())
        session_info[session] = {
            "date": df[df.session == session].date.unique()[0],
            "current_training_stage": df[df.session == session].current_training_stage.unique()[0],
            "n_trials": df[df.session == session].shape[0],
            "n_correct": int(df[df.session == session].correct.sum()),
            "performance": int(df[df.session == session].correct.sum()) / df[df.session == session].shape[0] * 100,
            "water": df[df.session == session].water.sum(),
            "stimulus_modality": df[df.session == session].stimulus_modality.unique()[0],
        }

    # get difficulty levels
    ev_levels = np.unique(np.concatenate(leftward_evidence_list).ravel())
    # Initialize the matrix
    mat_df = np.full([len(ev_levels), len(leftward_evidence_list)], np.nan)
    # Loop to fill it
    for i, evidence in enumerate(ev_levels):
        for j, choice in enumerate(leftward_choices_list):
            if evidence in leftward_evidence_list[j]:
                idx = np.where(leftward_evidence_list[j] == evidence)[0][0]
                mat_df[i, j] = choice[idx]

    # Transform to dataframe
    mat_df = pd.DataFrame(mat_df)
    mat_df = mat_df.set_index(ev_levels)
    mat_df.columns = df.session.unique()

    return mat_df, session_info


def calculate_time_between_trials_and_reaction_time(in_df: pd.DataFrame) -> pd.DataFrame:
    # TODO: separate this into two functions, one for time between trials and one for reaction time
    
    """
    Calculate Time Between Trials and Reaction Time.
    """
    # Check if the required columns are present
    utils.column_checker(in_df, required_columns={"Port1In", "Port1Out", "Port2In", "Port2Out", "Port3In", "Port3Out"})
    df = in_df.copy()  # Make a copy to avoid modifying the original DataFrame
    for date in pd.unique(df['date']):
        date_df = df[df['date'] == date].copy()
        port2outs = date_df['Port2Out'].apply(lambda x: np.max(ast.literal_eval(x)) if isinstance(x, str) else np.max(x))
        date_df['time_between_trials'] = port2outs.diff()
        df.loc[df['date'] == date, 'time_between_trials'] = date_df['time_between_trials']
    # Calculate the reaction time
    df['reaction_time'] = df.apply(utils.trial_reaction_time, axis=1)

    return df


def add_inter_trial_interval_column_to_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add an inter-trial interval column to the dataframe.
    """
    # Check if the required columns are present
    utils.column_checker(df_in, required_columns={"Port2In", "TRIAL_END"})
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    for date in pd.unique(df['date']):
        session_df = df[df['date'] == date].copy()
        # Calculate the inter-trial interval
        # shift the trial end time by one
        trial_end_shifted = session_df['TRIAL_END'].shift(1)
        # Animals can do multiple port2Ins in each trial. It could happen that the
        # animal gives up in the middle of these pokes. We will consider the last port2In as the one that counts
        # TODO: changed to the min
        port2ins_last = session_df['Port2In'].apply(lambda x: np.min(ast.literal_eval(x)) if isinstance(x, str) else np.min(x))
        iti_vector = port2ins_last - trial_end_shifted
        # fill the first value with NaN
        iti_vector.iloc[0] = np.nan
        # assign the new column to the original dataframe
        df.loc[df['date'] == date, 'inter_trial_interval'] = iti_vector
    
    return df


def add_trial_duration_column_to_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trial duration column to the dataframe.
    """
    df = df_in.copy()  # Make a copy to avoid modifying the original DataFrame
    # Check if the required columns are present
    utils.column_checker(df, required_columns={"TRIAL_START", "TRIAL_END"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    trial_duration_date = df['TRIAL_END'] - df['TRIAL_START']
    df['trial_duration'] = trial_duration_date

    return df



def add_day_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a day column to the dataframe.
    """
    # Check if the required columns are present
    utils.column_checker(df, required_columns={"date"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['year_month_day'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    return df


def add_trial_of_day_column_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trial of the day column to the dataframe.
    """
    # Check if the required columns are present
    utils.column_checker(df, required_columns={"year_month_day", "trial"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['trial_of_day'] = df.groupby('year_month_day')['trial'].transform(lambda x: x - x.min() + 1)
    return df


def add_trial_misses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a trial misses column to the dataframe.
    """
    # Check if the required columns are present
    utils.column_checker(df, required_columns={"STATE_stimulus_state_END", "TRIAL_END"})
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df["miss_trial"] = df.apply(utils.is_this_a_miss_trial, axis=1)

    return df


def get_start_and_end_of_sessions_df(df: pd.DataFrame) -> pd.DataFrame:
    list_of_occupancy = []
    for mouse in pd.unique(df['subject']):
        mouse_df = df[df['subject'] == mouse]
        for session in mouse_df['session'].unique():
            session_df = mouse_df[mouse_df['session'] == session]
            list_of_occupancy.append((mouse, session, utils.get_start_and_end_times(session_df)))
    # create a dataframe from the list of occupancy
    occupancy_df = pd.DataFrame(list_of_occupancy, columns=['subject', 'session', 'start_end_times'])
    # split the start_end_times column into two columns
    occupancy_df[['start_time', 'end_time']] = occupancy_df['start_end_times'].apply(pd.Series)
    # Calculate the duration of each event in minutes
    occupancy_df['duration'] = (occupancy_df['end_time'] - occupancy_df['start_time']).dt.total_seconds() / 60
    # Group by date and calculate the total duration of events for each day
    occupancy_df['date'] = occupancy_df['start_time'].dt.date
    # drop the start_end_times column
    return occupancy_df.drop(columns=['start_end_times'])


def get_daily_occupancy_percentages(occupancy_df: pd.DataFrame) -> pd.DataFrame:
    daily_durations = occupancy_df.groupby('date')['duration'].sum()
    # Calculate the percentage of the day that the events occupy (1440 minutes in a day)
    return (daily_durations / 1440) * 100


def get_occupancy_heatmap(occupancy_df: pd.DataFrame, window_size: int = 30) -> np.ndarray:
    utils.column_checker(occupancy_df, required_columns={"start_time", "end_time"})
    # Create a vector to represent the heatmap (1440 minutes in a day)
    heatmap_vector = np.zeros(1440)

    # Populate the heatmap vector with event occurrences
    for start, end in zip(occupancy_df['start_time'], occupancy_df['end_time']):
        start_minute_of_day = start.hour * 60 + start.minute
        end_minute_of_day = end.hour * 60 + end.minute
        
        if start_minute_of_day <= end_minute_of_day:
            heatmap_vector[start_minute_of_day:end_minute_of_day] += 1
        else:
            heatmap_vector[start_minute_of_day:] += 1
            heatmap_vector[:end_minute_of_day] += 1
    
    # normalize the heatmap vector to the number of days
    heatmap_vector /= len(occupancy_df.date.unique())

    # add the window size to the beginning and end of the vector
    heatmap_vector = np.concatenate((heatmap_vector[len(heatmap_vector)-window_size:], heatmap_vector, heatmap_vector[:window_size]))
    # apply the moving average
    heatmap_vector = np.convolve(heatmap_vector, np.ones(window_size)/window_size, mode='same')
    # cut the vector to the original size
    heatmap_vector = heatmap_vector[window_size:len(heatmap_vector)-window_size]

    return heatmap_vector


def get_occupancy_matrix(occupancy_df: pd.DataFrame, window_size: int = 30) -> pd.DataFrame:
    # do the same as in the get_occupancy_heatmap function, but for each day of training
    utils.column_checker(occupancy_df, required_columns={"start_time", "end_time"})
    # get all the possible dates using both start and end times
    all_dates = pd.concat([
        occupancy_df['start_time'].dt.date,
        occupancy_df['end_time'].dt.date
    ]).unique()
    # generate the matrix
    occupancy_matrix = pd.DataFrame(index=all_dates, columns=np.arange(0, 1440, 1))
    # fill the matrix with zeros
    occupancy_matrix.fillna(0, inplace=True)
    # Populate the matrix with event occurrences
    # for each row of the dataframe
    counter = 0
    for _, row in occupancy_df.iterrows():
        start_minute_of_day = row['start_time'].hour * 60 + row['start_time'].minute
        end_minute_of_day = row['end_time'].hour * 60 + row['end_time'].minute
        date = row['start_time'].date()
        
        if start_minute_of_day <= end_minute_of_day:
            occupancy_matrix.loc[date, start_minute_of_day:end_minute_of_day] += 1
        else:
            occupancy_matrix.loc[date, start_minute_of_day:] += 1
            # get the next day
            next_day = date + pd.Timedelta(days=1)
            occupancy_matrix.loc[next_day, :end_minute_of_day] += 1
    
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} rows out of {occupancy_df.shape[0]}")
    return occupancy_matrix


def reformat_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()  # Make a copy to avoid modifying the original DataFrame
    df['visual_stimulus'] = df['visual_stimulus'].apply(ast.literal_eval)
    # TODO: fix issues with this when there are no values

    return df


def analyze_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the dataframe adding new columns and filling missing data.
    """
    # df = reformat_df_columns(df)
    df = fill_missing_data(df)
    df = add_day_column_to_df(df)
    df = add_trial_misses(df)
    df = add_mouse_first_choice(df)
    df = add_mouse_last_choice(df)

    # add a column with the total number of trials
    for subject in pd.unique(df['subject']):
        subject_df = df[df['subject'] == subject].copy()
        # add a column with the total number of trials
        subject_df["total_trial"] = np.arange(1, subject_df.shape[0] + 1)
        # add the total trial column to the original dataframe
        df.loc[df['subject'] == subject, "total_trial"] = subject_df["total_trial"]
        df = add_trial_of_day_column_to_df(df)

    return df


def add_mouse_first_choice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the first choice made by the mouse in each trial.
    """

    utils.column_checker(df, required_columns={"Port1In", "Port3In", "STATE_stimulus_state_START"})
    df = df.copy()
    df['first_choice'] = df.apply(utils.first_poke_after_stimulus_state, axis=1)
    
    return df


def add_mouse_last_choice(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the last choice made by the mouse in each trial.
    """
    utils.column_checker(df, required_columns={"Port1In", "Port3In"})
    df = df.copy()
    df['last_choice'] = df.apply(utils.get_last_poke_of_trial, axis=1)
    
    return df


def get_performance_by_decision_df(df: pd.DataFrame, trial_group_size: int = 500) -> pd.DataFrame:
    """
    Get the performance by decision dataframe.
    """
    # check that there is only one subject
    if df['subject'].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    utils.column_checker(df, required_columns={"correct_side", "repeat_or_alternate", "correct", "total_trial"})
    # bin the data into groups of trial_group_size
    df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
    df['trial_group'] = df['total_trial'] // trial_group_size * trial_group_size
    # group by the trial group and calculate the mean of the correct column
    df_binned = df.groupby(['correct_side', 'repeat_or_alternate', 'trial_group'])[['correct']].mean().reset_index()
    # performance in between 0 and 100
    df_binned['correct'] = df_binned['correct'] * 100
    # get the different combinations of correct side and repeat or alternate
    df_binned['correct_side_repeat_or_alternate'] = df_binned['correct_side'] + '_' + df_binned['repeat_or_alternate']
    # get only the 4 combinations and remove nans etc
    df_binned = df_binned[df_binned['correct_side_repeat_or_alternate'].isin(['left_repeat', 'left_alternate', 'right_repeat', 'right_alternate'])]

    return df_binned


def get_triangle_polar_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    utils.column_checker(df, required_columns={"roa_choice_numeric", "subject"})
    # get the bias for each animal
    df_bias = df.groupby(['subject'])['roa_choice_numeric'].value_counts().reset_index(name='count')
    # transform the bias to be in the range of 0 to 2pi
    df_bias['bias_angle'] = df_bias['roa_choice_numeric'].replace({
        0: 2 * np.pi / 4,
        -1: 5 * np.pi / 4,
        1: 7 * np.pi / 4
    })
    # transform counts into percentages
    df_bias['percentage'] = df_bias['count'] / df_bias.groupby(['subject'])['count'].transform('sum')

    return df_bias


def get_bias_evolution_df(df: pd.DataFrame, groupby: Union[str, list[str]]) -> pd.DataFrame:
    """
    Gets how the bias of the animals (alternating, right bias, or left bias)
    evolves over time.

    Arguments:
    df: DataFrame with the data
    groupby: str or list, the column(s) to group by. Can be 'session' or 'trial_group'

    Returns:
    df: DataFrame with the bias evolution
    """
    groupby_items = ["subject"]
    if isinstance(groupby, str):
        groupby_items.append(groupby)
    elif isinstance(groupby, list):
        groupby_items.extend(groupby)
    utils.column_checker(df, required_columns=set(groupby_items + ["roa_choice_numeric"]))
    df_anchev = df.copy()
    df_anchev = df_anchev.groupby(groupby_items)['roa_choice_numeric'].value_counts().reset_index(name='count')
    # transform counts into percentages
    df_anchev['percentage'] = df_anchev['count'] / df_anchev.groupby(groupby_items)['count'].transform('sum')

    # pivot or melt the dataframe so that each subject and session has a y value, that will be the percentage when the bias
    # is 0, and the x value will be the differences between the percentages when the bias is 1 and -1
    df_bias_pivot = df_anchev.pivot(index=groupby_items, columns='roa_choice_numeric', values='percentage')

    # fill the NaN values with 0
    df_bias_pivot = df_bias_pivot.fillna(0)
    # calculate the difference between the percentages when the bias is 1 and -1
    df_bias_pivot['left_right_bias'] = df_bias_pivot[1] - df_bias_pivot[-1]
    df_bias_pivot['alternating_bias'] = df_bias_pivot[0]
    # reset index
    return df_bias_pivot.reset_index()


def points_to_lines_for_bias_evolution(df: pd.DataFrame, groupby: str) -> pd.DataFrame:
    # if more than one subject, raise an error
    if df['subject'].nunique() > 1:
        raise ValueError("The dataframe should contain only one subject.")
    utils.column_checker(df, required_columns={groupby})
    # convert the xs and ys points to lines by duplicating the endpoint using the next trial group
    dfbps = df.copy()
    dfbps[groupby] = dfbps[groupby].shift(1)
    # add the two dataframes together
    df_bias_pivot_merged = pd.concat([df, dfbps], ignore_index=True)
    df_bias_pivot_merged.dropna(inplace=True)
    
    return df_bias_pivot_merged


def create_transition_matrix(events: list) -> pd.DataFrame:
    # Initialize a matrix with zeros
    items = list(set(events))
    items.sort()  # Sort the items to ensure consistent ordering
    n = len(items)
    transition_matrix = np.zeros((n, n), dtype=int)
    
    # Map items to their indices in the matrix
    item_index = {item: i for i, item in enumerate(items)}
    
    # Count transitions from one item to the next
    for i in range(len(events) - 1):
        from_item = events[i]
        to_item = events[i + 1]
        transition_matrix[item_index[from_item], item_index[to_item]] += 1
    
    # Return the transition matrix as a pandas DataFrame for better readability
    return pd.DataFrame(transition_matrix, index=items, columns=items)


def add_visual_stimulus_difference(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    utils.column_checker(df_in, required_columns={"visual_stimulus"})
    df['visual_stimulus'] = df['visual_stimulus'].apply(ast.literal_eval)
    df["visual_stim_difference"] = df["visual_stimulus"].apply(lambda x: x[0] - x[1])
    # bin the data every 0.1
    df["vis_stim_dif_bin"] = np.round((df["visual_stim_difference"] // 0.1) * 0.1, 1)
    return df


def get_center_hold_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    utils.column_checker(df, required_columns={"Port2In", "Port2Out", "year_month_day"})
    df["port2_holds"] = df.apply(lambda row: utils.get_trial_port_hold(row, 2), axis=1)
    df["port2_holds_number"] = df.port2_holds.apply(len)
    # df["port2_holds_mean"] = df.port2_holds.apply(np.mean)
    # group by date and get the mean and 95 of the port2_holds
    def mean_and_cis_of_holds(group):
        port_holds_number = group["port2_holds"].apply(len)
        mean_n = np.nanmean(port_holds_number)
        bot95_n, top95_n = np.nanpercentile(port_holds_number, [5, 95])
        
        port_holds = group["port2_holds"].tolist()
        port_holds = [item for sublist in port_holds for item in sublist]
        if len(port_holds) == 0:
            mean_s, bot95_s, top95_s = np.nan, np.nan, np.nan
        else:
            mean_s = np.nanmean(port_holds)
            bot95_s, top95_s = np.nanpercentile(port_holds, [5, 95])
        return pd.Series({
            "number_of_pokes_mean": mean_n,
            "number_of_pokes_bot95": bot95_n,
            "number_of_pokes_top95": top95_n,
            "hold_time_mean": mean_s,
            "hold_time_bot95": bot95_s,
            "hold_time_top95": top95_s
        })
    return df.groupby("year_month_day").apply(mean_and_cis_of_holds).reset_index()


def get_reaction_times_by_date_df(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()  # Create a copy to avoid modifying the original DataFrame
    utils.column_checker(df, required_columns={"year_month_day"})
    # group by date and get the mean and 95 of the reaction times
    def mean_and_cis_of_rt_and_tbt(group):
        group = calculate_time_between_trials_and_reaction_time(group)
        if group.reaction_time.isna().all():
            mean_rt, bot95_rt, top95_rt = np.nan, np.nan, np.nan
        else:
            mean_rt = np.nanmean(group.reaction_time)
            bot95_rt, top95_rt = np.nanpercentile(group.reaction_time, [5, 95])
        if group.time_between_trials.isna().all():
            mean_tbt, bot95_tbt, top95_tbt = np.nan, np.nan, np.nan
        else:
            mean_tbt = np.nanmean(group.time_between_trials)
            bot95_tbt, top95_tbt = np.nanpercentile(group.time_between_trials, [5, 95])
        return pd.Series({
            "reaction_time_mean": mean_rt,
            "reaction_time_bot95": bot95_rt,
            "reaction_time_top95": top95_rt,
            "time_between_trials_mean": mean_tbt,
            "time_between_trials_bot95": bot95_tbt,
            "time_between_trials_top95": top95_tbt
        })
    return df.groupby("year_month_day").apply(mean_and_cis_of_rt_and_tbt).reset_index()


# if __name__ == "__main__":
#     from lecilab_behavior_analysis.utils import load_example_data
#     df = load_example_data("mouse1")
#     summary_matrix_df = summary_matrix(df)
#     print(summary_matrix_df)

def get_left_auditory_stim(df: pd.DataFrame, param: str) -> pd.DataFrame:
    pbd_df = df.copy()
    pbd_df[param+'_left'] = pbd_df.apply(
        lambda row: row[param + '_high'] if row['correct_side'] == 'left' else row[param + '_low'],
        axis=1
    )
    return pbd_df

def get_choice_before(df):
    df_copy = df.copy(deep=True)
    utils.column_checker(df_copy, required_columns={"first_choice", "last_choice", "previous_port_before_stimulus", "correct"})

    for mouse in df_copy['subject'].unique():
        for session in df_copy[df_copy.subject == mouse]['session'].unique():
            df_mouse_session = df_copy[np.logical_and(df_copy['subject'] == mouse, df_copy['session'] == session)]
            # df_mouse_session['correct_side_in_previous'] = df_mouse_session['correct_side'].shift(1, fill_value=np.nan)
            df_mouse_session['previous_correct'] = df_mouse_session['correct'].shift(1, fill_value=np.nan)
            df_mouse_session['previous_first_choice'] = df_mouse_session['first_choice'].shift(1, fill_value=np.nan)
            df_mouse_session['previous_last_choice'] = df_mouse_session['last_choice'].shift(1, fill_value=np.nan)
            df_mouse_session['previous_correct_side'] = df_mouse_session['correct_side'].shift(1, fill_value=np.nan)

            df_copy.loc[df_mouse_session.index, 'previous_correct'] = df_mouse_session['previous_correct']
            df_copy.loc[df_mouse_session.index, 'previous_first_choice'] = df_mouse_session['previous_first_choice']
            df_copy.loc[df_mouse_session.index, 'previous_last_choice'] = df_mouse_session['previous_last_choice']
            df_copy.loc[df_mouse_session.index, 'previous_port_before_stimulus'] = df_mouse_session['previous_port_before_stimulus']

            # Add the previous choice is left and correct as 1, or 0, remain NaN
            previous_left_choice_correct_numeric = np.where(
                df_mouse_session['previous_first_choice'].isna() | df_mouse_session['previous_correct'].isna(),
                None,
                ((df_mouse_session['previous_first_choice'] == 'left') & (df_mouse_session['previous_correct'] == True)).astype(int)
            )
            df_copy.loc[df_mouse_session.index, 'previous_left_choice_correct_numeric'] = previous_left_choice_correct_numeric

            # Add the previous choice is right and wrong as 1, or 0, remain NaN
            previous_right_choice_wrong_numeric = np.where(
                df_mouse_session['previous_first_choice'].isna() | df_mouse_session['previous_correct'].isna(),
                None,
                ((df_mouse_session['previous_first_choice'] == 'right') & (df_mouse_session['previous_correct'] == False)).astype(int)
            )
            df_copy.loc[df_mouse_session.index, 'previous_right_choice_wrong_numeric'] = previous_right_choice_wrong_numeric


            # Add the same and correct in choice from previous, where 1 is a same and correct choice and 0 is the rest
            previous_same_choice_correct_numeric = np.where(
                df_mouse_session['previous_first_choice'].isna() | df_mouse_session['previous_correct'].isna(),
                None,
                (((df_mouse_session['previous_first_choice'] == 'left') & (df_mouse_session['first_choice'] == 'left') & (df_mouse_session['previous_correct'] == True)) 
                | ((df_mouse_session['previous_first_choice'] == 'right') & (df_mouse_session['first_choice'] == 'right') & (df_mouse_session['previous_correct'] == True))
                ).astype(int)
            )
            df_copy.loc[df_mouse_session.index, 'previous_same_choice_correct_numeric'] = previous_same_choice_correct_numeric

            # Add the difference and wrong in choice from previous, where 1 is a different and wrong choice and 0 is the rest
            previous_diff_choice_wrong_numeric = np.where(
                df_mouse_session['previous_first_choice'].isna() | df_mouse_session['previous_correct'].isna(),
                None,
                (((df_mouse_session['previous_first_choice'] == 'left') & (df_mouse_session['first_choice'] == 'right') & (df_mouse_session['previous_correct'] == False)) 
                | ((df_mouse_session['previous_first_choice'] == 'right') & (df_mouse_session['first_choice'] == 'left') & (df_mouse_session['previous_correct'] == False))
                ).astype(int)
            )
            df_copy.loc[df_mouse_session.index, 'previous_diff_choice_wrong_numeric'] = previous_diff_choice_wrong_numeric

            # Add the same choice as previous, where 1 is the same choice and 0 is a different choice
            previous_same_choice_numeric = np.where(
                df_mouse_session['previous_first_choice'].isna() | df_mouse_session['first_choice'].isna(),
                None,
                (((df_mouse_session['previous_first_choice'] == 'left') & (df_mouse_session['first_choice'] == 'left')) 
                | ((df_mouse_session['previous_first_choice'] == 'right') & (df_mouse_session['first_choice'] == 'right'))
                ).astype(int)
            )
            df_copy.loc[df_mouse_session.index, 'previous_same_choice_numeric'] = previous_same_choice_numeric

    return df_copy

def parameters_for_fit(df):
    """
    Get the parameters for the fit
    """
    df_copy = df.copy(deep=True)
    df_copy = add_mouse_first_choice(df_copy)
    df_copy = add_mouse_last_choice(df_copy)
    df_copy = add_port_where_animal_comes_from(df_copy)
    if df['stimulus_modality'].unique() == 'visual':
        df_copy = get_performance_by_difficulty_ratio(df_copy)
        df_copy = get_performance_by_difficulty_diff(df_copy)
        df_copy['abs_visual_stimulus_ratio'] = df_copy['visual_stimulus_ratio'].abs()
        df_copy['visual_ratio_diff_interact'] = df_copy['visual_stimulus_ratio'] * (df_copy['visual_stimulus_diff'].abs())
        df_copy['left_bright'] = df_copy.apply(
            lambda row: ast.literal_eval(row['visual_stimulus'])[0] if row['correct_side'] == 'left' else ast.literal_eval(row['visual_stimulus'])[1],
            axis=1
        )
        df_copy['visual_ratio_bright_interact'] = df_copy['abs_visual_stimulus_ratio'] * df_copy['left_bright']
        df_copy['wrong_bright'] = df_copy['visual_stimulus'].apply(lambda x: abs(eval(x)[1]))
    elif df['stimulus_modality'].unique() == 'auditory':
        df_copy = add_auditory_real_statistics(df_copy)
        df_copy = get_left_auditory_stim(df_copy, 'total_percentage_of_tones')
        df_copy = get_left_auditory_stim(df_copy, 'number_of_tones')
        df_copy = get_left_auditory_stim(df_copy, 'percentage_of_timebins_with_evidence')
        df_copy['high_tones'] = df_copy['auditory_stimulus'].apply(lambda row: eval(row)['high_tones'])
        df_copy['high_tones_amplitude_sum'] = df_copy['high_tones'].apply(lambda row: np.sum(pd.DataFrame(row).values))
        df_copy['low_tones'] = df_copy['auditory_stimulus'].apply(lambda row: eval(row)['low_tones'])
        df_copy['low_tones_amplitude_sum'] = df_copy['low_tones'].apply(lambda row: np.sum(pd.DataFrame(row).values))
        df_copy['amplitude_strength'] = ((df_copy['high_tones_amplitude_sum']/df_copy['number_of_tones_high']) / (df_copy['low_tones_amplitude_sum']/df_copy['number_of_tones_low']))


    df_copy['previous_port_before_stimulus_numeric'] = df_copy['previous_port_before_stimulus'].apply(
                lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                )
    df_copy['roa_choice_numeric'] = df_copy['roa_choice'].apply(
                lambda x: 1 if x == 'repeat' else 0 if x == 'alternate' else np.nan
                )
    df_copy['last_choice_numeric'] = df_copy['last_choice'].apply(
                lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                )

    # Add the correct column as numeric, 1 for correct, 0 for incorrect
    df_copy['correct_numeric'] = df_copy['correct'].astype(int)

    # Add the past trials impact by time kernel
    

    df_copy = get_choice_before(df_copy)

    df_copy['previous_first_choice_numeric'] = df_copy['previous_first_choice'].apply(
                lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                )
    df_copy['previous_last_choice_numeric'] = df_copy['previous_last_choice'].apply(
                lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                )
    df_copy['first_choice_numeric'] = df_copy['first_choice'].apply(
                lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                )
    df_copy['previous_correct_numeric'] = df_copy['previous_correct'].astype('Int64')

    return df_copy

def get_time_kernel_impact(df:pd.DataFrame, y: str, max_lag, tau):
    df_copy = df.copy(deep=True)
    if y == 'first_choice_numeric':
        df_copy = add_mouse_first_choice(df_copy)
        df_copy['first_choice_numeric'] = df_copy['first_choice'].apply(
                lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                )
    elif y == 'correct_numeric':
        df_copy['correct_numeric'] = df_copy['correct'].astype(int)
    else:
        raise ValueError("impact should be either 'first_choice_numeric' and 'correct_numeric'")
    df_copy['time_kernel_impact'] = utils.previous_impact_on_time_kernel(df_copy[y], max_lag=max_lag, tau=tau)
    
    return df_copy

