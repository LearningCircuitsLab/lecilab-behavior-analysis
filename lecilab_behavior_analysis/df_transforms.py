import pandas as pd
import numpy as np
import ast
from typing import Tuple, Union
import lecilab_behavior_analysis.utils as utils

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
        np.nan,
        np.where(
            series == comparison_series,
            "repeat",
            "alternate"))

    return pd.Series(repeat_or_alternate, index=series.index)


def add_port_where_animal_comes_from(df_in: pd.DataFrame) -> pd.DataFrame:
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

def get_performance_by_difficulty_test(df_test: pd.DataFrame) -> pd.DataFrame:
    df_test['visual_stimulus_devi'] = df_test['visual_stimulus'].apply(lambda x: abs(round(eval(x)[0] / eval(x)[1])))
    df_test['visual_stimulus_devi'] = df_test.apply(
        lambda row: row['visual_stimulus_devi'] if row['correct_side'] == 'left' else -row['visual_stimulus_devi'],
        axis=1
    )
    leftward_choices = []
    leftward_evidence = []
    for i in df_test.groupby('visual_stimulus_devi'):
        if (i[0] > 0) & (len(i[1][i[1]['correct_side'] == 'left']) != 0):
            leftward_choices.append(len(i[1][(i[1]['correct_side'] == 'left') & (i[1]['correct'] == True)]) / len(i[1][i[1]['correct_side'] == 'left']))
            leftward_evidence.append(i[0])
        elif (i[0] < 0) & (len(i[1][i[1]['correct_side'] == 'right']) != 0):
            leftward_choices.append(-len(i[1][(i[1]['correct_side'] == 'right') & (i[1]['correct'] == True)]) / len(i[1][i[1]['correct_side'] == 'right']))
            leftward_evidence.append(i[0])
        else:
            pass
    return leftward_evidence, leftward_choices

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


def calculate_time_between_trials_and_reaction_time(in_df: pd.DataFrame, window: int = 25) -> pd.DataFrame:
    """
    Calculate Time Between Trials and Reaction Time.
    """
    # Check if the required columns are present
    utils.column_checker(df, required_columns={"Port1In", "Port1Out", "Port2In", "Port2Out", "Port3In", "Port3Out"})
    df = in_df.copy()  # Make a copy to avoid modifying the original DataFrame
    for date in pd.unique(df['date']):
        date_df = df[df['date'] == date]
        port2outs = date_df['Port2Out'].apply(lambda x: np.max(ast.literal_eval(x)) if isinstance(x, str) else np.max(x))
        date_df['Time_Between_Trials'] = port2outs.diff()
        df.loc[df['date'] == date, 'Time_Between_Trials'] = date_df['Time_Between_Trials']
    # Calculate the reaction time
    df['Reaction_Time'] = df.apply(utils.trial_reaction_time, axis=1)

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

    # add the window size to the beginning and end of the vector
    heatmap_vector = np.concatenate((heatmap_vector[len(heatmap_vector)-window_size:], heatmap_vector, heatmap_vector[:window_size]))
    # apply the moving average
    heatmap_vector = np.convolve(heatmap_vector, np.ones(window_size)/window_size, mode='same')
    # cut the vector to the original size
    heatmap_vector = heatmap_vector[window_size:len(heatmap_vector)-window_size]

    return heatmap_vector



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
    df = add_trial_of_day_column_to_df(df)
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

# if __name__ == "__main__":
#     from lecilab_behavior_analysis.utils import load_example_data
#     df = load_example_data("mouse1")
#     summary_matrix_df = summary_matrix(df)
#     print(summary_matrix_df)
