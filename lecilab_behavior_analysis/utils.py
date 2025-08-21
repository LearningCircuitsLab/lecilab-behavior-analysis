import numpy as np
import pandas as pd
import socket
from typing import List, Dict, Tuple, Union
import subprocess
import matplotlib.pyplot as plt
import ast
from scipy.optimize import minimize
import statsmodels.api as sm
import itertools
from collections import defaultdict
from sklearn.metrics import r2_score
import os
import lecilab_behavior_analysis.df_transforms as dft

IDIBAPS_TV_PROJECTS = "/archive/training_village/"


def get_session_performance(df, session: int) -> float:
    """
    Calculate the performance (proportion correct) for a specific session.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data with 'session' and 'correct' columns.
    session : int
        Session number to calculate performance for.
        
    Returns
    -------
    float
        Performance as proportion correct (0.0 to 1.0) for the specified session.
        
    Notes
    -----
    Performance is calculated as the mean of the 'correct' column for trials
    belonging to the specified session.
    """

    return df[df.session == session].correct.mean()


def get_day_performance(df, day: str) -> float:
    """
    Calculate the performance (proportion correct) for a specific day.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data with 'year_month_day' and 'correct' columns.
    day : str
        Day identifier to calculate performance for.
        
    Returns
    -------
    float
        Performance as proportion correct (0.0 to 1.0) for the specified day.
        
    Notes
    -----
    Performance is calculated as the mean of the 'correct' column for trials
    occurring on the specified day.
    """

    return df[df.year_month_day == day].correct.mean()


def get_session_number_of_trials(df, session: int) -> int:
    """
    Count the number of trials in a specific session.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data with 'session' column.
    session : int
        Session number to count trials for.
        
    Returns
    -------
    int
        Number of trials (rows) for the specified session.
        
    Notes
    -----
    Simply counts the number of rows in the DataFrame that belong
    to the specified session.
    """

    return df[df.session == session].shape[0]


def get_day_number_of_trials(df, day: str) -> int:
    """
    Count the number of trials for a specific day.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data with 'year_month_day' column.
    day : str
        Day identifier to count trials for.
        
    Returns
    -------
    int
        Number of trials (rows) for the specified day.
        
    Notes
    -----
    Simply counts the number of rows in the DataFrame that correspond
    to trials performed on the specified day.
    """
    return df[df.year_month_day == day].shape[0]


def get_start_and_end_times(df):
    """
    Get the start and end times of a behavioral session.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing trial data with columns 'session', 'trial', 
        'TRIAL_START', and 'TRIAL_END'.
        
    Returns
    -------
    tuple
        A tuple containing (start_time, end_time) as pandas datetime objects.
        
    Raises
    ------
    ValueError
        If the dataframe contains more than one session.
        
    Notes
    -----
    The function finds the earliest TRIAL_START and latest TRIAL_END times
    in the session and converts them from Unix timestamps to datetime objects.
    """
    # ensure that there is only one session in the dataframe
    if df['session'].nunique() != 1:
        raise ValueError("The dataframe contains more than one session.")
    # get the start time of the session as the start of the first trial
    start_time = df.loc[df['trial'] == np.min(df.trial), 'TRIAL_START'].values[0]
    # get the end time of the session as the end of the last trial
    end_time = df.loc[df['trial'] == np.max(df.trial), 'TRIAL_END'].values[0]
    # convert the start and end times to datetime
    start_time = pd.to_datetime(start_time, unit='s')
    end_time = pd.to_datetime(end_time, unit='s')
    return start_time, end_time


def get_block_size_truncexp_mean30() -> int:
    """
    Generate a block size following a truncated exponential distribution.
    
    Returns
    -------
    int
        Block size sampled from truncated exponential distribution with mean=30.
        
    Notes
    -----
    Uses a truncated exponential distribution with bounds [20, 50] and 
    optimal lambda=0.0607 to achieve a mean of approximately 30.
    The distribution parameters were optimized using ChatGPT.
    """
    lower_bound = 20
    upper_bound = 50
    optimal_lambda = 0.0607
    u = np.random.uniform(0, 1)  # Uniform sample
    block_size = (
        lower_bound
        - np.log(1 - u * (1 - np.exp(-optimal_lambda * (upper_bound - lower_bound))))
        / optimal_lambda
    )

    return int(block_size)


def get_right_bias(side_and_correct_dict: dict) -> float:
    """
    Calculate the right-side bias in behavioral choice data.
    
    Parameters
    ----------
    side_and_correct_dict : dict
        Dictionary with 'side' and 'correct' keys containing:
        - 'side': array-like of choice sides ('left' or 'right')
        - 'correct': array-like of boolean correctness values
        
    Returns
    -------
    float
        Bias towards the right side, ranging from -1 to 1.
        Positive values indicate right bias, negative values indicate left bias.
        
    Raises
    ------
    ValueError
        If dictionary doesn't have exactly 'side' and 'correct' keys.
        
    Notes
    -----
    The bias is calculated based on incorrect choices only. It measures
    the proportion of wrong choices made to each side, with the bias
    being left_wrong_proportion - right_wrong_proportion.
    """
    if len(side_and_correct_dict) != 2:
        raise ValueError("Input dict must have exactly two keys")
    if "side" not in side_and_correct_dict:
        raise ValueError("Input dict must have 'side' key")
    if "correct" not in side_and_correct_dict:
        raise ValueError("Input dict must have 'correct' key")
    first_pokes = side_and_correct_dict["side"]
    correct_choices = side_and_correct_dict["correct"]
    wrong_sides = first_pokes[~correct_choices]
    if len(wrong_sides) == 0 or "ignore" in first_pokes:
        return 0
    wrong_side_proportion = len(wrong_sides) / len(first_pokes)
    wrong_right_proportion = (
        wrong_side_proportion * np.nansum(wrong_sides == "right") / len(wrong_sides)
    )
    wrong_left_proportion = (
        wrong_side_proportion * np.nansum(wrong_sides == "left") / len(wrong_sides)
    )

    return wrong_left_proportion - wrong_right_proportion


def get_block_size_uniform_pm30(mean: int) -> int:
    """
    Generate a block size from a uniform distribution around a mean value.
    
    Parameters
    ----------
    mean : int
        The mean value around which to generate the block size.
        
    Returns
    -------
    int
        Block size sampled from uniform distribution [mean-30%, mean+30%].
        
    Notes
    -----
    The block size is drawn from a uniform distribution with bounds
    calculated as mean ± 30% of the mean value.
    """
    lower_bound = int(mean - 0.3 * mean)
    upper_bound = int(mean + 0.3 * mean)
    block_size = np.random.randint(lower_bound, upper_bound)

    return block_size


def column_checker(df: pd.DataFrame, required_columns: set):
    """
    Validate that required columns are present in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : set
        Set of column names that must be present in the DataFrame.
        
    Raises
    ------
    ValueError
        If any required columns are missing from the DataFrame.
        
    Notes
    -----
    This is a utility function commonly used to validate input DataFrames
    before processing to ensure all necessary columns are available.
    """
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The dataframe must have the following columns: "
            + ", ".join(required_columns)
        )


def get_text_from_subset_df(df: pd.DataFrame, reduced: bool = False) -> str:
    """
    Generate a text summary of behavioral session data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data for a single subject.
        Must contain columns: 'subject', 'session', 'date', 'current_training_stage', 'correct'.
    reduced : bool, optional
        If True, generates a reduced summary with fewer details (default: False).
        
    Returns
    -------
    str
        Formatted text summary containing session information, performance metrics,
        and behavioral statistics.
        
    Raises
    ------
    ValueError
        If the dataframe contains more than one subject.
        
    Notes
    -----
    The summary includes session details, trial counts, performance percentages,
    and additional behavioral metrics depending on the `reduced` parameter.
    """
    # make sure that there is only one subject in the dataframe
    if df.subject.nunique() != 1:
        raise ValueError("The dataframe contains more than one subject.")
    # get the session
    session = df.session.unique()
    # get the date
    date = df.date.unique()
    # get the current training stage
    current_training_stage = df.current_training_stage.unique()
    # get the number of trials
    n_trials = df.shape[0]
    # get the number of correct trials
    n_correct = int(df.correct.sum())
    # get the performance
    performance = n_correct / n_trials * 100
    # get the water consumed
    water = df.water.sum()
    # get the subject
    mouse_name = df.subject.unique()[0]

    if reduced:
        if len(session) > 3:
            session = f"{session[0]}-...-{session[-1]}"
            date = f"{date[0]}-...-{date[-1]}"

    # write the text
    text = f"""\
    Mouse: {mouse_name}
    Sessions: {session}
    Dates: {date}
    Training stages: {current_training_stage}
    Number of trials: {n_trials}
    Number of correct trials: {n_correct}
    Performance: {performance:.2f}%
    Water consumed: {water} μl
    """

    return text


def get_text_from_subject_df(df: pd.DataFrame) -> str:
    """
    Generate a text summary of overall subject performance across sessions.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing behavioral data for a single subject.
        Must contain columns: 'subject', 'session', 'water', 'year_month_day'.
        
    Returns
    -------
    str
        Formatted text summary showing subject name and daily averages:
        - Sessions per day
        - Trials per day  
        - Water consumption per day (μl)
        
    Raises
    ------
    ValueError
        If the dataframe contains more than one subject.
        
    Notes
    -----
    Provides high-level statistics averaged across all training days
    for a single subject, useful for progress monitoring and comparison.
    """
    # make sure that there is only one subject in the dataframe
    if df.subject.nunique() != 1:
        raise ValueError("The dataframe contains more than one subject.")
    column_checker(df, {"subject", "session", "water", "year_month_day"})
    # get the subject
    mouse_name = df.subject.unique()[0]
    # get the average water consumed per day
    days_of_training = df.year_month_day.nunique()
    avg_water_per_day = df.water.sum() / days_of_training if days_of_training > 0 else 0
    # get the number of trials per day on average
    avg_trials_per_day = df.shape[0] / days_of_training if days_of_training > 0 else 0
    # get the number of sessions per day on average
    n_sessions = df.session.nunique()
    avg_sessions_per_day = n_sessions / days_of_training if days_of_training > 0 else 0

    # write the text
    text = f"""\
        {mouse_name}
        {avg_sessions_per_day:.2f} sessions per day
        {avg_trials_per_day:.0f} trials per day
        {avg_water_per_day:.0f} μl of water per day
    """

    return text


def load_example_data(mouse_name) -> pd.DataFrame:
    """
    Load example behavioral data for a specified mouse.
    
    Parameters
    ----------
    mouse_name : str
        Name of the mouse to load example data for.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing example behavioral data for the specified mouse.
        
    Notes
    -----
    Loads data from the example_data directory structure, expecting files
    in the format: {outpath}/example_data/{mouse_name}/{mouse_name}_fakedata.csv
    """
    outpath = get_outpath()
    df = pd.read_csv(outpath + "/example_data/" + mouse_name + "/" + mouse_name + "_fakedata.csv", sep=";")

    return df


def get_sound_stats(sound_dict: dict) -> dict:
    """
    This method returns a dictionary with the actual sound statistics
    of two sound matrices like a cloud of tones

    Args:
        sound_dict (dict): Dictionary with the entries for the two sound matrices

    Returns:
        sound_stats (dict): Dictionary with the sound statistics
    """
    # if the entries are dictionaries, convert them to dataframes
    if isinstance(sound_dict["high_tones"], dict):
        high_mat = pd.DataFrame(sound_dict["high_tones"])
    else:
        high_mat = sound_dict["high_tones"]
    if isinstance(sound_dict["low_tones"], dict):
        low_mat = pd.DataFrame(sound_dict["low_tones"])
    else:
        low_mat = sound_dict["low_tones"]
    # make the matrices binary
    high_mat = (high_mat > 0).astype(int)
    low_mat = (low_mat > 0).astype(int)
    # get the sound statistics
    high_mat_stats = analyze_sound_matrix(high_mat)
    low_mat_stats = analyze_sound_matrix(low_mat)
    # calculate the evidence strength for the high sound
    total_high_evidence_strength = sound_evidence_strength(
        high_mat_stats["number_of_tones"],
        low_mat_stats["number_of_tones"])

    sound_stats = {
        "high_tones": high_mat_stats,
        "low_tones": low_mat_stats,
        "total_evidence_strength": total_high_evidence_strength
    }

    return sound_stats


def analyze_sound_matrix(matrix: pd.DataFrame) -> dict:
    """
    This method analyzes a sound matrix and returns a dictionary with the sound statistics

    Args:
        matrix (pd.DataFrame): Sound matrix to analyze

    Returns:
        sound_stats (dict): Dictionary with the matrix statistics
    """
    # calculate the number of 1s
    evidences = np.sum(matrix.values)
    # calculate the percentage of 1s
    perc = evidences / matrix.size
    # calculate the real probability
    # of having at least one 1 in each column
    colsums = matrix.sum(axis=0)
    real_prob = np.sum(colsums > 0) / len(colsums)

    return {"number_of_tones": evidences,
            "total_percentage_of_tones": perc,
            "percentage_of_timebins_with_evidence": real_prob}


def sound_evidence_strength(x, y):
    return (x - y) / (x + y)


def get_outpath():
    """
    Get the appropriate data output path based on the current hostname.
    
    Returns
    -------
    str
        File system path to the behavioral data directory for the current machine.
        Returns "default/path" if hostname is not recognized.
        
    Notes
    -----
    Maps known hostnames to their corresponding data directory paths.
    Used to handle different data storage locations across different
    computers and environments.
    """
    hostname = socket.gethostname()
    paths = {
        "lorena-ThinkPad-E550": "/home/emma/Desktop/EloiJacomet/data",
        "tectum": "/mnt/c/Users/HMARTINEZ/LeCiLab/data/behavioral_data",
        "localhost": "/home/kudongdong/data/LeciLab/behavioral_data",
        "setup2": "/home/kudongdong/Documents/data/LeciLab/behavioral_data"
    }
    return paths.get(hostname, "default/path")


def get_idibaps_cluster_credentials():
    hostname = socket.gethostname()
    if hostname == "tectum":
        return {
            "username": "hvergara",
            "host": "mini",
            # "port": 443,
        }
    elif hostname == "localhost":
        return {
            "username": "kudongdong",
            "host": "mini",
        }
    elif hostname == "setup2":
        return {
            "username": "kudongdong",
            "host": "mini",
        }
    else:
        raise ValueError("Unknown host")


def get_server_projects() -> List[str]:
    credentials = get_idibaps_cluster_credentials()    
    return get_folders_from_server(credentials, IDIBAPS_TV_PROJECTS)


def get_folders_from_server(credentials: dict, path: str) -> List[str]:
    # Create a single SSH connection to the remote server
    ssh_command = (
        f"ssh {credentials['username']}@{credentials['host']} "
        f"'ls {path}'"
    )
    result = subprocess.run(
        ssh_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Decode the output and split it into a list of folder names
    if result.returncode == 0:
        folders = result.stdout.decode("utf-8").strip().split("\n")
    else:
        print(f"Error: {result.stderr.decode('utf-8')}")
        folders = []
    # Filter out any unwanted folders (e.g., .git)
    folders = [folder for folder in folders if not folder.startswith(".")]

    return folders


def get_animals_in_project(project_name: str) -> List[str]:
    credentials = get_idibaps_cluster_credentials()
    folders = get_folders_from_server(
        credentials,
        f"{IDIBAPS_TV_PROJECTS}{project_name}/sessions/",
    )

    # filter out any unwanted folders (e.g., .csv)
    folders = [folder for folder in folders if not folder.endswith(".csv")]

    return folders


def rsync_cluster_data(
    project_name: str,
    file_path: str,
    credentials: dict,
    local_path: str,
) -> bool:
    """
    This method syncs the session data from the server to the local machine.
    """
    remote_path = f"{credentials['username']}@{credentials['host']}:{IDIBAPS_TV_PROJECTS}{project_name}/{file_path}"
    rsync_command = f"rsync -avz {remote_path} {local_path}"
    result = subprocess.run(rsync_command, shell=True)
    if result.returncode != 0:
        print(f"Error syncing data for {file_path}: {result.stderr.decode('utf-8')}")
    return result.returncode == 0


# def rsync_sessions_summary(
#     project_name: str,
#     credentials: dict,
#     local_path: str,
# ) -> bool:
#     """
#     This method syncs the session data from the server to the local machine.
#     """
#     remote_path = f"{credentials['username']}@{credentials['host']}:{IDIBAPS_TV_PROJECTS}{project_name}/sessions_summary.csv"
#     rsync_command = f"rsync -avz {remote_path} {local_path}"
#     result = subprocess.run(rsync_command, shell=True)
#     if result.returncode != 0:
#         print(f"Error syncing session summary data for {project_name}: {result.stderr.decode('utf-8')}")
#     return result.returncode == 0


def list_to_colors(ids: np.array, cmap: str) -> Tuple[List[tuple], Dict]:
    unique_cts = pd.unique(ids)
    colormap = plt.get_cmap(cmap)
    colors = [colormap(i) for i in range(len(unique_cts))]
    # Create a color dictionary
    color_dict = {unique_cts[i]: colors[i] for i in range(len(unique_cts))}
    # Set the colors for each current_training_stage
    color_list = [color_dict[x] for x in list(ids)]
    return color_list, color_dict


def is_this_a_miss_trial(series: pd.Series) -> bool:
    """
    checks if STATE_stimulus_state_END is the last state
    """
    try:
        end_of_stimulus_state = np.max(ast.literal_eval(series["STATE_stimulus_state_END"]))
        end_of_trial = series["TRIAL_END"]
        if np.round(end_of_stimulus_state, 6) == np.round(end_of_trial, 6):
            return True
        else:
            return False
    except ValueError:
        # if the value is not a list, return False
        return False


def trial_reaction_time(series: pd.Series) -> float:
    """
    This method calculates the reaction time of a trial.
    """
    try:
        out_time = np.max(ast.literal_eval(series["Port2Out"]))
    except Exception:
        return np.nan
    try:
        port1_ins = ast.literal_eval(series["Port1In"])
        # get the first Port1In after Port2Out
        port1_ins = np.array(port1_ins)
        port1_in = np.min(port1_ins[port1_ins > out_time])
    except Exception:
        port1_in = np.nan
    try:
        port3_ins = ast.literal_eval(series["Port3In"])
        # get the first Port3In after Port2Out
        port3_ins = np.array(port3_ins)
        port3_in = np.min(port3_ins[port3_ins > out_time])
    except Exception:
        port3_in = np.nan
    if np.isnan(port1_in) and np.isnan(port3_in):
        return np.nan
    return np.nanmin([port1_in, port3_in]) - out_time


def first_poke_after_stimulus_state(series: pd.Series) -> Union[str, None]:
    # convert the series to a dictionary
    ser_dict = series.to_dict()
    try:
        stim_state_array = ast.literal_eval(ser_dict["STATE_stimulus_state_START"])
    except ValueError:
        # if the value is not a list, return None
        return None
    if len(stim_state_array) == 0:
        return None
    start_time = min(stim_state_array)
    port1_in = get_dictionary_event_as_list(ser_dict, "Port1In")
    port3_in = get_dictionary_event_as_list(ser_dict, "Port3In")   
    
    port1_in_after = [i for i in port1_in if i > start_time]
    port3_in_after = [i for i in port3_in if i > start_time]
    
    if len(port1_in_after) == 0 and len(port3_in_after) == 0:
        return None
    elif len(port1_in_after) == 0:
        return "right"
    elif len(port3_in_after) == 0:
        return "left"
    
    if np.min(port1_in_after) < np.min(port3_in_after):
        return "left"
    elif np.min(port3_in_after) < np.min(port1_in_after):
        return "right"
    else:
        return None
    

def get_last_poke_of_trial(series: pd.Series) -> Union[str, None]:
    # convert the series to a dictionary
    ser_dict = series.to_dict()
    port1_in = get_dictionary_event_as_list(ser_dict, "Port1In")
    port3_in = get_dictionary_event_as_list(ser_dict, "Port3In")   
    
    if len(port1_in) == 0 and len(port3_in) == 0:
        return None
    elif len(port1_in) == 0:
        return "right"
    elif len(port3_in) == 0:
        return "left"
    
    if np.max(port1_in) > np.max(port3_in):
        return "left"
    elif np.max(port3_in) > np.max(port1_in):
        return "right"
    else:
        return None


def get_last_poke_before_stimulus_state(series: pd.Series) -> Union[str, None]:
    """
    Sometimes the animal can poke in the center port and not wait long enough
    to get the stimulus state. If trial is not aborted due to selected settings,
    the animal can keep poking.
    """
    # convert the series to a dictionary
    ser_dict = series.to_dict()
    try:
        stim_state_array = ast.literal_eval(ser_dict["STATE_stimulus_state_START"])
    except ValueError:
        # if the value is not a list, return None
        return None
    if len(stim_state_array) == 0:
        return None
    start_time = min(stim_state_array)
    port1_in = get_dictionary_event_as_list(ser_dict, "Port1In")
    port3_in = get_dictionary_event_as_list(ser_dict, "Port3In")   
    
    port1_in_before = [i for i in port1_in if i < start_time]
    port3_in_before = [i for i in port3_in if i < start_time]
    
    if len(port1_in_before) == 0 and len(port3_in_before) == 0:
        return None
    elif len(port1_in_before) == 0:
        return "right"
    elif len(port3_in_before) == 0:
        return "left"
    
    if np.max(port1_in_before) > np.max(port3_in_before):
        return "left"
    elif np.max(port3_in_before) > np.max(port1_in_before):
        return "right"
    else:
        return None


def get_dictionary_event_as_list(ser_dict: Dict, event: str) -> List:
    # check if the keys are in the dict
    if event in ser_dict.keys():
        try:
            event_list = ast.literal_eval(ser_dict[event])
            if type(event_list) is float:
                event_list = [event_list]
        except ValueError:
            # if the value is not a list, return an empty list
            event_list = []
    else:
        event_list = []
    return event_list


def get_repeat_or_alternate_to_numeric(row: pd.Series) -> float:
    if row['roa_choice'] == 'alternate':
        return 0
    elif row['roa_choice'] == 'repeat':
        if row['first_choice'] == 'left':
            return -1
        elif row['first_choice'] == 'right':
            return 1
    else:
        return np.nan

# Define the lapse logistic function with independent lapses for left and right
def lapse_logistic_independent(params, x, y):
    lapse_left, lapse_right, beta, x0 = params
    # Ensure lapse rates are within [0, 0.5]
    lapse_left = np.clip(lapse_left, 0, 0.5)
    lapse_right = np.clip(lapse_right, 0, 0.5)
    # Predicted probabilities
    p_left = lapse_left + (1 - lapse_left - lapse_right) / (1 + np.exp(-beta * (x - x0)))
    # Negative log-likelihood
    nll = -np.sum(y * np.log(p_left) + (1 - y) * np.log(1 - p_left))
    return nll

def fit_lapse_logistic_independent(x, y, initial_params=None):
    """
    Fit a lapse logistic model with independent lapses for left and right choices.
    
    Args:
        x (np.array): Independent variable (e.g., stimulus intensity).
        y (np.array): Dependent variable (e.g., choice probabilities).
        initial_params (list, optional): Initial guess for the parameters [lapse_left, lapse_right, beta, x0].
    
    Returns:
        tuple: Fitted parameters and the minimized negative log-likelihood.
    """
    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    if initial_params is None:
        initial_params = [0.05, 0.05, 1, 0]  # Default initial values
    
    result = minimize(
        lapse_logistic_independent,
        initial_params,
        args=(x, y),
        bounds=[(0, 0.5), (0, 0.5), (None, None), (None, None)]
    )

    # Extract fitted parameters
    lapse_left, lapse_right, beta, x0 = result.x
    # print(f"Lapse Left: {lapse_left}, Lapse Right: {lapse_right}, Slope (Beta): {beta}, PSE (x0): {x0}")

    # Generate predictions
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    p_left = lapse_left + (1 - lapse_left - lapse_right) / (1 + np.exp(-beta * (xs - x0)))
    return p_left, (lapse_left, lapse_right, beta, x0)

def get_trial_port_hold(row, port_number):
    if not type(row) == pd.Series:
        raise ValueError("row must be a pandas Series")
    ins = row["Port" + str(port_number) + "In"]
    outs = row["Port" + str(port_number) + "Out"]
    if ins == "[]" or outs == "[]":
        return np.nan
    if type(ins) == str:
        ins = ast.literal_eval(ins)
        outs = ast.literal_eval(outs)
    if type(ins) == float:
        ins = [ins]
    if type(outs) == float:
        outs = [outs]
    # if the first out is earlier than the first in, remove it
    if outs[0] < ins[0]:
        outs = outs[1:]
    # if they are different lengths, we need to pad the shorter one with NaNs
    if len(ins) > len(outs):
        outs = np.concatenate((outs, [np.nan] * (len(ins) - len(outs))))
    elif len(outs) > len(ins):
        ins = np.concatenate((ins, [np.nan] * (len(outs) - len(ins))))
    
    # now we can calculate the hold time
    return np.array(outs) - np.array(ins)


def logi_model_fit_input(df: pd.DataFrame, X, y, method='newton'):
    column_checker(df, {x for x in X})
    column_checker(df, {y})

    # drop NaN values if any
    df_for_fit = df.dropna(subset=X + [y])
    df_for_fit = df_for_fit[X + [y]].astype(float)

    # to make coefficients comparable
    df_for_fit[X] = df_for_fit[X].apply(lambda x: (x - x.mean()) / x.std())

    # Prepare the independent variables
    X_multi = df_for_fit[X]
    X_multi_const = sm.add_constant(X_multi)
    y_predict = df_for_fit[y]
    return X_multi_const, y_predict


def logi_model_fit(df: pd.DataFrame, X, y, method='newton'):
    X_multi_const, y_predict = logi_model_fit_input(df, X, y, method=method)

    # Fit the logistic regression model with multiple regressors
    logit_model_multi = sm.Logit(y_predict, X_multi_const).fit(method=method)

    # Display the summary, which includes p-values for all regressors
    results = logit_model_multi.summary(xname= ["intercept"] + X)
    
    return results, logit_model_multi

def hierarchical_partitioning(df, x_cols, y_col, method='newton'):
    contributions = defaultdict(list)
    n_vars = len(x_cols)

    for k in range(1, n_vars + 1):
        for subset in itertools.combinations(x_cols, k): # generate all the combinations of the variables
            subset = list(subset)
            # fit model input and generate model 
            X, y = logi_model_fit_input(df, subset, y_col)
            model = sm.Logit(y, X).fit(method=method)
            
            r2 = r2_score(y, model.predict(X))

            # contribution for each variable in the combinations
            for var in subset:
                reduced_subset = subset.copy()
                reduced_subset.remove(var)
                X_reduced = X[reduced_subset] if reduced_subset else X['const'] # if no feature after remove only calcalate constant
                model_reduced = sm.Logit(y, X_reduced).fit(method=method)
                r2_reduced = r2_score(y, model_reduced.predict(X_reduced))
                delta = r2 - r2_reduced
                contributions[var].append(delta)
    
    # normalize the contributions
    avg_contrib = {var: np.mean(contrib) for var, contrib in contributions.items()}
    total = sum(avg_contrib.values())
    norm_contrib = {var: val / total for var, val in avg_contrib.items()}
    return pd.Series(norm_contrib)

def previous_impact_on_time_kernel(series, max_lag=10, tau=5):
    kernel = np.exp(-np.arange(max_lag+1, 1, -1) / tau)
    padded = np.concatenate([[0]*max_lag, series])
    return np.array([
        np.dot(kernel, padded[i - max_lag:i])
        for i in range(max_lag, len(padded))
    ])

def verify_params_time_kernel(dic:dict, y:str):
    combinations = list(itertools.product(range(1,20), range(1, 20)))
    comb_dict = {}
    # iterate all the combinations of max_lag and tau of time kernel
    for comb in combinations:
        max_lag = comb[0]
        tau = comb[1]
        previous_impact_on_kernel_mice = []
        for df_name, df in zip(dic.keys(), dic.values()):
            if y == 'first_choice_numeric':
                # df = dft.add_mouse_first_choice(df)
                df['first_choice_numeric'] = df['first_choice'].apply(
                        lambda x: 1 if x == 'left' else 0 if x == 'right' else np.nan
                        )
            elif y == 'correct_numeric':
                df['correct_numeric'] = df['correct'].astype(int)
            else:
                raise ValueError("impact should be either 'first_choice_numeric' or 'correct_numeric'")
            
            for session in df['session'].unique():
                df_session = df[df['session'] == session]
                df_session['previous_impact_on_kernel'] = previous_impact_on_time_kernel(df_session[y], max_lag=max_lag, tau=tau)
                df.loc[df_session.index, 'previous_impact_on_kernel'] = df_session['previous_impact_on_kernel']
            _, model = logi_model_fit(df, X=['previous_impact_on_kernel'], y=y)
            previous_impact_on_kernel_mice.append(abs(model.params['previous_impact_on_kernel']))
        comb_dict[comb] = np.mean(previous_impact_on_kernel_mice)
    return comb_dict


def generate_tv_report(events: pd.DataFrame, sessions_summary: pd.DataFrame, hours: int = 24):
    events["date"] = pd.to_datetime(events["date"])
    sessions_summary["date"] = pd.to_datetime(sessions_summary["date"])

    # get the highest date in the events and sessions_summary dataframes
    max_date = max(events["date"].max(), sessions_summary["date"].max())
    time_hours_ago = pd.Timestamp.now() - pd.Timedelta(hours=hours)

    detections = events[
        (events["description"] == "Subject detected")
        & (events["date"] >= time_hours_ago)
    ]
    sessions = events[
        (events["type"] == "START") & (events["date"] >= time_hours_ago)
    ]
    sessions_summary = sessions_summary[sessions_summary["date"] >= time_hours_ago]

    subject_detections = detections.groupby("subject").size().to_dict()
    subject_sessions = sessions.groupby("subject").size().to_dict()
    subject_water = sessions_summary.groupby("subject")["water"].sum().to_dict()
    subject_weight = sessions_summary.groupby("subject")["weight"].mean().to_dict()

    # select only subjects that have detections
    subjects = set(subject_detections.keys())
    # sort it
    subjects = sorted(subjects)
    subject_detections = {subj: subject_detections.get(subj, 0) for subj in subjects}
    subject_sessions = {subj: subject_sessions.get(subj, 0) for subj in subjects}
    subject_water = {subj: subject_water.get(subj, 0) for subj in subjects}
    subject_weight = {subj: subject_weight.get(subj, np.nan) for subj in subjects}


    # generate a dataframe
    report_df = pd.DataFrame({
        "Subject": subject_detections.keys(),
        "Detections": subject_detections.values(),
        "Sessions": subject_sessions.values(),
        "Water Consumed (μl)": [subject_water.get(subj, 0) for subj in subject_detections.keys()],
        "Average Weight (g)": [subject_weight.get(subj, np.nan) for subj in subject_detections.keys()]
    })

    return report_df, max_date


def load_all_events(project_name: str) -> pd.DataFrame:
    """
    Load all events from the local machine for a given project.
    """
    outpath = get_outpath()
    events_path = f"{outpath}/{project_name}/events.csv"
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Events file for project {project_name} does not exist.")
    
    events_df = pd.read_csv(events_path, sep=";")

    # read events in old_events if they exist
    events_dfs_list = []
    old_events_path = f"{outpath}/{project_name}/old_events"
    # list files in the old_events folder
    if os.path.exists(old_events_path):
        old_events_files = [f for f in os.listdir(old_events_path) if f.endswith('.csv')]
        # sort files by name to ensure they are in the correct order
        old_events_files.sort()
        # read each file and concatenate to events_df
        for old_file in old_events_files:
            old_file_path = os.path.join(old_events_path, old_file)
            old_events_df = pd.read_csv(old_file_path, sep=";")
            events_dfs_list.append(old_events_df)
    events_df = pd.concat(events_dfs_list + [events_df], ignore_index=True)
    return events_df
    


if __name__ == "__main__":
    # Example usage
    print(get_server_projects())
    print(get_animals_in_project("visual_and_COT_data"))

def filter_variables_for_model(dic_fit:dict, X:list, y:str, max_lag=None, tau=None):
    corr_mat_list = []
    norm_contribution_df = pd.DataFrame([])
    for df_name, df_for_fit in zip(dic_fit.keys(), dic_fit.values()):
        if (max_lag is not None) & (tau is not None):
            df_for_fit = dft.get_time_kernel_impact(df_for_fit, y=y, max_lag=max_lag, tau=tau)
        
        corr_fit_X_df = df_for_fit[X].corr()
        corr_mat_list.append(corr_fit_X_df)

        norm_contribution = hierarchical_partitioning(df_for_fit, x_cols = X, y_col = y, method='bfgs')
        norm_contribution_df[df_name] = norm_contribution

    return corr_mat_list, norm_contribution_df


def find_next_end_task_time_in_events(events_df: pd.DataFrame, date: str, subject: str) -> Tuple[Union[str, None], Union[float, None]]:
    """
    Find the end task time in the events dataframe for a given date.
    """
    # filter events
    filtered_events = events_df[events_df['description'] == "The subject has returned home."]
    
    # get the first event after the given date
    end_event = filtered_events[filtered_events['date'] > date]
    # if there are no events after the given date, return None
    if end_event.empty:
        print(f"No end event found after date {date}.")
        return "No date", None
    # calculate the duration from the given date to the end event
    duration = (pd.to_datetime(end_event['date'].iloc[0]) - pd.to_datetime(date)).total_seconds()
    # get the subject as well
    subject_of_event = end_event['subject'].iloc[0]

    if subject_of_event != subject:
        print(f"Subject mismatch for {date}: expected {subject}, found {subject_of_event}.")
        return None, None

    # return the date of the first event
    return end_event['date'].iloc[0], duration


def get_session_box_usage(session_df: pd.DataFrame, session_duration_df: pd.DataFrame) -> pd.DataFrame:

    if session_df.date.unique().size != 1:
        raise ValueError("Session dataframe must contain data for a single date.")

    #TODO: do the column checker

    date = session_df.date.unique()[0]
    subject = session_df.subject.unique()[0]
    session_duration = session_duration_df[session_duration_df['date'] == date].duration.values[0]
    time_to_complete_first_trial = session_df.iloc[0].trial_duration
    start_of_first_trial = session_df.iloc[0].TRIAL_START
    last_trial_completed_time = session_df.iloc[-1].TRIAL_END - start_of_first_trial
    time_to_exit_box = session_duration - last_trial_completed_time
    # add accuracy as well
    accuracy = session_df['correct'].mean() * 100

    # add the time of engagement and disengagement, removing the first trial
    session_df = session_df.iloc[1:]  # remove the first trial for engagement calculation
    engaged_time = session_df[session_df['engaged'] == True]['trial_duration'].sum()
    disengaged_time = session_df[session_df['engaged'] == False]['trial_duration'].sum()

    unaccounted_time = session_duration - (time_to_complete_first_trial + time_to_exit_box +
                                           engaged_time + disengaged_time)
    
    total_session_time = time_to_complete_first_trial + time_to_exit_box + engaged_time + disengaged_time + unaccounted_time

    return pd.DataFrame({
        "date": [date] * 5,
        "subject": [subject] * 5,
        "time_type": [
            "time_to_complete_first_trial",
            "time_to_exit_box",
            "engaged_time",
            "disengaged_time",
            "unaccounted_time"
        ],
        "absolute_time": [
            time_to_complete_first_trial,
            time_to_exit_box,
            engaged_time,
            disengaged_time,
            unaccounted_time
        ],
        "percentage_of_time": [
            time_to_complete_first_trial / total_session_time * 100,
            time_to_exit_box / total_session_time * 100,
            engaged_time / total_session_time * 100,
            disengaged_time / total_session_time * 100,
            unaccounted_time / total_session_time * 100
        ],
        "accuracy": [accuracy] * 5,
    })