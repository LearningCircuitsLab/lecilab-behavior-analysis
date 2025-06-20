import numpy as np
import pandas as pd
import socket
from typing import List, Dict, Tuple, Union
import subprocess
import matplotlib.pyplot as plt
import ast
from scipy.optimize import minimize

IDIBAPS_TV_PROJECTS = "/archive/training_village/"


def get_session_performance(df, session: int) -> float:
    """
    This method calculates the performance of a session.
    """

    return df[df.session == session].correct.mean()


def get_day_performance(df, day: str) -> float:
    """
    This method calculates the performance of a day.
    """

    return df[df.year_month_day == day].correct.mean()


def get_session_number_of_trials(df, session: int) -> int:
    """
    This method calculates the number of trials in a session.
    """

    return df[df.session == session].shape[0]


def get_day_number_of_trials(df, day: str) -> int:
    """
    This method calculates the number of trials in a day.
    """
    return df[df.year_month_day == day].shape[0]


def get_start_and_end_times(df):
    """
    Get the start and end times of the session.
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
    This method returns a block size following a truncated exponential distribution.
    Optimized to get the mean of the distribution to be 30 using ChatGPT.
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
    This method calculates the right bias in the data.

    Args:
        dictionary with 'side' and 'correct' keys: the side ('left' or 'right') and correct answer (boolean)

    Returns:
        float: bias to the right side, from -1 to 1
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
    This method returns a block size following a uniform distribution,
    calculated as the mean of the distribution +/- 30% of the mean.

    Args:
        mean (int): Mean of the distribution
    """
    lower_bound = int(mean - 0.3 * mean)
    upper_bound = int(mean + 0.3 * mean)
    block_size = np.random.randint(lower_bound, upper_bound)

    return block_size


def column_checker(df: pd.DataFrame, required_columns: set):
    """
    This method checks if the required columns are present in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to check
        required_columns (set): Set of required columns
    """
    if not required_columns.issubset(df.columns):
        raise ValueError(
            "The dataframe must have the following columns: "
            + ", ".join(required_columns)
        )


def get_text_from_subset_df(df: pd.DataFrame) -> str:
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


def rsync_session_data(
    project_name: str,
    animal: str,
    credentials: dict,
    local_path: str,
) -> bool:
    """
    This method syncs the session data from the server to the local machine.
    """
    remote_path = f"{credentials['username']}@{credentials['host']}:{IDIBAPS_TV_PROJECTS}{project_name}/sessions/{animal}/{animal}.csv"
    rsync_command = f"rsync -avz {remote_path} {local_path}"
    result = subprocess.run(rsync_command, shell=True)
    if result.returncode != 0:
        print(f"Error syncing data for {animal}: {result.stderr.decode('utf-8')}")
    return result.returncode == 0


def rsync_sessions_summary(
    project_name: str,
    credentials: dict,
    local_path: str,
) -> bool:
    """
    This method syncs the session data from the server to the local machine.
    """
    remote_path = f"{credentials['username']}@{credentials['host']}:{IDIBAPS_TV_PROJECTS}{project_name}/sessions_summary.csv"
    rsync_command = f"rsync -avz {remote_path} {local_path}"
    result = subprocess.run(rsync_command, shell=True)
    if result.returncode != 0:
        print(f"Error syncing session summary data for {project_name}: {result.stderr.decode('utf-8')}")
    return result.returncode == 0


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


if __name__ == "__main__":
    # Example usage
    print(get_server_projects())
    print(get_animals_in_project("visual_and_COT_data"))