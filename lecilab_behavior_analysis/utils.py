import numpy as np
import pandas as pd


def get_session_performance(df, session: int) -> float:
    """
    TODO: move this to a different package
    This method calculates the performance of a session.
    """

    return df[df.session == session].correct.mean()


def get_session_number_of_trials(df, session: int) -> int:
    """
    This method calculates the number of trials in a session.
    """

    return df[df.session == session].shape[0]


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


def get_right_bias(side_and_correct_array: np.ndarray) -> float:
    """
    This method calculates the right bias in the data.

    Args:
        side_and_correct_array (np.ndarray): Array with the side (left or right) and correct answer (boolean)
    """
    if side_and_correct_array.shape[0] != 2:
        raise ValueError("Input array must have exactly two rows")

    first_pokes = side_and_correct_array[0, :]
    correct_list = side_and_correct_array[1, :]
    wrong_sides = first_pokes[correct_list == "False"]
    if len(wrong_sides) == 0:
        return 0
    wrong_side_proportion = len(wrong_sides) / len(first_pokes)
    wrong_right_proportion = (
        wrong_side_proportion * np.nansum(wrong_sides == "right") / len(wrong_sides)
    )
    wrong_left_proportion = (
        wrong_side_proportion * np.nansum(wrong_sides == "left") / len(wrong_sides)
    )

    return wrong_right_proportion - wrong_left_proportion


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


def get_text_from_df(df: pd.DataFrame, mouse_name: str) -> str:
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


def load_example_data(mouse_name) -> pd.DataFrame:
    outpath = "/mnt/c/Users/HMARTINEZ/LeCiLab/data"
    df = pd.read_csv(outpath + "/" + mouse_name + "/" + mouse_name + "_fakedata.csv", sep=";")

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
    # get the sound statistics
    high_mat_stats = analyze_sound_matrix(sound_dict["high_tones"])
    low_mat_stats = analyze_sound_matrix(sound_dict["low_tones"])
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