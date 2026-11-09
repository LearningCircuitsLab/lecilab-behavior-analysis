"""Helpers used by the tasks themselves (block sizes, side bias), not by the analysis."""
import numpy as np




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
