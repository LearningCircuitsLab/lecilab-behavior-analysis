# generate fake dataset for testing
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import get_outpath
import random

def generate_fake_dataset(outfile: str) -> None:
    """
    This method generates a fake dataset for testing purposes.
    """

    total_sessions = 25
    starting_date = pd.to_datetime("2021-01-01")
    current_training_stage_list = (
        ["Habituation"] * 3
        + ["TwoAFC_visual_easy"] * 5
        + ["TwoAFC_visual_hard"] * 3
        + ["TwoAFC_auditory_easy"] * 7
        + ["TwoAFC_auditory_hard"] * 4
        + ["TwoAFC_multisensory_easy"] * 8
    )

    # create empty dataframe
    df = pd.DataFrame(
        columns=[
            "current_training_stage",
            "water",
            "correct_side",
            "stimulus_modality",
            "difficulty",
            "session",
            "correct",
            "holding_time",
            "date",
            "trial",
            "Port1In",
            "Port1Out",
            "Port2In",
            "Port2Out",
            "Port3In",
            "Port3Out",
        ]
    )

    for i, session_counter in enumerate(np.arange(1, total_sessions + 1)):

        n_trials = np.random.randint(250, 500)

        trials = np.arange(1, n_trials + 1)
        # date
        date = [starting_date + pd.DateOffset(days=session_counter)] * n_trials
        # generate the trial types
        correct_side = np.random.choice(["left", "right"], n_trials)
        # generate the holding time
        holding_time = np.random.choice([0.5, 1, 1.5, 2], n_trials)
        # stimulus modality
        if "visual" in current_training_stage_list[i]:
            stimulus_modality = "visual"
        elif "auditory" in current_training_stage_list[i]:
            stimulus_modality = "auditory"
        else:
            stimulus_modality = "multisensory"
        # difficulty
        if "hard" in current_training_stage_list[i]:
            difficulty = np.random.choice(["easy", "medium", "hard"], n_trials)
        else:
            difficulty = ["easy"] * n_trials

        # generate the performance and modify it according to the difficulty
        prob_correct = np.min([0.5 + 0.03 * session_counter, 0.99])
        correct = np.empty(n_trials)
        prob_correctors = [1, 0.8, 0.6]
        for k, dif in enumerate(["easy", "medium", "hard"]):
            mask = [x==dif for x in difficulty]
            pc = prob_correctors[k] * prob_correct
            correct[mask] = np.random.choice(
                [True, False], np.sum(mask), p=[pc, 1 - pc]
            )

        session = [session_counter] * n_trials

        water = [0] * n_trials
        water = np.where(correct, 2, water)
        
        Port1In = []
        Port1Out = []
        Port2In = []
        Port2Out = []
        Port3In = []
        Port3Out = []

        #Create the timestamps for Port1, Port2 and Port3
        session_time = 0
        for trial in range(n_trials):
            session_time += random.randint(10, 50)
            Port2In.append(session_time)
            session_time += random.randint(10, 50)
            # Port2Out
            Port2Out.append(session_time)
            session_time += random.randint(40, 90)

            #Decision of Port 1 or Port 3 depending on the correct side
            if (correct_side[trial] == "left" and correct[trial] == True) or (correct_side[trial]== "right" and correct[trial] == False):
                # Port1In y Port1Out
                Port1In.append(session_time)
                session_time += random.randint(10, 50)
                Port1Out.append(session_time)
                Port3In.append(np.nan)
                Port3Out.append(np.nan)
            else:
                # Port3In y Port3Out
                Port3In.append(session_time)
                session_time += random.randint(10, 50)
                Port3Out.append(session_time)
                Port1In.append(np.nan)
                Port1Out.append(np.nan)

        # create the dataframe
        df_session = pd.DataFrame(
            {
                "current_training_stage": [current_training_stage_list[i]] * n_trials,
                "water": water,
                "correct_side": correct_side,
                "stimulus_modality": [stimulus_modality] * n_trials,
                "difficulty": difficulty,
                "session": session,
                "correct": correct,
                "holding_time": holding_time,
                "date": date,
                "trial": trials,
                "Port1In" : Port1In,
                "Port1Out": Port1Out,
                "Port2In": Port2In,
                "Port2Out": Port2Out,
                "Port3In": Port3In,
                "Port3Out": Port3Out,
            }
        )

        # concatenate the dataframes
        df = pd.concat([df, df_session])

        # save the dataframe
        df.to_csv(outfile, sep=";", index=False)


if __name__ == "__main__":
    outpath = get_outpath()
    mice = ["mouse1", "mouse2", "mouse3"]
    for mouse in mice:
        mouse_out_path = Path(outpath) / "example_data" / mouse
        mouse_out_path.mkdir(parents=True, exist_ok=True)
        generate_fake_dataset(mouse_out_path / f"{mouse}_fakedata.csv")
