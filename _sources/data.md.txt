# Data and its formats

## Preparing the trials

Each trial row has the task settings, the states and the events of the
trial (`Port1In`, `STATE_stimulus_state_START`..., saved as text such as
`"[1.2, 3.4]"`). `preprocessing.prepare` turns them into the columns every
analysis uses:

| Column | Meaning |
|---|---|
| `year_month_day`, `trial_of_day`, `total_trial` | Day, trial number in the day and in the whole history of the subject. |
| `first_choice`, `last_choice`, `last_choice_before_stimulus` | Side poked (left / right). |
| `roa_choice`, `roa_choice_numeric` | Repeat or alternate the previous choice. |
| `miss_trial`, `early_pokeout` | No choice; left the center port too early. |
| `reaction_time`, `time_between_trials`, `trial_duration` | Seconds. |
| `port2_holds` | Duration of every center poke of the trial. |

The event columns are read once, with a fast parser, and shared by every
column above. `prepare` only adds what is missing.

## Format changes

The tasks have changed what some columns mean over time (for example,
`visual_stimulus` went from (correct, incorrect) to (left, right), timeouts
are saved with `correct` empty, and old modality names such as `auditory_cot`
or `auditory_locaPT` are read as the current ones). Old rows cannot be
rewritten, so every change is described, with the function that reads both
versions, in `data_format_changes.py`. Add new changes there.

Each village started running the new task code on a different day
(`NEW_TASK_CODE_SINCE` in that file: village06 on 2026-11-09, villages 01, 02,
04 and 05 on 2026-11-25 at 14:00). Each row is read according to its date and its
village (`system_name`). A village that is not listed, such as a new one, is
read in the newest format; when a village with old data is updated, add the
moment it was updated to that list.

Trials recorded with the deprecated `use_sound_location` option (a cloud of
tones played from the side of the reward, project COT_unilateral_sound) are
read as the modality `auditory-local-cot`: their cue was where the sound came
from, not its frequency.

Performance counts timeouts as errors everywhere (`correct_as_number`).

## Setups

A project can train its subjects in several boxes. `io.loading.find_setups`
lists the folders of a project with an `events.csv` or a
`sessions_summary.csv`, and `load_project_logs` reads all of them together
(sorted by date, without the rows repeated in several files).

## Session settings

The village saves the settings of every session (training stage,
difficulties, punishment, reward...) in the `settings` column of each
setup's `sessions_summary.csv`, and in a `.json` file next to the session
(`sessions/<subject>/<subject>_<task>_<YYYYMMDD_HHMMSS>.json`). A session's
settings and trials are matched by subject and date (the start of the
session).

```python
from lecilab_behavior_analysis.io import settings
from lecilab_behavior_analysis.io.loading import load_project_logs

_, summary, _ = load_project_logs("COT_cannula_GAD2_data")
table = settings.session_settings_table(summary, "NUO057")  # one row per session
settings.changing_settings(table)  # the settings that changed at some point
settings.print_session_settings("COT_cannula_GAD2_data", "NUO057", "2026-04-09 04:01:23")
```

`print_session_settings` lists the changes since the previous session first:

```text
NUO057, session of 2026-04-09 04:01:23
3 changed since the previous session (*)
* current_training_stage   TwoAFC_visual_hard   (was TwoAFC_visual_easy)
* hard_trials_on           True   (was False)
* medium_trials_on         True   (was False)
  ambiguous_beginning_time 0.05
  ...
```

`get_session_settings` looks for one session in the summaries, then in its
`.json` file, and finally copies that file from the cluster (rsync over
ssh) if this computer can reach it.

To see how the settings change through training, together with the choices,
`metrics.choices.get_choices_by_difficulty_matrix` gives the proportion of
left choices by correct side and difficulty in each session, and
`plots.training_matrix_plot` draws it with some settings as rows above it
(`io.settings.settings_for_sessions`).
