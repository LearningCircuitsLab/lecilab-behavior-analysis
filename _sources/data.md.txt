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
`visual_stimulus` went from (correct, incorrect) to (left, right) on
2026-11-09, and timeouts are saved with `correct` empty). Old rows cannot be
rewritten, so every change is described, with its date and the function that
reads both versions, in `data_format_changes.py`. Add new changes there.

Performance counts timeouts as errors everywhere (`correct_as_number`).

## Setups

A project can train its subjects in several boxes. `io.loading.find_setups`
lists the folders of a project with an `events.csv` or a
`sessions_summary.csv`, and `load_project_logs` reads all of them together
(sorted by date, without the rows repeated in several files).
