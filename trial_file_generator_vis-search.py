import pandas as pd
import create_cb_matrix
import testable

# Visual Search

# Buttons and stimuli
button1 = "High confidence for Left"
button2 = "Medium confidence for Left"
button3 = "Low confidence for Left"
button4 = "Low confidence for Right"
button5 = "Medium confidence for Right"
button6 = "High confidence for Right"

fixation = "fixation_black"

# instructions + break
instructions_df = testable.insert_instructions()
mem_instructions = testable.insert_instructions_mem()
end_practice_break = testable.insert_end_practice()
race_df = testable.race_questions()
gender_df = testable.gender_questions()
hispanic_df = testable.hispanic_question()

# Subject related
subjectGroup = 5
subject_counter = 0

# initialize dataframes
practice_trials = pd.DataFrame([])
all_trials = pd.DataFrame([])


for t in range(subjectGroup):

    # Create counter balance matrix for each participant
    trial_cb, practice_cb = create_cb_matrix.cb_matrix()

    # initialize main dataframe
    par_trials = pd.DataFrame([])
    final_trial = pd.DataFrame([])
    mem_final_trial = pd.DataFrame([])

    # initialize counters
    subject_counter += 1
    break_counter = 0
    trial_num = 0

    for p in range(len(practice_cb)):

        start = testable.start_trial()
        cue = testable.cue_trial(practice_cb["Ref_Word"][p])

        # OBJECTS
        target = practice_cb["target"][p]
        pair = practice_cb["pair"][p]
        neutral1 = practice_cb["neutral1"][p]
        neutral2 = practice_cb["neutral2"][p]
        condition = practice_cb["condition"][p]

        objects = testable.object_trials(
            target, pair, neutral1, neutral2, condition, trial_num, practice_cb.iloc[[p]])

        trial = pd.concat([start, cue, objects], axis=0, sort=True)
        practice_trials = practice_trials.append(trial)
        practice_trials["subjectGroup"] = subject_counter

    for i in range(len(trial_cb)):
        # print("trial # " + str(i))
        trial_num += 1

        # START TRIAL (0/4) - ITI and alert for fixation prior to beginning of trial
        start = testable.start_trial()

        # CUE WORD
        cue = testable.cue_trial(trial_cb["Ref_Word"][i])

        # OBJECTS
        target = trial_cb["target"][i]
        pair = trial_cb["pair"][i]
        neutral1 = trial_cb["neutral1"][i]
        neutral2 = trial_cb["neutral2"][i]
        condition = trial_cb["condition"][i]

        objects = testable.object_trials(
            target, pair, neutral1, neutral2, condition, trial_num, trial_cb.iloc[[i]])

        # surprise memory test at the end
        memory = testable.insert_essential_columns()
        memory["type"] = ["test"]

        memory["stimList"] = f'{objects["stim_orig"][0]};{objects["stim_flip"][0]}'
        memory["button1"] = button1
        memory["button2"] = button2
        memory["button3"] = button3
        memory["button4"] = button4
        memory["button5"] = button5
        memory["button6"] = button6
        memory["mem_trial_num"] = trial_cb["mem_trial_order"][i]
        memory["sem_condition"] = trial_cb["condition"][i]
        memory["stimFormat"] = ".png"
        memory["stimPos"] = f'{-testable.mem_stim_pos}; {testable.mem_stim_pos}'
        memory["for_order"] = 1

        memory_start = start.copy()
        memory_start["stim1"] = "blank"
        memory_start["mem_trial_num"] = trial_cb["mem_trial_order"][i]
        memory_start["for_order"] = 0

        memoryz = pd.concat([memory_start, memory], axis=0, sort=True)

        # Insert break
        if i % 30 == 0 and i > 0:
            break_counter += 1
            blocks_left = len(trial_cb) / 30 - break_counter
            blocks_left_string = str(blocks_left)

            if blocks_left_string == 1:
                break_message = "1 more block remains. Final stretch! Click below to continue"
            else:
                break_message = blocks_left_string + \
                    " blocks remaining. Take a short break. Click the button below to continue"

            rest = testable.insert_essential_columns()
            rest["type"] = ["instructions", "learn"]
            rest["content"] = [break_message, ""]
            rest["button1"] = ["Next", ""]
            rest["stim1"] = ["", "blank"]
            rest["stimFormat"] = ["", "jpg"]
            rest["presTime"] = ["", "2000"]
            rest["stimPos"] = ["", "50, 50"]

            trial = pd.concat(
                [start, cue, objects, rest], axis=0, sort=True)
        else:
            trial = pd.concat([start, cue, objects], axis=0, sort=True)

        mem_final_trial = mem_final_trial.append(memoryz, sort=True)
        mem_final_trial.sort_values(
            ["mem_trial_num", "for_order"], inplace=True)

        final_trial = final_trial.append(trial, sort=True)

        # print('final trial: ' + str(len(final_trial)))
        # print('all_trials: ' + str(len(all_trials)))
        # mem_final_trial["subjectGroup"] = subject_counter

    print("adding parNo: " + str(subject_counter))
    par_trials = par_trials.append(final_trial)

    practice_trials['subjectGroup'] = subject_counter
    end_practice_break['subjectGroup'] = subject_counter
    par_trials['subjectGroup'] = subject_counter
    mem_instructions['subjectGroup'] = subject_counter
    mem_final_trial['subjectGroup'] = subject_counter

    all_trials = pd.concat(
        [all_trials, practice_trials, end_practice_break, par_trials, mem_instructions, mem_final_trial], axis=0, sort=True)


# Refine DataFrame for final trial file
all_trials = pd.concat([gender_df, race_df, hispanic_df,
                        instructions_df, all_trials], axis=0, sort=True)

all_trials = all_trials.set_index("subjectGroup")
all_trials = all_trials[
    ["type", "content", "trialText", "stimList", "stim1", "stimPos", "stimFormat", "ITI",
     "presTime", "keyboard", "key", "feedback", "feedbackTime", "feedbackOptions",
     "responseType", "responseOptions", "head",
     "button1", "button2", "button3", "button4", "button5", "button6", 'button7',
     "block_num", "trial_num", "mem_trial_num", "required",
     "target_loc", "pair_loc", "sem_condition", 'trial_type']]

# Extract result to clipboard
all_trials.to_clipboard(excel=True, sep="\t")

print("done")
