import pandas as pd
import create_cb_matrix
import testable
import random
# Visual Search for ROC curve

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
subjectGroup = 1
subject_counter = 0

# initialize dataframes
practice_trials = pd.DataFrame([])
all_trials = pd.DataFrame([])

# initialize random location for memory test object locations

obj_loc_matrix1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
obj_loc_matrix2 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
obj_loc_matrix3 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

random.shuffle(obj_loc_matrix1)
random.shuffle(obj_loc_matrix2)
random.shuffle(obj_loc_matrix3)

obj_locations = obj_loc_matrix1 + obj_loc_matrix2 + obj_loc_matrix3

for t in range(subjectGroup):

    # Create counter balance matrix for each participant
    trial_cb, practice_cb = create_cb_matrix.cb_matrix()

    practice_cb, search_trials, memory_trials_all, memory_test_trials = create_cb_matrix.cb_matrix_test(30)
    # initialize main dataframe
    par_trials = pd.DataFrame([])
    final_trial = pd.DataFrame([])
    mem_final_trial = pd.DataFrame([])

    # initialize counters
    subject_counter += 1
    break_counter = 0
    trial_num = 0
    test_num = 0
    for p in range(len(practice_cb)):

        start = testable.start_trial()
        cue = testable.cue_trial(practice_cb["Ref_Word"][p])

        # OBJECTS
        target = practice_cb["target"][p]
        pair = practice_cb["pair"][p]
        neutral1 = practice_cb["neutral1"][p]
        neutral2 = practice_cb["neutral2"][p]
        condition = practice_cb["condition"][p]

        objects = testable.object_trials_prac(
            target, pair, neutral1, neutral2, condition, trial_num, practice_cb.iloc[[p]])
        objects["type"] = ["learn"]
        trial = pd.concat([start, cue, objects], axis=0, sort=True)
        practice_trials = practice_trials.append(trial)
        # practice_trials["subjectGroup"] = subject_counter

    for i in range(len(trial_cb)):
        # print("trial # " + str(i))
        trial_num += 1

        # START TRIAL (0/4) - ITI and alert for fixation prior to beginning of trial
        start = testable.start_trial()

        # create all possible trials for subjectGroup
        if i in memory_trials_all['trial_number'].values:

            # filters for potential memory trials
            memory_trial_filter = (memory_trials_all['trial_number'] == i, ['target', 'pair', 'neutral1', 'neutral2', 'condition', 'trial_number', 'Ref_Word'])
            memory_rest_filter = (memory_trials_all['trial_number'] == i, ['subjectGroup', 'Mean_Rating_Tx', 'Mean_Rating_Thm', 'Difference_Score', 'mem_trials'])

            # total number of subjectGroup
            subjectGroup_total = len(memory_trials_all.loc[memory_trial_filter])

            # set up empty data frame
            objects_all = pd.DataFrame([])

            # loop through # of subject Group to create testable trial_file
            for subjGroup in range(subjectGroup_total):

                # cue
                cue = testable.cue_trial(memory_trials_all.loc[memory_trial_filter].values[subjGroup, 6])
                # print(cue['stim1'])
                cue['subjectGroup'] = memory_trials_all.loc[memory_rest_filter].values[subjGroup, 0]

                # target
                target = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 0]
                pair = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 1]
                neutral1 = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 2]
                neutral2 = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 3]
                condition = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 4]
                trial_num = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 5]
                extra_target = memory_trials_all.loc[memory_trial_filter].values[subjGroup, 7]

                print(extra_target)

                # rest of the information needed for trial (and analysis)
                rest_trials = memory_trials_all.loc[memory_rest_filter].values[subjGroup]

                # combine all target info into object df
                objects = testable.object_trials(target, pair, neutral1, neutral2, condition, trial_num, extra_target, rest_trials, num=1)

                # combine cue and object
                cue_objects = pd.concat([cue, objects])
                # print(cue_objects['stim1'])
                # print('')
                # Combine for all subjectGroup
                objects_all = pd.concat([objects_all, cue_objects])

                # print(subjGroup, test_num, "length of objects_all: ", len(objects_all), "length of objects: ", len(objects))
            objects_all.to_clipboard()

        else:

            # for all non-critical visual search trials
            search_trial_filter = (search_trials['trial_number'] == i, ['target', 'pair', 'neutral1', 'neutral2', 'condition', 'trial_number', "Ref_Word"])
            rest_trials_filter = (search_trials ['trial_number'] == i, ['Mean_Rating_Tx', 'Mean_Rating_Thm', 'Difference_Score', 'mem_trials'])

            # rest of the information needed for trial (and analysis)
            rest_trials = search_trials.loc[rest_trials_filter]

            cue = testable.cue_trial(search_trials.loc[search_trial_filter].values[0, 6])

            # OBJECTS
            target = search_trials.loc[search_trial_filter].values[0, 0]
            pair = search_trials.loc[search_trial_filter].values[0, 1]
            neutral1 = search_trials.loc[search_trial_filter].values[0, 2]
            neutral2 = search_trials.loc[search_trial_filter].values[0, 3]
            condition = search_trials.loc[search_trial_filter].values[0, 4]
            trial_num = search_trials.loc[search_trial_filter].values[0, 5]
            extra_target = '0'

            # print('search!', search_trials.loc[search_trial_filter].values[0])
            objects_all = testable.object_trials(target, pair, neutral1, neutral2, condition, trial_num, extra_target, rest_trials)


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

            if i in memory_trials_all['trial_number'].values:
                trial = pd.concat([start, objects_all, rest], axis=0, sort=True)
            else:
                trial = pd.concat([start, cue, objects_all, rest], axis=0, sort=True)
        else:
            if i in memory_trials_all['trial_number'].values:
                trial = pd.concat([start, objects_all], axis=0, sort=True)
            else:
                trial = pd.concat([start, cue, objects_all], axis=0, sort=True)


        final_trial = final_trial.append(trial, sort=True)

        # print('final trial: ' + str(len(final_trial)))
        # print('all_trials: ' + str(len(all_trials)))
        # mem_final_trial["subjectGroup"] = subject_counter

    print("adding parNo: " + str(subject_counter))
    par_trials = par_trials.append(final_trial)

    # practice_trials['subjectGroup'] = subject_counter
    # end_practice_break['subjectGroup'] = subject_counter
    # par_trials['subjectGroup'] = subject_counter
    # mem_instructions['subjectGroup'] = subject_counter

    all_trials = pd.concat(
        [all_trials, practice_trials, end_practice_break, par_trials], axis=0, sort=True)

    # time between end of visual search and beginning of memory search
    mem_start = testable.start_trial()

    # surprise memory test at the end
    memory_test_filter = (all_trials['trial_num'] == 89,
                           ['subjectGroup', 'stim_orig', 'stim_flip', 'sem_condition'])

    memory_test_trials = all_trials.loc[memory_test_filter]

    memory_test_all = pd.DataFrame([])
    memory = testable.insert_essential_columns()
    memory["type"] = ["test"]

    for subjs in range(len(memory_test_trials)):
        print('sub num: ', subjs)

        mem_subjGroup = memory_test_trials.values[subjs, 0]
        mem_stim_orig = memory_test_trials.values[subjs, 1]
        mem_stim_flip = memory_test_trials.values[subjs, 2]
        mem_sem_condition = memory_test_trials.values[subjs, 3]
        print('subjGroup: ', mem_subjGroup)
        # memory_test_trials.to_clipboard()

        if obj_locations[subjs] == 1:
            memory["stimList"] = f'{mem_stim_orig};{mem_stim_flip}'
            orig_stim_loc = 1
        else:
            memory["stimList"] = f'{mem_stim_flip};{mem_stim_orig}'
            orig_stim_loc = 2

        print(memory["stimList"])
        print('')
        memory["button1"] = button1
        memory["button2"] = button2
        memory["button3"] = button3
        memory["button4"] = button4
        memory["button5"] = button5
        memory["button6"] = button6
        memory["sem_condition"] = mem_sem_condition
        memory['subjectGroup'] = mem_subjGroup
        memory["stimFormat"] = ".png"
        memory["mem_trials"] = 2
        memory["stimPos"] = f'{-testable.mem_stim_pos}; {testable.mem_stim_pos}'
        memory['orig_stim_loc'] = orig_stim_loc
        memory['trialText'] = 'One of the below objects just appeared in the visual search task. Using your mouse, select a button below the objects that best represents your memory and confidence level.'

        memory_test_all = pd.concat([memory_test_all, memory])

    memoryz = pd.concat([mem_start, memory_test_all], axis=0, sort=True)

# Refine DataFrame for final trial file
all_trials = pd.concat([gender_df, race_df, hispanic_df,
                        instructions_df, all_trials, memoryz], axis=0, sort=True)
# .to_clipboard()


all_trials = all_trials.set_index("subjectGroup")
all_trials = all_trials[
    ["type", "content", "trialText", "stimList", "stim1", "stimPos", "stimFormat", "ITI",
     "presTime", "keyboard", "key", "feedback", "feedbackTime", "feedbackOptions",
     "responseType", "responseOptions",
     "button1", "button2", "button3", "button4", "button5", "button6", 'button7',
     "trial_num", 'mem_trials',
     "target_loc", "pair_loc", "sem_condition", 'orig_stim_loc', 'trial_type']]

# Extract result to clipboard
all_trials.to_clipboard(excel=True, sep="\t")
print("done")
