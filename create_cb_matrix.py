# Creates randomized CB matrix for each participant
# ver 1.0.0 = iter01 and iter02s
# current version: 3.0.0 iter05 (two objects)


import numpy as np
import pandas as pd
from itertools import cycle
import random

def cb_matrix():

    # how many pairs of objects
    num_of_pairs = 30

    # Load objects
    objects_dir = "/Users/joecool890/Dropbox/UC-Davis/projects/tnt_visual-search/stimuli/stimuli_trial_order_with_ratings_edit.csv"
    all_objects = pd.read_csv(objects_dir)

    # Grab thematic (high thm, low tax)
    thematic_objects = all_objects.sort_values(["Difference_Score", "Mean_Rating_Thm"], ascending=[
                                               True, True])[:num_of_pairs].sample(frac=1)
    thematic_objects["condition"] = "thematic"
    # print(thematic_objects)

    # Grab Taxonomic (high tax, low thm)
    taxonomic_objects = all_objects.sort_values(["Difference_Score", "Mean_Rating_Tx"], ascending=[
                                                False, True])[:num_of_pairs].sample(frac=1)
    taxonomic_objects["condition"] = "taxonomic"

    # Grab Neutral
    neutral = (all_objects["condition"] == "neutral")
    neutral_objects_all = all_objects[neutral].sample(frac=1)
    neutral_objects_all["condition"] = "neutral"
    neutral_objects = neutral_objects_all.head(30)

    # Rest are practice
    practice_objects = neutral_objects_all.tail(15)

    # concat into one
    all_objects_list = pd.concat(
        [thematic_objects, taxonomic_objects, neutral_objects])

    # Drop column names
    drop_list = ["Folder_Name_1", "Pair_Word", "Folder_Name_2", "SD_Rating_Tx",
                 "SD_Rating_Thm", "Num_Ratings_Tx", "Num_Ratings_Thm", "Index"]

    # Drop unnecessary columns
    all_objects_list.drop(columns=drop_list, inplace=True)
    practice_objects.drop(columns=drop_list, inplace=True)

    # equally distribute to the three blocks
    blocks = cycle([1, 2, 3])
    all_objects_list["block"] = [next(blocks)
                                 for block in range(len(all_objects_list))]

    # Sort by block
    all_objects_list = all_objects_list.sort_values(["block"])

    # Add trial numbers
    trial_number = [i for i in range(30)]
    trial_order = []
    for b in range(4):
        np.random.shuffle(trial_number)
        trial_order.extend(trial_number)

    trials = cycle(trial_order)
    all_objects_list["trial_order"] = [
        next(trials) for trial in range(len(all_objects_list))]

    # Add trial numbers for surprise memory test

    memory_trial_number = [i for i in range(90)]
    np.random.shuffle(memory_trial_number)

    all_objects_list["mem_trial_order"] = memory_trial_number

    all_objects_list = all_objects_list.sort_values(["block", "trial_order"])

    all_objects_list.reset_index(drop=True, inplace=True)
    practice_objects.reset_index(drop=True, inplace=True)

    # print(all_objects_list)
    return all_objects_list, practice_objects

def cb_matrix_test(subjectGroup):

    random.seed(12)
    # how many pairs of objects
    num_of_pairs = 30
    mem_trial = 10

    # Load objects
    objects_dir = "/Users/joecool890/Dropbox/UC-Davis/projects/tnt_visual-search/stimuli/stimuli_trial_order_with_ratings_edit.csv"
    all_objects = pd.read_csv(objects_dir)

    # Drop column names
    drop_list = ["Folder_Name_1", "Pair_Word", "Folder_Name_2", "SD_Rating_Tx",
                 "SD_Rating_Thm", "Num_Ratings_Tx", "Num_Ratings_Thm", "Index"]
    all_objects.drop(columns=drop_list, inplace=True)

    # Grab thematic (high thm, low tax)
    thematic_objects = all_objects.sort_values(["Difference_Score", "Mean_Rating_Thm"], ascending=[
                                               True, True])[:num_of_pairs]
    # Grab Taxonomic (high tax, low thm)
    taxonomic_objects = all_objects.sort_values(["Difference_Score", "Mean_Rating_Tx"], ascending=[
        False, True])[:num_of_pairs]

    # Grab Neutral
    neutral_filter = (all_objects["condition"] == "neutral")
    neutral_objects_all = all_objects[neutral_filter]

    # Assign conditions
    thematic_objects["condition"] = "thematic"
    taxonomic_objects["condition"] = "taxonomic"
    neutral_objects_all["condition"] = "neutral"

    # Divide neutral into practice
    neutral_objects = neutral_objects_all.head(30)
    practice_objects = neutral_objects_all.tail(15)

    # divide trials into memory and visual search trials
    thematic_objects_memory = thematic_objects[:mem_trial]
    thematic_objects_search = thematic_objects[mem_trial:]

    taxonomic_objects_memory = taxonomic_objects[:mem_trial]
    taxonomic_objects_search = taxonomic_objects[mem_trial:]

    neutral_objects_memory = neutral_objects[:mem_trial]
    neutral_objects_search = neutral_objects[mem_trial:]

    # concat visual search and memory trials
    memory_trials = pd.concat([thematic_objects_memory, taxonomic_objects_memory, neutral_objects_memory])
    search_trials = pd.concat([thematic_objects_search, taxonomic_objects_search, neutral_objects_search])

    # Demarcate whether memory trial or not
    search_trials['mem_trials'] = 0
    memory_trials['mem_trials'] = 1

    # concat into one
    all_objects_list = pd.concat([search_trials, memory_trials])
    # all_objects_list.to_clipboard()

    # Randomize trials, insert
    trial_order = list(range(0, 89))
    random.Random(5).shuffle(trial_order)
    critical_trials = trial_order[0:29] + [89]
    visual_search_trials = trial_order[29:]

    # create full list of trials for memory
    all_trials_list = []
    sub_num_list = []
    for subjs in range(subjectGroup):
        # print("subject", subjectGroup)
        all_trials_list = all_trials_list + critical_trials
        critical_trials.append(critical_trials.pop(0))
        sub_num = [subjs+1] * len(critical_trials)
        sub_num_list = sub_num_list + sub_num

    memory_trials = pd.concat([memory_trials] * subjectGroup)
    memory_trials.reset_index(inplace=True)
    all_trials_df = pd.DataFrame({'subjectGroup': sub_num_list, 'trial_number': all_trials_list})
    memory_trials_all = pd.concat([all_trials_df, memory_trials], axis=1)

    memory_trials_all.to_clipboard()
    # print(memory_trials_all)
    memory_test_trials = memory_trials_all[memory_trials_all['trial_number'] == 89]

    search_trials['trial_number'] = visual_search_trials

    practice_objects.reset_index(drop=True, inplace=True)

    return practice_objects, search_trials,  memory_trials_all, memory_test_trials

cb_matrix_test(2)
