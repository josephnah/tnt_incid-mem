import pandas as pd
import numpy as np
import random

start_time = 700
ITI = 500
cue_duration = 500
object_time = 2000

fixation = "fixation_black"

# Coordinates
fovea = "0, 0"
coor_origin = 230
object_loc_1 = f'{-coor_origin} {-coor_origin}'
object_loc_2 = f'{coor_origin} {-coor_origin}'
object_loc_3 = f'{-coor_origin} {coor_origin}'
object_loc_4 = f'{coor_origin} {coor_origin}'
all_coor = [object_loc_1, object_loc_2, object_loc_3, object_loc_4]
mem_stim_pos = 200
fovea = '0, 0'


def insert_essential_columns():
    essential_df = pd.DataFrame(columns=[
        'type', 'content', 'trialText', 'stimList', 'stim1', 'stimPos',
        'stimFormat', 'ITI', 'presTime', 'keyboard', 'key', 'feedback', 'feedbackTime', 'feedbackOptions',
        'responseType', 'responseOptions', 'head', 'button1', 'block_num', 'trial_num', 'condition', 'sem-rel',
        'cued-object', 'cued-object_loc', 'target_object_loc', 'obj-orientation', 'obj-pair_num',
        'stim1_pos', 'stim2_pos', 'stim3_pos', 'stim4_pos', 'target_kind', 'trial_type', 'required'
    ]
    )
    return essential_df


def insert_instructions():

    ### INSTRUCTIONS ###
    begin_message = "Click the button below for instructions"
    # race_question = "American Indian or Alaska Native;Asian;Black or African American;Native Hawaiian or Other Pacific Islander; White;More than one;Not Listed"
    # gender_question = "Male; Female; Non-Binary; Decline to State"
    instructions_np = np.array([
        # ["form", "", "", "", "", "", "", "", "dropdown",
        #     gender_question, "What is your gender identity?", 1],
        # ["form", "", "", "", "", "", "", "", "dropdown",
        #     race_question, "", 1],
        # ["form", "", "", "", "", "", "", "", "dropdown",
        #     "yes; no", "Do you identify as Hispanic", 1],
        ["instructions", begin_message, "", "",
            "Next", "", "", "", "", "", "", "", '1000'],
        ["instructions", "", "instructions01", ".png",
            "Next", "", "", "", "", "", "", "", ''],
        ["instructions", "", "instructions02", ".png",
            "Next", "", "", "", "", "", "", "", ''],
        ["learn", "", "blank", ".png", "", 2000,
            "allTrial", "50 50", "", "", "", "", '']
    ])

    instructions_df = pd.DataFrame(data=instructions_np, columns=["type", "content", "stim1", "stimFormat", "button1",
                                                                  "presTime", "fixation", "stimPos", "responseType",
                                                                  "responseOptions", "head", "required", 'ITI'])
    return instructions_df

def insert_instructions_mem():

    ### INSTRUCTIONS ###
    begin_message = "Click the button below for instructions"
    instructions_np = np.array([
        ["instructions", begin_message, "", "",
         "Next", "", "", "", "", "", "", ""],
        ["instructions", "", "instructions03", ".png",
         "Next", "", "", "", "", "", "", ""],
        ["instructions", "", "instructions04", ".png",
         "Next", "", "", "", "", "", "", ""],
        ["learn", "", "blank", ".png", "", 2000,
         "allTrial", "50 50", "", "", "", ""]
    ])

    instructions_df = pd.DataFrame(data=instructions_np, columns=["type", "content", "stim1", "stimFormat", "button1",
                                                                  "presTime", "fixation", "stimPos", "responseType",
                                                                  "responseOptions", "head", "required"])
    return instructions_df

def insert_end_practice():

    practice_message = "You have finished the practice. Click the button below to start the experiment"

    instructions_2_np = np.array([
        ["instructions", practice_message, "", "",
            "Next", "", "", "", "", "", "", ""],
        ["learn", "", "blank", ".png", "", 2000,
            "allTrial", "50 50", "", "", "", ""]
    ])

    instructions_2_df = pd.DataFrame(data=instructions_2_np, columns=["type", "content", "stim1", "stimFormat", "button1",
                                                                      "presTime", "fixation", "stimPos", "responseType",
                                                                      "responseOptions", "head", "required"])

    return instructions_2_df


def location_shuffle():
    locations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    np.random.shuffle(locations)
    if locations[0] == 0:
        object_loc = [0, 1, 2, 3]
    elif locations[0] == 1:
        object_loc = [0, 1, 3, 2]
    elif locations[0] == 2:
        object_loc = [0, 2, 1, 3]
    elif locations[0] == 3:
        object_loc = [0, 2, 3, 1]
    elif locations[0] == 4:
        object_loc = [1, 0, 2, 3]
    elif locations[0] == 4:
        object_loc = [1, 0, 3, 2]
    elif locations[0] == 5:
        object_loc = [1, 3, 0, 2]
    elif locations[0] == 6:
        object_loc = [1, 3, 2, 0]
    elif locations[0] == 7:
        object_loc = [1, 3, 2, 0]
    elif locations[0] == 8:
        object_loc = [2, 0, 1, 3]
    elif locations[0] == 9:
        object_loc = [2, 0, 3, 1]
    elif locations[0] == 10:
        object_loc = [2, 3, 0, 1]
    elif locations[0] == 11:
        object_loc = [2, 3, 1, 0]
    elif locations[0] == 12:
        object_loc = [3, 1, 0, 2]
    elif locations[0] == 13:
        object_loc = [3, 1, 2, 0]
    elif locations[0] == 14:
        object_loc = [3, 2, 0, 1]
    elif locations[0] == 15:
        object_loc = [3, 2, 1, 0]

    return object_loc


def target_shuffle():
    ''' Shuffle target type'''
    target_type = [1, 2, 3, 4]
    np.random.shuffle(target_type)

    stim5 = 'target_' + str(target_type[0])
    stim6 = 'target_' + str(target_type[1])
    stim7 = 'target_' + str(target_type[2])
    stim8 = 'target_' + str(target_type[3])

    return [stim5, stim6, stim7, stim8]


def object_flip(str):
    ''' randomly flip object '''
    if random.random() > .5:
        object = str
    else:
        object = str + '_flip'
    return object


def start_trial():
    start = insert_essential_columns()
    start["type"] = ["learn"]
    start["stimList"] = fixation
    start["stimFormat"] = ".png"
    start["presTime"] = start_time
    start["ITI"] = ITI
    start["stimPos"] = fovea

    return start

def cue_trial(cue_word):

    cue = insert_essential_columns()
    cue["type"] = ["learn"]
    cue["stim1"] = cue_word
    cue["stimFormat"] = "word"
    cue["presTime"] = cue_duration
    cue["stimPos"] = fovea

    return cue


def race_questions():
    race = insert_essential_columns()
    race['trialText'] = ['With which race do you identify?']
    race['button1'] = 'American Indian or Alaska Native'
    race['button2'] = 'Asian'
    race['button3'] = 'Black or African American'
    race['button4'] = 'Native Hawaiian or Other Pacific Islander'
    race['button5'] = 'White'
    race['button6'] = 'More than one'
    race['button7'] = 'Not Listed'
    race['stim1'] = 'blank'
    race['stimFormat'] = '.png'
    race['type'] = ['test']
    race['trial_type'] = 'demographics'

    return race


def gender_questions():
    gender = insert_essential_columns()
    gender['trialText'] = ['What is your gender identity?']
    gender["type"] = ["test"]
    gender['button1'] = 'Male'
    gender['button2'] = 'Female'
    gender['button3'] = 'Non-Binary'
    gender['button4'] = 'Decline to State'
    gender['stim1'] = 'blank'
    gender['stimFormat'] = '.png'
    gender['trial_type'] = 'demographics'

    return gender


def hispanic_question():
    hispanic = insert_essential_columns()
    hispanic["type"] = ["test"]
    hispanic['trialText'] = ['Do you identify as Hispanic?']
    hispanic['button1'] = 'Yes'
    hispanic['button2'] = 'No'
    hispanic['stim1'] = 'blank'
    hispanic['stimFormat'] = '.png'
    hispanic['trial_type'] = 'demographics'

    return hispanic


def object_trials(object1, object2, object3, object4, semCond, trial_num, matrix):

    objects = insert_essential_columns()
    objects["type"] = ["test"]
    objects["stimFormat"] = ".png"

    object_loc = location_shuffle()

    stim1 = object_flip(object1)
    stim2 = object_flip(object2)
    stim3 = object_flip(object3)
    stim4 = object_flip(object4)

    objects['stim_orig'] = stim1

    if stim1[-5:] == '_flip':
        objects['stim_flip'] = stim1.replace('_flip', '')

    else:
        objects['stim_flip'] = stim1 + '_flip'

    # targets
    targets = target_shuffle()
    stim5 = targets[0]
    stim6 = targets[1]
    stim7 = targets[2]
    stim8 = targets[3]
    stim9 = fixation

    objects["sem_condition"] = semCond

    objects["stimList"] = f'{stim1};{stim2};{stim3};{stim4};{stim5};{stim6};{stim7};{stim8};{stim9}'
    stimPos = f'{all_coor[object_loc[0]]}; {all_coor[object_loc[1]]}; {all_coor[object_loc[2]]}; {all_coor[object_loc[3]]}'
    objects["stimPos"] = f'{stimPos}; {stimPos}; {fovea}'

    objects["presTime"] = object_time
    objects["target_loc"] = object_loc[0] + 1
    objects["pair_loc"] = object_loc[1] + 1

    objects["keyboard"] = "1 2 3 4"
    objects['key'] = stim5[-1]

    objects["feedback"] = "incorrect: incorrect"
    objects["feedbackTime"] = 1000
    objects["feedbackOptions"] = "center"

    objects["trial_num"] = trial_num
    objects = pd.merge(objects.assign(
        A=1), matrix.assign(A=1), on="A").drop("A", 1)

    return objects


# if __name__ == '__main__':
