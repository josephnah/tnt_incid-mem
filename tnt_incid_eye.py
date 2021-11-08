__author__="joecool890"
#!/usr/bin/env python2
# semantic-obj-scn
# Version 1.1.5
# fixed email portion of code + got rid of paidPar var
from psychopy import core, data, event, visual
import os, datetime as dt
import numpy as np
import random
import sys
import testable
# sys.path.append('/Users/joecool890/Code/Python/uc-davis/tnt_incid_mem/pylinkwrapper')
# import pylinkwrapper



# --- Experiment Specifications ---
# RAs, edit + double check before running
parNo = 1
parAge = 33
parGen = 1 # 1 for Male, 2 for Female, 3 for
parHand = 1 # 1 for Right, 2 for Left
eye_track_var = 0 # 0 for debugging, 1 for data collection

# -- Only change when necessary
prac_repeat = 0 # 1 if skipping practice

# Set date
date = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # get date as YMD_H_M_S

#  --- Set Path ---
root_path = os.getcwd()
# print(root_path)
stim_path = '/stim/'
file_path = '/data/'
target_path = '/stim/targets/'
eye_track_data_path = root_path + '/eye_track_data'
#  --- Experiment Variables ---
exp_name = "exp_eyetrack"
exp_ver = 1
exp_iter = 1
reps = 1
block = 3
block_size = 30

# --- Data setup ---
dataMatrix  = {}
if parNo < 10:
    raw_data = root_path + file_path + "00" + str(parNo) + "_" + date + "_" + exp_name
    rawName = "00" + str(parNo) + "_" + date + "_" + exp_name
elif parNo >= 10:
    raw_data = root_path + file_path + "0" + str(parNo) + "_" + date + "_" + exp_name
    rawName = "0" + str(parNo) + "_" + date + "_" + exp_name
elif parNo >= 100:
    raw_data = root_path + file_path + str(parNo) + "_" + date + "_" + exp_name
    rawName = str(parNo) + "_" + date + "_" + exp_name

# name of monitor
monitor_name = 'testMonitor'
# full screen or not
screen_var = 0

# target variables
circ_fill_color = '#808080'
circ_line_color = '#808080'
circ_rad = .4
text_size = .5

# --- Set Duration (s) ---
cue_time = .5
object_time = 2
ISI = 2
if eye_track_var == 1:
    ITI = .1
else:
    ITI = 1

# --- Set Visual Angle ---
fix_size = 1
resolution = 4
obj_size = 5

xpos = 5
ypos = 5

# -- Set Virtual Window ---
win = visual.Window(
    [1280, 1024],
    monitor=monitor_name,
    screen=0,
    units="deg",
    color="white",
    fullscr=screen_var
)

# --- set up stimuli ---

# fixations
fix = visual.ImageStim(win, image=root_path + stim_path + "fixations/fixation_black.png", size=fix_size, pos=[0, 0], units="deg")
wrongFix = visual.ImageStim(win, image=root_path + stim_path + "fixations/wrongFix.png", size=fix_size, pos=[0, 0], units="deg")

# Cue word
cue_word = visual.TextStim(win, color=(-1, -1, -1), alignText="center")

# stim locations
up_left_pos = [-xpos, ypos]
up_right_pos = [xpos, ypos]
bot_left_pos = [-xpos, -ypos]
bot_right_pos = [xpos, -ypos]

all_stim_loc = [up_left_pos, up_right_pos, bot_left_pos, bot_right_pos]

# objects
target = visual.ImageStim(win, size=obj_size, units="deg")
pair = visual.ImageStim(win, size=obj_size, units="deg")
neutral1 = visual.ImageStim(win, size=obj_size, units="deg")
neutral2 = visual.ImageStim(win, size=obj_size, units="deg")

# targets
target_bg1 = visual.Circle(win=win, units="deg", radius=circ_rad, pos=up_left_pos, fillColor=circ_fill_color, lineColor=circ_line_color)
target_bg2 = visual.Circle(win=win, units="deg", radius=circ_rad, pos=up_right_pos, fillColor=circ_fill_color, lineColor=circ_line_color)
target_bg3 = visual.Circle(win=win, units="deg", radius=circ_rad, pos=bot_left_pos, fillColor=circ_fill_color, lineColor=circ_line_color)
target_bg4 = visual.Circle(win=win, units="deg", radius=circ_rad, pos=bot_right_pos, fillColor=circ_fill_color, lineColor=circ_line_color)

target1 = visual.TextStim(win, color=(-1, -1, -1), alignText="center", height=text_size)
target2 = visual.TextStim(win, color=(-1, -1, -1), alignText="center", height=text_size)
target3 = visual.TextStim(win, color=(-1, -1, -1), alignText="center", height=text_size)
target4 = visual.TextStim(win, color=(-1, -1, -1), alignText="center", height=text_size)

# --- hide mouse ---
event.Mouse(visible=False)

# --- File for balancing factors based on orientation ---
prac_matrix_file = root_path + "/balanceFactors-scn-obj_prac.csv"
matrix_file = root_path + "/balance_factors-tnt_eye.csv"

trialMatrix = data.importConditions(matrix_file)
trials = data.TrialHandler(
    trialList=trialMatrix,
    nReps=reps,
    method="random"
)

# add trials to the experiment handler to store data
currExp = data.ExperimentHandler(
    name="tnt-eyetrack",
    version="1.0",
    extraInfo=dataMatrix,
    saveWideText=True,
    dataFileName=raw_data
)

currExp.addLoop(trials)
rtClock = core.Clock()  # sets up response clock for RT printout

# --- Blocks and Current Trial ---
block = 0
trial = 0
curTrial = len(trialMatrix)
total = 0
disp_acc = []
disp_rt = []

# Initiate eye-tracker link and open EDF
if eye_track_var == 1:
    # set up when tracker came on
    trackerOnClock = core.Clock()
    tracker = pylinkwrapper.Connect(win, rawName)

    # calibrate eye-tracker
    tracker.calibrate()

# --- Start Experimental Trial Loop ---
for thisTrial in trials:
    rt = rtClock.getTime()
    remBlock = 3 - block  # display remaining blocks
    ITI = 1     # Reset ITI to normal

    # --- Display Remaining Blocks ---
    if np.mod(curTrial, block_size) == 0:
        if trial == 0:
            message = visual.TextStim(
                win,
                text=str(remBlock) + " blocks remaining" + "\n\n\n\n\n\nPress space to continue",
                color=(-1, -1, -1),  # black
                alignText="center"
            )
        elif trial > 0:
            message = visual.TextStim(
                win,
                text=str(remBlock) + " block(s) remaining" + "\n\nPress space to continue",
                color=(-1, -1, -1),  # black
                alignText="center"
            )
        message.draw()
        fix.draw()
        win.flip()
        event.waitKeys(keyList="spacebar")

        # --- Recalibration of variables (block #, average RT, etc) ---
        block = block + 1
        disp_rt = []
        disp_acc = []
        win.flip()
        core.wait(2)

    # --- Trial counter ---
    curTrial = curTrial - 1
    trial = trial + 1

    # print(thisTrial['target'])

    # set images
    target.setImage(root_path + stim_path + str(thisTrial['target']) + ".png")
    pair.setImage(root_path + stim_path + str(thisTrial["pair"]) + ".png")
    neutral1.setImage(root_path + stim_path + str(thisTrial["neutral1"]) + ".png")
    neutral2.setImage(root_path + stim_path + str(thisTrial["neutral2"]) + ".png")

    # Set location of images
    object_loc = testable.location_shuffle()

    # object position set
    target.pos = all_stim_loc[object_loc[0]]
    pair.pos = all_stim_loc[object_loc[1]]
    neutral1.pos = all_stim_loc[object_loc[2]]
    neutral2.pos = all_stim_loc[object_loc[3]]

    # target number position set
    target1.pos = all_stim_loc[object_loc[0]]
    target2.pos = all_stim_loc[object_loc[1]]
    target3.pos = all_stim_loc[object_loc[2]]
    target4.pos = all_stim_loc[object_loc[3]]

    # randomize target text
    target_loc_matrix = [1, 2, 3, 4]
    random.shuffle(target_loc_matrix)
    target1.text = target_loc_matrix[0]
    target2.text = target_loc_matrix[1]
    target3.text = target_loc_matrix[2]
    target4.text = target_loc_matrix[3]

    if eye_track_var == 1:
        # check that fixation is maintained before starting exp
        tracker.fix_check(size=5, ftime=1, button='p')
        # Eye tracker trial set-up
        stxt = 'Trial %d' % str(thisTrial)
        tracker.set_status(stxt)
        tracker.set_trialid()
        tracker.send_var('condition', thisTrial['condition'])
        tracker.send_var('trial_num', trial)

        # draw interest areas and start recording
        tracker.draw_ia(0, 0, fix_size, 100, 0, 'fixation')
        tracker.draw_ia(all_stim_loc[0][0], all_stim_loc[0][1], obj_size, 1, 5, 'target_obj')
        tracker.draw_ia(all_stim_loc[1][0], all_stim_loc[1][1], obj_size, 2, 3, 'pair_obj')
        tracker.draw_ia(all_stim_loc[2][0], all_stim_loc[2][1], obj_size, 3, 0, 'neutral1_obj')
        tracker.draw_ia(all_stim_loc[3][0], all_stim_loc[3][1], obj_size, 4, 0, 'neutral2_obj')

        # start recording
        tracker.record_on()
        tracker_time = trackerOnClock.getTime()

    timer = core.Clock()

    # # display fixation (might delete based on fixation code above)
    # timer.add(ITI)
    # while timer.getTime() < 0:
    #     fix.draw()
    #     win.flip()

    # display cue word
    timer.add(cue_time)
    while timer.getTime() < 0:
        cue_word.setText(str(thisTrial['Ref_Word']))
        cue_word.draw()
        win.flip()

    # display objects + targets
    timer.add(object_time)
    win.callOnFlip(rtClock.reset)  # this is when RT is being collected
    keys = []
    event.clearEvents()
    while timer.getTime() < 0:
        fix.draw()
        target.draw()
        pair.draw()
        neutral1.draw()
        neutral2.draw()
        target_bg1.draw()
        target_bg2.draw()
        target_bg3.draw()
        target_bg4.draw()
        target1.draw()
        target2.draw()
        target3.draw()
        target4.draw()
        win.flip()

        # --- Get response ---
        while timer.getTime() < 0:
            keys = event.getKeys(keyList=["1", "2", "3", "4", 'q'])
            if len(keys) > 0:
                keyDown = keys[0]  # take the first keypress as the response
                if eye_track_var == 1:
                    tracker.record_off()
                    eyetracker_time = tracker_time.getTime()
                rt = rtClock.getTime()
                timer.reset()
                if keyDown == "q":
                    core.quit()
            elif len(keys) == 0:
                keyDown = None
                rt = 9999

    #    --- Response Check ---
    if keyDown == target1.text:
        corr = 1
    else:
        corr = 0
        ITI = 1
    print(corr)
    # ITI
    timer.add(ITI)
    while timer.getTime() < 0:
        if corr == 100:
            fix.draw()
        elif corr == 0:
            message = visual.TextStim(win,
                                      text="incorrect",
                                      color=(-1, -1, -1), alignText="center", wrapWidth=100)
            message.draw()

        win.flip()
    print(' ')

    # --- Store Response, RT and other Data ---
    trials.addData("exp_ver", exp_ver)
    trials.addData("exp_iter", exp_iter)
    trials.addData("par_ID", parNo)
    trials.addData("par_age", parAge)
    trials.addData("par_gen", parGen)
    trials.addData("par_hand", parHand)
    trials.addData("blockNo", block)
    trials.addData("trialNo", trial)
    trials.addData("accuracy", corr)
    trials.addData("RT", rt * 1000)
    trials.addData("target_obj_pos", str(object_loc[0]))
    trials.addData("target_pos", str(object_loc[0]))
    trials.addData("pair_obj_pos", str(object_loc[1]))
    trials.addData("pair_pos", str(object_loc[1]))
    if eye_track_var == 1:
        trials.addData("eye_track_time", eyetracker_time * 1000) # since recording started
        tracker.set_trialresult()
    currExp.nextEntry()

if eye_track_var == 1:
    if tracker:
        tracker.end_experiment(eye_track_data_path)
win.close()
