import os
import pandas as pd
import numpy as np
import my_functions
import pingouin as pg
import time

# Show or don't show
# behav_results = input("show behavioral results (y/n)?") or 'n'
behav_results = 'y'

if behav_results == 'y':
    show_behav = 1
elif behav_results == 'n':
    show_behav = 0
else:
    print('enter y or n please')


# Set pandas options
pd.set_option('display.max_columns', 242)
# disable chained assignment warning
pd.options.mode.chained_assignment = None
data_ext = '.csv'

# Directories to data
eye_data_dir = '/Users/joecool890/Dropbox/UC-Davis/projects/tnt_incid-mem/raw-data/eye_tracking/'
behav_data_dir = '/Users/joecool890/Dropbox/UC-Davis/projects/tnt_incid-mem/raw-data/eye_tracking/behav_data/'

# load behavioral data for analysis
behav_data = my_functions.load_filepath(behav_data_dir)
behav_raw_df = pd.concat(map(lambda x: pd.read_csv(x), behav_data))
behav_raw_df.set_index('par_ID', inplace=True)
total_par = len(behav_data)

print('\n# of participants: ', total_par, '\n')

# filters for behavioral data
accuracy_filter = behav_raw_df['accuracy'] == 1
corr_behav_data = behav_raw_df[accuracy_filter]

# Grouping conditions
raw_sem_cond = behav_raw_df.groupby(['par_ID', 'condition'])
corr_sem_cond = corr_behav_data.groupby(['par_ID', 'condition'])

# dropped trials
dropped_trials = 30 - corr_sem_cond['accuracy'].count().unstack()
dropped_trials['total'] = dropped_trials.sum(axis=1) / 90 * 100

# accuracy and RT
accuracy_df = raw_sem_cond['accuracy'].mean() * 100
RT_df = corr_sem_cond['RT'].mean()

# ANOVA for RT and acc (behavioral)
RT_df_anova = RT_df.reset_index()
ANOVA_RT = pg.rm_anova(data=RT_df_anova, dv='RT', within='condition', subject='par_ID').round(4)

pairwise_results_RT = pg.pairwise_ttests(data=RT_df_anova, dv='RT', within='condition',
                                         subject='par_ID', marginal=True, padjust='bonf')

accuracy_df_anova = accuracy_df.reset_index()
ANOVA_ACC = pg.rm_anova(data=accuracy_df_anova, dv='accuracy', within='condition', subject='par_ID').round(4)

pairwise_results_ACC = pg.pairwise_ttests(data=accuracy_df_anova, dv='accuracy', within='condition',
                                         subject='par_ID', marginal=True, padjust='bonf')

# Print data and results
if show_behav == 1:
    print('# of dropped trials \n', dropped_trials, '\n')
    print('RT based on semantic conditions \n', RT_df.unstack(), '\n')
    print('ANOVA for RT \n', ANOVA_RT, '\n\n')
    print("Pairwise testings for RT \n", pairwise_results_RT, '\n')
    print('Accuracy based on semantic conditions \n', round(accuracy_df.unstack(), 2), '\n')
    print("ANOVA for Acc \n", ANOVA_ACC, '\n')
    print("Pairwise testings for Acc \n", pairwise_results_ACC, '\n')
    print('Done with Behavioral data, moving on to eye data \n\n\n')

# load eye data
data_files = my_functions.load_filepath(eye_data_dir)
eye_raw_df = pd.concat(map(lambda x: pd.read_csv(x), data_files))

# Rename columns
eye_raw_df.rename(columns={'RECORDING_SESSION_LABEL': 'par_ID'}, inplace=True)
eye_raw_df.set_index('par_ID', inplace=True)
eye_raw_df.rename(columns={'trial_num': 'trialNo'}, inplace=True)
# remove practice trials from trial count
eye_raw_df['TRIAL_INDEX'] = eye_raw_df['TRIAL_INDEX'] - 15
# Drop useless columns
eye_raw_df.drop(columns=my_functions.eye_drop_list, inplace=True)

# Get correct trial number, use to remove inaccurate trials from eye track, skip for now
accurate_trial_numbers = behav_raw_df[accuracy_filter][['trialNo', 'RT']]

# filter out practice trials
trial_filter = eye_raw_df['condition'] != 'practice'
eye_df = eye_raw_df[trial_filter]

# filter out incorrect trials from df
eye_df = eye_df.reset_index()
accurate_eye_df = pd.DataFrame([])
x = time.time()
for row in accurate_trial_numbers.itertuples():
    par_id = row[0]
    trial_index = row[1]
    par_trial = eye_df[((eye_df['par_ID'] == par_id) & (eye_df['TRIAL_INDEX'] == trial_index))]
    accurate_eye_df = pd.concat([accurate_eye_df, par_trial])
time_elapsed = time.time() - x
print(f'\nDone! Only accurate trials filtered in : {(round(time_elapsed, 5))} seconds')

# Get total trial count

# 1. Calculate total fixation duration - Joy
# last fixation duration - Joy
# whether they looked at the distractors and how long they looked at it
# how many times they looked at an object (2 different looks, 0 looks)
# useful to know what difference it makes if it's 1 look or 2, but
# compare nearest and target just report the visual angle

total_trials = accurate_eye_df.drop_duplicates(['par_ID', 'TRIAL_INDEX'])
total_trial_count = total_trials.groupby(['par_ID'])['TRIAL_INDEX'].count().reset_index(name='total_acc_trials')

# Get interesting data
target_fix_filter = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'target_obj'
pair_fix_filter = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'pair_obj'
first_fix_filter = accurate_eye_df['CURRENT_FIX_INDEX'] == 2 # 1 is fixation

# apply filter
target_fix_trials = accurate_eye_df[target_fix_filter]
target_fix_trials.to_clipboard()
print(A)
# take care of fixations not in IA
# replace_values = (target_fix_trials['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'target_obj') & (target_fix_trials['CURRENT_FIX_NEAREST_INTEREST_AREA_DISTANCE'] < 2)

# find trials w/ repeated fixation on target
repeated_trials = target_fix_trials.groupby(['par_ID', 'TRIAL_INDEX']).size().reset_index(name='num_of_repeat')

# merge back to target fix trials
target_fix_trials = pd.merge(target_fix_trials, repeated_trials, how='left', on=['par_ID', 'TRIAL_INDEX'])

# more filters
rep_fix_filter = target_fix_trials['num_of_repeat'] != 1


repeated_fix_trials = target_fix_trials[rep_fix_filter]
single_fix_trials = target_fix_trials[~rep_fix_filter]

# eye_sem_cond = single_fixations.groupby(['par_ID', 'condition'])['CURRENT_FIX_DURATION'].mean()
# eye_rm_anova = eye_sem_cond.reset_index()
# print(eye_rm_anova)
# ANOVA_RT = pg.rm_anova(data=eye_rm_anova, dv='CURRENT_FIX_DURATION', within='condition', subject='par_ID').round(4)
#
# pairwise_results_RT = pg.pairwise_ttests(data=eye_rm_anova, dv='CURRENT_FIX_DURATION', within='condition',
#                                          subject='par_ID', marginal=True, padjust='bonf')
#
# print('RT based on semantic conditions \n', eye_sem_cond.unstack().mean(), '\n')
# print(ANOVA_RT)
# print(pairwise_results_RT)


## NOTES from Malcolm 2009 and 2010

# Proportion of Trials in Which Target ROI Was Fixated First
# Search initiation time = time from appearance of the search scene until the first saccade away from the initial fixation point (the initial saccade latency) and measures the time needed to begin search.
# Scanning time = time from the end of first saccade to first fixation on the target object and represents the actual search process.
# Verification time = participant’s gaze duration on the target object, reflecting the time needed to decide that the fixated object is actually the target.
# Total trial duration = the RT measure reported in most previous visual search studies, is equal to the sum of these three epochs (Figure 1).