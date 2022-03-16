import os
import pandas as pd
import numpy as np
import my_functions
import pingouin as pg
import time
import matplotlib.pyplot as plt
import seaborn as sns

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

# 1. Calculate total fixation duration - Joy (done)
# last fixation duration - Joy
# whether they looked at the distractors and how long they looked at it
# how many times they looked at an object (2 different looks, 0 looks)
# useful to know what difference it makes if it's 1 look or 2, but
# compare nearest and target just report the visual angle

# Change type for numeric columns
accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'] = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'].replace(['.'], 0)
accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'] = pd.to_numeric(accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'])

# Calculate total accurate trials
total_trials = accurate_eye_df.drop_duplicates(['par_ID', 'TRIAL_INDEX'])
total_trial_count = total_trials.groupby(['par_ID'])['TRIAL_INDEX'].count().reset_index(name='total_acc_trials')

# Fixation on object (filter for separate interest areas)
target_fix_filter = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'target_obj'
pair_fix_filter = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'pair_obj'
neutral_fix_filter = (accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'neutral1_obj') | (accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'neutral2_obj')

first_fix_filter = accurate_eye_df['CURRENT_FIX_INDEX'] == 2 # 1 is fixation

# apply filter and grab appropriate trials
target_fix_trials = accurate_eye_df[target_fix_filter]
pair_fix_trials = accurate_eye_df[pair_fix_filter]
neutral_fix_trials = accurate_eye_df[neutral_fix_filter]

# Get target dwell time
target_dwell_time = target_fix_trials[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME']]
target_dwell_time.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'target_fix_dur'}, inplace=True)
target_dwell_time = target_dwell_time.drop_duplicates()
target_dwell_time.set_index('par_ID', inplace=True)
target_dwell_time_RT = target_dwell_time.groupby(['par_ID', 'condition'])['target_fix_dur'].mean().unstack()

# Get Pair dwell time
pair_dwell_time = pair_fix_trials[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME']]
pair_dwell_time.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'pair_fix_dur'}, inplace=True)
pair_dwell_time = pair_dwell_time.drop_duplicates()
pair_dwell_time.set_index('par_ID', inplace=True)
pair_dwell_time_RT = pair_dwell_time.groupby(['par_ID', 'condition'])['pair_fix_dur'].mean().unstack()

# Get Neutral Pair dwell time
neutral_dwell_time = neutral_fix_trials[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME']]
neutral_dwell_time.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'neu_fix_dur'}, inplace=True)
neutral_dwell_time = neutral_dwell_time.drop_duplicates()
neutral_dwell_time.set_index('par_ID', inplace=True)
neutral_dwell_time_RT = neutral_dwell_time.groupby(['par_ID', 'condition'])['neu_fix_dur'].mean().unstack()

# Combine all three conditions
all_RT = pd.concat([target_dwell_time_RT, pair_dwell_time_RT, neutral_dwell_time_RT], axis=1)
all_RT.columns = ['target_neu_RT', 'target_tax_RT', 'target_thm_RT', 'pair_neu_RT', 'pair_tax_RT', 'pair_thm_RT',
                  'neu-pair_neu_RT', 'neu-pair_tax_RT', 'neu-pair_thm_RT']
all_RT.to_clipboard()


print(f'\nShall we draw graphs now?')


# Figure setup
colors = ["#FF221A", "#6A9551", "#D2AC3A"]
conditions_x = ['Neutral', 'Taxonomic', 'Thematic']
conditions = 3
sns.set_palette(sns.color_palette(colors))
sns.set_context('talk')
sns.set_style('white')
fig_0, axes_0 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

# Graph Parameters
errbar_color = 'black'
errbar_line_width = 2
errbar_capsize = 5
errbar_capthick = 2
font_color = 'black'
trans_param = False

# figure 1a
# Figure 1a, data
f_RT_means = RT_df.unstack().mean()
f_sem_RT_means = RT_df.unstack().sem()

# Draw graph and error bar
axes_0[0].bar(np.arange(conditions), f_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_0[0].errorbar(np.arange(conditions), f_RT_means, yerr=f_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_0[0].set_title('RT for Visual Search', size=20, color=font_color)

# x axis stuff
axes_0[0].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_0, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

# y-axis stuff
axes_0[0].set_ylabel('RT (ms)', color=font_color)

# Figure 1b
# Data
f_ACC_means = accuracy_df.unstack().mean()
f_sem_ACC_means = accuracy_df.unstack().sem()

# Draw graph and error bar
axes_0[1].bar(np.arange(conditions), f_ACC_means, color=colors, edgecolor='black', linewidth=2)
axes_0[1].errorbar(np.arange(conditions), f_ACC_means, yerr=f_sem_ACC_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_0[1].set_title('Accuracy for Visual Search', size=20, color=font_color)

# x axis stuff
axes_0[1].set_xlabel('Semantic Conditions', color=font_color)

# y-axis stuff
axes_0[1].set_ylabel('Accuracy (%)', color=font_color)
axes_0[1].set_ylim(0, 100)

sns.despine()
plt.tight_layout(h_pad=2.0)
plt.savefig('f_RT-ACC.png', transparent=trans_param)
plt.show()


# Figure 2a setup
fig_1, axes_1 = plt.subplots(figsize=(14, 6), nrows=1, ncols=3)

# data
target_RT_means = target_dwell_time_RT.mean()
target_sem_RT_means = target_dwell_time_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_1[0].bar(np.arange(conditions), target_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_1[0].errorbar(np.arange(conditions), target_RT_means, yerr=target_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_1[0].set_title('Target Dwell Time', size=20, color=font_color)

# x axis stuff
# axes_1[0].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

# y-axis stuff
axes_1[0].set_ylabel('RT (ms)', color=font_color)

# Figure 2b, data
pair_RT_means = pair_dwell_time_RT.mean()
pair_sem_RT_means = pair_dwell_time_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_1[1].bar(np.arange(conditions), pair_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_1[1].errorbar(np.arange(conditions), pair_RT_means, yerr=pair_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_1[1].set_title('Sem Pair Dwell Time', size=20, color=font_color)

# x axis stuff
# axes_1[1].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)


# Figure 2c, data
neutral_RT_means = neutral_dwell_time_RT.mean()
neutral_sem_RT_means = neutral_dwell_time_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_1[2].bar(np.arange(conditions), neutral_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_1[2].errorbar(np.arange(conditions), neutral_RT_means, yerr=neutral_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_1[2].set_title('Neutral Distractor Dwell Time', size=20, color=font_color)

# x axis stuff
# axes_1[2].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)


# Add # of participants to graph
axes_1[2].text(2, -100, 'n = ' + str(total_par), color=font_color)

# limit y axis
axes_1[0].set_ylim(0, 650)
axes_1[1].set_ylim(0, 650)
axes_1[2].set_ylim(0, 650)

# Finalize and print
sns.despine()
plt.tight_layout(h_pad=2.0)
plt.savefig('f_fix_RT.png', transparent=trans_param)
plt.show()

## NOTES from Malcolm 2009 and 2010

# Proportion of Trials in Which Target ROI Was Fixated First
# Search initiation time = time from appearance of the search scene until the first saccade away from the initial fixation point (the initial saccade latency) and measures the time needed to begin search.
# Scanning time = time from the end of first saccade to first fixation on the target object and represents the actual search process.
# Verification time = participant’s gaze duration on the target object, reflecting the time needed to decide that the fixated object is actually the target.
# Total trial duration = the RT measure reported in most previous visual search studies, is equal to the sum of these three epochs (Figure 1).