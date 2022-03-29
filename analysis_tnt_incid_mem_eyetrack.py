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
    print('RT based on semantic conditions \n', RT_df.unstack().mean(), '\n')
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

# 1. Calculate total fixation duration - Joy (done)
# last fixation duration - Joy (done)
# whether they looked at the distractors and how long they looked at it (done)
# how many times they looked at an object (2 different looks, 0 looks) (done)
# Accurate as of 2022-03-29
# useful to know what difference it makes if it's 1 look or 2, but
# compare nearest and target just report the visual angle

# Change type for numeric columns
accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'] = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'].replace(['.'], 0)
accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'] = pd.to_numeric(accurate_eye_df['CURRENT_FIX_INTEREST_AREA_DWELL_TIME'])

# set counter for total fixation count
accurate_eye_df['total_fixation_count'] = 1
# Calculate total accurate trials
total_trials = accurate_eye_df.drop_duplicates(['par_ID', 'TRIAL_INDEX'])
total_trial_count = total_trials.groupby(['par_ID'])['TRIAL_INDEX'].count().reset_index(name='total_acc_trials')

# Fixation on object (filter for separate interest areas)
target_fix_filter = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'target_obj'
pair_fix_filter = accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'pair_obj'
neutral_fix_filter = (accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'neutral1_obj') | (accurate_eye_df['CURRENT_FIX_INTEREST_AREA_LABEL'] == 'neutral2_obj')

first_fix_filter = accurate_eye_df['CURRENT_FIX_INDEX'] == 2 # 1 is fixation

# apply filter and grab appropriate trials
target_fix_trials_raw = accurate_eye_df[target_fix_filter]
pair_fix_trials_raw = accurate_eye_df[pair_fix_filter]
neutral_fix_trials_raw = accurate_eye_df[neutral_fix_filter]

# Get target fixation trials
target_fix_trials = target_fix_trials_raw[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'total_fixation_count']]
target_fix_trials.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'target_fix_dur'}, inplace=True)

# Figure out fixation count on target (Replace with dedicated column)
target_fixation_count = target_fix_trials.groupby(['par_ID','TRIAL_INDEX'])['total_fixation_count'].count().reset_index(name='total_fixation_count')
target_fix_trials = target_fix_trials.drop_duplicates()
target_fix_trials = target_fix_trials.drop(columns=['total_fixation_count'])
target_fix_trials = pd.merge(target_fix_trials, target_fixation_count, how='inner', on=['par_ID', 'TRIAL_INDEX'])
target_fix_trials.set_index('par_ID', inplace=True)

target_fix_RT = target_fix_trials.groupby(['par_ID', 'condition'])['target_fix_dur'].mean().unstack()
target_fix_count = target_fix_trials.groupby(['par_ID', 'condition'])['total_fixation_count'].mean().unstack()

# Get semantic pair fixation trials
pair_fix_trials = pair_fix_trials_raw[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'total_fixation_count']]
pair_fix_trials.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'pair_fix_dur'}, inplace=True)

# Figure out fixation count on semantic pair
pair_fixation_count = pair_fix_trials.groupby(['par_ID','TRIAL_INDEX'])['total_fixation_count'].count().reset_index(name='total_fixation_count')
pair_fix_trials = pair_fix_trials.drop_duplicates()
pair_fix_trials = pair_fix_trials.drop(columns=['total_fixation_count'])
pair_fix_trials = pd.merge(pair_fix_trials, pair_fixation_count, how='inner', on=['par_ID', 'TRIAL_INDEX'])
pair_fix_trials.set_index('par_ID', inplace=True)

# Get meaningful data
pair_fix_trials_RT = pair_fix_trials.groupby(['par_ID', 'condition'])['pair_fix_dur'].mean().unstack()
pair_fix_count = pair_fix_trials.groupby(['par_ID', 'condition'])['total_fixation_count'].mean().unstack()

# Get Neutral Pair dwell time
neutral_fix_trials = neutral_fix_trials_raw[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INTEREST_AREA_DWELL_TIME', 'total_fixation_count']]
neutral_fix_trials.rename(columns={'CURRENT_FIX_INTEREST_AREA_DWELL_TIME': 'neu_fix_dur'}, inplace=True)

# Get neutral distractor fixation trials
neutral_fixation_count = neutral_fix_trials.groupby(['par_ID','TRIAL_INDEX'])['total_fixation_count'].count().reset_index(name='total_fixation_count')
neutral_fix_trials = neutral_fix_trials.drop_duplicates()
neutral_fix_trials = neutral_fix_trials.drop(columns=['total_fixation_count'])
neutral_fix_trials = pd.merge(neutral_fix_trials, neutral_fixation_count, how='inner', on=['par_ID', 'TRIAL_INDEX'])
neutral_fix_trials.set_index('par_ID', inplace=True)

neutral_fix_trials_RT = neutral_fix_trials.groupby(['par_ID', 'condition'])['neu_fix_dur'].mean().unstack()
neutral_fix_count = neutral_fix_trials.groupby(['par_ID', 'condition'])['total_fixation_count'].mean().unstack()

# Combine all three conditions
all_RT = pd.concat([target_fix_RT, pair_fix_trials_RT, neutral_fix_trials_RT], axis=1)
all_RT.columns = ['target_neu_RT', 'target_tax_RT', 'target_thm_RT', 'pair_neu_RT', 'pair_tax_RT', 'pair_thm_RT',
                  'neu-pair_neu_RT', 'neu-pair_tax_RT', 'neu-pair_thm_RT']
all_RT.to_clipboard()

# ANOVA for RT and acc
target_dwell_anova = target_fix_trials.groupby(['par_ID', 'condition'])['target_fix_dur'].mean().reset_index()
target_dwell_ANOVA_RT = pg.rm_anova(data=target_dwell_anova, dv='target_fix_dur', within='condition', subject='par_ID').round(2)

target_pairwise_results_RT = pg.pairwise_ttests(data=target_dwell_anova, dv='target_fix_dur', within='condition',
                                         subject='par_ID', marginal=True, padjust='bonf')

pair_dwell_anova = pair_fix_trials.groupby(['par_ID', 'condition'])['pair_fix_dur'].mean().reset_index()
pair_dwell_ANOVA_RT = pg.rm_anova(data=pair_dwell_anova, dv='pair_fix_dur', within='condition', subject='par_ID').round(2)

pair_pairwise_results_RT = pg.pairwise_ttests(data=pair_dwell_anova, dv='pair_fix_dur', within='condition',
                                         subject='par_ID', marginal=True, padjust='bonf')

neutral_dwell_anova = neutral_fix_trials.groupby(['par_ID', 'condition'])['neu_fix_dur'].mean().reset_index()
neutral_dwell_ANOVA_RT = pg.rm_anova(data=neutral_dwell_anova, dv='neu_fix_dur', within='condition', subject='par_ID').round(2)

# ANOVA for fixation count
target_fix_count_anova = target_fix_trials.groupby(['par_ID', 'condition'])['total_fixation_count'].mean().reset_index()
target_fix_count_ANOVA = pg.rm_anova(data=target_fix_count_anova, dv='total_fixation_count', within='condition', subject='par_ID').round(2)

pair_fix_count_anova = pair_fix_trials.groupby(['par_ID', 'condition'])['total_fixation_count'].mean().reset_index()
pair_fix_count_ANOVA = pg.rm_anova(data=pair_fix_count_anova, dv='total_fixation_count', within='condition', subject='par_ID').round(2)

neutral_fix_count_anova = neutral_fix_trials.groupby(['par_ID', 'condition'])['total_fixation_count'].mean().reset_index()
neutral_fix_count_ANOVA = pg.rm_anova(data=neutral_fix_count_anova, dv='total_fixation_count', within='condition', subject='par_ID').round(2)
neutral_fix_pairwise_results = pg.pairwise_ttests(data=neutral_fix_count_anova, dv='total_fixation_count', within='condition',
                                         subject='par_ID', marginal=True, padjust='bonf')
# Print data and results
print('Dwell time on targets \n', target_fix_RT.mean(), '\n')
print('ANOVA for target dwell time \n', target_dwell_ANOVA_RT, '\n\n')
print("Pairwise testings for RT \n", target_pairwise_results_RT, '\n')

print('Dwell time on semantic pairs \n', pair_fix_trials_RT.mean(), '\n')
print('ANOVA for semantic pair dwell time \n', pair_dwell_ANOVA_RT, '\n\n')
print("Pairwise testings for RT \n", pair_pairwise_results_RT, '\n')

print('Dwell time on neutral distractor \n', neutral_fix_trials_RT.mean(), '\n')
print('ANOVA for neutral distractor dwell time \n', neutral_dwell_ANOVA_RT, '\n\n')


# Print data and results
print('ANOVA for target fixation count \n', target_fix_count_ANOVA, '\n\n')
print('ANOVA for semantic pair fixation count \n', pair_fix_count_ANOVA, '\n\n')
print('ANOVA for neutral distractor fixation count \n', neutral_fix_count_ANOVA, '\n\n')


# Calculate
# Final
# Fixation
# Duration

target_final_fix_trials = target_fix_trials_raw[['par_ID', 'TRIAL_INDEX', 'condition', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_DURATION', 'TRIAL_FIXATION_TOTAL']]
target_final_fix_trials = target_final_fix_trials.drop_duplicates(['par_ID', 'TRIAL_INDEX'], keep='last')
target_final_fix_trials = target_final_fix_trials[(target_final_fix_trials['CURRENT_FIX_INDEX'] - target_final_fix_trials['TRIAL_FIXATION_TOTAL'] == 0) | (target_final_fix_trials['CURRENT_FIX_INDEX'] - target_final_fix_trials['TRIAL_FIXATION_TOTAL'] == -1)]
# target_final_fix_trials = target_final_fix_trials[(target_final_fix_trials['CURRENT_FIX_INDEX'] - target_final_fix_trials['TRIAL_FIXATION_TOTAL'] == 0)]
target_final_fix_trials.set_index('par_ID', inplace=True)

target_final_fix_RT = target_final_fix_trials.groupby(['par_ID', 'condition'])['CURRENT_FIX_DURATION'].mean().unstack()

target_final_fix_anova = target_final_fix_trials.groupby(['par_ID', 'condition'])['CURRENT_FIX_DURATION'].mean().reset_index()

target_final_fix_ANOVA_RT = pg.rm_anova(data=target_final_fix_anova, dv='CURRENT_FIX_DURATION', within='condition', subject='par_ID').round(2)
target_final_fix_pairwise_results_RT = pg.pairwise_ttests(data=target_final_fix_anova, dv='CURRENT_FIX_DURATION', within='condition',
                                         subject='par_ID', marginal=True, padjust='bonf')

# target_final_fix_RT.to_clipboard()

print('Final Fixation duration (ms) \n', target_final_fix_RT.mean(), '\n')
print('ANOVA for Final Fixation duration \n', target_final_fix_ANOVA_RT, '\n\n')
print("Pairwise testings for RT \n", target_final_fix_pairwise_results_RT, '\n')

print(f'\nShall we draw graphs now?')

target_final_fix_ANOVA_RT.to_clipboard()
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
props = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 20, 'shrinkB': 20, 'linewidth': 2,
             "color": font_color}
props2 = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 25, 'shrinkB': 25, 'linewidth': 2,
             "color": font_color}
props3 = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 40, 'shrinkB': 40, 'linewidth': 2,
             "color": font_color}
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

# Draw significance bars
axes_0[0].text(0.5, 1380, '*', size=20, color=font_color)
axes_0[0].annotate('', xy=(0, 1250), xytext=(1, 1250), arrowprops=props2)
axes_0[0].text(1.5, 1380, '*', size=20, color=font_color)
axes_0[0].annotate('', xy=(1, 1200), xytext=(2, 1200), arrowprops=props2)

axes_0[0].set_ylim(0, 1500)

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
axes_0[1].set_ylim(0, 110)

# Draw significance bars
axes_0[1].annotate('', xy=(0, 85), xytext=(1, 85), arrowprops=props)
axes_0[1].text(1.25, 100, '*', size=20, color=font_color)
axes_0[1].annotate('', xy=(.5, 85), xytext=(2, 85), arrowprops=props3)
sns.despine()
plt.tight_layout(h_pad=2.0)
plt.savefig('f_RT-ACC.png', transparent=trans_param)
plt.show()


# Figure 2a setup
fig_1, axes_1 = plt.subplots(figsize=(14, 6), nrows=1, ncols=3)

# data
target_RT_means = target_fix_RT.mean()
target_sem_RT_means = target_fix_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_1[0].bar(np.arange(conditions), target_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_1[0].errorbar(np.arange(conditions), target_RT_means, yerr=target_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_1[0].set_title('Target Fix. Time', size=20, color=font_color)

# x axis stuff
# axes_1[0].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

# y-axis stuff
axes_1[0].set_ylabel('Fixation Duration (ms)', color=font_color)

# Draw significance bars
axes_1[0].text(0.75, 670, '*', size=20, color=font_color)
axes_1[0].annotate('', xy=(0, 590), xytext=(1.5, 590), arrowprops=props2)
axes_1[0].annotate('', xy=(1, 580), xytext=(2, 580), arrowprops=props)

# Figure 2b, data
pair_RT_means = pair_fix_trials_RT.mean()
pair_sem_RT_means = pair_fix_trials_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_1[1].bar(np.arange(conditions), pair_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_1[1].errorbar(np.arange(conditions), pair_RT_means, yerr=pair_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_1[1].set_title('Sem Pair Fix. Time', size=20, color=font_color)

# x axis stuff
# axes_1[1].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

# Draw significance bars
axes_1[1].text(0.5, 350, '*', size=20, color=font_color)
axes_1[1].annotate('', xy=(0, 250), xytext=(.95, 250), arrowprops=props)
axes_1[1].text(1.5, 350, '*', size=20, color=font_color)
axes_1[1].annotate('', xy=(1.05, 250), xytext=(2, 250), arrowprops=props)

# Figure 2c, data
neutral_RT_means = neutral_fix_trials_RT.mean()
neutral_sem_RT_means = neutral_fix_trials_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_1[2].bar(np.arange(conditions), neutral_RT_means, color=colors, edgecolor='black', linewidth=2)
axes_1[2].errorbar(np.arange(conditions), neutral_RT_means, yerr=neutral_sem_RT_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_1[2].set_title('Neutral Distractor Fix. Time', size=20, color=font_color)

# x axis stuff
# axes_1[2].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)


# Add # of participants to graph
axes_1[2].text(2, -100, 'n = ' + str(total_par), color=font_color)

# limit y axis
axes_1[0].set_ylim(0, 700)
axes_1[1].set_ylim(0, 700)
axes_1[2].set_ylim(0, 700)

# Finalize and print
sns.despine()
plt.tight_layout(h_pad=2.0)
plt.savefig('f_fix_RT.png', transparent=trans_param)
plt.show()

# Figure 3a setup
fig_2, axes_2 = plt.subplots(figsize=(14, 6), nrows=1, ncols=3)

# data
target_fix_means = target_fix_count.mean()
target_sem_fix_means = target_fix_count.sem() # need to fix to within participants)

# Draw graph and error bar
axes_2[0].bar(np.arange(conditions), target_fix_means, color=colors, edgecolor='black', linewidth=2)
axes_2[0].errorbar(np.arange(conditions), target_fix_means, yerr=target_sem_fix_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_2[0].set_title('Target Fix. Count', size=20, color=font_color)

# x axis stuff
plt.setp(axes_2, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

# y-axis stuff
axes_2[0].set_ylabel('Fixation Count', color=font_color)

# Figure 2b, data
pair_fix_means = pair_fix_count.mean()
pair_sem_fix_means = pair_fix_count.sem() # need to fix to within participants)

# Draw graph and error bar
axes_2[1].bar(np.arange(conditions), pair_fix_means, color=colors, edgecolor='black', linewidth=2)
axes_2[1].errorbar(np.arange(conditions), pair_fix_means, yerr=pair_sem_fix_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_2[1].set_title('Sem Pair Fix. Count', size=20, color=font_color)

# x axis stuff
# axes_1[1].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_2, xticks=[i for i in range(conditions)], xticklabels=conditions_x)


# Figure 2c, data
neutral_fix_means = neutral_fix_count.mean()
neutral_sem_fix_means = neutral_fix_count.sem() # need to fix to within participants)

# Draw graph and error bar
axes_2[2].bar(np.arange(conditions), neutral_fix_means, color=colors, edgecolor='black', linewidth=2)
axes_2[2].errorbar(np.arange(conditions), neutral_fix_means, yerr=neutral_sem_fix_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_2[2].set_title('Neu. Distractor Fix. Count', size=20, color=font_color)

# x axis stuff
# axes_1[2].set_xlabel('Semantic Conditions', color=font_color)
plt.setp(axes_2, xticks=[i for i in range(conditions)], xticklabels=conditions_x)


# Add # of participants to graph
axes_2[2].text(2, -.3, 'n = ' + str(total_par), color=font_color)

# limit y axis
axes_2[0].set_ylim(0, 2)
axes_2[1].set_ylim(0, 2)
axes_2[2].set_ylim(0, 2)

# Finalize and print
sns.despine()
plt.tight_layout(h_pad=2.0)
plt.savefig('f_fix_count.png', transparent=trans_param)
plt.show()

# Figure 4 setup
fig_3, axes_3 = plt.subplots(figsize=(6, 6), nrows=1, ncols=1)

# data
target_final_fix_means = target_final_fix_RT.mean()
target_sem_final_fix_means = target_final_fix_RT.sem() # need to fix to within participants)

# Draw graph and error bar
axes_3.bar(np.arange(conditions), target_final_fix_means, color=colors, edgecolor='black', linewidth=2)
axes_3.errorbar(np.arange(conditions), target_final_fix_means, yerr=target_sem_final_fix_means, fmt=' ', ecolor=errbar_color,
                   elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

# title stuff
axes_3.set_title('Final Fixation Duration', size=20, color=font_color)

# x axis stuff
plt.setp(axes_3, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

# y-axis stuff
axes_3.set_ylabel('Fixation Duration (ms)', color=font_color)

# Add # of participants to graph
axes_3.text(2, -100, 'n = ' + str(total_par), color=font_color)

# limit y axis
axes_3.set_ylim(0, 650)

# Finalize and print
sns.despine()
plt.tight_layout(h_pad=2.0)
plt.savefig('f_final_fix_dur.png', transparent=trans_param)
plt.show()

## NOTES from Malcolm 2009 and 2010

# Proportion of Trials in Which Target ROI Was Fixated First
# Search initiation time = time from appearance of the search scene until the first saccade away from the initial fixation point (the initial saccade latency) and measures the time needed to begin search.
# Scanning time = time from the end of first saccade to first fixation on the target object and represents the actual search process.
# Verification time = participant’s gaze duration on the target object, reflecting the time needed to decide that the fixated object is actually the target.
# Total trial duration = the RT measure reported in most previous visual search studies, is equal to the sum of these three epochs (Figure 1).