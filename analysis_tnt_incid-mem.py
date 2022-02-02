# extract data for ROC curve analysis
import pandas as pd
import glob
# import testable_analysis
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# import metrics to calculate area under the curve (AUC)
from sklearn import metrics

# Set pandas options
pd.set_option('display.max_columns', 37)
# disable chained assignment warning
pd.options.mode.chained_assignment = None

# bring up prompt to draw graphs or not
# print('Draw Graphs? (1 for yes): ')
draw_graph = 1

# directory to data
data_dir = '/Users/joecool890/Dropbox/UC-Davis/projects/tnt_incid-mem/raw-data/*.csv'

# Graph Parameters
errbar_color = 'black'
errbar_line_width = 2
errbar_capsize = 5
errbar_capthick = 2
font_color = 'black'
trans_param = False

# load all files
csv_paths = glob.glob(data_dir)
exp_ver = csv_paths[0][-24:-18]

participants = len(csv_paths)
df = []

for file in range(participants):
    data = pd.read_csv(csv_paths[file], skiprows=3)
    data['par_num'] = csv_paths[file][-10:-4]
    data['par_date'] = csv_paths[file][-17:-11]
    df.append(data)

raw_df = pd.concat(df, sort=True)
# Set par_num as index
raw_df.set_index('par_num', inplace=True)
raw_df.to_clipboard()

# Setup demographics dataframe
demographics = raw_df['trial_type'] == 'demographics'
demographics_df = raw_df[demographics].dropna(axis=1, how='all')

# Setup visual search trials
visual_search_trials = raw_df['trial_num'] > 0
visual_search_df = raw_df[visual_search_trials].dropna(axis=1, how='all')

# DF for all correct trials (for RT)
accuracy_filter = visual_search_df['correct'] == 1
corr_visual_search_df = visual_search_df[accuracy_filter]

# Visual Search DF for RT
corr_sem_condition_group = corr_visual_search_df.groupby(['par_num', 'sem_condition'])
corr_visual_search_df_group = corr_sem_condition_group['RT'].mean().unstack()
corr_visual_search_df_group.columns = ['RT_neu', 'RT_tax', 'RT_thm']

# Visual Search DF for accuracy
sem_condition_group = visual_search_df.groupby(['par_num', 'sem_condition'])
visual_search_df_group = sem_condition_group['correct'].mean().unstack() * 100
visual_search_df_group.columns = ['acc_neu', 'acc_tax', 'acc_thm']

# File output as csv for JASP
csv_output = pd.concat([corr_visual_search_df_group, visual_search_df_group], axis=1)
csv_output.to_clipboard()

# DF for Statistical Analysis (RT)
anova_corr_visual_search_df_group = corr_sem_condition_group['RT'].mean()
anova_corr_visual_search_df_group = anova_corr_visual_search_df_group.reset_index()

# RM ANOVA for RT
ANOVA_RT = pg.rm_anova(data=anova_corr_visual_search_df_group, dv='RT', within='sem_condition',
                       subject='par_num').round(4)
pairwise_results_RT = pg.pairwise_ttests(data=anova_corr_visual_search_df_group, dv='RT', within='sem_condition',
                                         subject='par_num', marginal=True, padjust='bonf')

print('now I present to thee, the holy ANOVA for RT:')
print(ANOVA_RT)
print('')
# Pairwise t-tests
print('followed by pairwise t-tests:')
print(pairwise_results_RT)

# DF for statistical analysis (Accuracy)
anova_visual_search_df_group = sem_condition_group['correct'].mean() * 100
anova_visual_search_df_group = anova_visual_search_df_group.reset_index()

# RM ANOVA for ACC
ANOVA_ACC = pg.rm_anova(data=anova_visual_search_df_group, dv='correct', within='sem_condition',
                        subject='par_num').round(4)
pairwise_results_ACC = pg.pairwise_ttests(data=anova_visual_search_df_group, dv='correct', within='sem_condition',
                                          subject='par_num', marginal=True, padjust='bonf')

print('')
print('Next, the holy ANOVA for Accuracy')
print(ANOVA_ACC)
print('')
# Pairwise t-tests
print('followed by pairwise t-tests')
print(pairwise_results_ACC)

# This section calculates ROC curve with memory trials
# Prep memory trials data frame
memory_trials = raw_df['mem_trials'] == 2
memory_df = raw_df[memory_trials].dropna(axis=1, how='all')

# Mark participants w/ final accurate visual search trials
accurate_final_trial_filter = (visual_search_df['correct'] == 1) & (visual_search_df['trial_num'] == 89)
accurate_final_trial_df = visual_search_df.loc[accurate_final_trial_filter]
final_trial_df = visual_search_df[accurate_final_trial_filter]

# index for correct final trials to find participants only correct on final trials
final_trial_index = final_trial_df.index
memory_df = memory_df.loc[final_trial_index]

# # re-write orig_stim location based on experimenter error for participants on certain dates
if exp_ver == '392251':
    memory_keep_orig_stim = memory_df[(memory_df['par_date'] != '220128') & (memory_df['par_date'] != '220129') & (memory_df['par_date'] != '220130') & (memory_df['par_date'] != '220131') & (memory_df['par_date'] != '220201')]
    memory_rewrite_orig_stim = memory_df[(memory_df['par_date'] == '220128') | (memory_df['par_date'] == '220129') | (memory_df['par_date'] == '220130') | (memory_df['par_date'] == '220131') | (memory_df['par_date'] == '220201')]

    # find subject group
    sub_group_4 = memory_rewrite_orig_stim.subjectGroup == 'nsg:4'
    sub_group_6 = memory_rewrite_orig_stim.subjectGroup == 'nsg:6'
    sub_group_8 = memory_rewrite_orig_stim.subjectGroup == 'nsg:8'
    sub_group_12 = memory_rewrite_orig_stim.subjectGroup == 'nsg:12'
    sub_group_14 = memory_rewrite_orig_stim.subjectGroup == 'nsg:14'
    sub_group_21 = memory_rewrite_orig_stim.subjectGroup == 'nsg:21'
    sub_group_27 = memory_rewrite_orig_stim.subjectGroup == 'nsg:27'
    sub_group_29 = memory_rewrite_orig_stim.subjectGroup == 'nsg:29'
    sub_group_30 = memory_rewrite_orig_stim.subjectGroup == 'nsg:30'

    memory_rewrite_orig_stim.loc[sub_group_4, 'orig_stim_loc'] = 1
    memory_rewrite_orig_stim.loc[sub_group_6, 'orig_stim_loc'] = 1
    memory_rewrite_orig_stim.loc[sub_group_8, 'orig_stim_loc'] = 2
    memory_rewrite_orig_stim.loc[sub_group_12, 'orig_stim_loc'] = 2
    memory_rewrite_orig_stim.loc[sub_group_14, 'orig_stim_loc'] = 2
    memory_rewrite_orig_stim.loc[sub_group_21, 'orig_stim_loc'] = 2
    memory_rewrite_orig_stim.loc[sub_group_27, 'orig_stim_loc'] = 1
    memory_rewrite_orig_stim.loc[sub_group_29, 'orig_stim_loc'] = 1
    memory_rewrite_orig_stim.loc[sub_group_30, 'orig_stim_loc'] = 2

    memory_rewrite_orig_stim.to_clipboard()

    memory_df = pd.concat([memory_rewrite_orig_stim, memory_keep_orig_stim])

# Keep original response and create new familiarity column
memory_df['confidence_values'] = memory_df['response']

# In case orig_stim_loc doesn't exist
memory_df['orig_stim_loc'].fillna(0, inplace=True)

# Separate out when test item is on left or right
memory_objleft = memory_df['orig_stim_loc'] == 1
memory_objrght = memory_df['orig_stim_loc'] == 2

obj_left_df = memory_df[memory_objleft]
obj_rght_df = memory_df[memory_objrght]

# Set Dictionary for old/new + confidences to replace Testable's default response
conf_left_dict = {1: 'hc_old', 2: 'mc_old', 3: 'lc_old', 4: 'lc_new', 5: 'mc_new', 6: 'hc_new'}
conf_rght_dict = {1: 'hc_new', 2: 'mc_new', 3: 'lc_new', 4: 'lc_old', 5: 'mc_old', 6: 'hc_old'}

obj_left_df.replace({'confidence_values': conf_left_dict}, inplace=True)
obj_rght_df.replace({'confidence_values': conf_rght_dict}, inplace=True)

# Separate the two stimuli
obj_left_df['orig_stim'] = obj_left_df['stimList'].str.split(pat=';', expand=True)[0]
obj_left_df['flipped_stim'] = obj_left_df['stimList'].str.split(pat=';', expand=True)[1]

obj_rght_df['orig_stim'] = obj_rght_df['stimList'].str.split(pat=';', expand=True)[1]
obj_rght_df['flipped_stim'] = obj_rght_df['stimList'].str.split(pat=';', expand=True)[0]
# concat back into memory_df
memory_df = pd.concat([obj_left_df, obj_rght_df])

# Clean up the stimuli list by removing "_flip"
memory_df['test_image'] = memory_df['orig_stim'].str.replace('_flip', '')
memory_df.to_clipboard()

# Drop useless columns
# List of columns to drop
drop_list = [
    'rowNo', 'type', 'stimPos', 'stimFormat', 'stimPos_actual', 'ITI_f', 'ITI_fDuration', 'timestamp', 'button1',
    'button2', 'button3', 'button4', 'button5', 'button6', 'flipped_stim', 'orig_stim', 'stimList'
]
# Replace test_image with proper name dictionary
name_dict = {'AGUINEAP2': 'guinea_pig', 'AHEADSET': 'headset', 'Acougar': 'cougar',
             'Agardeningrak': 'garden_rake', 'Agardeningshe': 'garden_shears',
             '26421370.thl': 'handcuffs', 'AJOYSTIC1': 'joystick', 'AGARBCAN': 'garbage_can',
             '23070064.thl': 'bone', 'coffee': 'coffee_mach', 'ABAT': 'bat'}

memory_df.drop(columns=drop_list, inplace=True)
memory_df.replace({'test_image': name_dict}, inplace=True)
memory_df.to_clipboard()

# count # of responses
roc_df = memory_df.groupby(['sem_condition', 'test_image', 'confidence_values']).size().reset_index(name='counts')
roc_df = roc_df.pivot_table(values='counts', columns='confidence_values', index=['sem_condition', 'test_image']).reset_index()
roc_df = roc_df[['sem_condition', 'test_image', 'hc_old', 'mc_old', 'lc_old', 'lc_new', 'mc_new', 'hc_new']]
roc_df['total_trials'] = roc_df[['hc_old', 'mc_old', 'lc_old', 'lc_new', 'mc_new', 'hc_new']].sum(axis=1)
roc_df.fillna(0, inplace=True)
roc_df.to_clipboard()
# print(a)
roc_df['all_hits'] = roc_df[['hc_old', 'mc_old', 'lc_old']].sum(axis=1)
roc_df['all_faa'] = roc_df[['hc_new', 'mc_new', 'lc_new']].sum(axis=1)

roc_df['hit_rate'] = roc_df['all_hits'] / roc_df['total_trials']
roc_df['faa_rate'] = roc_df['all_faa'] / roc_df['total_trials']

roc_df.groupby(['sem_condition'])[['hc_old', 'mc_old', 'lc_old', 'hc_new', 'mc_new', 'lc_new']].sum()
roc_df.to_clipboard()

print(' ')
print('Total count of confidence values')
print(roc_df.groupby(['sem_condition'])[['hc_old', 'mc_old', 'lc_old', 'hc_new', 'mc_new', 'lc_new']].sum())
print(' ')

# hits and false alarm grouped by semantic conditions
roc_hfa_df_group = round(roc_df.groupby(['sem_condition'])['all_hits'].sum() / roc_df.groupby(['sem_condition'])[
    'total_trials'].sum() * 100, 2)
print(' ')
print('Hit rate by semantic groups')
print(roc_hfa_df_group)
print(' ')

# hits and false alarm grouped by semantic conditions
roc_hfa_df_group2 = roc_df.groupby(['sem_condition', 'test_image'])['all_hits'].sum() / \
                    roc_df.groupby(['sem_condition', 'test_image'])['total_trials'].sum() * 100
roc_hfa_df_group2.unstack().mean()
# roc_df.to_clipboard()

# Divide up into Hit and False Alarm DataFrames
roc_old_df = roc_df[['sem_condition', 'hc_old', 'mc_old', 'lc_old']]
roc_new_df = roc_df[['sem_condition', 'hc_new', 'mc_new', 'lc_new']]
# roc_df_prob = roc_df.div(30).cumsum(axis=1).reset_index

# calculate all # of trials of HITS and FALSE ALARMS and add as column
roc_old_df['total_count'] = roc_old_df[['hc_old', 'mc_old', 'lc_old']].sum(axis=1)
roc_new_df['total_count'] = roc_new_df[['hc_new', 'mc_new', 'lc_new']].sum(axis=1)

# Calculate Hit rate and false alarm rate
roc_old_df[['hc_old_prob', 'mc_old_prob', 'lc_old_prob']] = roc_old_df[['hc_old', 'mc_old', 'lc_old']].div(
    roc_old_df['total_count'].values, axis=0).cumsum(axis=1)
roc_new_df[['hc_new_prob', 'mc_new_prob', 'lc_new_prob']] = roc_new_df[['hc_new', 'mc_new', 'lc_new']].div(
    roc_new_df['total_count'].values, axis=0).cumsum(axis=1)

roc_old_df.to_clipboard()
print(a)
# set up DF for ROC data
roc_auc = pd.DataFrame(columns=['par_num', 'sem_condition', 'auc'])

# for p in roc_old_df.par_num.unique():
#     sub_old = roc_old_df[roc_old_df['par_num'] == p]
#     sub_new = roc_new_df[roc_new_df['par_num'] == p]
#     sub_new.to_clipboard()
#
#     neu_params_dict = {'par_num': p, 'sem_condition': 'neutral', 'auc': metrics.auc(sub_new.iloc[0, 6:10], sub_old.iloc[0, 6:10])}
#     neu_auc_df = pd.DataFrame(neu_params_dict, index=[0])
#
#     tax_params_dict = {'par_num': p, 'sem_condition': 'taxonomic', 'auc': metrics.auc(sub_new.iloc[1, 6:10], sub_old.iloc[1, 6:10])}
#     tax_auc_df = pd.DataFrame(tax_params_dict, index=[0])
#
#     thm_params_dict = {'par_num': p, 'sem_condition': 'thematic', 'auc': metrics.auc(sub_new.iloc[2, 6:10], sub_old.iloc[2, 6:10])}
#     thm_auc_df = pd.DataFrame(thm_params_dict, index=[0])
#
#     all_auc_df = neu_auc_df.append(tax_auc_df).append(thm_auc_df)
#
#     roc_auc = roc_auc.append(all_auc_df)
#
# auc_ANOVA = pg.rm_anova(data=roc_auc, dv='auc', within='sem_condition', subject='par_num')
# print('')
# print(auc_ANOVA)
# look at mean AUC for each condition
# print(roc_auc.groupby(['sem_condition']).mean())

# Graph ROC Curve
roc_all_df = pd.concat([roc_old_df[['hc_old_prob', 'mc_old_prob', 'lc_old_prob']], roc_new_df], axis=1)
roc_all_df.to_clipboard()
# print(a)
x = roc_all_df.groupby('sem_condition')['hc_new_prob', 'mc_new_prob', 'lc_new_prob'].mean().reset_index()
y = roc_all_df.groupby('sem_condition')['hc_old_prob', 'mc_old_prob', 'lc_old_prob'].mean().reset_index()
print(x)
print(y)
# print(a)
x_melt = x.melt(id_vars=['sem_condition'])
y_melt = y.melt(id_vars=['sem_condition'])

x_melt.replace({'hc_new_prob': 1, 'mc_new_prob': 2, 'lc_new_prob': 3}, inplace=True)
y_melt.replace({'hc_old_prob': 1, 'mc_old_prob': 2, 'lc_old_prob': 3}, inplace=True)

x_melt.rename(columns={'value': 'faa_rate'}, inplace=True)
y_melt.rename(columns={'value': 'hit_rate'}, inplace=True)

all_data = y_melt.merge(x_melt, on=['sem_condition', 'confidence_values'])
all_data.sort_values(['sem_condition', 'confidence_values'], ascending=[True, True], inplace=True)
# all_data.to_clipboard()
# Calculate Hit/False Alarm by Objects
roc_objects_df = memory_df.groupby(['sem_condition', 'confidence_values', 'test_image']).size().reset_index(name='counts')
roc_objects_df = roc_objects_df.pivot_table(values='counts', columns='confidence_values',
                                            index=['sem_condition', 'test_image']).reset_index()
roc_objects_df = roc_objects_df[['sem_condition', 'test_image', 'hc_old', 'mc_old', 'lc_old', 'lc_new', 'mc_new', 'hc_new']]
roc_objects_df.fillna(0, inplace=True)
roc_objects_df['total_hit_count'] = roc_objects_df[['hc_old', 'mc_old', 'lc_old']].sum(axis=1)
roc_objects_df['total_faa_count'] = roc_objects_df[['hc_new', 'mc_new', 'lc_new']].sum(axis=1)
roc_objects_df['hit_rate'] = roc_objects_df['total_hit_count'] / (
            roc_objects_df['total_hit_count'] + roc_objects_df['total_faa_count'])
roc_objects_df['faa_rate'] = roc_objects_df['total_faa_count'] / (
            roc_objects_df['total_hit_count'] + roc_objects_df['total_faa_count'])

tax_filt = roc_objects_df['sem_condition'] == 'taxonomic'
thm_filt = roc_objects_df['sem_condition'] == 'thematic'

tax_objects_df = roc_objects_df[tax_filt].sort_values('hit_rate', ascending=False)
thm_objects_df = roc_objects_df[thm_filt].sort_values('hit_rate', ascending=False)

if draw_graph == 1:
    # Graph Thematic Objects

    # Set seaborn defaults
    sns.set_context('poster')
    sns.set_theme(style='white')

    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw graph using seaborn
    sns.barplot(data=thm_objects_df, x='test_image', y='hit_rate')
    sns.despine()

    # Customize x axis
    plt.xlabel("Thematic Objects", fontsize=20)
    plt.xticks(rotation=45)
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='x', which='major', labelsize=12)
    # ax.margins(x=0)

    # customize y axis
    plt.ylabel("Hit Rate", fontsize=20)
    ax.set_ylim(0, 1)

    # Draw chance level
    ax.plot([0, 3], [0.5, 0.5], transform=ax.transAxes, color='red', dashes=(2, 1))
    plt.tight_layout(h_pad=2.0)
    plt.savefig('f_thm_hit-rate.png', transparent=trans_param)
    plt.show()

    # Draw Taxonomic Objects
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw graph using seaborn
    sns.barplot(data=tax_objects_df, x='test_image', y='hit_rate')
    sns.despine()

    # Customize x axis
    plt.xlabel("Taxonomic Objects", fontsize=20)
    plt.xticks(rotation=45)
    ax.tick_params(axis='y', which='major', labelsize=15)
    ax.tick_params(axis='x', which='major', labelsize=12)
    # ax.margins(x=0)

    # customize y axis
    plt.ylabel("Hit Rate", fontsize=20)
    ax.set_ylim(0, 1)

    # Draw chance level
    ax.plot([0, 3], [0.5, 0.5], transform=ax.transAxes, color='red', dashes=(2, 1))
    plt.tight_layout(h_pad=2.0)
    plt.savefig('f_tax_hit-rate.png', transparent=trans_param)
    plt.show()

    sns.set_context('poster')
    sns.set_style('white')

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.despine()

    all_data.to_clipboard()
    sns.scatterplot(data=all_data.query('sem_condition == "neutral"'), x='faa_rate',
                    y='hit_rate', color="#FF221A", ax=ax, edgecolor='black', zorder=11)
    sns.scatterplot(data=all_data.query('sem_condition == "taxonomic"'), x='faa_rate',
                    y='hit_rate', color="#6A9551", ax=ax, edgecolor='black', zorder=11)
    sns.scatterplot(data=all_data.query('sem_condition == "thematic"'), x='faa_rate',
                    y='hit_rate', color="#D2AC3A", ax=ax, edgecolor='black', zorder=12)

    sns.lineplot(data=all_data.query('sem_condition == "neutral"'),
                 x='faa_rate', y='hit_rate', color="#FF221A", ax=ax)
    sns.lineplot(data=all_data.query('sem_condition == "taxonomic"'),
                 x='faa_rate', y='hit_rate', color="#6A9551", ax=ax, dashes=(3, 1))
    sns.lineplot(data=all_data.query('sem_condition == "thematic"'),
                 x='faa_rate', y='hit_rate', color="#D2AC3A", ax=ax, dashes=(2, 1))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='gray', dashes=(2, 1))

    ax.lines[1].set_linestyle("--")
    ax.lines[2].set_linestyle(":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ax.set(title='ROCs')
    ax.set_xlabel("False Alarm Rate", color=font_color)
    ax.set_ylabel("Hit Rate", color=font_color)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='gray', dashes=(2, 1))
    ax.legend()
    plt.legend(title='Semantic Relationship', fontsize='15', title_fontsize='14',
               labels=['Neutral', 'Taxonomic', 'Thematic'])
    plt.savefig('f_ROC.png', transparent=trans_param)
    plt.show()

    # Figure 1 Parameters
    colors = ["#FF221A", "#6A9551", "#D2AC3A"]
    conditions_x = ['Neutral', 'Taxonomic', 'Thematic']
    conditions = 3
    sns.set_palette(sns.color_palette(colors))
    sns.set_context('talk')
    sns.set_style('white')
    fig_1, axes_1 = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

    # Figure 1a, data
    f_RT_means = corr_sem_condition_group['RT'].mean().unstack().mean()
    f_sem_RT_means = corr_sem_condition_group['RT'].mean().unstack().sem()

    # Draw graph and error bar
    axes_1[0].bar(np.arange(conditions), f_RT_means, color=colors, edgecolor='black', linewidth=2)
    axes_1[0].errorbar(np.arange(conditions), f_RT_means, yerr=f_sem_RT_means, fmt=' ', ecolor=errbar_color,
                       elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

    # title stuff
    axes_1[0].set_title('RT for Visual Search', size=20, color=font_color)

    # x axis stuff
    axes_1[0].set_xlabel('Semantic Conditions', color=font_color)
    plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

    # y-axis stuff
    axes_1[0].set_ylabel('RT (ms)', color=font_color)

    # Figure 1b
    # Data
    f_ACC_means = sem_condition_group['correct'].mean().unstack().mean() * 100
    f_sem_ACC_means = sem_condition_group['correct'].mean().unstack().sem() * 100

    # Draw graph and error bar
    axes_1[1].bar(np.arange(conditions), f_ACC_means, color=colors, edgecolor='black', linewidth=2)
    axes_1[1].errorbar(np.arange(conditions), f_ACC_means, yerr=f_sem_ACC_means, fmt=' ', ecolor=errbar_color,
                       elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

    # title stuff
    axes_1[1].set_title('Accuracy for Visual Search', size=20, color=font_color)

    # x axis stuff
    axes_1[1].set_xlabel('Semantic Conditions', color=font_color)

    # y-axis stuff
    axes_1[1].set_ylabel('Accuracy (%)', color=font_color)

    # Add # of participants to graph
    axes_1[1].text(2, -20, 'n = ' + str(participants), color=font_color)

    # Finalize and print
    sns.despine()
    plt.tight_layout(h_pad=2.0)
    plt.savefig('f_RT-ACC.png', transparent=trans_param)
    plt.show()

    # Figure SOM (hit and false alarm rates)
    fig_SOM, axes_SOM = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

    # Data
    f_hit_means = roc_hfa_df.groupby(['par_num', 'sem_condition'])[['hit_rate']].mean().unstack().mean()
    f_hit_sem = roc_hfa_df.groupby(['par_num', 'sem_condition'])[['hit_rate']].mean().unstack().sem()

    f_faa_means = roc_hfa_df.groupby(['par_num', 'sem_condition'])[['faa_rate']].mean().unstack().mean()
    f_faa_sem = roc_hfa_df.groupby(['par_num', 'sem_condition'])[['faa_rate']].mean().unstack().sem()

    # Draw graph and error bar
    axes_SOM[0].bar(np.arange(conditions), f_hit_means, color=colors, edgecolor='black', linewidth=2)
    axes_SOM[0].errorbar(np.arange(conditions), f_hit_means, yerr=f_hit_sem, fmt=' ', ecolor=errbar_color,
                         elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

    # title stuff
    axes_SOM[0].set_title('Overall Hit Rate', size=20, color=font_color)

    # x axis stuff
    axes_SOM[0].set_xlabel('Semantic Conditions', color=font_color)
    plt.setp(axes_SOM, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

    # y-axis stuff
    axes_SOM[0].set_ylabel('Hit Rate', color=font_color)

    # Draw graph and error bar
    axes_SOM[1].bar(np.arange(conditions), f_faa_means, color=colors, edgecolor='black', linewidth=2)
    axes_SOM[1].errorbar(np.arange(conditions), f_faa_means, yerr=f_faa_sem, fmt=' ', ecolor=errbar_color,
                         elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

    # title stuff
    axes_SOM[1].set_title('Overall False Alarm Rate', size=20, color=font_color)

    # x axis stuff
    axes_SOM[1].set_xlabel('Semantic Conditions', color=font_color)
    plt.setp(axes_SOM, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

    # y-axis stuff
    axes_SOM[1].set_ylabel('False Alarm Rate', color=font_color)
    # Finalize and print
    sns.despine()
    plt.tight_layout(h_pad=2.0)
    plt.savefig('f_HIT-FAA.png', transparent=trans_param)
    plt.show()
    # roc_hfa_df.groupby(['par_num', 'sem_condition'])[['hit_rate']].mean().to_clipboard()
    # anova_corr_visual_search_df_group.to_clipboard()
