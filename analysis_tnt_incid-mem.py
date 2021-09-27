# extract data for ROC curve analysis
import pandas as pd
import glob
import testable_analysis
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
print('Draw Graphs? (1 for yes): ')
draw_graph = float(input())

# directory to data
data_dir = testable_analysis.data_path

# Graph Parameters
errbar_color = 'black'
errbar_line_width = 2
errbar_capsize = 5
errbar_capthick = 2



# load all files
csv_paths = glob.glob(data_dir)
participants = len(csv_paths)
df = []

for file in range(participants):
    data = pd.read_csv(csv_paths[file], skiprows=2)
    data['par_num'] = csv_paths[file][-10:-4]
    df.append(data)

raw_df = pd.concat(df, sort=True)
# Set par_num as index
raw_df.set_index('par_num', inplace=True)

# Divide up the demographics
demographics = raw_df['trial_type'] == 'demographics'
demographics_df = raw_df[demographics].dropna(axis=1, how='all')

# from visual search
search_trials = raw_df['trial_num'] > 0
search_df = raw_df[search_trials].dropna(axis=1, how='all')

# Analysis filters
accuracy_filter = search_df['correct'] == 1

# DF for all correct trials (for RT)
corr_search_df = search_df[accuracy_filter]

# Visual Search DF for RT
corr_sem_condition_group = corr_search_df.groupby(['par_num', 'sem_condition'])
corr_search_df_group = corr_sem_condition_group['RT'].mean().unstack()
corr_search_df_group.columns = ['RT_neu', 'RT_tax', 'RT_thm']

# Visual Search DF for accuracy
sem_condition_group = search_df.groupby(['par_num', 'sem_condition'])
search_df_group = sem_condition_group['correct'].mean().unstack() * 100
search_df_group.columns = ['acc_neu', 'acc_tax', 'acc_thm']

# File output as csv for JASP
csv_output = pd.concat([corr_search_df_group, search_df_group], axis=1)
csv_output.to_clipboard()
# DF for Statistical Analysis (RT)
anova_corr_search_df_group = corr_sem_condition_group['RT'].mean()
anova_corr_search_df_group = anova_corr_search_df_group.reset_index()

# RM ANOVA for RT
ANOVA_RT = pg.rm_anova(data=anova_corr_search_df_group, dv='RT', within='sem_condition',subject='par_num').round(4)
pairwise_results_RT = pg.pairwise_ttests(data=anova_corr_search_df_group, dv='RT', within='sem_condition', subject='par_num', marginal= True, padjust = 'bonf')


print('now I present to thee, the holy ANOVA for RT:')
print(ANOVA_RT)
print('')
# Pairwise t-tests
print('followed by the lesser pairwise t-tests:')
print(pairwise_results_RT)

# DF for statistical analysis (Accuracy)
anova_search_df_group = sem_condition_group['correct'].mean() * 100
anova_search_df_group = anova_search_df_group.reset_index()

# RM ANOVA for ACC
ANOVA_ACC = pg.rm_anova(data=anova_search_df_group, dv='correct', within='sem_condition', subject='par_num').round(4)
pairwise_results_ACC = pg.pairwise_ttests(data=anova_search_df_group, dv='correct', within='sem_condition', subject='par_num', marginal= True, padjust = 'bonf')

print('')
print('Next, the holy ANOVA for Accuracy')
print(ANOVA_ACC)
print('')
# Pairwise t-tests
print('followed by the lesser pairwise t-tests')
print(pairwise_results_ACC)

# Now calculate ROC curve starting with memory trials

# and memory trials
memory_trials = raw_df['mem_trial_num'] >= 0
memory_df_prepro = raw_df[memory_trials].dropna(axis=1, how='all')

# Keep original response and create new familiarity column
memory_df_prepro['old_new'] = memory_df_prepro['response']

# In case orig_stim_loc doesn't exist
memory_df_prepro['orig_stim_loc'].fillna(0, inplace=True)

# Separate out when test item is on left or right
memory_objleft = memory_df_prepro['orig_stim_loc'] == 1
memory_objrght = memory_df_prepro['orig_stim_loc'] == 2

obj_left_df = memory_df_prepro[memory_objleft]
obj_rght_df = memory_df_prepro[memory_objrght]


# Set Dictionary for old/new + confidences and replace with Testable response
conf_left_dict = {1: '3_old', 2: '2_old', 3: '1_old', 4: '1_new', 5: '2_new', 6: '3_new'}
conf_rght_dict = {1: '3_new', 2: '2_new', 3: '1_new', 4: '1_old', 5: '2_old', 6: '3_old'}

obj_left_df.replace({'old_new': conf_left_dict}, inplace=True)
obj_rght_df.replace({'old_new': conf_rght_dict}, inplace=True)

# concat back into memory_df
memory_df = pd.concat([obj_left_df, obj_rght_df])

# List of columns to drop
drop_list= ['rowNo', 'type', 'stimPos','stimFormat', 'stimPos_actual', 'ITI_f', 'ITI_fDuration', 'timestamp', 'button1', 'button2', 'button3', 'button4', 'button5', 'button6','flipped_stim', 'orig_stim', 'stimList']

# Separate stimList column
memory_df['orig_stim'] = memory_df['stimList'].str.split(pat=';', expand=True)[0]
memory_df['flipped_stim'] = memory_df['stimList'].str.split(pat=';', expand=True)[1]

# remove "_flip"
memory_df['test_image'] = memory_df['orig_stim'].str.replace('_flip', '')

# Drop now useless columns
memory_df.drop(columns=drop_list, inplace=True)

# count # of responses
roc_df = memory_df.groupby(['par_num', 'sem_condition', 'old_new']).size().reset_index(name='counts')
roc_df = roc_df.pivot_table(values='counts', columns='old_new', index=['par_num', 'sem_condition']).reset_index()
roc_df = roc_df[["par_num", 'sem_condition', '3_old', '2_old', '1_old', '1_new', '2_new', '3_new']]
roc_df.fillna(0, inplace=True)

roc_old_df = roc_df[["par_num", 'sem_condition', '3_old', '2_old', '1_old']]
roc_new_df = roc_df[["par_num", 'sem_condition', '1_new', '2_new', '3_new']]
# roc_df_prob = roc_df.div(30).cumsum(axis=1).reset_index

# calculate all # of trials of HITS and FALSE ALARMS and add as column
roc_old_df['total_count'] = roc_old_df[['3_old', '2_old', '1_old']].sum(axis=1)
roc_new_df['total_count'] = roc_new_df[['3_new', '2_new', '1_new']].sum(axis=1)

# Calculate Hit rate and false alarm rate
roc_old_df[['3_old_prob', '2_old_prob', '1_old_prob']] = roc_old_df[['3_old', '2_old', '1_old']].div(roc_old_df['total_count'].values, axis=0).cumsum(axis=1)
roc_new_df[['3_new_prob', '2_new_prob', '1_new_prob']] = roc_new_df[['3_new', '2_new', '1_new']].div(roc_new_df['total_count'].values, axis=0).cumsum(axis=1)

# set up DF for ROC data
roc_auc = pd.DataFrame(columns=['par_num', 'sem_condition', 'auc'])

for p in roc_old_df.par_num.unique():
    sub_old = roc_old_df[roc_old_df['par_num'] == p]
    sub_new = roc_new_df[roc_new_df['par_num'] == p]

    neu_params_dict = {'par_num': p, 'sem_condition': 'neutral', 'auc': metrics.auc(sub_new.iloc[0, 6:10], sub_old.iloc[0, 6:10])}
    neu_auc_df = pd.DataFrame(neu_params_dict, index=[0])

    tax_params_dict = {'par_num': p, 'sem_condition': 'taxonomic', 'auc': metrics.auc(sub_new.iloc[1, 6:10], sub_old.iloc[1, 6:10])}
    tax_auc_df = pd.DataFrame(tax_params_dict, index=[0])

    thm_params_dict = {'par_num': p, 'sem_condition': 'thematic', 'auc': metrics.auc(sub_new.iloc[2, 6:10], sub_old.iloc[2, 6:10])}
    thm_auc_df = pd.DataFrame(thm_params_dict, index=[0])

    all_auc_df = neu_auc_df.append(tax_auc_df).append(thm_auc_df)

    roc_auc = roc_auc.append(all_auc_df)

auc_ANOVA = pg.rm_anova(data=roc_auc, dv='auc', within='sem_condition', subject='par_num')
print('')
print(auc_ANOVA)
# look at mean AUC for each condition
# print(roc_auc.groupby(['sem_condition']).mean())

# Graph ROC Curve
roc_all_df = pd.concat([roc_old_df[['3_old_prob', '2_old_prob', '1_old_prob']], roc_new_df], axis=1)
x = roc_all_df.groupby('sem_condition')['3_new_prob', '2_new_prob', '1_new_prob'].mean().reset_index()
y = roc_all_df.groupby('sem_condition')['3_old_prob', '2_old_prob', '1_old_prob'].mean().reset_index()

x_melt = x.melt(id_vars=['sem_condition'])
y_melt = y.melt(id_vars=['sem_condition'])

x_melt.replace({'3_new_prob': 1, '2_new_prob': 2, '1_new_prob': 3}, inplace=True)
y_melt.replace({'3_old_prob': 1, '2_old_prob': 2, '1_old_prob': 3}, inplace=True)

x_melt.rename(columns={'value': 'faa_rate'}, inplace=True)
y_melt.rename(columns={'value': 'hit_rate'}, inplace=True)

all_data = y_melt.merge(x_melt, on=['sem_condition', 'old_new'])
all_data.sort_values(['sem_condition', 'old_new'], ascending=[True, True], inplace=True)

# Calculate Hit/False Alarm by Objects
roc_objects_df = memory_df.groupby(['sem_condition', 'old_new', 'test_image']).size().reset_index(name='counts')
roc_objects_df = roc_objects_df.pivot_table(values='counts', columns='old_new', index=['sem_condition', 'test_image']).reset_index()
roc_objects_df = roc_objects_df[['sem_condition','test_image', '3_old', '2_old', '1_old', '1_new', '2_new', '3_new']]
roc_objects_df.fillna(0, inplace=True)
roc_objects_df['total_hit_count'] = roc_objects_df[['3_old', '2_old', '1_old']].sum(axis=1)
roc_objects_df['total_faa_count'] = roc_objects_df[['3_new', '2_new', '1_new']].sum(axis=1)
roc_objects_df['hit_rate'] = roc_objects_df['total_hit_count'] / (roc_objects_df['total_hit_count'] + roc_objects_df['total_faa_count'])
roc_objects_df['faa_rate'] = roc_objects_df['total_faa_count'] / (roc_objects_df['total_hit_count'] + roc_objects_df['total_faa_count'])

tax_filt = roc_objects_df['sem_condition'] == 'taxonomic'
thm_filt = roc_objects_df['sem_condition'] == 'thematic'

tax_objects_df = roc_objects_df[tax_filt].sort_values('hit_rate', ascending=False)
thm_objects_df = roc_objects_df[thm_filt].sort_values('hit_rate', ascending=False)

if draw_graph == 1:
    # Graph Thematic Objects

    # Set seaborn defaults
    sns.set_context('poster')
    sns.set_theme(style='white')

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw graph using seaborn
    sns.barplot(data=thm_objects_df, x='test_image', y='hit_rate')
    sns.despine()

    # Customize x axis
    plt.xlabel("Thematic Objects", fontsize=20)
    plt.xticks(rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # ax.margins(x=0)

    # customize y axis
    plt.ylabel("Hit Rate", fontsize=20)
    ax.set_ylim(0, 1)

    # Draw chance level
    ax.plot([0, 3], [0.5, 0.5], transform=ax.transAxes, color='gray', dashes=(2, 1))
    plt.savefig('f_thm_hit-rate.png')
    plt.tight_layout(h_pad=2.0)
    plt.show()

    # Draw Taxonomic Objects
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw graph using seaborn
    sns.barplot(data=tax_objects_df, x='test_image', y='hit_rate')
    sns.despine()

    # Customize x axis
    plt.xlabel("Taxonomic Objects", fontsize=20)
    plt.xticks(rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=15)
    # ax.margins(x=0)

    # customize y axis
    plt.ylabel("Hit Rate", fontsize=20)
    ax.set_ylim(0, 1)

    # Draw chance level
    ax.plot([0, 3], [0.5, 0.5], transform=ax.transAxes, color='gray', dashes=(2, 1))
    plt.tight_layout(h_pad=2.0)
    plt.savefig('f_tax_hit-rate.png')
    plt.show()

    sns.set_context('poster')
    sns.set_style('white')

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.despine()


    sns.scatterplot(data = all_data.query('sem_condition == "neutral"'), x ='faa_rate',
                    y='hit_rate', color="#FF221A",ax=ax, edgecolor = 'black', zorder=11)
    sns.scatterplot(data = all_data.query('sem_condition == "taxonomic"'), x ='faa_rate',
                    y= 'hit_rate',color="#6A9551",ax=ax, edgecolor = 'black', zorder=11)
    sns.scatterplot(data = all_data.query('sem_condition == "thematic"'), x ='faa_rate',
                    y= 'hit_rate', color="#D2AC3A",ax=ax, edgecolor = 'black', zorder=12)

    sns.lineplot(data = all_data.query('sem_condition == "neutral"'),
                 x ='faa_rate', y='hit_rate', color = "#FF221A",ax = ax)
    sns.lineplot(data = all_data.query('sem_condition == "taxonomic"'),
                 x ='faa_rate', y='hit_rate',color = "#6A9551",ax = ax, dashes = (3,1))
    sns.lineplot(data = all_data.query('sem_condition == "thematic"'),
                 x ='faa_rate', y='hit_rate', color = "#D2AC3A",ax = ax, dashes = (2,1))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='gray', dashes=(2, 1))

    ax.lines[1].set_linestyle("--")
    ax.lines[2].set_linestyle(":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # ax.set(title='ROCs')
    ax.set_xlabel("False Alarm Rate")
    ax.set_ylabel("Hit Rate")
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='gray', dashes=(2,1))
    ax.legend()
    plt.legend(title='Semantic Relationship', fontsize='15', title_fontsize='14', labels=['Neutral', 'Taxonomic', 'Thematic'])
    plt.savefig('f_ROC.png')
    plt.show()

    # Figure 1 Parameters
    colors = ["#FF221A", "#6A9551", "#D2AC3A"]
    conditions_x = ['Neutral', 'Taxonomic', 'Thematic']
    conditions = 3
    sns.set_palette(sns.color_palette(colors))
    sns.set_context('talk')
    sns.set_style('white')
    fig_1, axes_1 = plt.subplots(figsize=(12, 6),  nrows=1, ncols=2)

    # Figure 1a, data
    f_RT_means = corr_sem_condition_group['RT'].mean().unstack().mean()
    f_sem_RT_means = corr_sem_condition_group['RT'].mean().unstack().sem()

    # Draw graph and error bar
    axes_1[0].bar(np.arange(conditions), f_RT_means, color=colors, edgecolor='black', linewidth=2)
    axes_1[0].errorbar(np.arange(conditions), f_RT_means, yerr=f_sem_RT_means, fmt=' ', ecolor=errbar_color, elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

    # title stuff
    axes_1[0].set_title('RT for Visual Search', size=20)

    # x axis stuff
    axes_1[0].set_xlabel('Semantic Conditions')
    plt.setp(axes_1, xticks=[i for i in range(conditions)], xticklabels=conditions_x)

    # y-axis stuff
    axes_1[0].set_ylabel('RT (ms)')

    # Figure 1b
    # Data
    f_ACC_means = sem_condition_group['correct'].mean().unstack().mean() * 100
    f_sem_ACC_means = sem_condition_group['correct'].mean().unstack().sem() * 100

    # Draw graph and error bar
    axes_1[1].bar(np.arange(conditions), f_ACC_means, color=colors, edgecolor='black', linewidth=2)
    axes_1[1].errorbar(np.arange(conditions), f_ACC_means, yerr=f_sem_ACC_means, fmt=' ', ecolor=errbar_color, elinewidth=errbar_line_width, capsize=errbar_capsize, capthick=errbar_capthick)

    # title stuff
    axes_1[1].set_title('Accuracy for Visual Search', size=20)

    # x axis stuff
    axes_1[1].set_xlabel('Semantic Conditions')

    # y-axis stuff
    axes_1[1].set_ylabel('Accuracy (%)')

    # Add # of participants to graph
    axes_1[1].text(2, -20, 'n = ' + str(participants))

    # Finalize and print
    sns.despine()
    plt.tight_layout(h_pad=2.0)
    plt.savefig('f_RT-ACC.png')
    plt.show()
    # anova_corr_search_df_group.to_clipboard()