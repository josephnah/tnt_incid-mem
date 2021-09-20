# extract data for ROC curve analysis
import pandas as pd
import glob
import testable_analysis
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
# import metrics to calculate area under the curve (AUC)
from sklearn import metrics


draw_graph = 0

# directory to data
data_dir = testable_analysis.data_path

# Set pandas option
pd.set_option('display.max_columns', 37)

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

# and memory trials
memory_trials = raw_df['mem_trial_num'] >= 0
memory_df = raw_df[memory_trials].dropna(axis=1, how='all')

# Analysis filters
accuracy_filter = search_df['correct'] == 1

# DF for all correct trials (for RT)
corr_search_df = search_df[accuracy_filter]

corr_sem_condition_group = corr_search_df.groupby(['par_num', 'sem_condition'])
sem_condition_group = search_df.groupby(['par_num', 'sem_condition'])

# Visual Search DF for RT
corr_search_df_group = corr_sem_condition_group['RT'].mean().unstack()
corr_search_df_group.columns = ['RT_neu', 'RT_tax', 'RT_thm']

# Visual Search DF for accuracy
search_df_group = sem_condition_group['correct'].mean().unstack() * 100
search_df_group.columns = ['acc_neu', 'acc_tax', 'acc_thm']

# File output as csv for JASP
csv_result = pd.concat([corr_search_df_group, search_df_group], axis=1)

# Statistical Analysis
anova_corr_search_df_group = corr_sem_condition_group['RT'].mean()
anova_corr_search_df_group = anova_corr_search_df_group.reset_index()

anova_search_df_group = sem_condition_group['correct'].mean() * 100
anova_search_df_group = anova_search_df_group.reset_index()

# RM ANOVA for RT
# print('now I present to thee, the holy ANOVA')
ANOVA_RT = pg.rm_anova(data=anova_corr_search_df_group, dv='RT', within='sem_condition',subject='par_num').round(4)
# print(ANOVA_RT)
# print('')
# Pairwise t-tests
# print('followed by the lesser pairwise t-tests')
# print('')
pairwise_results_RT = pg.pairwise_ttests(data=anova_corr_search_df_group, dv='RT', within='sem_condition', subject='par_num', marginal= True, padjust = 'bonf')
# print(pairwise_results_RT)

# RM ANOVA for ACC
ANOVA_ACC = pg.rm_anova(data=anova_search_df_group, dv='correct', within='sem_condition', subject='par_num').round(4)
# Pairwise t-tests
results = pg.pairwise_ttests(data=anova_search_df_group, dv='correct', within='sem_condition', subject='par_num', marginal= True, padjust = 'bonf')

# Now calculate ROC curve
# Drop lists
drop_list_1 = ['rowNo', 'type', 'stimPos','stimFormat', 'stimPos_actual', 'ITI_f', 'ITI_fDuration', 'timestamp']
drop_list_2 = ['button1', 'button2', 'button3', 'button4', 'button5', 'button6','flipped_stim', 'orig_stim', 'stimList']

memory_df.drop(columns=drop_list_1, inplace=True)

# Add orig_stim_loc (Need to update when new participants come around)
memory_df['orig_stim_loc'] = 1
#FILLNA or something like that goes here

# Separate stimList column
memory_df['orig_stim'] = memory_df['stimList'].str.split(pat=';', expand=True)[0]
memory_df['flipped_stim'] = memory_df['stimList'].str.split(pat=';', expand=True)[1]

# remove "_flip"
memory_df['test_image'] = memory_df['orig_stim'].str.replace('_flip', '')

# Keep original response and create new familiarity column
memory_df['familiarity'] = memory_df['response']

# rescore automatic Testable scoring to our scoring system
# memory_df.replace({'familiarity': {1:6, 2:5, 3:4, 4:3, 5:2, 6:1}}, inplace=True)

# Set Dictionary for old/new + confidences and replace with Testable response
conf_dict = {1: '3_old', 2: '2_old', 3: '1_old', 4: '1_new', 5: '2_new', 6: '3_new'}
memory_df["old_new"] = memory_df['familiarity']
memory_df.replace({'old_new': conf_dict}, inplace=True)

# Drop now useless columns
memory_df.drop(columns=drop_list_2, inplace=True)

# count # of responses
roc_df = memory_df.groupby(['par_num', 'sem_condition', 'old_new']).size().reset_index(name='counts')
roc_df = roc_df.pivot_table(values='counts', columns='old_new', index=['par_num', 'sem_condition']).reset_index()
roc_df = roc_df[["par_num", 'sem_condition', '3_old', '2_old', '1_old', '1_new', '2_new', '3_new']]
roc_df.fillna(0, inplace=True)

roc_old_df = roc_df[["par_num", 'sem_condition', '3_old', '2_old', '1_old']]
roc_new_df = roc_df[["par_num", 'sem_condition', '1_new', '2_new', '3_new']]
# roc_df_prob = roc_df.div(30).cumsum(axis=1).reset_index
roc_df.to_clipboard()
# calculate all # of trials of HITS and FALSE ALARMS and add as column
roc_old_df['total_count'] = roc_old_df[['3_old', '2_old', '1_old']].sum(axis=1)
roc_new_df['total_count'] = roc_new_df[['3_new', '2_new', '1_new']].sum(axis=1)


# Calculate Hit rate for reals this time

roc_old_df[['3_old_prob', '2_old_prob', '1_old_prob']] = roc_old_df[['3_old', '2_old', '1_old']].div(roc_old_df['total_count'].values, axis=0).cumsum(axis=1)
roc_new_df[['3_new_prob', '2_new_prob', '1_new_prob']] = roc_new_df[['3_new', '2_new', '1_new']].div(roc_new_df['total_count'].values, axis=0).cumsum(axis=1)

# print(roc_old_df.mean())
roc_auc = pd.DataFrame(columns = ['par_num', 'sem_condition', 'auc'])

for p in roc_old_df.par_num.unique():
    sub_old = roc_old_df[roc_old_df['par_num'] == p]
    sub_new = roc_new_df[roc_new_df['par_num'] == p]

    neu_params_dict = {'par_num': p, 'sem_condition': 'neutral', 'auc': metrics.auc(sub_new.iloc[0,6:10], sub_old.iloc[0,6:10])}
    neu_auc_df = pd.DataFrame(neu_params_dict, index=[0])

    tax_params_dict = {'par_num': p, 'sem_condition': 'taxonomic', 'auc': metrics.auc(sub_new.iloc[1,6:10], sub_old.iloc[1,6:10])}
    tax_auc_df = pd.DataFrame(tax_params_dict, index=[0])

    thm_params_dict = {'par_num': p, 'sem_condition': 'thematic', 'auc': metrics.auc(sub_new.iloc[2,6:10], sub_old.iloc[2,6:10])}
    thm_auc_df = pd.DataFrame(thm_params_dict, index=[0])

    all_auc_df = neu_auc_df.append(tax_auc_df).append(thm_auc_df)

    roc_auc = roc_auc.append(all_auc_df)

auc_ANOVA = pg.rm_anova(data=roc_auc, dv='auc', within='sem_condition', subject='par_num')

# look at mean AUC for each condition
print(roc_auc.groupby(['sem_condition']).mean())

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


if draw_graph == 1:
    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.set_style('whitegrid')


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
    plt.savefig('output.png')
    plt.show()

    # Generate graphs for RT
    # Figure 1?
    colors = ["#FF221A", "#6A9551", "#D2AC3A"]
    # sns.set_palette(sns.color_palette(colors))
    sns.set_context('poster')
    fig, ax = plt.subplots(figsize=(12, 12))

    sns.set_style('white')
    # sns.despine()

    sns.barplot(data=anova_corr_search_df_group, x='sem_condition', y= 'RT',
                palette=colors, order = ['neutral', 'taxonomic','thematic'],
                ci=99, errwidth=3)

    # ax.set_title('RT for Visual Search')
    ax.set_xticklabels(['Neutral', 'Taxonomic','Thematic'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("Semantic Conditions")
    plt.ylabel("RT (ms)")

    plt.savefig('output.png')
    plt.show()