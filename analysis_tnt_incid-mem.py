# extract data for ROC curve analysis
import pandas as pd
import glob
import testable_analysis
import pingouin as pg

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
print('now I present to thee, the holy ANOVA')
ANOVA_RT = pg.rm_anova(data=anova_corr_search_df_group, dv='RT', within='sem_condition',subject='par_num').round(4)
print(ANOVA_RT)
print('')
# Pairwise t-tests
print('followed by the lesser pairwise t-tests')
print('')
pairwise_results_RT = pg.pairwise_ttests(data=anova_corr_search_df_group, dv='RT', within='sem_condition', subject='par_num', marginal= True, padjust = 'bonf')
print(pairwise_results_RT)

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
memory_df.replace({'familiarity': {1:6, 2:5, 3:4, 4:3, 5:2, 6:1}}, inplace=True)

# Set Dictionary for old/new + confidences
conf_dict = {1:'def_new', 2: 'maybe_new', 3:'idk_new', 4: 'idk_old', 5: 'maybe_old', 6: 'def_old'}
memory_df["old_new"] = memory_df['familiarity']
memory_df.replace({'old_new': conf_dict}, inplace=True)

memory_df.drop(columns=drop_list_2, inplace=True)

print(memory_df)