# extract data for ROC curve analysis
import pandas as pd
import glob
import testable_analysis
from statsmodels.stats.anova import AnovaRM

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

aovrm = AnovaRM(anova_corr_search_df_group, 'RT', 'par_num', within=['sem_condition'])
res = aovrm.fit()
print(res)
