# extract data for ROC curve analysis
import pandas as pd
import glob

data_dir = '/Users/joecool890/Dropbox/UC-Davis/projects/tnt_visual-search/raw-data/*.csv'

# load all files
csv_paths = glob.glob(data_dir)

participants = len(csv_paths)

df = []

for file in range(participants):
    data = pd.read_csv(csv_paths[file], skiprows=2)
    data['par_num'] = csv_paths[file][-10:-4]
    df.append(data)

df_raw = pd.concat(df)

df_raw.set_index('par_num', inplace=True)


# Divide up the visual search and memory trials
demographics = df_raw['trial_type'] == 'demographics'
search_trials = df_raw['trial_num'] > 0
memory_trials = df_raw['mem_trial_num'] >= 0
