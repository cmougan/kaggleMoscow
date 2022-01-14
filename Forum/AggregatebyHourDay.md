# One of the magic (LB from 2.85759 to 1.30288)
https://www.kaggle.com/c/now-you-are-playing-with-power/discussion/300700

Our team improved the LB from 2.85759 to 1.30288 by incorporating the following code.

BASE_DIR = Path('../input/now-you-are-playing-with-power')
train = pd.read_csv(BASE_DIR / 'train.csv')
test = pd.read_csv(BASE_DIR / 'test.csv')
sub = pd.read_csv('../input/power-exp002/submission.csv')

grp3_df = train.groupby(['obs_day', 'obs_hour', 'obs_minute'])['output_gen'].mean().reset_index()
grp3_df = grp3_df.rename(columns={'output_gen': '3_output_gen'})
train = train.merge(grp3_df, on=['obs_day', 'obs_hour', 'obs_minute'], how='left')
test = test.merge(grp3_df, on=['obs_day', 'obs_hour', 'obs_minute'], how='left')

sub.loc[test['3_output_gen'].notnull(), 'output_gen'] = test.loc[test['3_output_gen'].notnull(), '3_output_gen'].values
sub.to_csv('submission.csv', index=False)
But they couldn't improve their score any further. What did the top teams work on?#