import pandas as pd
from datetime import timedelta


df = pd.read_csv('log.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d_%H:%M:%S')

video_param = 'video'
video_event_type = 2

# filter
filtered = df[(df['parameter'] == video_param) & (df['event_type'] == video_event_type)]

print(filtered)
grouped_dict = filtered.groupby('date')['user'].apply(set).to_dict()

max_date = max(grouped_dict, key=lambda k: len(grouped_dict[k]))
print(f"date with max users is {max_date}, amount: {len(grouped_dict[max_date])}")

# answer: date with max users is 2020-04-03 17:52:49, amount: 2