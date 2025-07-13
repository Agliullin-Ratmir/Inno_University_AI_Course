import pandas as pd
from datetime import timedelta

df = pd.read_csv('log.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d_%H:%M:%S')
print(df)
specific_date1 = pd.to_datetime('2020-04-19').date()
specific_date2 = pd.to_datetime('2020-04-20').date()

filtered_df = df[(df['date'].dt.date == specific_date1) | (df['date'].dt.date == specific_date2)]

df = filtered_df.sort_values('date').reset_index(drop=True)
print(df)
delta1 = timedelta(minutes=30)

# grouped_dict = {name: group for name, group in df.groupby('user')}
# print(filtered_df)

grouped_dict = df.groupby('user')['date'].apply(list).to_dict()
#print(grouped_dict)
# result_intervals = df[df['date'].diff() < delta1]
# print(result_intervals)
# print(f"Result: {len(result_intervals)}")

sessions = []
valid_values = []
for key in grouped_dict:
    values = grouped_dict[key]
    if len(values) > 1 and values[0].date() == specific_date1:
        valid_values.append(values)

for el in valid_values:
    session_list = []
  #  print(el)
    for i in range(0, len(el) - 2):
        for j in range(i, len(el) - 2):
            if el[j] - el[i] < delta1 and el[i].date() == specific_date1:
                session_list.append(el[i])
        sessions.append(session_list)
session_amount = 0
for session in sessions:
    session_amount = session_amount + len(session)

print(f"amount of sessions beginning on 2020-04-19: {session_amount}")

# answer: amount of sessions beginning on 2020-04-19: 6573