import pandas as pd
from datetime import timedelta

df = pd.read_csv('log.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d_%H:%M:%S')

specific_date1 = pd.to_datetime('2020-04-19').date()
specific_date2 = pd.to_datetime('2020-04-20').date()

filtered_df = df[(df['date'].dt.date == specific_date1) | (df['date'].dt.date == specific_date2)]

df = filtered_df.sort_values('date').reset_index(drop=True)
delta1 = timedelta(minutes=30)

grouped_dict = df.groupby('user')['date'].apply(list).to_dict()

sessions = []
valid_values = []
for key in grouped_dict:
    values = grouped_dict[key]
    if len(values) > 1 and values[0].date() == specific_date1:
        valid_values.append(values)

for el in valid_values:
    session_list = []
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

# solution made with Gregory's help
print("solution made with Gregory's help")
df['diff'] = df.groupby('user')['date'].diff()


df['sessions'] = (df['diff'] >= pd.Timedelta(minutes=30)) | df['diff'].isna()
df['sessions_amount'] = df['sessions'].cumsum()
filtered_sessions_df = df[df['date'].dt.date == specific_date1]
last_row = filtered_sessions_df.iloc[-1]
print(f"amount of sessions beginning on 2020-04-19(better version): {last_row['sessions_amount']}")
# answer: amount of sessions beginning on 2020-04-19(better version): 6271

