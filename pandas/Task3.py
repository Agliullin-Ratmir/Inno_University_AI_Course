import pandas as pd
from datetime import timedelta

df = pd.read_csv('log.csv')

df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d_%H:%M:%S')

df = df.sort_values('date').reset_index(drop=True)

delta1 = timedelta(minutes=5)
df['diff'] = df['date'].diff()
print(f"Result: {df}")

result_dict = {}
for index, row in df.iterrows():
    current_date = row['date']
    result_dict[current_date] = 0
    for i in range(index, len(df)):
        if df.iloc[i]['date'] - current_date <= delta1:
            result_dict[current_date] = result_dict[current_date] + 1
        else:
            break
    print(f"index: {index} current_date: {current_date}, result: {result_dict[current_date] }")
print(f"size of result_dict is: {len(result_dict)}")
max_amount = 0
result_key = None
for key in result_dict:
    if result_dict[key] >= max_amount:
        result_key = key
        max_amount = result_dict[key]

print(f"date: {result_key}, max amount: {max_amount}")

# date: 2020-04-09 13:06:41, max amount: 75