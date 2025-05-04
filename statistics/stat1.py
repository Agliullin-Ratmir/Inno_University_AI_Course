import numpy as np
import pandas as pd
from scipy import stats
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt

import seaborn as sns

# From here https://fred.stlouisfed.org/series/IRLTLT01USM156N Long-Term Government Bond Yields
df = pd.read_csv("bonds.csv")  

# Extract the column values, skipping NaNs and converting to float
values = df['IRLTLT01USM156N'].dropna().values.astype(float)

average = np.mean(values)
std_deviation = np.std(values)
mode_result = stats.mode(values, keepdims=False)

print("Average value:", average)
print("Standard Deviation:", std_deviation)
print("Mode:", mode_result.mode, " (Count:", mode_result.count, ")")

# profile = ProfileReport(df, title="Profiling Report")
# profile.to_file("bonds_report.html")

# Apply Seaborn theme
sns.set_theme(style="darkgrid")

# Creating a simple Matplotlib plot from here https://www.geeksforgeeks.org/plotting-with-seaborn-and-matplotlib/?preview_id=1367457&preview=true
x = []
y = values.tolist()
size = len(y) + 1
for i in range(1, size):
    x.append(i)

plt.plot(x, y, marker='o', linestyle='-', color='blue', label="Trend")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Matplotlib Plot with Seaborn Theme")
plt.legend()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(values, kde=True, bins=30, color='purple')

# Adding Mean Line using Matplotlib
mean_value = np.mean(values)
plt.axvline(mean_value, color='red', linestyle='dashed', linewidth=2)
plt.text(mean_value + 0.1, 50, f'Mean: {mean_value:.2f}', color='red')

plt.title("Distribution with Seaborn and Matplotlib Customization")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()