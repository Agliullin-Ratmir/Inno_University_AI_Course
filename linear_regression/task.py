import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Настройка для корректного отображения русских символов
plt.rcParams['font.family'] = 'DejaVu Sans'

df = pd.read_csv('developers_salary.csv')

df.head(10)
df.info()
df.describe()

min_experience = df['опыт_лет'].min()
max_experience = df['опыт_лет'].max()
average_salary = df['зарплата'].mean()

print(f"Минимальный опыт работы: {min_experience} лет")
print(f"Максимальный опыт работы: {max_experience} лет")
print(f"Средняя зарплата: {average_salary:,.0f} руб.")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))


axes[0].scatter(df['опыт_лет'], df['зарплата'], color='blue', alpha=0.7)
axes[0].set_title('Зависимость зарплаты от опыта')
axes[0].set_xlabel('Опыт_лет (лет)')
axes[0].set_ylabel('Зарплата (руб.)')
axes[0].grid(True)

axes[1].boxplot(df['зарплата'], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'))
axes[1].set_title('Boxplot зарплат')
axes[1].set_ylabel('Зарплата (руб.)')
axes[1].set_xticks([1], ['Зарплата'])
axes[1].grid(True, axis='y')


axes[2].hist(df['зарплата'], bins=8, color='skyblue', edgecolor='black', alpha=0.7)
axes[2].set_title('Гистограмма распределения зарплат')
axes[2].set_xlabel('Зарплата (руб.)')
axes[2].set_ylabel('Частота')
axes[2].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Задание 2
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()
categorical_columns = ['образование', 'город', 'язык_программирования',
                       'размер_компании', 'английский']
df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns,
                            drop_first=True)
cols = df_encoded.columns
print(f"Колонки после кодирования: {cols}")
print(f"Форма после декодирования: {df_encoded.shape}")


numeric_cols = ['опыт_лет', 'возраст', 'образование_код', 'зарплата']
correlation = df_encoded.corr()[numeric_cols]
corr_sorted = correlation.sort_values(by='зарплата', ascending=False).head(10)
plt.figure(figsize=(5, 4))
sns.heatmap(corr_sorted, annot=True, cmap='coolwarm', center=0, fmt=".2f", square=True)
plt.title('Корреляция между выбранными переменными')
plt.show()
