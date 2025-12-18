import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Загрузка данных
df = pd.read_csv("cs-test.csv")

# Удалим лишний столбец (первый, судя по данным — индекс CUST_ID)
df = df.drop(columns=df.columns[0], errors='ignore')

# Автоматическое определение целевой переменной и признаков
target_col = 'seriousDlqin2yrs'

# Колонки признаков (все, кроме целевой)
feature_cols = [col for col in df.columns if col != target_col]

# Заполним пропуски (если есть) — Median для числовых
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Убедимся, что типы корректны
df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0).astype(int)
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# --- Нормализация числовых признаков (StandardScaler) ---
scaler = StandardScaler()
numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
df_scaled = df.copy()
df_scaled[numeric_features] = scaler.fit_transform(df[numeric_features])

print("✅ Данные загружены, очищены и нормализованы.\n")

# --- Гипотеза 1 ---
print("🔍 Гипотеза 1:")
print("Если хотя бы одна из колонок NumberOfTime30_59DaysPastDueNotWorse, "
      "NumberOfTimes90DaysLate, NumberOfTime60_89DaysPastDueNotWorse > 0, "
      "то seriousDlqin2yrs == 1 в >50% случаев.")

cols1 = ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']
# Приведём названия к реальному формату (возможно, в данных без подчёркиваний)
# Попробуем оба варианта
mapped_cols1 = []
for c in cols1:
    if c in df.columns:
        mapped_cols1.append(c)
    elif c.replace('_', '') in df.columns:
        mapped_cols1.append(c.replace('_', ''))
    else:
        raise ValueError(f"Колонка {c} не найдена в данных. Доступные: {list(df.columns)}")

mask1 = (df[mapped_cols1[0]] > 0) | (df[mapped_cols1[1]] > 0) | (df[mapped_cols1[2]] > 0)
subset1 = df[mask1]

if len(subset1) > 0:
    prop1 = (subset1[target_col] == 1).mean()
    print(f"→ Количество таких наблюдений: {len(subset1)}")
    print(f"→ Доля seriousDlqin2yrs == 1: {prop1:.2%}")
    print(f"→ Гипотеза {'✅ подтверждена' if prop1 > 0.5 else '❌ опровергнута'}\n")
else:
    print("→ Нет наблюдений, удовлетворяющих условию гипотезы 1.\n")

# --- Гипотеза 2 ---
print("🔍 Гипотеза 2:")
print("Если age > 50, MonthlyIncome < 50000 и DebtRatio < 0.3, "
      "то seriousDlqin2yrs == 0 в >50% случаев.")

mask2 = (df['age'] > 50) & (df['MonthlyIncome'] < 50000) & (df['DebtRatio'] < 0.3)
subset2 = df[mask2]

if len(subset2) > 0:
    prop2 = (subset2[target_col] == 0).mean()
    print(f"→ Количество таких наблюдений: {len(subset2)}")
    print(f"→ Доля seriousDlqin2yrs == 0: {prop2:.2%}")
    print(f"→ Гипотеза {'✅ подтверждена' if prop2 > 0.5 else '❌ опровергнута'}\n")
else:
    print("→ Нет наблюдений, удовлетворяющих условию гипотезы 2.\n")

# --- Гипотеза 3 ---
print("🔍 Гипотеза 3:")
print("Если MonthlyIncome < 100000, DebtRatio > 0.5, "
      "NumberOfOpenCreditLinesAndLoans > 0 и NumberOfDependents > 1, "
      "то seriousDlqin2yrs == 1 в >50% случаев.")

mask3 = (
        (df['MonthlyIncome'] < 100000) &
        (df['DebtRatio'] > 0.5) &
        (df['NumberOfOpenCreditLinesAndLoans'] > 0) &
        (df['NumberOfDependents'] > 1)
)
subset3 = df[mask3]

if len(subset3) > 0:
    prop3 = (subset3[target_col] == 1).mean()
    print(f"→ Количество таких наблюдений: {len(subset3)}")
    print(f"→ Доля seriousDlqin2yrs == 1: {prop3:.2%}")
    print(f"→ Гипотеза {'✅ подтверждена' if prop3 > 0.5 else '❌ опровергнута'}\n")
else:
    print("→ Нет наблюдений, удовлетворяющих условию гипотезы 3.\n")

# --- Сохранение нормализованных данных (опционально) ---
df_scaled.to_csv("cs-test_normalized.csv", index=False)
print("📁 Нормализованные данные сохранены в 'cs-test_normalized.csv'")