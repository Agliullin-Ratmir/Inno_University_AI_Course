import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif_safe(X):
    # Make a copy to avoid modifying original
    X = X.copy()

    # Convert bool to int
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    # Ensure all columns are numeric
    try:
        X = X.astype(float)
    except ValueError as e:
        raise ValueError(f"Non-numeric data found. Check your features: {e}")

    # Drop rows with NaN
    X = X.dropna()

    # Compute VIF
    vif_data = []
    for i, col in enumerate(X.columns):
        try:
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'feature': col, 'VIF': vif})
        except Exception as e:
            print(f"Could not compute VIF for {col}: {e}")
            vif_data.append({'feature': col, 'VIF': np.nan})

    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
    return vif_df


def print_metrics(y_true, y_pred, label):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{label}:")
    print(f"  R²  = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

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
feature_columns = df_encoded.columns.tolist()
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


#Задание 3
from sklearn.model_selection import train_test_split
X = df_encoded.drop('зарплата', axis=1)
y = df_encoded['зарплата']
X = X.drop('образование_код', axis=1, errors='ignore')

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% — тестовая выборка
    random_state=42       # для воспроизводимости
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = LinearRegression()
model.fit(X_train, y_train)


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print_metrics(y_train, y_train_pred, "=== Метрики на обучающей выборке ===")
print_metrics(y_test, y_test_pred, "\n=== Метрики на тестовой выборке ===")

coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
})


coef_df = pd.concat([coef_df], ignore_index=True)

coef_df = coef_df.sort_values(by='coefficient', key=abs, ascending=False)

top_10 = coef_df.nlargest(10, 'coefficient')


plt.figure(figsize=(10, 6))
plt.barh(top_10['feature'], top_10['coefficient'], color=np.where(top_10['coefficient'] >= 0, 'green', 'red'))
plt.xlabel('Coefficient value')
plt.title('Top-10 Most Important Features (by |coefficient|)')
plt.axvline(0, color='black', linewidth=0.8)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# VIF
vif_df = calculate_vif_safe(X_train)
high_vif = vif_df[vif_df['VIF'] > 10]
print(high_vif)

# улучшенная модель
X_train_improved = X_train.drop(['возраст'], axis=1)
X_test_improved = X_test.drop(['возраст'], axis=1)

model2 = LinearRegression()
model2.fit(X_train_improved, y_train)

y_train_pred2 = model2.predict(X_train_improved)
y_test_pred2 = model2.predict(X_test_improved)

print_metrics(y_train, y_train_pred2, "=== Метрики на улучшенной обучающей выборке ===")
print_metrics(y_test, y_test_pred2, "\n=== Метрики на улучшенной тестовой выборке ===")

# 5
import statsmodels.api as sm

X_train_numeric = X_train.apply(pd.to_numeric, errors='coerce')

# Для y_train
y_train_numeric = pd.to_numeric(y_train, errors='coerce')

# Удалите строки, где появились NaN после преобразования
combined = pd.concat([X_train_numeric, y_train_numeric], axis=1).dropna()
X_clean = combined.iloc[:, :-1]
y_clean = combined.iloc[:, -1]
X_clean = X_clean.astype(float)
y_clean = y_clean.astype(float)


X_clean_sm = sm.add_constant(X_clean)

model_stats = sm.OLS(y_clean, X_clean_sm).fit()
print(model_stats.summary())

p_values = model_stats.pvalues
significant_features = p_values[p_values < 0.05].index.tolist()
significant_features = [f for f in significant_features if f != 'const']

insignificant_features = p_values[p_values > 0.05].index.tolist()
insignificant_features = [f for f in insignificant_features if f != 'const']

print("Значимые признаки (p-value < 0.05):")
print(significant_features)
print("Незначимые признаки (p-value > 0.05):")
print(insignificant_features)

if significant_features:
    X_train_sig = X_train[significant_features]
    X_test_sig = X_test[significant_features]

    model_sig = LinearRegression()
    model_sig.fit(X_train_sig, y_train)

    y_train_pred_sig = model_sig.predict(X_train_sig)
    y_test_pred_sig = model_sig.predict(X_test_sig)


    print_metrics(y_train, y_train_pred_sig, "=== Метрики на значимых признаках на улучшенной обучающей выборке ===")
    print_metrics(y_test, y_test_pred_sig, "\n=== Метрики на значимых признаках на улучшенной тестовой выборке ===")


plt.figure(figsize=(12, 5))

# 6 --- График 1: Базовая модель (все признаки) ---
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.6, color='steelblue', edgecolor='k', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальная модель')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Базовая модель (все признаки)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# 6 --- График 2: Модель на значимых признаках ---
plt.subplot(1, 2, 2)
plt.scatter(y_test_pred, y_test - y_test_pred, alpha=0.6, color='seagreen', edgecolor='k', s=40)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Идеальная модель')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('Модель на значимых признаках')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 7


new_candidate = pd.DataFrame([{
    'опыт': 5,
    'возраст': 28,
    'образование': 'Магистр',
    'город': 'Москва',
    'язык_программирования': 'Python',
    'размер_компании': 'Крупная',
    'английский': 'B1-B2'
}])

# Apply the same encoding
new_candidate_encoded = pd.get_dummies(
    new_candidate,
    columns=categorical_columns,
    drop_first=True
)

new_candidate_aligned = new_candidate_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

print(f"new_candidate_aligned: {new_candidate_aligned.columns.tolist()}")
new_candidate_aligned = new_candidate_aligned.drop('образование_код', axis=1, errors='ignore')
new_candidate_aligned = new_candidate_aligned.drop('зарплата', axis=1, errors='ignore')
predicted_salary = model.predict(new_candidate_aligned)
print(f"Предсказанная зарплата: {predicted_salary[0]:.0f} тыс. руб.")