import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
import re
from datetime import datetime
from ydata_profiling import ProfileReport

def validate_email(value):

    if not isinstance(value, str):
        return False
    pattern = r'^[^@]+@[^@]+\.[^@]+$'
    return re.match(pattern, value) is not None


def validate_phone(value):

    if not isinstance(value, str):
        return False
    if not value.startswith('+7'):
        return False
    digits_only = re.sub(r'\D', '', value)  # Оставляем только цифры
    return len(digits_only) == 11 and digits_only.startswith('7')


def validate_age(value):

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return 0 <= value <= 120


def validate_purchase_amount(value):

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return value >= 0


def validate_registration_date(value):

    if isinstance(value, str):
        try:
            reg_date = datetime.strptime(value, '%Y-%m-%d')
        except ValueError:
            return False
    elif isinstance(value, datetime):
        reg_date = value
    else:
        return False

    now = datetime.now()
    return reg_date <= now


validators = {
    'email': validate_email,
    'phone': validate_phone,
    'age': validate_age,
    'purchase_amount': validate_purchase_amount,
    'registration_date': validate_registration_date
}

def calculate_accuracy(summary, dataframe):
    report = {}

    for column in dataframe.columns:
        if column not in validators:
            continue

        validator_func = validators[column]
        series = dataframe[column]


        validatable_mask = series.notna() & (series != '')
        validatable_values = series[validatable_mask]

        total_validatable = len(validatable_values)
        if total_validatable == 0:
            report[column] = {
                'total_validatable': 0,
                'valid_count': 0,
                'accuracy_percent': 0.0,
                'invalid_examples': []
            }
            continue


        is_valid = validatable_values.apply(validator_func)
        valid_count = is_valid.sum()
        invalid_values = validatable_values[~is_valid]

        invalid_examples = invalid_values.head(5).tolist()

        accuracy_percent = (valid_count / total_validatable) * 100

        report[column] = {
            'total_validatable': total_validatable,
            'valid_count': valid_count,
            'accuracy_percent': round(accuracy_percent, 2),
            'invalid_examples': invalid_examples
        }

    first_col_name = summary.columns[0]
    summary['Точность (%)'] = summary[first_col_name].map(report)

    return report


def calculate_completeness(summary, dataframe):
    completeness_report = {}

    for column in dataframe.columns:
        total_count = len(dataframe[column])

        series = dataframe[column].replace('', np.nan)

        missing_count = series.isna().sum()
        completeness_percent = ((total_count - missing_count) / total_count) * 100 if total_count > 0 else 0.0

        completeness_report[column] = round(completeness_percent, 2)

    first_col_name = summary.columns[0]

    summary['Полнота (%)'] = summary[first_col_name].map(completeness_report)

    return completeness_report


def plot_completeness(completeness_dict):

    columns = list(completeness_dict.keys())
    values = list(completeness_dict.values())

    colors = []
    for val in values:
        if val >= 95:
            colors.append('green')
        elif val >= 85:
            colors.append('orange')
        else:
            colors.append('red')


    plt.figure(figsize=(10, 6))
    bars = plt.bar(columns, values, color=colors, edgecolor='black')


    plt.title('Полнота данных по столбцам', fontsize=16, fontweight='bold')
    plt.xlabel('Столбцы', fontsize=12)
    plt.ylabel('Процент полноты (%)', fontsize=12)
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def get_stat(summary, dataframe, plot_boxplots=True):
    for_summary = {}
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("⚠️  В DataFrame нет числовых столбцов.")
        return pd.DataFrame()

    results = []

    print("🔍 Анализ выбросов по методу Тьюки (IQR):\n")

    # Подготовка графиков, если нужно
    if plot_boxplots:
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 1) // 2  # 2 графика в строке
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]

        fig.suptitle('Boxplots для числовых столбцов (с выбросами)', fontsize=16, fontweight='bold')

    for i, col in enumerate(numeric_cols):
        # Расчет статистик
        Q1 = dataframe[col].quantile(0.25)
        Q3 = dataframe[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Поиск выбросов
        outliers_mask = (dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)
        outliers_values = dataframe.loc[outliers_mask, col]
        n_outliers = outliers_values.count()
        for_summary[col] = n_outliers

        print(f"📈 Столбец: '{col}'")
        print(f"   Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
        print(f"   Границы: [{lower_bound:.4f}, {upper_bound:.4f}]")
        print(f"   Найдено выбросов: {n_outliers}")

        if n_outliers > 0:
            examples = outliers_values.head(5).tolist()
            print(f"   Примеры выбросов: {examples}")
            if n_outliers > 5:
                print(f"   ... и ещё {n_outliers - 5} выбросов")
        else:
            print("   Выбросов не обнаружено")
        print("-" * 60)

        results.append({
            'column': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'n_outliers': n_outliers,
            'outliers_examples': outliers_values.head(5).tolist() if n_outliers > 0 else []
        })

    first_col_name = summary.columns[0]

    summary['IQR выбросы'] = summary[first_col_name].map(for_summary)

    if plot_boxplots:
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=dataframe, y=col)
            #plt.set_title(f'{col} (выбросов: {n_outliers})', fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.7)

    # Удаляем пустые subplot'ы, если столбцов нечётное количество
    # if plot_boxplots and len(numeric_cols) % 2 != 0 and len(numeric_cols) > 1:
    #     for j in range(len(numeric_cols), len(axes)):
    #         fig.delaxes(axes[j])

    if plot_boxplots:
        plt.tight_layout()
        plt.show()

    # Возвращаем DataFrame со статистикой
    return pd.DataFrame(results)

def get_stat_z(summary, dataframe, compare_with_iqr=True, plot_histograms=True):
    for_summary = {}
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("⚠️  В DataFrame нет числовых столбцов.")
        return pd.DataFrame()

    # Если нужно сравнение — вызываем IQR-анализ
    iqr_results = None
    if compare_with_iqr:
        print("📊 Сначала выполняем IQR-анализ для сравнения...")
        iqr_results = get_stat(summary, dataframe, plot_boxplots=False)
        iqr_dict = iqr_results.set_index('column')['n_outliers'].to_dict()

    results = []

    print("\n🔍 Анализ выбросов по Z-score (|Z| > 3):\n")

    if plot_histograms:
        n_cols = len(numeric_cols)
        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 5 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        fig.suptitle('Гистограммы с границами Z-score = ±3', fontsize=16, fontweight='bold')

    for i, col in enumerate(numeric_cols):
        # Шаг 1: Расчёт Z-score
        mean_val = dataframe[col].mean()
        std_val = dataframe[col].std()
        if std_val == 0:
            print(f"⚠️  Столбец '{col}': std=0, Z-score не определён. Пропускаем.")
            z_scores = pd.Series([0]*len(dataframe), index=dataframe.index)
        else:
            z_scores = (dataframe[col] - mean_val) / std_val

        # Шаг 2: Поиск выбросов
        outliers_mask = z_scores.abs() > 3
        outliers_values = dataframe.loc[outliers_mask, col]
        n_outliers_z = outliers_values.count()

        # Вывод информации
        print(f"Столбец: '{col}'")
        print(f"   Среднее: {mean_val:.4f}, Стандартное отклонение: {std_val:.4f}")
        print(f"   Выбросов (|Z| > 3): {n_outliers_z}")

        if n_outliers_z > 0:
            examples = outliers_values.head(5).tolist()
            print(f"   Примеры выбросов: {examples}")
            if n_outliers_z > 5:
                print(f"   ... и ещё {n_outliers_z - 5} выбросов")
        else:
            print("   Выбросов не обнаружено")

        # Сравнение с IQR
        if compare_with_iqr and iqr_results is not None:
            n_outliers_iqr = iqr_dict.get(col, 0)
            print(f"   ➤ Сравнение: IQR нашёл {n_outliers_iqr}, Z-score нашёл {n_outliers_z}")

        print("-" * 60)

        # if col == 'age':
        #     common = list()
        #     diff = list()
        #     for item in outliers_values:
        #         if

        # Сохраняем результат
        results.append({
            'column': col,
            'mean': mean_val,
            'std': std_val,
            'n_outliers_z': n_outliers_z,
            'outliers_examples_z': outliers_values.head(5).tolist() if n_outliers_z > 0 else []
        })
        for_summary[col] = n_outliers_z

    first_col_name = summary.columns[0]
    summary['Z выбросы'] = summary[first_col_name].map(for_summary)

    #     # Шаг 3: Построение гистограммы
    if plot_histograms:
            ax = axes[i]
            sns.histplot(dataframe[col], kde=True, ax=ax, color='skyblue', edgecolor='black')

            # Добавляем границы ±3σ
            left_bound = mean_val - 3 * std_val
            right_bound = mean_val + 3 * std_val

            ax.axvline(left_bound, color='red', linestyle='--', linewidth=2, label=f'μ - 3σ = {left_bound:.2f}')
            ax.axvline(right_bound, color='red', linestyle='--', linewidth=2, label=f'μ + 3σ = {right_bound:.2f}')

            if n_outliers_z > 0:
                ax.scatter(outliers_values, [0]*len(outliers_values),
                           color='red', s=50, zorder=5, label='Выбросы (|Z|>3)', marker='^')

            ax.set_title(f'{col} (Z-выбросов: {n_outliers_z})', fontweight='bold')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)

    if plot_histograms and len(numeric_cols) % 2 != 0 and len(numeric_cols) > 1:
        for j in range(len(numeric_cols), len(axes)):
            fig.delaxes(axes[j])

    if plot_histograms:
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)

def get_duplicates(summary, df):
    columns_to_check = ['customer_id', 'email', 'phone']

    for col in columns_to_check:
        total = len(df[col])
        unique = df[col].nunique()
        uniqueness = (unique / total) * 100
        print(f"Уникальность {col}: {uniqueness:.2f}%")
        row_index = summary.index[summary['Столбец'] == col][0]

        summary.loc[row_index, 'Уникальность (%)'] = uniqueness



    print("\n" + "="*50 + "\n")

    duplicates = df[df.duplicated(keep=False)]
    if duplicates.empty:
        print("Дубликатов нет.")
    else:
        print(duplicates)

    print("\n" + "="*50 + "\n")

def get_actual(df):
    df['registration_date'] = pd.to_datetime(df['registration_date'])

    print("Исходные данные:")
    print(df)
    print("\n" + "="*60 + "\n")


    now = pd.Timestamp.now()

    future_threshold = now
    past_threshold = now - pd.DateOffset(years=10)

    df['is_actual'] = (df['registration_date'] <= future_threshold) & \
                      (df['registration_date'] >= past_threshold)

    total_records = len(df)
    actual_records = df['is_actual'].sum()
    actual_percent = (actual_records / total_records) * 100

    print(f"Текущая дата: {now.strftime('%Y-%m-%d')}")
    print(f"Актуальный диапазон: с {past_threshold.strftime('%Y-%m-%d')} по {future_threshold.strftime('%Y-%m-%d')}")
    print(f"Актуальных записей: {actual_records} из {total_records}")
    print(f"Актуальность данных: {actual_percent:.2f}%")
    print("\n" + "="*60 + "\n")

    non_actual = df[~df['is_actual']]
    print("Неактуальные записи:")
    if non_actual.empty:
        print("Все записи актуальны.")
    else:
        print(non_actual[['customer_id', 'registration_date', 'is_actual']])

def is_consistent(row):
    status = row['status']
    amount = row['purchase_amount']

    if status == 'VIP':
        return amount > 20_000
    elif status == 'Premium':
        return 5_000 <= amount <= 20_000
    elif status == 'Regular':
        return amount < 5_000
    else:
        return False


df = pd.read_csv('./customer_data.csv')
summary = pd.DataFrame()
summary['Столбец'] = df.columns

print(df.head(10))

completeness = calculate_completeness(summary=summary, dataframe=df)


print(completeness)
print(calculate_accuracy(summary, df))

summary['Уникальность (%)'] = pd.NA
get_duplicates(summary, df)

get_actual(df)

# согласованность данных
df['is_consistent'] = df.apply(is_consistent, axis=1)
total_records = len(df)
consistent_records = df['is_consistent'].sum()
consistency_percent = (consistent_records / total_records) * 100
print(f"📊 Согласованность данных: {consistency_percent:.2f}%")

get_stat(summary, dataframe=df)
get_stat_z(summary, dataframe=df)
summary.to_csv('quality_summary.csv', index=False)

plot_completeness(completeness)


profile = ProfileReport(df, title='Отчет о качестве данных')
profile.to_file("auto_report.html")

