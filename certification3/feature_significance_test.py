import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from util import read_postgres_to_pandas_psycopg2
import os

# Настройка стиля графиков
plt.rcParams.update({'font.size': 10})
sns.set_style("whitegrid")

# Конфигурация БД
db_config = {
    'dbname': 'credit_scoring',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}

TARGET_COL = 'seriousDlqin2yrs'

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2_val, _, _, _ = stats.chi2_contingency(confusion_matrix)
    n = confusion_matrix.sum().sum()
    phi2 = chi2_val / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def is_binary(series):
    return set(series.dropna().unique()) <= {0, 1}

def shapiro_test_for_normality(series, sample_size_limit=2000):
    if len(series) > sample_size_limit:
        return False
    try:
        _, p = stats.shapiro(series.sample(n=min(len(series), sample_size_limit), random_state=42))
        return p > 0.05
    except:
        return False

def run_statistical_tests_and_plot():
    print("Загрузка данных...")
    featured_df = read_postgres_to_pandas_psycopg2(db_config, 'public.featured_data')
    cleaned_df = read_postgres_to_pandas_psycopg2(db_config, 'public.cleaned_data')
    y = cleaned_df[TARGET_COL].astype(int)

    results = []

    for col in featured_df.columns:
        x = featured_df[col]
        dtype = 'numeric' if np.issubdtype(x.dtype, np.number) else 'categorical'
        if is_binary(x):
            dtype = 'binary'

        valid = ~(x.isna() | y.isna())
        x_clean = x[valid]
        y_clean = y[valid]

        if len(x_clean) < 10:
            continue

        try:
            if dtype == 'numeric':
                corr = stats.pointbiserialr(y_clean, x_clean).correlation
                group0 = x_clean[y_clean == 0]
                group1 = x_clean[y_clean == 1]

                if len(group0) < 2 or len(group1) < 2:
                    p_val = np.nan
                    test_name = 'insufficient'
                else:
                    normal0 = shapiro_test_for_normality(group0)
                    normal1 = shapiro_test_for_normality(group1)
                    if normal0 and normal1 and len(x_clean) <= 2000:
                        _, p_val = stats.ttest_ind(group0, group1, equal_var=False)
                        test_name = 't-test'
                    else:
                        _, p_val = stats.mannwhitneyu(group0, group1, alternative='two-sided')
                        test_name = 'Mann-Whitney'
            else:
                corr = cramers_v(x_clean, y_clean)
                try:
                    if len(x_clean.unique()) == 2 and len(y_clean.unique()) == 2:
                        _, p_val = stats.fisher_exact(pd.crosstab(x_clean, y_clean))
                        test_name = 'Fisher'
                    else:
                        _, p_val, _, _ = stats.chi2_contingency(pd.crosstab(x_clean, y_clean))
                        test_name = 'Chi2'
                except:
                    p_val = np.nan
                    test_name = 'error'

            significant = p_val < 0.05 if not np.isnan(p_val) else False
            neg_log_p = -np.log10(p_val) if not np.isnan(p_val) and p_val > 0 else 0

            results.append({
                'feature': col,
                'type': dtype,
                'test': test_name,
                'p_value': p_val,
                'neg_log10_p': neg_log_p,
                'correlation_with_target': corr,
                'significant': significant
            })

        except Exception as e:
            print(f"Ошибка при обработке {col}: {e}")
            results.append({
                'feature': col,
                'type': dtype,
                'test': 'error',
                'p_value': np.nan,
                'neg_log10_p': 0,
                'correlation_with_target': np.nan,
                'significant': False
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('neg_log10_p', ascending=False)
    results_df.to_csv('feature_significance_results.csv', index=False)
    print("\nРезультаты сохранены в feature_significance_results.csv")

    # === График 1: –log10(p-value) barplot ===
    plt.figure(figsize=(10, max(6, len(results_df) * 0.3)))
    colors = results_df['significant'].map({True: 'tab:red', False: 'tab:blue'})
    plt.barh(results_df['feature'], results_df['neg_log10_p'], color=colors, alpha=0.8)
    plt.axvline(-np.log10(0.05), color='gray', linestyle='--', label='p = 0.05')
    plt.xlabel('–log₁₀(p-value)')
    plt.title('Статистическая значимость признаков (по p-value)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pvalue_barplot.png', dpi=150)
    plt.show()

    # === График 2: Корреляция vs –log10(p-value) ===
    valid_corr = results_df.dropna(subset=['correlation_with_target', 'neg_log10_p'])
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        valid_corr['correlation_with_target'],
        valid_corr['neg_log10_p'],
        c=valid_corr['significant'].map({True: 'red', False: 'blue'}),
        alpha=0.7,
        edgecolors='k',
        linewidth=0.5
    )
    plt.axhline(-np.log10(0.05), color='gray', linestyle='--', label='p = 0.05')
    plt.xlabel('Корреляция с целевой переменной\n(Point-biserial / Cramér’s V)')
    plt.ylabel('–log₁₀(p-value)')
    plt.title('Связь признаков с целевой переменной')
    plt.legend(['Порог p=0.05', 'Значимые', 'Незначимые'], loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('correlation_vs_pvalue.png', dpi=150)
    plt.show()

    # === График 3: Гистограмма p-value ===
    pvals = results_df['p_value'].dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(pvals, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(0.05, color='red', linestyle='--', label='α = 0.05')
    plt.xlabel('p-value')
    plt.ylabel('Частота')
    plt.title('Распределение p-value по признакам')
    plt.legend()
    plt.tight_layout()
    plt.savefig('pvalue_histogram.png', dpi=150)
    plt.show()

    print("\nГрафики сохранены:")
    print("- pvalue_barplot.png")
    print("- correlation_vs_pvalue.png")
    print("- pvalue_histogram.png")

if __name__ == "__main__":
    run_statistical_tests_and_plot()