from datasets import load_dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import Dataset

def process_dataset():
    RFSD = load_dataset('irlspbru/RFSD', split='train', streaming=True)

    examples = []
    rfds_iter = iter(RFSD)
    pd.set_option('display.max_columns', None)
    for _ in range(1000):
        try:
            examples.append(next(rfds_iter))
        except StopIteration:
            print("Reached end of dataset early.")
            break

    df = pd.DataFrame(examples)
    print(f"Loaded {len(df)} examples into DataFrame.")
    df.to_csv('df.csv', sep='\t', index=False)

    print("Non null")
    filtered_df = df.loc[:, df.count() > 500]
    print(filtered_df.columns.tolist())

    useful_columns = ['inn', 'ogrn', 'creation_date', 'dissolution_date', 'age', 'eligible', 'exemption_criteria', 'financial', 'filed', 'imputed', 'outlier', 'okved', 'okved_section', 'okopf', 'okogu', 'okfc', 'oktmo']
    df = df[useful_columns]
    print(df.dtypes)
    print(df.head(5))

    with open("head.txt", "w") as logFile:
        logFile.write(str(df.head(5)))


    print("=== DataFrame Info ===")
    print(df.info())

    print("=== Numerical Columns Summary ===")
    print(df.describe(include='all'))

    # конвертируем колонки с типом дата в числовой тип
    datetime_cols = ['creation_date', 'dissolution_date']

    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col])
        reference = pd.Timestamp('1970-01-01')
        df[col] = (df[col] - reference).dt.days.astype(float)

    df_clean = df.replace(['none', 'None', ''], np.nan)
    df_numeric = df_clean.apply(pd.to_numeric, errors='coerce')


    correlation_matrix = df_numeric.corr()


    threshold = 0.7
    corr_pairs = []

    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_pairs.append((col1, col2, corr_val))

    print("Highly correlated pairs:")
    for col1, col2, corr_val in corr_pairs:
        print(f"{col1} vs {col2}: {corr_val:.2f}")

    # визуализируем корреляцию
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=False,
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5
    )
    plt.title("Матрица корреляции между 20 колонками", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    process_dataset()