
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from util import insert_dataframe_to_postgres, read_postgres_to_pandas_psycopg2

def transform():
    db_config = {
        'dbname': 'credit_scoring',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost', # change to 'db' for using inside docker
        'port': '5432'
    }
    df = read_postgres_to_pandas_psycopg2(db_config, 'public.raw_data')
    df['seriousDlqin2yrs'] = df['seriousDlqin2yrs'].astype('category')
    print('Amount of positive: ' + str((df['seriousDlqin2yrs'] == 1).sum()))
    print('Amount of negative: ' + str((df['seriousDlqin2yrs'] == 0).sum()))

    # 3. EDA
    print("=== EDA ===")
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())


    # Целевая переменная (замените 'target' на имя вашей целевой колонки)
    target_col = 'seriousDlqin2yrs'
    if target_col in df.columns:
        print(f"Распределение целевой переменной '{target_col}':")
        print(df[target_col].value_counts())

    # Удаление дубликатов
    df_clean = df.drop_duplicates()

    # Обработка пропусков
    for col in df_clean.columns:
        if df_clean[col].dtype in ['object', 'category']:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown')
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    # Обработка выбросов (только для числовых колонок)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    # df_clean = df_clean.astype(float) # Убрана, т.к. 'seriousDlqin2yrs' теперь категориальная
    print('HEAD of df_clean')
    print(df_clean.head(5))
    print('types of df_clean')
    print(df_clean.dtypes)
    print(df_clean.info())
    print(df_clean.describe())

    data_tuples = [tuple(row) for row in df_clean.values]
    print(f"Type of data_tuples: {type(data_tuples)}")
    first_item = data_tuples[0]
    print(f"First item in data_tuples: {first_item[0]}")
    print(f"Type of First item in data_tuples: {type(first_item[0])}")

    df_clean = df_clean.drop(columns=['id'])

    # Визуализация распределений
    df_clean.hist(bins=20, figsize=(15, 10))
    plt.tight_layout()
    plt.show()

    # Корреляция
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Корреляция признаков')
    plt.show()

    # Сохранение очищенных данных в PostgreSQL
    insert_dataframe_to_postgres(df_clean, 'public.cleaned_data', db_config)

    print("Загрузка очищенных данных из датасета в БД закончена")