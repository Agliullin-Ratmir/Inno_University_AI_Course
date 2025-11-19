from util import insert_dataframe_to_postgres, read_postgres_to_pandas_psycopg2
import pandas as pd
def prepare():
    db_config = {
        'dbname': 'credit_scoring',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'db',
        'port': '5432'
    }
    df_clean = read_postgres_to_pandas_psycopg2(db_config, 'public.cleaned_data')

    featured_df = df_clean.copy()
    featured_df = featured_df.drop(columns=['seriousDlqin2yrs'])
    featured_df = featured_df.drop(columns=['id'])

    featured_df = featured_df.apply(pd.to_numeric, errors='coerce')


    insert_dataframe_to_postgres(featured_df, 'public.featured_data', db_config)
    print("Загрузка подготовленных данных из датасета в БД закончена")