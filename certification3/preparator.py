from util import insert_dataframe_to_postgres, read_postgres_to_pandas_psycopg2
from sklearn.preprocessing import StandardScaler
import pandas as pd
def prepare():
    db_config = {
        'dbname': 'credit_scoring',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost', # change to 'db' for using inside docker
        'port': '5432'
    }

    df_clean = read_postgres_to_pandas_psycopg2(db_config, 'public.cleaned_data')

    target = df_clean['seriousDlqin2yrs'].copy()

    featured_df = df_clean.drop(columns=['seriousDlqin2yrs', 'id']).copy()
    featured_df = featured_df.apply(pd.to_numeric, errors='coerce')

    scaler = StandardScaler()
    featured_df_scaled = scaler.fit_transform(featured_df)

    featured_df = pd.DataFrame(
        featured_df_scaled,
        columns=featured_df.columns,
        index=featured_df.index
    )

    featured_df['seriousDlqin2yrs'] = target

    print(featured_df.info())
    print(featured_df.describe())

    insert_dataframe_to_postgres(featured_df, 'public.featured_data', db_config)
    print("Загрузка подготовленных (нормализованных) данных из датасета в БД закончена")