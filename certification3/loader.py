import pandas as pd
from util import insert_dataframe_to_postgres

def load():
    print("Загрузка из датасета в БД стартовала")
    # Загрузка данных из датасета в БД
    df = pd.read_csv('./cs-training.csv')
    df.columns = [
        'seriousDlqin2yrs',
        'RevolvingUtilizationOfUnsecuredLines',
        'age',
        'NumberOfTime30_59DaysPastDueNotWorse',
        'DebtRatio',
        'MonthlyIncome',
        'NumberOfOpenCreditLinesAndLoans',
        'NumberOfTimes90DaysLate',
        'NumberRealEstateLoansOrLines',
        'NumberOfTime60_89DaysPastDueNotWorse',
        'NumberOfDependents'
    ]

    db_config = {
        'dbname': 'credit_scoring',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'db',
        'port': '5432'
    }

    insert_dataframe_to_postgres(df, 'public.raw_data', db_config)

    print("Загрузка из датасета в БД закончена")