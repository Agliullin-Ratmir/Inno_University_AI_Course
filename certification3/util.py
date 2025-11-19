import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_batch
import psycopg2
from sqlalchemy import create_engine
def insert_dataframe_to_postgres(df: pd.DataFrame, table_name: str, db_config: dict):
    """
    Вставляет датафрейм pandas в таблицу PostgreSQL посредством SQL-запросов INSERT.
    :param df: pandas.DataFrame — данные для вставки
    :param table_name: str — имя таблицы в PostgreSQL
    :param db_config: dict — параметры подключения к БД (dbname, user, password, host, port)
    """
    # --- Подключение к БД ---
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    # --- Определение структуры таблицы на основе типов данных в DataFrame ---
    # Сопоставление типов pandas с PostgreSQL
    type_mapping = {
        'object': 'TEXT',  # включая строки
        'int64': 'INTEGER',
        'int32': 'INTEGER',
        'float64': 'NUMERIC',
        'float32': 'NUMERIC',
        'bool': 'BOOLEAN',
        'datetime64[ns]': 'TIMESTAMP',
        'timedelta64[ns]': 'INTERVAL'
    }
    # Игнорируем 'id SERIAL PRIMARY KEY' — PostgreSQL сам генерирует
    columns = df.columns
    types = [type_mapping.get(str(df[col].dtype), 'TEXT') for col in columns]
    # Создаём строку для SQL-запроса на создание таблицы
    columns_with_types = ", ".join([f'"{col}" {dtype}' for col, dtype in zip(columns, types)])
    print(f"columns_with_types: {columns_with_types}")
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id SERIAL PRIMARY KEY,
        {columns_with_types}
    );
    """
    cursor.execute(create_table_sql)
    conn.commit()
    # --- Подготовка данных ---
    # Заменяем 'NA' или 'NaN' на None (NULL в SQL)
    df = df.replace({pd.NA: None, np.nan: None})
    # Конвертируем датафрейм в список кортежей
    data_tuples = [tuple(row) for row in df.values]
    # print(f"data_tuples: {data_tuples}")
    has_numpy_float64 = any(isinstance(element, np.floating) for row_tuple in data_tuples for element in row_tuple)
    if has_numpy_float64:
        print("Обнаружены элементы типа numpy.float64 в data_tuples.")
        # Преобразуем кортежи, заменяя numpy.float64 на Python float
        converted_data_tuples = [
            tuple(float(x) if isinstance(x, np.floating) else x for x in original_tuple)
            for original_tuple in data_tuples
        ]
        print("Кортежи преобразованы: numpy.float64 -> float.")
    else:
        print("Элементы типа numpy.float64 в data_tuples НЕ обнаружены.")
        converted_data_tuples = data_tuples # Используем оригинальные кортежи
    # --- SQL для вставки ---
    # Заголовки в кавычки, чтобы избежать конфликта с ключевыми словами
    quoted_columns = [f'"{col}"' for col in columns]
    placeholders = ', '.join(['%s'] * len(columns))
    insert_sql = f"""
    INSERT INTO {table_name} ({', '.join(quoted_columns)}) VALUES ({placeholders});
    """
    # --- Вставка данных ---
    execute_batch(cursor, insert_sql, converted_data_tuples, page_size=1000)
    # --- Фиксируем изменения и закрываем соединение ---
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Данные из DataFrame успешно вставлены в таблицу '{table_name}'.")

def read_postgres_to_pandas_sqlalchemy(connection_string, table_name):
    """
    Read PostgreSQL table to pandas DataFrame using SQLAlchemy

    Args:
        connection_string: PostgreSQL connection string (e.g., 'postgresql://user:password@host:port/database')
        table_name: Name of the table to read

    Returns:
        pandas.DataFrame
    """
    # Create engine
    engine = create_engine(connection_string)

    # Read table into DataFrame
    df = pd.read_sql_table(table_name, engine)

    return df

# Method 2: Using psycopg2 with pandas
def read_postgres_to_pandas_psycopg2(db_config: dict, table_name):
    """
    Read PostgreSQL table to pandas DataFrame using psycopg2

    Args:
        host: Database host
        database: Database name
        user: Username
        password: Password
        port: Port number
        table_name: Name of the table to read

    Returns:
        pandas.DataFrame
    """
    # Create connection
    conn = psycopg2.connect(**db_config)

    # Read table into DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    # Close connection
    conn.close()

    return df