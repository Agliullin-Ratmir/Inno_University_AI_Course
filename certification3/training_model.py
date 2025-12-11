from util import insert_dataframe_to_postgres, read_postgres_to_pandas_psycopg2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sqlalchemy import create_engine
from scipy import stats
import psycopg2
from psycopg2.extras import execute_batch
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# --- Добавлен импорт для нейронной сети ---
from sklearn.neural_network import MLPClassifier
# --- Добавлен импорт для более подробного логирования ---
import time


def train():
    db_config = {
        'dbname': 'credit_scoring',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost', # change to 'db' for using inside docker
        'port': '5432'
    }
    target_col = 'seriousDlqin2yrs'
    featured_df = read_postgres_to_pandas_psycopg2(db_config, 'public.featured_data')
    df_clean = read_postgres_to_pandas_psycopg2(db_config, 'public.cleaned_data')
    target = df_clean[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        featured_df,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target
    )

    # --- 2. Обучение базовой модели ---
    model = LogisticRegression(max_iter=1000)  # увеличиваем max_iter на всякий случай
    print('Amount of positive target: ' + str((target == 1.0).sum()))
    print('Amount of negative target: ' + str((target == 0.0).sum()))
    model.fit(X_train, y_train)

    # --- 3. Предсказания базовой модели ---
    y_pred = model.predict(X_test)

    # --- 4. Метрики базовой модели ---
    grid_search_start_time = time.time()
    accuracy = accuracy_score(y_test, y_pred)
    y_test = pd.to_numeric(y_test).astype(int)
    y_pred = pd.to_numeric(y_pred).astype(int)
    precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    print("=== Метрики базовой модели (LogisticRegression) ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    grid_search_time = time.time() - grid_search_start_time
    print(f"Время LinearRegression: {grid_search_time:.2f} секунд")
    print("\n=== Отчет классификации ===")
    print(classification_report(y_test, y_pred))


    # --- 5. Анализ ошибок базовой модели ---
    # Найдем индексы, где предсказание не совпадает с истиной
    wrong_predictions = X_test[y_test != y_pred]
    if not wrong_predictions.empty:
        print("\n=== Примеры неправильно классифицированных наблюдений ===")
        print(wrong_predictions.head(10))  # вывести первые 10 неправильных
    else:
        print("\n=== Нет неправильно классифицированных наблюдений в тестовом наборе ===")

    # --- Создание и обучение StandardScaler ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Определение моделей и их параметров для поиска ---
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [10],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'random_state': [42]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100],
                'max_depth': [5],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'learning_rate': [0.1],
                'random_state': [42]
            }
        },
        'SVM': {
            'model': SVC(random_state=42),
            'params': {
                'C': [1.0],
                'kernel': ['rbf'],
                'gamma': ['scale'],
                'random_state': [42]
            }
        },
        # --- Добавление нейронной сети ---
        'NeuralNetwork': {
            'model': MLPClassifier(random_state=42, max_iter=500), # max_iter увеличен для сходимости
            'params': {
                'hidden_layer_sizes': [(50, 50), (100,)], # Пример: 2 скрытых слоя по 50 нейронов или 1 слой из 100
                'activation': ['relu'],
                'solver': ['adam'],
                'alpha': [0.0001, 0.001], # Параметр регуляризации
                'learning_rate': ['adaptive'], # Адаптивная скорость обучения
                'random_state': [42]
            }
        }
    }

    # --- 3. Обучение и оценка моделей ---
    results = {}
    for name, model_info in models.items():
        print(f"\n=== Обработка модели: {name} ===")
        model = model_info['model']
        params = model_info['params']

        # Для SVM и NeuralNetwork используем масштабированные данные
        if name in ['SVM', 'NeuralNetwork']:
            search_X_train = X_train_scaled
            search_X_test = X_test_scaled
            print(f"Используются масштабированные данные для {name}.")
        else:
            search_X_train = X_train
            search_X_test = X_test
            print(f"Используются исходные данные для {name}.")

        # --- GridSearchCV ---
        grid_search_start_time = time.time()
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params,
            cv=3,  # 3-фолд кросс-валидация
            scoring='f1',  # Оптимизация по F1-score
            n_jobs=-1, # Использовать все доступные ядра
            verbose=0 # Уменьшено для краткости
        )
        grid_search.fit(search_X_train, y_train)
        grid_search_time = time.time() - grid_search_start_time

        # --- Лучшая модель ---
        best_model = grid_search.best_estimator_

        # --- Предсказания ---
        y_pred = best_model.predict(search_X_test)
        y_pred = pd.to_numeric(y_pred).astype(int)

        # --- Метрики ---
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results[name] = {
            'model': best_model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'grid_search_time': grid_search_time # Время поиска гиперпараметров
        }

        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Время GridSearchCV: {grid_search_time:.2f} секунд")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"\n--- Отчет классификации модели {name} ---")
        print(classification_report(y_test, y_pred))

    # --- 4. Сравнение моделей ---
    print("\n=== Сравнение моделей ===")
    comparison_df = pd.DataFrame({
        name: [
            results[name]['accuracy'],
            results[name]['precision'],
            results[name]['recall'],
            results[name]['f1'],
            results[name]['grid_search_time'] # Добавлено время
        ]
        for name in results
    }, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'GridSearch Time (s)']).round(4) # Округление для лучшего отображения
    print(comparison_df)

    # --- 5. Анализ важности признаков (только для Random Forest и Gradient Boosting) ---
    for name in ['RandomForest', 'GradientBoosting']:
        if name in results:
            model = results[name]['model']
            if hasattr(model, 'feature_importances_'):
                print(f"\n=== Важность признаков ({name}) ===")
                importances = model.feature_importances_
                feature_names = X_train.columns
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values(by='importance', ascending=False)
                print(importance_df)