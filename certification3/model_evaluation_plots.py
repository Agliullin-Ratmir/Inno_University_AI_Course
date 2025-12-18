import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)
from util import read_postgres_to_pandas_psycopg2

# Настройка стиля
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 10})

# Конфигурация
db_config = {
    'dbname': 'credit_scoring',
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432'
}
TARGET_COL = 'seriousDlqin2yrs'
MODEL_NAMES = ['RandomForest', 'GradientBoosting', 'SVM', 'NeuralNetwork']
OUTPUT_DIR = 'evaluation_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cm_{model_name}.png", dpi=150)
    plt.close()

def plot_roc_curve(y_true, y_score, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve — {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_{model_name}.png", dpi=150)
    plt.close()
    return roc_auc

def plot_precision_recall_curve(y_true, y_score, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where='post', color='b', alpha=0.8)
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve — {model_name}\n(AP = {avg_precision:.3f})')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pr_{model_name}.png", dpi=150)
    plt.close()
    return avg_precision

def analyze_errors(X_test, y_true, y_pred, model_name, feature_columns):
    # Ложно-положительные (FP): предсказано 1, но на самом деле 0
    fp_mask = (y_pred == 1) & (y_true == 0)
    # Ложно-отрицательные (FN): предсказано 0, но на самом деле 1
    fn_mask = (y_pred == 0) & (y_true == 1)

    X_test_df = pd.DataFrame(X_test, columns=feature_columns)

    # Выберем топ-5 числовых признаков (с наибольшей дисперсией)
    numeric_cols = X_test_df.select_dtypes(include=[np.number]).columns[:10]

    if len(numeric_cols) == 0:
        return

    n_cols = min(5, len(numeric_cols))
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(numeric_cols[:n_cols]):
        # FP
        axes[0, i].hist(X_test_df.loc[fp_mask, col].dropna(), bins=20, alpha=0.7, color='red', label='FP')
        axes[0, i].set_title(f'FP: {col}')
        # FN
        axes[1, i].hist(X_test_df.loc[fn_mask, col].dropna(), bins=20, alpha=0.7, color='orange', label='FN')
        axes[1, i].set_title(f'FN: {col}')

    plt.suptitle(f'Анализ ошибок — {model_name}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{OUTPUT_DIR}/errors_{model_name}.png", dpi=150)
    plt.close()

def main():
    print("Загрузка данных...")
    featured_df = read_postgres_to_pandas_psycopg2(db_config, 'public.featured_data')
    cleaned_df = read_postgres_to_pandas_psycopg2(db_config, 'public.cleaned_data')
    y = cleaned_df[TARGET_COL].astype(int)

    # Разделение на train/test (так же, как в train())
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        featured_df, y, test_size=0.2, random_state=42, stratify=y
    )
    feature_columns = list(X_train.columns)

    # Загрузка scaler
    try:
        scaler = joblib.load("scaler.joblib")
    except FileNotFoundError:
        print("⚠️ scaler.joblib не найден. Убедитесь, что он был сохранён в train().")
        scaler = None

    results_summary = []

    for name in MODEL_NAMES:
        model_path = f"model_{name}.joblib"
        if not os.path.exists(model_path):
            print(f"Модель {name} не найдена — пропускаем.")
            continue

        print(f"\nОбработка модели: {name}")
        model = joblib.load(model_path)

        # Подготовка данных
        if name in ['SVM', 'NeuralNetwork']:
            if scaler is None:
                print(f"⚠️ Пропуск {name}: нет scaler'а")
                continue
            X_test_input = scaler.transform(X_test)
        else:
            X_test_input = X_test

        # Предсказания
        y_pred = model.predict(X_test_input).astype(int)

        # Для ROC и PR нужны вероятности или decision function
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_input)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test_input)
        else:
            y_score = y_pred  # fallback

        # 1. Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, name)

        # 2. ROC-AUC
        roc_auc = plot_roc_curve(y_test, y_score, name)

        # 3. Precision-Recall
        avg_prec = plot_precision_recall_curve(y_test, y_score, name)

        # 4. Анализ ошибок (только если есть ошибки)
        if (y_pred != y_test).any():
            analyze_errors(X_test, y_test, y_pred, name, feature_columns)

        # Метрики
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results_summary.append({
            'Model': name,
            'Accuracy': acc,
            'F1-score': f1,
            'ROC-AUC': roc_auc,
            'Avg Precision': avg_prec
        })

        print(f"  → ROC-AUC: {roc_auc:.4f}, AP: {avg_prec:.4f}, F1: {f1:.4f}")

    # Итоговая таблица
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(f"{OUTPUT_DIR}/model_comparison.csv", index=False)
    print(f"\n✅ Все графики сохранены в папку '{OUTPUT_DIR}'")
    print(summary_df.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    main()