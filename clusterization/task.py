import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('./Online Retail.xlsx')

print(f"Размер датасета: {df.shape}")
print("\nПервые строки:")
print(df.head())
print("\nИнформация о колонках:")
print(df.info())

print("\nПропущенные значения:")
print(df.isnull().sum())

df = df.dropna(subset=['CustomerID'])

df_clean = df[(df['Quantity'] >= 0) & (df['UnitPrice'] >= 0)]

print(f"\nРазмер после очистки: {df_clean.shape}")
print(f"Уникальных клиентов: {df_clean['CustomerID'].nunique()}")


max_date = df_clean['InvoiceDate'].max()
current_date = max_date + pd.Timedelta(days=1)
print(f"Референсная дата для расчета Recency: {current_date}")


df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (current_date - x.max()).days,
    'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
})


rfm.columns = ['Recency', 'Frequency', 'Monetary']

print("\nСтатистика RFM-метрик:")
print(rfm.describe())
print("\nПримеры клиентов:")
print(rfm.head(10))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(rfm['Recency'], bins=30, color='skyblue', edgecolor='black')
axes[0].set_title('Распределение Recency')
axes[0].set_xlabel('Дни с последней покупки')


axes[1].hist(rfm['Frequency'], bins=30, color='lightgreen', edgecolor='black')
axes[1].set_title('Распределение Frequency')
axes[1].set_xlabel('Количество покупок')


axes[2].hist(rfm['Monetary'], bins=30, color='salmon', edgecolor='black')
axes[2].set_title('Распределение Monetary')
axes[2].set_xlabel('Общая сумма покупок')


plt.tight_layout()
plt.show()

from sklearn.preprocessing import StandardScaler

rfm_metrics = rfm[['Recency', 'Frequency', 'Monetary']]

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_metrics)

rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'], index=rfm.index)

from sklearn.cluster import KMeans
inertias = []

K_range = range(2, 9)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(rfm_scaled_df)  # используем нормализованные RFM-метрики
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Количество кластеров (K)')
plt.ylabel('Inertia (внутрикластерная сумма квадратов)')
plt.title('Elbow Method для выбора оптимального K')
plt.grid(True)
plt.show()

optimal_k = 5
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['Cluster'] = kmeans_final.fit_predict(rfm_scaled_df)
print(f"\nРаспределение клиентов по кластерам:")
print(rfm['Cluster'].value_counts().sort_index())
print(f"\nДоля клиентов в каждом кластере:")
print(rfm['Cluster'].value_counts(normalize=True).sort_index())

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
rfm_pca = pca.fit_transform(rfm_scaled_df)
rfm_pca_df = pd.DataFrame(rfm_pca, columns=['PC1', 'PC2'], index=rfm_scaled_df.index)
print(f"\nОбъясненная дисперсия: {pca.explained_variance_ratio_}")
print(f"Суммарно: {pca.explained_variance_ratio_.sum():.2%}")

plt.figure(figsize=(10, 7))
sns.scatterplot(
    data=rfm_pca_df,
    x='PC1',
    y='PC2',
    palette='tab10',
    alpha=0.7,
    s=50
)

plt.title('Кластеры RFM (визуализация PCA, 2 компоненты)')
plt.xlabel('Главная компонента 1')
plt.ylabel('Главная компонента 2')
plt.legend(title='Кластер')
plt.grid(True)
plt.show()


# 3
rfm_cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()
print(rfm_cluster_means)

sns.set(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))


metrics = ['Recency', 'Frequency', 'Monetary']
titles = ['Recency по кластерам', 'Frequency по кластерам', 'Monetary по кластерам']


for ax, metric, title in zip(axes, metrics, titles):
    sns.boxplot(data=rfm, x='Cluster', y=metric, ax=ax, palette='Set2')
    ax.set_title(title)
    ax.set_xlabel('Кластер')
    ax.set_ylabel(metric)

plt.tight_layout()
plt.show()

cluster_means_norm = (rfm_cluster_means - rfm_cluster_means.mean()) / rfm_cluster_means.std()

# Или, если хотите оставить абсолютные значения — используйте просто `cluster_means`

plt.figure(figsize=(10, 6))
sns.heatmap(
    cluster_means_norm.T,  # транспонируем: метрики по строкам, кластеры по столбцам
    annot=rfm_cluster_means.T.round(1),  # отображаем реальные значения (округлённые)
    fmt='',  # формат уже задан в annot
    cmap='RdYlBu_r',  # цветовая схема: синий = низкое, красный = высокое
    cbar_kws={'label': 'Стандартизированное значение'}
)

plt.title('Тепловая карта средних RFM-значений по кластерам', fontsize=14)
plt.xlabel('Кластер')
plt.ylabel('RFM-метрика')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


cluster_names = {
    0: "Новички",
    1: "Спящие клиенты",
    2: "VIP-клиенты",
    3: "Постоянные клиенты",
    4: "В зоне риска"
}
rfm['Segment'] = rfm['Cluster'].map(cluster_names)
print("\n=== ИТОГОВАЯ СЕГМЕНТАЦИЯ ===")
print(rfm.groupby('Segment')[['Recency', 'Frequency',
                              'Monetary']].mean().round(2))
print(f"\nРаспределение клиентов по сегментам:")
print(rfm['Segment'].value_counts())