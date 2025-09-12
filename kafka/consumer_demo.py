import json
from kafka import KafkaConsumer
from collections import defaultdict, Counter
import pandas as pd
import datetime
import matplotlib.pyplot as plt

user_behavior = defaultdict(list)
# История действий по пользователям
session_data = defaultdict(dict)
# Данные сессий
hourly_activity = defaultdict(int)
# Активность по часам
conversion_funnel = Counter()
# Воронка конверсии
revenue_data = []
# Данные о выручке
product_performance = defaultdict(lambda: {
    'views': 0, 'cart_adds': 0, 'purchases': 0, 'revenue': 0
})

consumer = KafkaConsumer(
    'user-actions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='demo-analytics-group'
)

action_stats = Counter()
product_views = Counter()
user_sessions = defaultdict(list)
user_actions = list()
def calculate_conversion_rates(user_actions):
    user_action_counts = defaultdict(lambda: {'view': 0, 'cart': 0, 'purchase': 0})

    # Count actions per user
    for action_obj in user_actions:
        user_id = action_obj['user_id']
        action = action_obj['action']

        if action == 'view_product':
            user_action_counts[user_id]['view'] += 1
        elif action == 'add_to_cart':
            user_action_counts[user_id]['cart'] += 1
        elif action == 'purchase':
            user_action_counts[user_id]['purchase'] += 1

    def safe_divide(a, b):
        return a / b if b > 0 else 0.0

    user_metrics = {}
    for user_id, counts in user_action_counts.items():
        view = counts['view']
        cart = counts['cart']
        purchase = counts['purchase']

        user_metrics[user_id] = {
            'view_to_cart': safe_divide(cart, view),
            'cart_to_purchase': safe_divide(purchase, cart),
            'overall_conversion': safe_divide(purchase, view),
            '_counts': {
                'view_product': view,
                'add_to_cart': cart,
                'purchase': purchase
            }
        }

    return user_metrics


def calculate_average_session_value(user_actions):

    session_purchase_prices = defaultdict(list)
    for action_obj in user_actions:
        if action_obj['action'] == 'purchase':
            session_id = action_obj['session_id']
            price = action_obj['price']
            if price is not None:
                session_purchase_prices[session_id].append(price)

    avg_price_per_session = {}
    for session_id, prices in session_purchase_prices.items():
        if prices:
            avg_price = sum(prices) / len(prices)
            avg_price_per_session[session_id] = avg_price

    return avg_price_per_session

def find_top_customers(user_actions):
    user_action_counts = Counter(action['user_id'] for action in user_actions)
    return user_action_counts.most_common(5)

def detect_abandonment_sessions1(actions_list):
    df = pd.DataFrame([
        {'session_id': a['session_id'], 'action': a['action']}
        for a in actions_list
    ])

    add_to_cart_sessions = set(df[df['action'] == 'add_to_cart']['session_id'])

    purchase_sessions = set(df[df['action'] == 'purchase']['session_id'])

    return list(add_to_cart_sessions - purchase_sessions)

def get_user_ids_with_logouts(user_actions):
    user_total_actions = defaultdict(int)
    user_logout_actions = defaultdict(int)

    for action_obj in user_actions:
        user_id = action_obj['user_id']
        action = action_obj['action']

        user_total_actions[user_id] += 1
        if action == 'logout':
            user_logout_actions[user_id] += 1

    result_users = []
    for user_id in user_total_actions:
        total = user_total_actions[user_id]
        logouts = user_logout_actions[user_id]

        if total > 0 and logouts / total > 0.5:
            result_users.append(user_id)

    return result_users

def get_purchase_cancellation(user_actions):


    user_action_counts = defaultdict(lambda: defaultdict(int))

    for action_obj in user_actions:
        user_id = action_obj['user_id']
        action = action_obj['action']
        user_action_counts[user_id][action] += 1

    result_users = []

    for user_id, action_counts in user_action_counts.items():
        logout_count = action_counts.get('logout', 0)
        purchase_count = action_counts.get('purchase', 0)
        add_to_cart_count = action_counts.get('add_to_cart', 0)


        if logout_count <= 10:
            continue

        if add_to_cart_count == 0:
            continue

        purchase_to_cart_ratio = purchase_count / add_to_cart_count

        if purchase_to_cart_ratio < 0.2:
            result_users.append(user_id)

    return result_users


def find_users_with_amount_actions(user_actions, threshold=10):

    user_minute_counts = defaultdict(lambda: defaultdict(int))

    for action_obj in user_actions:
        user_id = action_obj['user_id']
        dt = action_obj['timestamp']
        dt = datetime.datetime.fromisoformat(dt)
        minute_key = dt.replace(second=0, microsecond=0)

        user_minute_counts[user_id][minute_key] += 1

    result_users = []
    for user_id, minute_counts in user_minute_counts.items():
        if any(count > threshold for count in minute_counts.values()):
            result_users.append(user_id)

    return result_users

def find_users_no_purchase_in_last_events(user_actions):

    user_actions_map = defaultdict(list)

    for action_obj in user_actions:
        user_actions_map[action_obj['user_id']].append(action_obj)

    result_users = []

    for user_id, actions in user_actions_map.items():
        sorted_actions = sorted(actions, key=lambda x: x['timestamp'], reverse=True)


        last_50 = sorted_actions[:50]

        has_purchase_in_last_50 = any(action['action'] == 'purchase' for action in last_50)

        if not has_purchase_in_last_50:
            result_users.append(user_id)

    return result_users


def check_for_alerts(recent_events):
    return get_user_ids_with_logouts(recent_events), get_purchase_cancellation(recent_events), find_users_with_amount_actions(recent_events), find_users_no_purchase_in_last_events(recent_events)

def export_session_data_to_csv(data, **kwargs):
        default_params = {
       'index': False,  # Don't write row indices
       'encoding': 'utf-8'  # Use UTF-8 encoding
   }

        columns = ['user_id', 'action', 'product', 'timestamp', 'session_id', 'price']
        default_params.update(kwargs)
        df = pd.DataFrame.from_records(data)
     #   df = df.transpose()
        df.to_csv('user_sessions.csv', columns=columns, **default_params)

def save_real_time_metrics(metrics):
    with open('metrics_snapshot.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print("Metrics saved to 'metrics_snapshot.json'")

print("Starting to consume messages...")
print("Press Ctrl+C to stop and see analytics")

def plot_conversion(rows):
    user_ids = list(rows.keys())
    view_to_cart = list()
    cart_to_purchase = list()
    overall_conversion = list()
    for user_id in user_ids:
        row = rows[user_id]
        view_to_cart.append(str(row['view_to_cart']))
        cart_to_purchase.append(str(row['cart_to_purchase']))
        overall_conversion.append(str(row['overall_conversion']))


    plt.figure(figsize=(10, 5))
    plt.bar(user_ids, overall_conversion, color='teal', edgecolor='black')
    plt.bar(user_ids, cart_to_purchase, color='red', edgecolor='black')
    plt.bar(user_ids, view_to_cart, color='green', edgecolor='black')
    plt.title('Overall Conversion Rate per User')
    plt.xlabel('User ID')
    plt.ylabel('Conversion Rate (Purchase / View)')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_conversion(rows):
    user_ids = list(rows.keys())
    view_to_cart = list()
    cart_to_purchase = list()
    overall_conversion = list()
    for user_id in user_ids:
        row = rows[user_id]
        view_to_cart.append(str(row['view_to_cart']))
        cart_to_purchase.append(str(row['cart_to_purchase']))
        overall_conversion.append(str(row['overall_conversion']))
    return user_ids, view_to_cart, cart_to_purchase, overall_conversion


    # plt.figure(figsize=(10, 5))
    # plt.bar(user_ids, overall_conversion, color='teal', edgecolor='black')
    # plt.bar(user_ids, cart_to_purchase, color='red', edgecolor='black')
    # plt.bar(user_ids, view_to_cart, color='green', edgecolor='black')
    # plt.title('Overall Conversion Rate per User')
    # plt.xlabel('User ID')
    # plt.ylabel('Conversion Rate (Purchase / View)')
    # plt.ylim(0, 1.1)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    #
    # plt.show()

def show_top_products(user_actions):
    purchases = [item for item in user_actions if item['action'] == 'purchase']


    revenue_by_product = defaultdict(float)
    for purchase in purchases:
        if purchase['price'] is not None and purchase['product'] is not None:
            revenue_by_product[purchase['product']] += purchase['price']


    top_products = sorted(revenue_by_product.items(), key=lambda x: x[1], reverse=True)


    products_sorted = [item[0] for item in top_products]
    revenue_sorted = [item[1] for item in top_products]

    return products_sorted, revenue_sorted


    # plt.figure(figsize=(10, 6))
    # bars = plt.barh(products_sorted, revenue_sorted, color='mediumseagreen', edgecolor='black', linewidth=0.8)
    #
    #
    # plt.title('Топ продуктов по выручке (только покупки)', fontsize=16, fontweight='bold')
    # plt.xlabel('Выручка ($)', fontsize=12)
    # plt.ylabel('Продукт', fontsize=12)
    #
    #
    # plt.grid(axis='x', linestyle='--', alpha=0.6)
    # plt.tight_layout()
    #
    # plt.show()

def show_top_active_users(data):
    user_actions_count = defaultdict(int)
    user_revenue = defaultdict(float)

    for record in data:
        user_id = record['user_id']
        action = record['action']
        price = record['price']

        if action != 'logout':
            user_actions_count[user_id] += 1


        if action == 'purchase' and price is not None:
            user_revenue[user_id] += price

    # Подготовим данные для scatter plot
    users = list(user_actions_count.keys())
    actions = [user_actions_count[u] for u in users]
    revenue = [user_revenue[u] for u in users]

    scale_factor = 15
    sizes = [max(r * scale_factor, 20) for r in revenue]

    return users, actions, sizes, revenue

    # # Построение scatter plot
    # plt.figure(figsize=(12, 7))
    # scatter = plt.scatter(
    #     x=users,
    #     y=actions,
    #     s=sizes,           # размер зависит от выручки
    #     c=revenue,         # цвет зависит от выручки (градиент)
    #     cmap='viridis',
    #     alpha=0.7,
    #     edgecolors='black',
    #     linewidth=0.5
    # )
    #
    # # Настройки графика
    # plt.title('Активность пользователей по количеству действий и выручке', fontsize=16, fontweight='bold')
    # plt.xlabel('User ID', fontsize=12)
    # plt.ylabel('Количество действий (без logout)', fontsize=12)
    # plt.grid(True, linestyle='--', alpha=0.6)
    #
    # # Добавляем цветовую шкалу (colorbar)
    # cbar = plt.colorbar(scatter)
    # cbar.set_label('Выручка ($)', rotation=270, labelpad=20)
    #
    #
    # # Улучшаем ось X — делаем её целочисленной
    # plt.xticks(sorted(set(users)))
    #
    # # Отображаем график
    # plt.tight_layout()
    # plt.show()

def show_session_activities(data):
    session_action_counts = defaultdict(int)

    for record in data:
        session_id = record['session_id']
        session_action_counts[session_id] += 1  # Учитываем ВСЕ действия, включая logout

    # Получаем список количеств действий по сессиям
    counts = list(session_action_counts.values())
    return counts

    # # Построение гистограммы
    # plt.figure(figsize=(10, 6))
    # bins = range(1, max(counts) + 2)  # от 1 до max+1 (чтобы каждое целое число было в отдельном бине)
    # n, bins_edges, patches = plt.hist(counts, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
    #
    # # Настройки графика
    # plt.title('Распределение количества действий в сессии', fontsize=16, fontweight='bold')
    # plt.xlabel('Количество действий в сессии', fontsize=12)
    # plt.ylabel('Количество сессий', fontsize=12)
    #
    #
    # # Устанавливаем метки по оси X только на целые числа
    # plt.xticks(range(1, max(counts)+1))
    #
    # # Добавляем сетку
    # plt.grid(axis='y', linestyle='--', alpha=0.6)
    #
    # # Отображаем график
    # plt.tight_layout()
    # plt.show()

def show_activities_hours(data):
    hourly_activity = defaultdict(int)
    for record in data:
        ts_str = record['timestamp']
        try:
            dt = datetime.datetime.fromisoformat(ts_str)
            hour = dt.hour
            hourly_activity[hour] += 1
        except ValueError:
            continue

    # Готовим данные для line plot
    hours = list(range(24))  # Все часы от 0 до 23
    counts = [hourly_activity[h] for h in hours]
    print(f"counts: {counts}")
    return hours, counts

    # # Построение line plot
    # plt.figure(figsize=(12, 6))
    # plt.plot(hours, counts, marker='o', linestyle='-', color='navy', linewidth=2, markersize=6, label='Количество действий')
    #
    # # Настройки графика
    # plt.title('Распределение активности пользователей по часам суток', fontsize=16, fontweight='bold')
    # plt.xlabel('Час дня', fontsize=12)
    # plt.ylabel('Количество действий', fontsize=12)
    # plt.xticks(hours)  # Показываем все часы
    # plt.grid(True, linestyle='--', alpha=0.6)
    #
    #
    # # Улучшаем внешний вид
    # plt.legend()
    # plt.tight_layout()
    #
    # # Отображаем график
    # plt.show()

def show_actions_pie(data):
    action_counts = {}
    actions = ['view_product', 'add_to_cart', 'purchase', 'search', 'logout']
    for act in actions:
        action_counts[act] = 0

    for record in data:
        action = record['action']
        if action in action_counts:
            action_counts[action] += 1

    # Подготовка данных для pie chart
    labels = list(action_counts.keys())
    sizes = list(action_counts.values())

    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#FF99CC']

    return sizes, labels, colors

    # # Создание pie chart
    # plt.figure(figsize=(8, 8))
    # wedges, texts, autotexts = plt.pie(
    #     sizes,
    #     labels=labels,
    #     colors=colors,
    #     autopct='%1.1f%%',           # Отображать проценты
    #     startangle=90,               # Начинать с верхней точки
    #     explode=(0.05, 0.05, 0.05, 0.05, 0.05),  # Лёгкий "вылет" секторов для стиля
    #     shadow=True,
    #     textprops={'fontsize': 11}
    # )
    #
    # # Настройки графика
    # plt.title('Распределение типов действий пользователей', fontsize=16, fontweight='bold', pad=20)
    #
    # # Улучшаем читаемость процентов
    # for autotext in autotexts:
    #     autotext.set_color('white')
    #     autotext.set_fontweight('bold')
    #
    # # Добавляем легенду (если метки слишком длинные или перекрываются)
    # plt.legend(wedges, labels, title="Типы действий", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    #
    # # Убедимся, что круг — круг, а не овал
    # plt.axis('equal')
    #
    # # Отображаем график
    # plt.tight_layout()
    # plt.show()


message_count = 0

try:
    limit = 200
    dt = datetime.datetime.now()
    print(f"Start: {dt}")
    for message in consumer:
        event = message.value
        message_count += 1
        if message_count >= limit:
            break
        action_stats[event['action']] += 1

        if event.get('product'):
            product_views[event['product']] += 1

        user_sessions[event['user_id']].append(event['action'])
        user_actions.append(event)
        if event.get('price'):
            revenue_data.append(event['price'])

        print(f"Message {message_count}: {event}")

        if message_count % 10 == 0:
            print(f"\n--- Stats after {message_count} messages ---")
            print("Top actions:")
            for action, count in action_stats.most_common(3):
                print(f"  {action}: {count}")
            print(f"Active users: {len(user_sessions)}")
            print("-" * 50)
    save_real_time_metrics(user_actions)
    export_session_data_to_csv(user_actions)
    logouts, purchase_cancellations, top_users, users_no_purchase = check_for_alerts(user_actions)
    print(f"logouts: {logouts}, purchase_cancellations: {purchase_cancellations}, top_users: {top_users}, users_no_purchase: {users_no_purchase}")
    avg_session_value = calculate_average_session_value(user_actions)
    print(f"avg_session_value: {avg_session_value}")
    top_customers = find_top_customers(user_actions)
    print(f"top_customers: {top_customers}")
    conversions = calculate_conversion_rates(user_actions)
    print(f"overall_conversion: {conversions}")


    user_ids, view_to_cart, cart_to_purchase, overall_conversion = plot_conversion(conversions)
    products_sorted, revenue_sorted = show_top_products(user_actions)

    users, actions, sizes, revenue = show_top_active_users(user_actions)
    counts_sessions = show_session_activities(user_actions)
    hours, counts = show_activities_hours(user_actions)
    sizes_pie, labels, colors = show_actions_pie(user_actions)

    # show graphics
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle('📊 Анализ пользовательской активности и продаж (6 визуализаций)', fontsize=20, fontweight='bold')

    # plt.bar(user_ids, overall_conversion, color='teal', edgecolor='black')
    # plt.bar(user_ids, cart_to_purchase, color='red', edgecolor='black')
    # plt.bar(user_ids, view_to_cart, color='green', edgecolor='black')
    # plt.title('Overall Conversion Rate per User')
    # plt.xlabel('User ID')
    # plt.ylabel('Conversion Rate (Purchase / View)')
    # plt.ylim(0, 1.1)
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # --- График 1
    axes[0, 0].bar(user_ids, overall_conversion, color='teal', edgecolor='black')
    axes[0, 0].bar(user_ids, cart_to_purchase, color='red', edgecolor='black')
    axes[0, 0].bar(user_ids, view_to_cart, color='green', edgecolor='black')
    axes[0, 0].set_title('Воронка конверсии', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('user_id')
    axes[0, 0].set_ylabel('Конверсия')
    axes[0, 0].grid(axis='x', linestyle='--', alpha=0.7)

    # --- График 2: Топ продуктов по выручке (горизонтальный bar) ---
    axes[0, 1].barh(products_sorted, revenue_sorted, color='steelblue', edgecolor='black', linewidth=0.8)
    axes[0, 1].set_title('Топ продуктов по выручке', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Выручка ($)')
    axes[0, 1].grid(axis='x', linestyle='--', alpha=0.6)
    for i, v in enumerate(revenue_sorted):
        axes[0, 1].text(v + 100, i, f'${v:,}', va='center', fontsize=9)

    # --- График 3: Активность пользователей (scatter) ---
    scatter = axes[0, 2].scatter(users, actions, s=sizes, c=revenue, cmap='viridis', alpha=0.7, edgecolors='black', linewidth=0.5)
    axes[0, 2].set_title('Активность пользователей\n(размер = выручка, цвет = выручка)', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('User ID')
    axes[0, 2].set_ylabel('Кол-во действий')
    axes[0, 2].grid(True, linestyle='--', alpha=0.6)
    plt.colorbar(scatter, ax=axes[0, 2], label='Выручка ($)')

    # --- График 4: Распределение по часам (line plot) ---
    axes[1, 0].plot(hours, counts, marker='o', linestyle='-', color='navy', linewidth=2, markersize=5)
    axes[1, 0].set_title('Активность по часам суток', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Час дня')
    axes[1, 0].set_ylabel('Количество действий')
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    for i, v in enumerate(counts):
        if v > 0:
            axes[1, 0].text(i, v + 0.3, str(v), ha='center', va='bottom', fontsize=8)

    # --- График 5: Действия в сессии  ---
    bins = range(1, max(counts_sessions) + 2)
    n, bins_edges, patches = axes[1, 1].hist(counts_sessions, bins=bins, color='skyblue', edgecolor='black', alpha=0.8)
    axes[1, 1].set_title('Количество действий в сессии', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Количество действий')
    axes[1, 1].set_ylabel('Количество сессий')
    axes[1, 1].set_xticks(range(1, max(counts_sessions)+1))
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)
    for i, v in enumerate(n):
        if v > 0:
            axes[1, 1].text(bins_edges[i] + 0.3, v + 0.1, str(int(v)), ha='center', va='bottom', fontsize=9)

    # --- График 6: Распределение типов действий (pie chart) ---
    wedges, texts, autotexts = axes[1, 2].pie(
        sizes_pie,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05, 0.05, 0.05, 0.05),
        shadow=True
    )
    axes[1, 2].set_title('Распределение типов действий', fontsize=14, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    axes[1, 2].legend(wedges, labels, title="Типы", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    # --- Общие настройки ---
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    dt = datetime.datetime.now()
    print(f"Finish: {dt}")

except KeyboardInterrupt:
    print(f"\nStopped after {message_count} messages")