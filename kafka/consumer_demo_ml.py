import json
from kafka import KafkaConsumer
import numpy as np

# Р”Р»СЏ РґРµРјРѕРЅСЃС‚СЂР°С†РёРё РёСЃРїРѕР»СЊР·СѓРµРј РїСЂРѕСЃС‚СѓСЋ Р»РѕРіРёРєСѓ РІРјРµСЃС‚Рѕ СЂРµР°Р»СЊРЅРѕР№ ML-Р±РёР±Р»РёРѕС‚РµРєРё

consumer = KafkaConsumer(
    'user-actions',
    bootstrap_servers=['localhost:9092'],
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='latest',  # Р§РёС‚Р°РµРј С‚РѕР»СЊРєРѕ РЅРѕРІС‹Рµ СЃРѕРѕР±С‰РµРЅРёСЏ
    group_id='ml-pipeline-group'
)


def process_for_ml(event_data):
    """РџСЂРµРѕР±СЂР°Р·СѓРµРј СЃРѕР±С‹С‚РёРµ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ РІ РїСЂРёР·РЅР°РєРё РґР»СЏ ML РјРѕРґРµР»Рё"""
    features = []

    # РџСЂРёР·РЅР°Рє 1: С†РµРЅР° (РµСЃР»Рё РµСЃС‚СЊ)
    price = event_data.get('price', 0)
    features.append(price if price else 100)  # СЃСЂРµРґРЅСЏСЏ С†РµРЅР° РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ

    # РџСЂРёР·РЅР°Рє 2: РєР°С‚РµРіРѕСЂРёСЏ С‚РѕРІР°СЂР° (С…РµС€РёСЂСѓРµРј РЅР°Р·РІР°РЅРёРµ)
    product = event_data.get('product', 'unknown')
    category_feature = hash(product) % 100 if product else 0
    features.append(category_feature)

    # РџСЂРёР·РЅР°Рє 3: СЃРµРіРјРµРЅС‚ РїРѕР»СЊР·РѕРІР°С‚РµР»СЏ (РїРѕ user_id)
    user_segment = event_data['user_id'] % 10
    features.append(user_segment)

    # РџСЂРёР·РЅР°Рє 4: С‚РёРї РґРµР№СЃС‚РІРёСЏ
    action_mapping = {
        'view_product': 1,
        'add_to_cart': 2,
        'purchase': 3,
        'search': 4,
        'logout': 5
    }
    action_feature = action_mapping.get(event_data['action'], 0)
    features.append(action_feature)

    return np.array(features).reshape(1, -1)


def simple_recommendation_model(features):
    """РџСЂРѕСЃС‚Р°СЏ РёРјРёС‚Р°С†РёСЏ ML-РјРѕРґРµР»Рё РґР»СЏ СЂРµРєРѕРјРµРЅРґР°С†РёР№"""
    # Р’ СЂРµР°Р»СЊРЅРѕСЃС‚Рё Р·РґРµСЃСЊ Р±С‹Р»Р° Р±С‹ РѕР±СѓС‡РµРЅРЅР°СЏ РјРѕРґРµР»СЊ
    score = np.sum(features) % 100

    if score > 70:
        return "premium_products"
    elif score > 40:
        return "popular_products"
    else:
        return "budget_products"


print("ML Pipeline started. Processing events for recommendations...")

for message in consumer:
    event = message.value

    # РџСЂРµРѕР±СЂР°Р·СѓРµРј СЃРѕР±С‹С‚РёРµ РІ РїСЂРёР·РЅР°РєРё
    features = process_for_ml(event)

    # РџРѕР»СѓС‡Р°РµРј СЂРµРєРѕРјРµРЅРґР°С†РёСЋ РѕС‚ "РјРѕРґРµР»Рё"
    recommendation = simple_recommendation_model(features)

    print(f"User {event['user_id']} performed '{event['action']}' в†’ Recommend: {recommendation}")
    print(f"  Features used: {features.flatten()}")
    print("-" * 60)