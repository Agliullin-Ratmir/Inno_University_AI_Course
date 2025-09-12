import json
import random
import time
from kafka import KafkaProducer
from datetime import datetime

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)


def generate_user_action():
    actions = ['view_product', 'add_to_cart', 'purchase', 'search', 'logout']
    products = ['laptop', 'smartphone', 'tablet', 'headphones', 'keyboard', 'mouse']

    return {
        'user_id': str(random.randint(1, 100)),
        'action': random.choice(actions),
        'product': random.choice(products) if random.random() > 0.3 else None,
        'timestamp': datetime.now().isoformat(),
        'session_id': f"session_{random.randint(1, 20)}",
        'price': random.randint(50, 1500) if random.random() > 0.7 else None
    }


print("Starting to send events...")
print("Press Ctrl+C to stop")

try:
    for i in range(200):
        event = generate_user_action()
        producer.send('user-actions', value=event)
        print(f"Event {i + 1}: {event}")
        time.sleep(2)

except KeyboardInterrupt:
    print("\nStopping producer...")

producer.flush()
producer.close()
print("Producer stopped!")

