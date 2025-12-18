import requests

BASE_URL = "http://localhost:8000"

endpoints = ["/load", "/prepare", "/transform", "/train"]

def test_api():
    for endpoint in endpoints:
        url = BASE_URL + endpoint
        print(f"Отправка GET-запроса к {url}...")
        try:
            response = requests.get(url)
            print(f"Статус: {response.status_code}")
            print(f"Ответ: {response.text}\n")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при запросе к {url}: {e}\n")

if __name__ == "__main__":
    test_api()