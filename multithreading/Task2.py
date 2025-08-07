
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
import time

urls = ['https://httpbin.org/delay/1', 'https://httpbin.org/uuid', 'https://httpbin.org/bytes/128', 'https://httpbin.org/get', 'https://httpbin.org/delay/3']

def download_page(url):
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            data = response.read()
            return len(data)
    except Exception as e:
        return 0

def download_pages_parallel(n):
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(download_page, urls))

    end_time = time.perf_counter()

    for url, length in zip(urls, results):
        print(f"URL: {url}: {length} bytes")

    print(f"Time {n} threads: {end_time - start_time:.2f} sec")

def download_pages():
    start_time = time.perf_counter()
    results = list()

    for page in urls:
        results.append(download_page(page))

    end_time = time.perf_counter()

    # Выводим результаты
    for url, length in zip(urls, results):
        print(f"{url} -> {length} байт")

    print(f"Последовательно: {end_time - start_time:.2f} секунд")

if __name__ == "__main__":
    download_pages_parallel(3) # Time 3 threads: 12.72 sec
    download_pages_parallel(5) # Time 5 threads: 10.81 sec
    download_pages() # Последовательно: 32.25 секунд

# URL: https://httpbin.org/delay/1: 326 bytes
# URL: https://httpbin.org/uuid: 53 bytes
# URL: https://httpbin.org/bytes/128: 128 bytes
# URL: https://httpbin.org/get: 276 bytes
# URL: https://httpbin.org/delay/3: 326 bytes