import pandas as pd
import time
import csv
import string
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv('/home/ratmir/Desktop/all/AI university/AI_homework/lenta-ru-news.csv')
filename = '/home/ratmir/Desktop/all/AI university/AI_homework/lenta-ru-news.csv'

def using_pandas():
    print(df.columns)
    titles = df['title']

    translator = str.maketrans('', '', string.punctuation)

    all_words = []
    for title in titles:
        cleaned = str(title).translate(translator).lower()
        all_words.extend(cleaned.split())

    word_count = Counter(all_words)

    top_10 = word_count.most_common(10)


    for word, count in top_10:
        print(f"{word}: {count}")

def without_pandas_one_thread():
    all_words = []

    translator = str.maketrans('', '', string.punctuation)

    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)

        header = next(reader)

        try:
            title_index = header.index('title')
        except ValueError:
            raise ValueError("Колонка 'title' не найдена в CSV-файле")

        row_count = 0
        for row in reader:

            if len(row) <= title_index:
                row_count += 1
                continue

            title = row[title_index]

            if not title or title.strip() == '':
                row_count += 1
                continue

            cleaned = title.translate(translator).lower()
            words = cleaned.split()
            all_words.extend(words)

            row_count += 1

    if not all_words:
        print("Не найдено ни одного слова.")
    else:
        word_count = Counter(all_words)

        top_10 = word_count.most_common(10)

        print("Топ-10 самых частых слов:")
        for word, count in top_10:
            print(f"{word}: {count}")

def process_row(row):
    if not row or not row.strip():
        return []

    translator = str.maketrans('', '', string.punctuation)
    cleaned = row.translate(translator).lower()
    words = cleaned.split()
    return words

def without_pandas_in_threads(number_of_threads):
    rows_to_process = []
    all_words = []
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Заголовок

        try:
            title_index = header.index('title')
        except ValueError:
            raise ValueError("Колонка 'title' не найдена в CSV-файле")

        row_count = 0
        for row in reader:

            if len(row) <= title_index:
                row_count += 1
                continue
            title = row[title_index]
            rows_to_process.append(title)
            row_count += 1


        if not rows_to_process:
            print("Нет данных для обработки.")
            return


    with ThreadPoolExecutor(max_workers=number_of_threads) as executor:
            future_to_process = {executor.submit(process_row, item): item for item in rows_to_process}


            for future in as_completed(future_to_process):
                try:
                    words = future.result()
                    all_words.extend(words)
                except Exception as exc:
                    title = future_to_process[future]
                    print(f"Ошибка при обработке строки '{title}': {exc}")

    if not all_words:
            print("Не найдено ни одного слова после обработки.")
    else:
            word_count = Counter(all_words)
            top_10 = word_count.most_common(10)

            print("Топ-10 самых частых слов:")
            for word, count in top_10:
                print(f"{word}: {count}")

start_time = time.perf_counter()
#using_pandas() #Time: 11.41 sec
#without_pandas_one_thread() # Time: 45.18 sec
#without_pandas_in_threads(5) # Time: 121.71 sec
without_pandas_in_threads(3) # Time: 105.29 sec
end_time = time.perf_counter()
print(f"Time: {end_time - start_time:.2f} sec")

# Скажу честно, с Polars не сделал, т.к. у меня проблема какая-та с совместимостью и с версией питона. Я еще какое-то время поразбираюсь с этой проблемой,
# если не получится, то обращусь за помощью к вам.