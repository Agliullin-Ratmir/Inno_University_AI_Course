import requests
import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

API_KEY = "1d9abf121912c2889d0a52329906b456"
cities = ["Moscow", "Saint Petersburg", "Sochi", "Kazan", "Novosibirsk"]
PATH_TO_RAW = "./weather_tourism_pipeline/data/raw/openweather_api/"
PATH_TO_CLEANED = "./weather_tourism_pipeline/data/cleaned/"
PATH_TO_ENRICHED = "./weather_tourism_pipeline/data/enriched/"
PATH_TO_AGGREGATED = "./weather_tourism_pipeline/data/aggregated/"

CITIES_MONTH = {'Москва': [1,2,3,4,5,6,7,8,9,10,11,12],
                 'Казань': [9,10,11],
                 'Новосибирск': [1,2,12],
                 'Санкт-Петербург': [6,7,8],
                 'Сочи': [3,4,5]}
CITIES_RUSSIAN = {'Москва': 'Moscow', 'Казань': 'Kazan', 'Новосибирск': 'Novosibirsk', 'Санкт-Петербург': 'Saint_Petersburg', 'Сочи': 'Sochi'}
CITIES_INFO = {'Moscow': {'federal_district':'ЦФО', 'timezone': '+2', 'population': 15000000, 'season': 'Круглогодично', 'comfort': 8, 'activity':'Достопримечательности'},
                'Kazan': {'federal_district':'ПФО', 'timezone': '+2', 'population': 1000000, 'season': 'Осень', 'comfort': 7, 'activity':'Кухня'},
                'Novosibirsk': {'federal_district':'СФО', 'timezone': '+6', 'population': 1000000, 'season': 'Зима', 'comfort': 6, 'activity':'Прогулки по городу'},
                'Saint_Petersburg': {'federal_district':'СЗФО', 'timezone': '+2', 'population': 3000000, 'season': 'Лето', 'comfort': 5, 'activity':'Морские прогулки'},
                'Sochi': {'federal_district':'ЮФО', 'timezone': '+2', 'population': 560000, 'season': 'Весна', 'comfort': 9, 'activity':'Море'}}
CITIES_SEASON = {'Moscow': 'Круглогодично', 'Kazan': 'Осень', 'Novosibirsk': 'Зима', 'Saint_Petersburg': 'Лето', 'Sochi': 'Весна'}

def save_to_json_file(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_to_log_file(filename, data):
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(str(data))
def collect_weather_data(day_path, timestamp):


    for city in cities:
        url =f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric&lang=ru"
        response = requests.get(url)
        if response.status_code == 200:
            weather_data = response.json()
            weather_data['_metadata'] = {
                'collection_time': datetime.now().isoformat(),
                'source': 'openweathermap.org',
                'city_query': city
            }
            filename = f"{day_path}/weather_{city}_{timestamp}.json"
            save_to_json_file(filename, weather_data)
def create_cleaned(data_folder_path, log_file, cleaned_filename):
    data_folder = Path(data_folder_path)
    if not data_folder.exists() or not data_folder.is_dir():
        save_to_log_file(log_file, f"Folder '{data_folder_path}' does not exist or is not a directory.\n")
    records = []

    count = 0
    for json_file in data_folder.glob("*.json"):
        count += 1
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            main = data.get('main', {})
            temp = main.get('temp', None)
            feels_like = main.get('feels_like', None)
            city_name = data.get('name', 'Unknown')
            humidity = f"{main.get('humidity', None)}%"
            pressure = f"{int(round(main.get('pressure', None)/1,333))} мм рт.ст."
            wind = data.get('wind', {})
            wind_speed = f"{wind.get('speed', None)} м/c"
            weather = data.get('weather', {})
            weather_description = weather[0].get('description', None)
            metadata = data.get('_metadata', None)
            collection_time = metadata.get('collection_time', None)

            temp_int = int(round(temp))
            feels_like_int = int(round(feels_like))
            records.append({
                    'city_name': city_name,
                    'temperature': temp_int,
                    'feels_like': feels_like_int,
                    'pressure': pressure,
                    'wind_speed': wind_speed,
                    'humidity': humidity,
                    'weather_description': weather_description,
                    'collection_time': collection_time
            })

        except Exception as e:
            print(f"❌ Error reading {json_file.name}: {e}")


    save_to_log_file(log_file, f"Количество исходных записей: {count}\n")
    save_to_log_file(log_file, f"Количество очищенных записей: {len(records)}\n")

    if records:
        df = pd.DataFrame(records)

        df.to_csv(cleaned_filename, index=False, encoding='utf-8')
        save_to_log_file(log_file, f"Successfully saved {len(records)} records to '{cleaned_filename}\n'")
        save_to_log_file(log_file, df)
        return df
    else:
        save_to_log_file(log_file, "No valid weather records found.\n")

def create_enriched(enriched_csv, df, month):
    if df is None:
        return
    df['federal_district'] = None
    df['timezone'] = None
    df['population'] = None
    df['season'] = None
    df['comfort'] = None
    df['recommended_activity'] = None
    df['comfort_index'] = None
    df ['tourist_season_match'] = None
    for index, row in df.iterrows():
        city_name = row['city_name']
        print(f"city_name: {city_name}")
        city = CITIES_RUSSIAN.get(city_name)
        print(f"city: {city}")
        info = CITIES_INFO.get(city)
        print(f"info: {info}")
        df.at[index, 'federal_district'] = info.get('federal_district')
        df.at[index, 'timezone'] = info.get('timezone')
        df.at[index, 'population'] = info.get('population')
        df.at[index, 'season'] = info.get('season')
        df.at[index, 'comfort'] = info.get('comfort')
        df.at[index, 'recommended_activity'] = info.get('activity')
        df.at[index, 'comfort_index'] = float(row['temperature']) - float(row['wind_speed'].replace('м/c', '')) + float(row['humidity'].replace('%', ''))
        df.at[index, 'tourist_season_match'] = month in CITIES_MONTH
    df.to_csv(enriched_csv, index=False, encoding='utf-8')
    return df

def create_aggregated(df):
    df_sorted = df.sort_values('comfort_index', ascending=False)
    rating_df = df_sorted[['city_name', 'recommended_activity']]
    rating_df['best_time'] = None
    for index, row in rating_df.iterrows():
        rating_df.at[index, 'best_time'] = str(CITIES_MONTH.get('city_name'))
    rating_df.to_csv(f"{PATH_TO_AGGREGATED}city_tourism_rating.csv", index=False, encoding='utf-8')


    df_sorted['comfort_index']  = pd.to_numeric(df_sorted['comfort_index'], errors='coerce')
    top_3 = df_sorted.nlargest(3, 'comfort_index', keep="first")
    lowest_row = df_sorted.nsmallest(1, 'comfort_index')
    result_df = pd.concat([top_3, lowest_row], ignore_index=True)
    result_df.to_csv(f"{PATH_TO_AGGREGATED}travel_recommendations.csv", index=False, encoding='utf-8')


timestamp = datetime.now().strftime("%Y%m%d_%H%M")
dt = datetime.strptime(timestamp, "%Y%m%d_%H%M")
year = dt.year
month = dt.month
day = dt.day
year_path = f"{PATH_TO_RAW}{year}"
month_path = f"{year_path}/{month}"
day_path = f"{month_path}/{day}"
cleaned_path = f"{PATH_TO_CLEANED}"
cleaned_filename = f"{cleaned_path}weather_cleaned_{year}{month}{day}.csv"
cleaned_log = f"{cleaned_path}cleaning_log_{year}{month}{day}.txt"
enriched_csv = f"{PATH_TO_ENRICHED}weather_enriched_{year}{month}{day}.csv"
cities_reference_csv = f"{PATH_TO_ENRICHED}cities_reference.csv"
if not os.path.exists(year_path):
    os.makedirs(year_path)
if not os.path.exists(month_path):
    os.makedirs(month_path)
if not os.path.exists(day_path):
    os.makedirs(day_path)
if not os.path.exists(cleaned_path):
    os.makedirs(cleaned_path)
if not os.path.exists(PATH_TO_ENRICHED):
    os.makedirs(PATH_TO_ENRICHED)
if not os.path.exists(PATH_TO_AGGREGATED):
    os.makedirs(PATH_TO_AGGREGATED)

collect_weather_data(day_path, timestamp)
df = create_cleaned(day_path, cleaned_log, cleaned_filename)
df = create_enriched(enriched_csv, df, month)
create_aggregated(df)
