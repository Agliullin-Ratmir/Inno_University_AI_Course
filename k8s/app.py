from fastapi import FastAPI
from pydantic import BaseModel
import random
import time
app = FastAPI(title="House Price Predictor")
class HouseData(BaseModel):
    square_meters: float
rooms: int
year_built: int
class PredictionResponse(BaseModel):
    predicted_price: float
currency: str = "USD"
@app.get("/health")
def health():
    return {"status": "healthy"}
@app.get("/ready")
def ready():
    # Имитируем проверку готовности
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseData):
    # Простая формула для предсказания (для демонстрации)
    # В реальности здесь была бы обученная ML модель
    base_price = 1000 # цена за квадратный метр
    age_factor = max(0.5, 1 - (2024 - data.year_built) * 0.01)
    room_bonus = data.rooms * 5000
    predicted_price = (data.square_meters * base_price * age_factor) + room_bonus
    # Имитируем задержку обработки
    time.sleep(0.1)
    return PredictionResponse(predicted_price=round(predicted_price, 2))
@app.get("/")
def root():
    return {"service": "House Price Predictor", "version": "1.0.0"}