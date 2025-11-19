from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import os
from loader import load
from preparator import prepare
from transformer import transform
from training_model import train


app = FastAPI()

# uvicorn main:app --reload

@app.get("/load", response_class=PlainTextResponse)
async def load_data():
    load()
    return "Data has been loaded"

@app.get("/transform", response_class=PlainTextResponse)
async def transform_data():
    transform()
    return "Data has been transformed"

@app.get("/prepare", response_class=PlainTextResponse)
async def prepare_data():
    prepare()
    return "Data has been prepared"

@app.get("/train", response_class=PlainTextResponse)
async def train_data():
    train()
    return "Data has been trained"