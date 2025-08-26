from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
import os

app = FastAPI()

FILE_PATH = "./head.txt"

@app.get("/content", response_class=PlainTextResponse)
async def get_content():
    if not os.path.exists(FILE_PATH):
        return "File not found."

    try:
        with open(FILE_PATH, "r", encoding="utf-8") as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"