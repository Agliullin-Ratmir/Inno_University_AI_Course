from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
app = FastAPI()

@app.get("/ping", response_class=PlainTextResponse)
async def get_content():
    return "pong"

#docker build --network host -t fastapi-app .


#docker run -d -p 8000:8000 fastapi-app