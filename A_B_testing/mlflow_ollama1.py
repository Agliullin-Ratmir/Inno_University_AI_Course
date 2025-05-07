# Got from here: https://www.mlflow.org/docs/latest/tracing/integrations/ollama

import mlflow
from openai import OpenAI

mlflow.openai.autolog()

client = OpenAI(
    base_url="http://localhost:11434/api/chat",  # The local Ollama REST endpoint, 404 not found error, it's trying to call /chat/completions instead of api/chat
    api_key="some key",  # Required to instantiate OpenAI client, it can be a random string
)

response = client.chat.completions.create(
    model="llama3.1",
    messages=[
        {"role": "system", "content": "You are a science teacher."},
        {"role": "user", "content": "Why is the sky blue?"},
    ],
)