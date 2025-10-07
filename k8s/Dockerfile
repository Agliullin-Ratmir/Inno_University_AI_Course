FROM python:3.11-slim
WORKDIR /app


RUN pip install --no-cache-dir \
    "uvicorn[standard]==0.24.0" \
    "pydantic==2.5.0" \
    "fastapi==0.112.4"


COPY app.py .
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t ratmir123/house-price-api:v1 .