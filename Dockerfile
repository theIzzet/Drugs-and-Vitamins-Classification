FROM node:20-alpine as frontend-builder

WORKDIR /app/frontend

COPY UI/vitaminui/package*.json ./
RUN npm install

COPY UI/vitaminui/ .
RUN npm run build


FROM python:3.11-slim as final-stage

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


COPY main.py .
COPY drug_cnn.py .

COPY checkpoint.weights.h5 . 

COPY --from=frontend-builder /app/frontend/dist /app/static

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]