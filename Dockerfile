FROM python:3.11-slim

WORKDIR /app

COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/requirements-api.txt

COPY . .

EXPOSE 8000 