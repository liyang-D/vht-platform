FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app/apps/evaluation/backend

COPY apps/evaluation/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/evaluation/backend ./
COPY shared /app/shared

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8050"]
