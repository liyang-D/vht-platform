FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app/apps/simple-chat/backend

COPY apps/simple-chat/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY apps/simple-chat/backend ./
COPY shared /app/shared

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
