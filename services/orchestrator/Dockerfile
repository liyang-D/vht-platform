FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY services/orchestrator/requirements.txt ./services/orchestrator/requirements.txt
RUN pip install --no-cache-dir -r services/orchestrator/requirements.txt

COPY services ./services
COPY shared ./shared

CMD ["uvicorn", "services.orchestrator.main:app", "--host", "0.0.0.0", "--port", "8000"]
