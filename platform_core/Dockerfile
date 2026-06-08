FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY platform_core/requirements.txt ./platform_core/requirements.txt
RUN pip install --no-cache-dir -r platform_core/requirements.txt

COPY platform_core ./platform_core

CMD ["python", "-m", "platform_core.database.init_db"]
