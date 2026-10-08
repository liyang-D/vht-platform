FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Europe/London
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY apps/ace/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -m nltk.downloader wordnet

COPY apps/ace/backend /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8070"]
