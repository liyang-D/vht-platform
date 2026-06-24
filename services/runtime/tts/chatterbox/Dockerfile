FROM nvcr.io/nvidia/pytorch:25.04-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

COPY services/runtime/tts/chatterbox/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps torchaudio==2.7.0 \
    && pip install --no-cache-dir --no-deps s3tokenizer \
    && pip install --no-cache-dir --no-deps "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master" \
    && pip install --no-cache-dir --no-deps chatterbox-tts==0.1.7

COPY services/runtime/tts/chatterbox/main.py ./main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
