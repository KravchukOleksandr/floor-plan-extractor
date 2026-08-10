FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models/huggingface
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY floorplan floorplan
COPY weights weights

EXPOSE 8000
CMD ["uvicorn", "floorplan.api:app", "--host", "0.0.0.0", "--port", "8000"]
