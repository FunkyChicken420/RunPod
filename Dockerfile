FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY worker.py .
RUN mkdir -p /tmp
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV PORT_HEALTH=8081
EXPOSE 8080 8081
CMD ["python", "worker.py"]
