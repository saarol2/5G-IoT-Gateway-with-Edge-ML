FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir paho-mqtt requests

COPY gateway/app ./app

CMD ["python", "-m", "app.main"]
