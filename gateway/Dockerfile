FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY app ./app
RUN pip install --no-cache-dir paho-mqtt requests

CMD ["python", "-m", "app.main"]
