FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY gateway/app ./app
COPY machine-learning/models ./machine-learning/models
RUN pip install --no-cache-dir paho-mqtt requests numpy joblib torch scikit-learn

CMD ["python", "-m", "app.main"]
