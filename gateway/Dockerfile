FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir paho-mqtt requests numpy joblib scikit-learn==1.5.2

COPY gateway/app ./app
COPY machine-learning/models ./machine-learning/models

CMD ["python", "-m", "app.main"]
