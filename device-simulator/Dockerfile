FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY device-simulator/device.py .
COPY machine-learning/data/sensor_pca_cleaned.csv ./data.csv

RUN pip install paho-mqtt
RUN pip install pandas

CMD ["python", "device.py"]
