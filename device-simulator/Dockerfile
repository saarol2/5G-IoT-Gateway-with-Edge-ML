FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY device.py .
RUN pip install paho-mqtt

CMD ["python", "device.py"]
