FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY gateway.py .
RUN pip install paho-mqtt

CMD ["python", "gateway.py"]
