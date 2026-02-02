FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY gateway.py .
RUN pip install paho-mqtt requests

CMD ["python", "gateway.py"]
