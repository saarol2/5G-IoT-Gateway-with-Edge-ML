FROM python:3.11

WORKDIR /app

COPY azure-function/function_app.py .
COPY azure-function/requirements.txt .
COPY /machine-learning/models ./machine-learning/models

RUN pip install --upgrade pip
RUN pip install -r requirements.txt