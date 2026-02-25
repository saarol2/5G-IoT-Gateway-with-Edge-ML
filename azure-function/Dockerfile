FROM mcr.microsoft.com/azure-functions/python:4-python3.10

WORKDIR /home/site/wwwroot

COPY azure-function/ .

COPY machine-learning/models ./machine-learning/models

RUN pip install -r requirements.txt