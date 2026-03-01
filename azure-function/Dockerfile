FROM mcr.microsoft.com/azure-functions/python:4-python3.10

WORKDIR /home/site/wwwroot

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY azure-function/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY azure-function/ .
COPY machine-learning/models ./machine-learning/models