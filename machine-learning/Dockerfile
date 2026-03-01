FROM python:3.11

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn==1.5.2 \
    matplotlib \
    joblib

COPY . .

CMD ["python", "supervised_lstm_training.py"]
