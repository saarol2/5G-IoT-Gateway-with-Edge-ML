import os

BROKER = os.getenv("MQTT_BROKER", "mqtt")
SHARED_TOPIC = os.getenv("MQTT_TOPIC", "$share/gateways/sensors/waterpump")

GATEWAY_ID = os.getenv("HOSTNAME", "gateway")
MAX_READINGS = int(os.getenv("MAX_READINGS", "500"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
SEND_INTERVAL = int(os.getenv("SEND_INTERVAL", "5"))

CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "http://azure-function/api/iot-data")

MODEL_PATH = "machine-learning/models/50_window_lstm_model.pt"
SCALER_PATH = "machine-learning/models/scaler.pkl"
THRESHOLD = 0.21
SEQ_LENGTH = 50
