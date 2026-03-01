import time
import torch
import torch.nn as nn
import numpy as np
import joblib
from collections import deque
from typing import Any, Dict, List
from .buffer import ReadingBuffer
from .cloud_client import send_to_cloud
from .config import (MODEL_PATH, SCALER_PATH, THRESHOLD, SEQ_LENGTH)

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
scaler = joblib.load(SCALER_PATH)

device_buffers = {}

def should_send(queue_size: int, buffer_usage: float, time_since: float, batch_size: int, send_interval: int) -> bool:
    return (
        queue_size >= batch_size or
        buffer_usage > 0.8 or
        (queue_size > 0 and time_since >= send_interval)
    )

def run_sender_loop(
    buf: ReadingBuffer,
    gateway_id: str,
    cloud_endpoint: str,
    batch_size: int,
    send_interval: int,
):
    last_send = time.time()

    while True:
        now = time.time()
        qsize = buf.size()
        usage = buf.usage()

        if not should_send(qsize, usage, now - last_send, batch_size, send_interval):
            time.sleep(0.5)
            continue

        batch: List[Dict[str, Any]] = buf.peek_batch(batch_size)
        if not batch:
            time.sleep(0.5)
            continue

        if usage > 0.8:
            print(f"[{gateway_id}] WARNING buffer {usage*100:.0f}% ({qsize}/{buf.maxlen})")

        predictions = []
        for reading in batch:
            device_id = reading.get("device_id")
            pc1 = reading.get("pc1")
            pc2 = reading.get("pc2")
            timestamp = reading.get("timestamp")

            if device_id is None or pc1 is None or pc2 is None:
                continue

            if device_id not in device_buffers:
                device_buffers[device_id] = deque(maxlen=SEQ_LENGTH)

            device_buffers[device_id].append([pc1, pc2, timestamp])

            if len(device_buffers[device_id]) == SEQ_LENGTH:
                sequence_data = list(device_buffers[device_id])

                sequence = np.array([[x[0], x[1]] for x in sequence_data])
                last_timestamp = sequence_data[-1][2]

                sequence = scaler.transform(sequence)
                tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    prob = torch.sigmoid(output).item()

                reading["anomaly_prob"] = prob
                reading["anomaly"] = prob > THRESHOLD
                predictions.append({
                    "device_id": device_id,
                    "probability": prob,
                    "anomaly": prob > THRESHOLD,
                    "inference_timestamp": last_timestamp
                })

        payload = {
            "gateway_id": gateway_id,
            "timestamp": time.time(),
            "readings": batch,
            "predictions": predictions
        }

        ok = False
        try:
            ok = send_to_cloud(cloud_endpoint, payload)
        except Exception as e:
            print(f"[{gateway_id}] cloud exception: {e}")

        if ok:
            print(f"[{gateway_id}] sent {len(batch)} readings")
            buf.drop(len(batch))
            last_send = time.time()
        else:
            print(f"[{gateway_id}] cloud send failed")
            time.sleep(1.0)
