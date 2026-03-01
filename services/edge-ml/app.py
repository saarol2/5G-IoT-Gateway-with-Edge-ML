import os
import torch
import torch.nn as nn
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

MODEL_PATH = os.getenv("MODEL_PATH", "machine-learning/models/50_window_lstm_model.pt")
SCALER_PATH = os.getenv("SCALER_PATH", "machine-learning/models/scaler.pkl")
THRESHOLD = float(os.getenv("THRESHOLD", "0.21"))


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


print("[edge-ml] loading model...")
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=2)
_model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
_model.to(_device)
_model.eval()
_scaler = joblib.load(SCALER_PATH)
print(f"[edge-ml] model loaded on {_device}, threshold={THRESHOLD}")

app = FastAPI(title="Edge ML Inference Service")


class PredictRequest(BaseModel):
    device_id: str
    sequence: List[List[float]]


class PredictResponse(BaseModel):
    device_id: str
    probability: float
    anomaly: bool


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    seq = np.array(req.sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 2:
        raise HTTPException(status_code=422, detail="sequence must have shape [N, 2]")

    seq = _scaler.transform(seq)
    tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(_device)

    with torch.no_grad():
        output = _model(tensor)
        prob = float(torch.sigmoid(output).item())

    anomaly = prob > THRESHOLD
    return PredictResponse(device_id=req.device_id, probability=prob, anomaly=anomaly)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(_device)}
