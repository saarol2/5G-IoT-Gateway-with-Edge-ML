from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import os

app = FastAPI(title="Edge ML Inference")

MODEL_VERSION = os.getenv("MODEL_VERSION", "edge-v1")
EDGE_THRESH = float(os.getenv("EDGE_THRESH", "3.0"))

class Req(BaseModel):
    gateway_id: str
    readings: List[Dict[str, Any]]

@app.post("/predict")
def predict(req: Req):
    xs = []
    for r in req.readings:
        if isinstance(r.get("pc1"), (int, float)) and isinstance(r.get("pc2"), (int, float)):
            xs.append([float(r["pc1"]), float(r["pc2"])])
    if not xs:
        return {"model_version": MODEL_VERSION, "anomaly": False, "score": 0.0}

    arr = np.array(xs, dtype=float)
    mag = np.linalg.norm(arr, axis=1)
    z = (mag - mag.mean()) / (mag.std() + 1e-6)
    score = float(z.max())
    return {"model_version": MODEL_VERSION, "anomaly": score > EDGE_THRESH, "score": score}
