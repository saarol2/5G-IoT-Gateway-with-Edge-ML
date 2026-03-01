import requests
from collections import deque
from typing import Dict, Optional, Tuple

_device_buffers: Dict[str, deque] = {}


def process_reading(
    device_id: str,
    pc1: float,
    pc2: float,
    seq_length: int,
    edge_ml_endpoint: str,
) -> Optional[Tuple[float, bool]]:

    if device_id not in _device_buffers:
        _device_buffers[device_id] = deque(maxlen=seq_length)

    _device_buffers[device_id].append([pc1, pc2])

    if len(_device_buffers[device_id]) < seq_length:
        return None

    sequence = list(_device_buffers[device_id])
    try:
        resp = requests.post(
            f"{edge_ml_endpoint}/predict",
            json={"device_id": device_id, "sequence": sequence},
            timeout=5,
        )
        if resp.status_code == 200:
            data = resp.json()
            return float(data["probability"]), bool(data["anomaly"])
        print(f"[edge_ml_client] predict HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as exc:
        print(f"[edge_ml_client] predict error for {device_id}: {exc}")

    return None


def buffer_stats(seq_length: int) -> Optional[dict]:
    if not _device_buffers:
        return None
    fills = [len(q) for q in _device_buffers.values()]
    ready = sum(1 for f in fills if f >= seq_length)
    return {
        "devices": len(fills),
        "ready": ready,
        "min_fill": min(fills),
        "max_fill": max(fills),
    }
