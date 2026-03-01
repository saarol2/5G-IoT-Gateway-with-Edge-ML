import time
from typing import Any, Dict, List
from .buffer import ReadingBuffer
from .cloud_client import send_to_cloud
from .edge_ml_client import process_reading, buffer_stats
from .config import SEQ_LENGTH, EDGE_ML_ENDPOINT

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

            result = process_reading(device_id, pc1, pc2, SEQ_LENGTH, EDGE_ML_ENDPOINT)
            if result is not None:
                prob, anomaly = result
                reading["anomaly_prob"] = prob
                reading["anomaly"] = anomaly
                predictions.append({
                    "device_id": device_id,
                    "probability": prob,
                    "anomaly": anomaly,
                    "inference_timestamp": timestamp,
                })
                label = "ANOMALY" if anomaly else "normal"
                print(f"[{gateway_id}] edge predict {device_id}: {prob:.4f} → {label}")

        stats = buffer_stats(SEQ_LENGTH)
        if stats:
            print(f"[{gateway_id}] buffers: {stats['devices']} devices, {stats['ready']}/{stats['devices']} ready, min={stats['min_fill']} max={stats['max_fill']} (need {SEQ_LENGTH})")

        payload = {
            "gateway_id": gateway_id,
            "timestamp": time.time(),
            "readings": batch,
            "predictions": predictions
        }

        ok = False
        err = ""
        try:
            ok, err = send_to_cloud(cloud_endpoint, payload)
        except Exception as e:
            err = str(e)

        if ok:
            print(f"[{gateway_id}] sent {len(batch)} readings")
            buf.drop(len(batch))
            last_send = time.time()
        else:
            print(f"[{gateway_id}] cloud send failed: {err}")
            time.sleep(1.0)
