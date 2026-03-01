import time
from typing import Any, Dict, List
from .buffer import ReadingBuffer
from .cloud_client import send_to_cloud
from .edge_ml_client import process_reading, buffer_stats
from .config import SEQ_LENGTH, EDGE_ML_ENDPOINT
from .metrics import GatewayMetrics

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
    metrics = GatewayMetrics(gateway_id, report_interval=10, warmup_duration=60)
    
    print(f"[{gateway_id}] Metrics collection starting with 60s warmup period")
    print(f"[{gateway_id}] Reports will be printed every 10s after warmup")

    while True:
        now = time.time()
        qsize = buf.size()
        usage = buf.usage()
        
        # Record buffer usage
        metrics.record_buffer_usage(usage)
        
        # Print metrics report periodically
        if metrics.should_report():
            metrics.print_report()

        if not should_send(qsize, usage, now - last_send, batch_size, send_interval):
            time.sleep(0.5)
            continue

        batch: List[Dict[str, Any]] = buf.peek_batch(batch_size)
        if not batch:
            time.sleep(0.5)
            continue

        if usage > 0.8:
            print(f"[{gateway_id}] WARNING buffer {usage*100:.0f}% ({qsize}/{buf.maxlen})")
        
        batch_start = time.time()
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
                
                # Record prediction metrics
                metrics.record_prediction(anomaly)
                
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
        
        # Record latencies
        cloud_timestamp = time.time()
        for reading in batch:
            if "timestamp" in reading:
                metrics.record_latency(reading["timestamp"], cloud_timestamp)

        ok = False
        err = ""
        try:
            ok, err = send_to_cloud(cloud_endpoint, payload)
        except Exception as e:
            err = str(e)

        batch_processing_time = time.time() - batch_start

        if ok:
            print(f"[{gateway_id}] sent {len(batch)} readings (processing: {batch_processing_time*1000:.1f}ms)")
            metrics.record_batch_sent(len(batch), batch_processing_time)
            buf.drop(len(batch))
            last_send = time.time()
        else:
            print(f"[{gateway_id}] cloud send failed: {err}")
            metrics.record_batch_failed(len(batch))
            time.sleep(1.0)
