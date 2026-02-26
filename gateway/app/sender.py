import time
from typing import Any, Dict, List
from .buffer import ReadingBuffer
from .cloud_client import send_to_cloud

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

        payload = {
            "gateway_id": gateway_id,
            "timestamp": time.time(),
            "readings": batch,
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
