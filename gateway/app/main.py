import threading
from .config import (
    BROKER, SHARED_TOPIC,
    GATEWAY_ID, MAX_READINGS,
    CLOUD_ENDPOINT, EDGE_ML_ENDPOINT,
    BATCH_SIZE, SEND_INTERVAL
)
from .buffer import ReadingBuffer
from .mqtt_consumer import create_client
from .sender import run_sender_loop

def main():
    buf = ReadingBuffer(MAX_READINGS)

    def on_json(data: dict):
        buf.append(data)

    client = create_client(on_json)

    # attach simple callbacks
    ud = client._userdata
    ud["on_connect"] = lambda rc: print(f"[{GATEWAY_ID}] connected rc={rc}, sub={SHARED_TOPIC}") if rc == 0 else print(f"[{GATEWAY_ID}] connect failed rc={rc}")
    ud["on_disconnect"] = lambda rc: print(f"[{GATEWAY_ID}] disconnected rc={rc}")
    ud["on_error"] = lambda s: print(f"[{GATEWAY_ID}] {s}")

    sender_thread = threading.Thread(
        target=run_sender_loop,
        daemon=True,
        args=(buf, GATEWAY_ID, CLOUD_ENDPOINT, EDGE_ML_ENDPOINT, BATCH_SIZE, SEND_INTERVAL)
    )
    sender_thread.start()

    print(f"[{GATEWAY_ID}] edge={EDGE_ML_ENDPOINT} cloud={CLOUD_ENDPOINT}")
    client.connect(BROKER, 1883, 60)
    client.subscribe(SHARED_TOPIC)
    client.loop_forever()

if __name__ == "__main__":
    main()