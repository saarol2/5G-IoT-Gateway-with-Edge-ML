import json
import paho.mqtt.client as mqtt
import requests
import threading
import time
import os
from collections import deque

BROKER = "mqtt"
TOPIC = "sensors/temperature"
MAX_READINGS = 500
CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "http://localhost:7071/api/iot-data")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
SEND_INTERVAL = int(os.getenv("SEND_INTERVAL", "5"))  # seconds

readings = deque(maxlen=MAX_READINGS)
send_lock = threading.Lock()

def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        client.subscribe(TOPIC)
        print(f"Gateway connected and subscribed to {TOPIC}")
    else:
        print(f"Gateway connection failed with code {rc}")

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"Received: {data}")
        with send_lock:
            readings.append(data)
            print(f"Stored in memory. Total readings: {len(readings)}")
    except Exception as e:
        print(f"Error processing message: {e}")

def on_disconnect(client, userdata, rc, properties=None):
    print(f"Gateway disconnected with code {rc}")

def send_to_cloud():
    """Send batched data to cloud endpoint via REST API with adaptive sending"""
    last_send_time = time.time()
    
    while True:
        current_time = time.time()
        time_since_last_send = current_time - last_send_time
        
        with send_lock:
            queue_size = len(readings)
            buffer_usage = queue_size / MAX_READINGS
        
        # Adaptive sending logic:
        # 1. Send immediately if batch is full
        # 2. Send if buffer is >80% full (prevent data loss)
        # 3. Send if SEND_INTERVAL has passed and data exists
        should_send = (
            queue_size >= BATCH_SIZE or  # Batch ready
            buffer_usage > 0.8 or  # Buffer almost full
            (queue_size > 0 and time_since_last_send >= SEND_INTERVAL)  # Time elapsed
        )
        
        if not should_send:
            time.sleep(0.5)  # Check more frequently
            continue
        
        with send_lock:
            if len(readings) == 0:
                time.sleep(SEND_INTERVAL)
                continue
            batch = list(readings)[:BATCH_SIZE]
        
        # Warning if buffer is getting full
        if buffer_usage > 0.8:
            print(f" WARNING: Buffer {buffer_usage*100:.0f}% full ({queue_size}/{MAX_READINGS})")
        
        print(f"Attempting to send {len(batch)} readings to cloud...")
            
        try:
            payload = {
                "gateway_id": "gateway_1",
                "timestamp": time.time(),
                "readings": batch
            }
            
            response = requests.post(
                CLOUD_ENDPOINT,
                json=payload,
                timeout=10,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                print(f"Successfully sent {len(batch)} readings to cloud")
                with send_lock:
                    for _ in range(min(len(batch), len(readings))):
                        readings.popleft()
                last_send_time = time.time()
            else:
                print(f"Failed to send data. Status: {response.status_code}, Response: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"Error sending data to cloud: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

print("Gateway starting...")

# Start cloud sender thread
cloud_thread = threading.Thread(target=send_to_cloud, daemon=True)
cloud_thread.start()
print(f"Cloud sender thread started. Sending every {SEND_INTERVAL}s to {CLOUD_ENDPOINT}")

client.connect(BROKER, 1883, 60)
client.loop_forever()