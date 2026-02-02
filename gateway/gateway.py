import json
import paho.mqtt.client as mqtt
import requests
import threading
import time
import os
from collections import deque

BROKER = "mqtt"
TOPIC = "sensors/temperature"
MAX_READINGS = 200
CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "http://localhost:7071/api/iot-data")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
SEND_INTERVAL = int(os.getenv("SEND_INTERVAL", "30"))  # seconds

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
        readings.append(data)
        print(f"Stored in memory. Total readings: {len(readings)}")
    except Exception as e:
        print(f"Error processing message: {e}")

def on_disconnect(client, userdata, rc, properties=None):
    print(f"Gateway disconnected with code {rc}")

def send_to_cloud():
    """Send batched data to cloud endpoint via REST API"""
    while True:
        time.sleep(SEND_INTERVAL)
        
        with send_lock:
            if len(readings) == 0:
                print("No data to send")
                continue
            
            # Get batch of readings
            batch = list(readings)[:BATCH_SIZE]
            
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
                # Remove sent readings from buffer
                with send_lock:
                    for _ in range(min(len(batch), len(readings))):
                        readings.popleft()
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