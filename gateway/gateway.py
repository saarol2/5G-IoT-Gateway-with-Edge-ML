import json
import paho.mqtt.client as mqtt
from collections import deque

BROKER = "mqtt"
TOPIC = "sensors/temperature"
MAX_READINGS = 200

readings = deque(maxlen=MAX_READINGS)

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

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message
client.on_disconnect = on_disconnect

print("Gateway starting...")
client.connect(BROKER, 1883, 60)
client.loop_forever()