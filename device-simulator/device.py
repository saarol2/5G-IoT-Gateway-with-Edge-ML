import os, time, json, random
import paho.mqtt.client as mqtt
import pandas as pd

BROKER = os.getenv("MQTT_BROKER", "mqtt")
TOPIC = "sensors/temperature"
DEVICE_ID = os.getenv("DEVICE_ID", os.getenv("HOSTNAME", "sensor_unknown"))

DATA_PATH = os.getenv("DATA_PATH", "data.csv")
df = pd.read_csv(DATA_PATH)

connected = False

def on_connect(client, userdata, flags, rc, properties=None):
    global connected
    connected = True
    print(f"[{DEVICE_ID}] Connected to {BROKER}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.connect(BROKER, 1883, 60)
client.loop_start()

while not connected:
    time.sleep(0.1)

index = 0
while True:
    row = df.iloc[index % len(df)]

    data = {
        "device_id": DEVICE_ID,
        "pc1": float(row["pc1"]),
        "pc2": float(row["pc2"]),
        "timestamp": time.time()
    }

    client.publish(TOPIC, json.dumps(data))

    index += 1
    time.sleep(random.uniform(1,5))

"""
while True:
    data = {
        "device_id": DEVICE_ID,
        "temperature": round(random.uniform(60, 90), 2),
        "timestamp": time.time()
    }

    client.publish(TOPIC, json.dumps(data))
    #print(f"[{DEVICE_ID}] Sent {data['temperature']}°C")

    time.sleep(random.uniform(1,5))
"""