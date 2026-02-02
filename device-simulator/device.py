import os, time, json, random
import paho.mqtt.client as mqtt

BROKER = os.getenv("MQTT_BROKER", "mqtt")
TOPIC = "sensors/temperature"
DEVICE_ID = os.getenv("DEVICE_ID", os.getenv("HOSTNAME", "sensor_unknown"))

connected = False

def on_connect(client, userdata, flags, rc, properties=None):
    global connected
    connected = True
    print("Device connected")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.connect(BROKER, 1883, 60)
client.loop_start()

while not connected:
    time.sleep(0.1)

while True:
    data = {
        "device_id": DEVICE_ID,
        "temperature": round(random.uniform(60, 90), 2),
        "timestamp": time.time()
    }

    client.publish(TOPIC, json.dumps(data))
    print(f"{DEVICE_ID} sent {data['temperature']}°C")

    time.sleep(random.uniform(1,5))