import time, json, random
import paho.mqtt.client as mqtt

BROKER, TOPIC = "mqtt", "sensors/temperature"
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
    data = {"device_id": "sensor_1", "temperature": round(random.uniform(60, 90), 2), "timestamp": time.time()}
    client.publish(TOPIC, json.dumps(data))
    print(f"Sent: {data['temperature']}°C")
    time.sleep(5)