from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import os, time, json, secrets
import paho.mqtt.client as mqtt

app = FastAPI(title="Middleware Layer")

BROKER = os.getenv("MQTT_BROKER", "mqtt")
TOPIC = os.getenv("MQTT_TOPIC", "sensors/temperature")

DEVICE_KEYS: dict[str, str] = {}

mqttc = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqttc.connect(BROKER, 1883, 60)
mqttc.loop_start()

class RegisterResp(BaseModel):
    device_id: str
    api_key: str

@app.post("/devices/register", response_model=RegisterResp)
def register():
    device_id = f"dev_{secrets.token_hex(4)}"
    api_key = secrets.token_hex(16)
    DEVICE_KEYS[device_id] = api_key

    print(json.dumps({"event":"device_registered","device_id":device_id,"ts":time.time()}))
    return RegisterResp(device_id=device_id, api_key=api_key)

class Ingest(BaseModel):
    device_id: str
    pc1: float
    pc2: float
    timestamp: float | None = None

@app.post("/ingest")
def ingest(body: Ingest, x_api_key: str = Header(default="")):
    expected = DEVICE_KEYS.get(body.device_id)
    if not expected or x_api_key != expected:
        print(json.dumps({"event":"auth_failed","device_id":body.device_id,"ts":time.time()}))
        raise HTTPException(status_code=401, detail="Invalid API key")

    msg = {
        "device_id": body.device_id,
        "pc1": body.pc1,
        "pc2": body.pc2,
        "timestamp": body.timestamp or time.time()
    }

    # centralized logging here
    print(json.dumps({"event":"ingest_ok","device_id":body.device_id,"ts":time.time()}))

    mqttc.publish(TOPIC, json.dumps(msg))
    return {"status": "published", "topic": TOPIC}
