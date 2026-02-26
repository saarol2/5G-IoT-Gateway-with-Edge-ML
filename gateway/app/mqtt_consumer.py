import json
import paho.mqtt.client as mqtt
from typing import Callable

def create_client(on_json_message: Callable[[dict], None]):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(c, userdata, flags, rc, properties=None):
        userdata["on_connect"](rc)

    def on_message(c, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
            userdata["on_json"](data)
        except Exception as e:
            userdata["on_error"](f"bad message: {e}")

    def on_disconnect(c, userdata, rc, properties=None):
        userdata["on_disconnect"](rc)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.user_data_set({
        "on_json": on_json_message,
        "on_connect": lambda rc: None,
        "on_disconnect": lambda rc: None,
        "on_error": lambda s: None
    })
    return client