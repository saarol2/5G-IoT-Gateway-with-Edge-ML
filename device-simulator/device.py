import os
import time
import random
import requests
import pandas as pd

MIDDLEWARE_URL = os.getenv("MIDDLEWARE_URL", "http://middleware:8080")
DATA_PATH = os.getenv("DATA_PATH", "data.csv")

print("[device] registering...")

resp = requests.post(f"{MIDDLEWARE_URL}/devices/register", timeout=10)
resp.raise_for_status()

reg = resp.json()
DEVICE_ID = reg["device_id"]
API_KEY = reg["api_key"]

print(f"[device] Registered: {DEVICE_ID}")

df = pd.read_csv(DATA_PATH)
index = random.randint(0, len(df)-1)

while True:
    row = df.iloc[index % len(df)]

    payload = {
        "device_id": DEVICE_ID,
        "pc1": float(row["pc1"]),
        "pc2": float(row["pc2"]),
        "timestamp": time.time()
    }

    try:
        r = requests.post(
            f"{MIDDLEWARE_URL}/ingest",
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=10
        )

        if r.status_code != 200:
            print(f"[{DEVICE_ID}] Auth/ingest failed: {r.status_code} {r.text}")

    except Exception as e:
        print(f"[{DEVICE_ID}] Middleware error: {e}")

    index += 1
    time.sleep(random.uniform(1,5))
