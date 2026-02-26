import requests
from typing import Any, Dict

def send_to_cloud(endpoint: str, payload: Dict[str, Any]) -> bool:
    if not endpoint:
        return True  # no cloud endpoint configured – skip silently
    r = requests.post(
        endpoint,
        json=payload,
        timeout=10,
        headers={"Content-Type": "application/json"},
    )
    return r.status_code == 200
