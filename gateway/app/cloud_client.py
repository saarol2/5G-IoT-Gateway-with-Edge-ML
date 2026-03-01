import requests
from typing import Any, Dict, Tuple

def send_to_cloud(endpoint: str, payload: Dict[str, Any]) -> Tuple[bool, str]:
    """Returns (success, error_message). error_message is empty on success."""
    if not endpoint:
        return True, ""  # no cloud endpoint configured – skip silently
    r = requests.post(
        endpoint,
        json=payload,
        timeout=10,
        headers={"Content-Type": "application/json"},
    )
    if r.status_code == 200:
        return True, ""
    return False, f"HTTP {r.status_code}: {r.text[:200]}"
