import requests
from typing import Any, Dict, List

def call_edge_ml(endpoint: str, gateway_id: str, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        r = requests.post(
            endpoint,
            json={"gateway_id": gateway_id, "readings": batch},
            timeout=5,
            headers={"Content-Type": "application/json"},
        )
        if r.status_code == 200:
            return r.json()
        return {"error": f"edge_status={r.status_code}"}
    except Exception as e:
        return {"error": f"edge_exception={str(e)}"}
