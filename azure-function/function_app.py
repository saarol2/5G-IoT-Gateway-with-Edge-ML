import azure.functions as func
import logging
import json

import torch
import torch.nn as nn
import numpy as np
from collections import deque
from sklearn.preprocessing import StandardScaler
import joblib

app = func.FunctionApp()

MODEL_PATH = "machine-learning/models/best_lstm_model.pt"
SCALER_PATH = "machine-learning/models/scaler.pkl"
SEQ_LENGTH = 1000
# Found to be the optimal threshold in testing to maximize the weighted F1 score
THRESHOLD = 0.32

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(input_size=2, hidden_size=64, num_layers=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

scaler = joblib.load(SCALER_PATH)

logging.info('Model loaded.')

device_buffers = {}
# changed to AONONYMOUS from FUNCTION for testing
@app.route(route="iot-data", auth_level=func.AuthLevel.ANONYMOUS)
def iot_data(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function to receive IoT data from gateway via REST API,
    and run LSTM inference.
    """
    logging.info('Python HTTP trigger function processed a request.')

    try:
        req_body = req.get_json()
        
        gateway_id = req_body.get('gateway_id')
        timestamp = req_body.get('timestamp')
        readings = req_body.get('readings', [])
        
        if not gateway_id or not readings:
            return func.HttpResponse(
                "Missing required fields: gateway_id or readings",
                status_code=400
            )

        predictions = []
        for reading in readings:
            device_id = reading.get('device_id')
            pc1 = reading.get('pc1')
            pc2 = reading.get('pc2')

            """
            temperature = reading.get('temperature')
            reading_timestamp = reading.get('timestamp')
            """
            if device_id is None or pc1 is None or pc2 is None:
                logging.warning(f"Skipping invalid reading: {reading}")
                continue

            if device_id not in device_buffers:
                device_buffers[device_id] = deque(maxlen=SEQ_LENGTH)

            device_buffers[device_id].append([pc1, pc2])

            if len(device_buffers[device_id]) == SEQ_LENGTH:
                sequence = np.array(device_buffers[device_id])

                sequence = scaler.transform(sequence)

                sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(sequence_tensor)
                    prob = torch.sigmoid(output).item()

                predictions.append({
                    "device_id": device_id,
                    "probability": prob,
                    "anomaly": prob > THRESHOLD
                })

        response = {
            "status": "success",
            "gateway_id": gateway_id,
            "readings_amount": len(readings),
            "predictions_amount": len(predictions),
            "predictions": predictions
        }

        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )

        """
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": f"Processed {len(readings)} readings",
                "gateway_id": gateway_id
            }),
            status_code=200,
            mimetype="application/json"
        )
        """
        
    except ValueError:
        return func.HttpResponse(
            "Invalid JSON payload",
            status_code=400
        )
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return func.HttpResponse(
            f"Internal server error: {str(e)}",
            status_code=500
        )
