from datetime import datetime, timezone
import azure.functions as func
import logging
import json
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import joblib
from flask import Flask
from database import db
from models import Gateway, Device, Reading, Prediction

flask_app = Flask(__name__)
flask_app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:////app/data/iot.db"
flask_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(flask_app)

with flask_app.app_context():
    db.create_all()
    print("Database created")

app = func.FunctionApp()

MODEL_PATH = "machine-learning/models/1000_window_lstm_model.pt"
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

print("Cloud model loaded.")
logging.info('Cloud model loaded.')

device_buffers = {}
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
        edge_predictions = req_body.get('predictions', [])

        print(f"Received {len(readings)} readings from gateway {gateway_id}.")
        
        if not gateway_id or not readings:
            return func.HttpResponse(
                "Missing required fields: gateway_id or readings",
                status_code=400
            )

        cloud_predictions = []

        for reading in readings:
            device_id = reading.get('device_id')
            pc1 = reading.get('pc1')
            pc2 = reading.get('pc2')
            reading_timestamp = reading.get('timestamp')

            if device_id is None or pc1 is None or pc2 is None:
                logging.warning(f"Skipping invalid reading: {reading}")
                continue

            if device_id not in device_buffers:
                device_buffers[device_id] = deque(maxlen=SEQ_LENGTH)

            device_buffers[device_id].append([pc1, pc2, reading_timestamp])

            if len(device_buffers[device_id]) == SEQ_LENGTH:
                sequence_data = list(device_buffers[device_id])

                sequence = np.array([[x[0], x[1]] for x in sequence_data])
                last_timestamp = sequence_data[-1][2]

                sequence = scaler.transform(sequence)

                tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    prob = torch.sigmoid(output).item()

                cloud_predictions.append({
                    "device_id": device_id,
                    "probability": prob,
                    "anomaly": prob > THRESHOLD,
                    "inference_timestamp": last_timestamp
                })
        
        with flask_app.app_context():

            gateway = Gateway.query.filter_by(gateway_id=gateway_id).first()
            if not gateway:
                gateway = Gateway(gateway_id=gateway_id)
                db.session.add(gateway)
                db.session.commit()
                print(f" Gateway {gateway_id} created in database.")

            # readings
            for reading in readings:
                device_id = reading.get("device_id")
                pc1 = reading.get("pc1")
                pc2 = reading.get("pc2")

                device_obj = Device.query.filter_by(device_id=device_id).first()
                if not device_obj:
                    device_obj = Device(device_id=device_id, gateway=gateway)
                    db.session.add(device_obj)
                    db.session.commit()
                    print(f"Device {device_id} added to database.")

                new_reading = Reading(device=device_obj, pc1=pc1, pc2=pc2)
                db.session.add(new_reading)
                print(f"{len(readings)} readings saved to database.")

            # edge predictions
            for pred in edge_predictions:
                device_id = pred.get("device_id")
                device_obj = Device.query.filter_by(device_id=device_id).first()

                if device_obj:
                    new_pred = Prediction(
                        device=device_obj,
                        probability=pred["probability"],
                        anomaly=pred["anomaly"],
                        source="edge",
                        timestamp=datetime.fromtimestamp(pred["inference_timestamp"], timezone.utc)
                    )
                    db.session.add(new_pred)
            print(f"{len(edge_predictions)} edge predictions saved to database.")

            # cloud predictions
            for pred in cloud_predictions:
                device_id = pred.get("device_id")
                device_obj = Device.query.filter_by(device_id=device_id).first()

                if device_obj:
                    new_pred = Prediction(
                        device=device_obj,
                        probability=pred["probability"],
                        anomaly=pred["anomaly"],
                        source="cloud",
                        timestamp=datetime.fromtimestamp(pred["inference_timestamp"], timezone.utc)
                    )
                    db.session.add(new_pred)
            print(f"{len(cloud_predictions)} cloud predictions saved to database.")
            db.session.commit()

        response = {
            "status": "success",
            "gateway_id": gateway_id,
            "readings_amount": len(readings),
            "cloud_predictions_amount": len(cloud_predictions),
            "cloud_predictions": cloud_predictions,
            "edge_predictions_amount": len(edge_predictions),
            "edge_predictions": edge_predictions
        }

        return func.HttpResponse(
            json.dumps(response),
            status_code=200,
            mimetype="application/json"
        )
        
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
