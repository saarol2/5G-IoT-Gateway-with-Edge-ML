import azure.functions as func
import logging
import json

app = func.FunctionApp()

@app.route(route="iot-data", auth_level=func.AuthLevel.FUNCTION)
def iot_data(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function to receive IoT data from gateway via REST API
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
        
        logging.info(f"Received {len(readings)} readings from {gateway_id}")
        
        # Process each reading
        for reading in readings:
            device_id = reading.get('device_id')
            temperature = reading.get('temperature')
            reading_timestamp = reading.get('timestamp')
            
            logging.info(f"Device: {device_id}, Temp: {temperature}°C, Time: {reading_timestamp}")
                   
        return func.HttpResponse(
            json.dumps({
                "status": "success",
                "message": f"Processed {len(readings)} readings",
                "gateway_id": gateway_id
            }),
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
