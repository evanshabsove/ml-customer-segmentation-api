# Customer Segmentation Flask API

A Flask REST API for customer segmentation predictions using machine learning.

## Features

- POST endpoint for customer segmentation predictions
- Input validation and preprocessing
- Model loading and reloading capabilities
- Health check endpoint
- Error handling and logging
- Ready for model integration

## API Endpoints

### Health Check
```
GET /
```
Returns the status of the API and model loading state.

### Prediction
```
POST /predict
```
Makes a customer segmentation prediction based on input parameters.

**Request Body:**
```json
{
    "customer_id": "12345",
    "gender": "Female",
    "age": 35,
    "annual_income": 50,
    "spending_score": 75
}
```

**Response:**
```json
{
    "customer_id": "12345",
    "prediction": {
        "segment": 3,
        "segment_name": "Young Female Shoppers",
        "confidence": [0.1, 0.2, 0.1, 0.6, 0.0]
    },
    "input_data": {
        "gender": "Female",
        "age": 35,
        "annual_income": 50,
        "spending_score": 75
    },
    "timestamp": "2025-08-08T10:30:00.123456"
}
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create the models directory:
```bash
mkdir models
```

3. Place your trained model files in the models directory:
   - `models/customer_segmentation_model.pkl` - The trained model
   - `models/scaler.pkl` - The feature scaler (optional)

4. Run the application:
```bash
python app.py
```

The API will be available at `http://localhost:8080`

## Model Integration

When you receive the model files from your team:

1. Save the trained model as `models/customer_segmentation_model.pkl`
2. If you have a scaler, save it as `models/scaler.pkl`
3. The app will automatically try to load them on startup
4. Use the `/model/reload` endpoint to reload models without restarting the app

## Input Parameters

- **customer_id**: Unique identifier for the customer (string/number)
- **gender**: Customer gender ("Male", "Female", "M", or "F" - case insensitive)
- **age**: Customer age (integer, 0-120)
- **annual_income**: Annual income in thousands of dollars (e.g., 50 for $50,000)
- **spending_score**: Spending score from 1-100 (integer)

## Customer Segments

The model predicts one of these segments:
- 0: Low Income, Low Spending
- 1: Low Income, High Spending
- 2: High Income, Low Spending
- 3: High Income, High Spending
- 4: Average Income, Average Spending

## Testing

You can test the API using curl:

```bash
# Health check
curl -X GET http://localhost:8080/

# Make a prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "12345",
    "gender": "Female",
    "age": 35,
    "annual_income": 50,
    "spending_score": 75
  }'
```

## Error Handling

The API includes comprehensive error handling for:
- Missing or invalid input parameters
- Model loading failures
- Prediction errors
- Invalid HTTP methods
- Non-existent endpoints

## Production Deployment

For production deployment, use gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:8080 app:app
```
