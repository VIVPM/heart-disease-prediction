# Heart Disease Prediction - Backend API

FastAPI backend for serving heart disease predictions.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure model file exists
# models/catboost_best_model.cbm should be present
```

## Running the API

### Development Mode

```bash
# From project root
uvicorn backend.api:app --reload --port 8000

# Or from backend directory
cd backend
uvicorn api:app --reload --port 8000
```

### Production Mode

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Endpoints

### 1. Root
```
GET /
```
Returns API status and available endpoints.

### 2. Single Patient Prediction
```
POST /predict
Content-Type: application/json

{
  "Age": 58,
  "Sex": 1,
  "BP": 152,
  "Cholesterol": 239,
  "FBS_over_120": 0,
  "Max_HR": 158,
  "Exercise_angina": 1,
  "ST_depression": 3.6,
  "Number_of_vessels_fluro": 2,
  "Chest_pain_type": 4,
  "EKG_results": 0,
  "Slope_of_ST": 2,
  "Thallium": 7
}
```

**Response:**
```json
{
  "heart_disease_prediction": "Presence",
  "heart_disease_probability": 0.8542,
  "risk_level": "Critical",
  "recommendation": "🚨 URGENT: Immediate medical attention required!"
}
```

### 3. Batch Prediction
```
POST /predict/batch
Content-Type: multipart/form-data

file: patients.csv
```

**CSV Format:**
```csv
id,Age,Sex,Chest pain type,BP,Cholesterol,FBS over 120,EKG results,Max HR,Exercise angina,ST depression,Slope of ST,Number of vessels fluro,Thallium
1,58,1,4,152,239,0,0,158,1,3.6,2,2,7
2,52,1,1,125,325,0,2,171,0,0.0,1,0,3
```

**Response:**
```json
{
  "total": 2,
  "successful": 2,
  "failed": 0,
  "summary": {
    "predicted_with_disease": 1,
    "predicted_without_disease": 1,
    "percentage_with_disease": 50.0
  },
  "predictions": [
    {
      "patient_id": 1,
      "heart_disease_prediction": "Presence",
      "heart_disease_probability": 0.8542,
      "risk_level": "Critical",
      "recommendation": "🚨 URGENT: Immediate medical attention required!"
    },
    {
      "patient_id": 2,
      "heart_disease_prediction": "Absence",
      "heart_disease_probability": 0.2341,
      "risk_level": "Low",
      "recommendation": "✅ Low risk. Continue regular health monitoring."
    }
  ]
}
```

### 4. Model Info
```
GET /model/info
```

Returns model information including feature names.

### 5. Health Check
```
GET /health
```

Returns API health status.

## Interactive API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Testing with cURL

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 58,
    "Sex": 1,
    "BP": 152,
    "Cholesterol": 239,
    "FBS_over_120": 0,
    "Max_HR": 158,
    "Exercise_angina": 1,
    "ST_depression": 3.6,
    "Number_of_vessels_fluro": 2
  }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@sample_patients.csv"
```

## Testing with Python

```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
data = {
    "Age": 58,
    "Sex": 1,
    "BP": 152,
    "Cholesterol": 239,
    "FBS_over_120": 0,
    "Max_HR": 158,
    "Exercise_angina": 1,
    "ST_depression": 3.6,
    "Number_of_vessels_fluro": 2
}

response = requests.post(url, json=data)
print(response.json())

# Batch prediction
url = "http://localhost:8000/predict/batch"
files = {"file": open("sample_patients.csv", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## Environment Variables

Create a `.env` file in the backend directory (optional):

```env
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Model Settings
MODEL_PATH=../models/catboost_best_model.cbm
```

## Deployment

### Docker (Optional)

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

The API can be deployed to:
- AWS Lambda (with Mangum adapter)
- Google Cloud Run
- Azure App Service
- Heroku
- Render
- Railway

## Notes

- The model file must be present in `../models/catboost_best_model.cbm`
- CORS is enabled for all origins (adjust in production)
- The API loads the model once at startup for efficiency
- Batch predictions process all patients even if some fail

## Troubleshooting

**Model not found:**
- Ensure `catboost_best_model.cbm` exists in the `models/` directory
- Check the path in `config.py`

**Import errors:**
- Make sure you're running from the project root
- Check that all dependencies are installed

**Port already in use:**
- Change the port: `uvicorn backend.api:app --port 8001`
