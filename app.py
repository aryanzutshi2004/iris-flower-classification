from fastapi import FastAPI, HTTPException
from joblib import load
from pydantic import BaseModel
import pandas as pd
import logging


# Logging Configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Load Model and Artifacts

try:
    data = load("./artifacts/model.joblib")
    model = data["model"]
    cols_to_scale = data["columns_to_scale"]
    scaler = data["scaler"]
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or artifacts: {e}")
    raise RuntimeError(f"Error loading model artifacts: {e}")


# Input Schema using Pydantic

class IrisFlowers(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Prediction Mapping
mapper_species = {
    0: 'Versicolor',
    1: 'Setosa',
    2: 'Virginica',
}


# Predict Flower Endpoint
@app.post("/predict")
def species_predict(data_user: IrisFlowers):
    try:
        logging.info(f"Received input: {data_user}")

        df = pd.DataFrame([{
            "sepal_length": data_user.sepal_length,
            "sepal_width": data_user.sepal_width,
            "petal_length": data_user.petal_length,
            "petal_width": data_user.petal_width
        }])

        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

        prediction_num = model.predict(df)[0]
        prediction_flower = mapper_species.get(prediction_num, "Unknown")

        probabilities = model.predict_proba(df)[0]
        predicted_class_prob = round(probabilities[prediction_num] * 100, 2)

        logging.info(f"Prediction: {prediction_flower} ({predicted_class_prob}%)")
        return {
            "prediction": prediction_flower,
            "probability": f"{predicted_class_prob} %"
        }
    except ValueError as ve:
        logging.warning(f"Value error: {ve}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {ve}")

    except Exception as err:
        logging.error(f"Unhandled exception: {err}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {err}")



# Home Route
@app.get("/")
def home():
    logging.info("Health check route accessed.")
    return {"message": "API is working"}
