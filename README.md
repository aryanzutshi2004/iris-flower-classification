# 🌸 Iris Flower Classification API (FastAPI Deployment)

This project demonstrates how to train a machine learning model on the **Iris dataset** and deploy it using **FastAPI**. The API predicts the species of a flower based on its sepal and petal measurements.

---

## 🚀 Features

- Trained a classification model using **scikit-learn**
- Deployed the model using **FastAPI**
- Includes **MinMaxScaler** for preprocessing
- Predicts flower species: *Setosa*, *Versicolor*, or *Virginica*
- Returns both the predicted class and its **probability**
- Includes logging and basic error handling
- Swagger UI for testing (`/docs`)

---

## 📦 Tech Stack

- Python 3.10+
- Scikit-learn
- Pandas & NumPy
- FastAPI
- Pydantic
- Joblib

---

## 🧠 How It Works

- User sends a POST request with the flower measurements.
- The API scales the input, makes a prediction, and returns:
  - 🌼 Predicted species
  - 📊 Prediction probability

---

## 📁 Project Structure

.
├── artifacts/
│ └── model.joblib # Serialized model, scaler, and columns
├── main.py # FastAPI app
├── requirements.txt
├── README.md
└── app.log # Log file
---

## 🔧 API Usage

### ▶️ Run the App

```bash
uvicorn main:app --reload

## Installation

git clone https://github.com/yourusername/iris-flower-fastapi.git
cd iris-flower-fastapi
pip install -r requirements.txt
