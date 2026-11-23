from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib

app = FastAPI(title="Spam Detection API")
model = joblib.load("spam_model.pkl")


class Email(BaseModel):
    text: str


@app.get("/")
def home():
    return {"status": "running"}


@app.post("/predict")
def predict(email: Email):
    if not email.text.strip():
        raise HTTPException(status_code=400, detail="Email text is empty")
    answer = int(model.predict([email.text])[0])
    chance = float(model.predict_proba([email.text])[0][1])
    label = "spam" if answer == 1 else "not spam"
    return {"prediction": label, "spam_probability": round(chance, 4)}
