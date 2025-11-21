from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

# Import your function
from main3 import predict_urgency_from_github

app = FastAPI(title="Urgency Detection API", version="1.0.0")


# Request body schema
class PredictRequest(BaseModel):
    Massage_Id: str
    Tanent_id: str
    mailbox_id: str


@app.post("/predict")
def predict(request: PredictRequest):
    try:
        result = predict_urgency_from_github(
            text=request.text,
            model_dir="./result/tf_bert_urgency_model",
            threshold=0.4,
            scaler_method="power",
            use_raw_threshold=False,
        )
        return result
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "Urgency Detection API is running"}
