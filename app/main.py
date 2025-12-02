# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from pathlib import Path
from typing import List

app = FastAPI(title="ML API - Iris Example")


class Features(BaseModel):
    features: List[float]


# Cargar el modelo al iniciar
MODEL_PATH = Path(__file__).parent / "model.joblib"
if not MODEL_PATH.exists():
    raise RuntimeError(
        f"No se encontró el modelo en {MODEL_PATH}. "
        "Asegúrate de ejecutar primero train_model.py"
    )

model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Features):
    # Para Iris se esperan 4 features.
    if len(payload.features) != 4:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban 4 features, pero se recibieron {len(payload.features)}",
        )

    prediction = model.predict([payload.features])[0]
    # Opcional: también podrías retornar probabilidades
    proba = model.predict_proba([payload.features])[0].tolist()

    return {
        "prediction": int(prediction),
        "probabilities": proba,
    }
