from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from .schemas import FullInput, MinimalInput, PredictionOut
from .service import PredictionService

app = FastAPI(title="Sound Realty House Price API")
service = PredictionService()  # reads env MODEL_URI, loads demographics cache

@app.get("/health")
def health():
    return {"status": "ok", "model_version": service.model_version}

@app.get("/model/metadata")
def model_metadata():
    return service.get_metadata()

@app.post("/predict/full", response_model=PredictionOut)
def predict_full(inp: FullInput):
    try:
        return service.predict_full(inp.model_dump())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/minimal", response_model=PredictionOut)
def predict_minimal(payload: Dict[str, Any]):
    try:
        return service.predict_minimal(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
