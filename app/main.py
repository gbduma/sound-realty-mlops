from fastapi import FastAPI, HTTPException, Header, Query
from typing import Dict, Any
from .schemas import FullInput, MinimalInput, PredictionOut
from .service import PredictionService
from typing import Optional

app = FastAPI(title="Sound Realty House Price API")
service = PredictionService()  # reads env MODEL_URI, loads demographics cache

@app.get("/health")
def health(model: Optional[str] = Query(default=None)):
    # Resolve the selected model (query overrides default)
    (_, _, _), chosen = service._choose_model(model)
    return {
        "status": "ok",
        "default_model": service.default_model_name,
        "selected_model": chosen,
        "loaded_models": list(service.models.keys()),
    }

@app.get("/model/metadata")
def model_metadata(model: Optional[str] = Query(default=None), x_model: Optional[str] = Header(default=None)):
    name = x_model or model
    (model_obj, feat_order, version), chosen = service._choose_model(name)
    meta = service.get_metadata_for(chosen)  # add this method
    meta["selected_model"] = chosen
    return meta

@app.post("/predict/full", response_model=PredictionOut)
def predict_full(inp: FullInput, model: Optional[str] = Query(default=None), x_model: Optional[str] = Header(default=None)):
    name = x_model or model
    (model_obj, feat_order, version), chosen = service._choose_model(name)
    return service.predict_full_with_model(inp.model_dump(), chosen)

@app.post("/predict/minimal", response_model=PredictionOut)
def predict_minimal(payload: dict, model: Optional[str] = Query(default=None), x_model: Optional[str] = Header(default=None)):
    name = x_model or model
    (model_obj, feat_order, version), chosen = service._choose_model(name)
    return service.predict_minimal_with_model(payload, chosen)
