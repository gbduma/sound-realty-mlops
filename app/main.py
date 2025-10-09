from fastapi import FastAPI, HTTPException, Header, Query, Body
from typing import Optional, Dict, Any
from .schemas import FullInput, MinimalInput, PredictionOut
from .service import PredictionService

app = FastAPI(title="Sound Realty House Price API", 
              version="1.0.0",
              description="""
              Predict house prices in the Seattle area
              **To Demo:**
              - Use the `model` query param (or `X-Model` header) to switch between `baseline_knn` and `enhanced_auto`.
              - Try `/model/metadata` for each model to see metrics/leaderboard.
              - The `/predict/*` endpoints return an `aux` block with comps and quality info.
              """)
service = PredictionService()  # reads env MODEL_URI, loads demographics cache

@app.get("/health", tags=["system"])
def health(model: Optional[str] = Query(default=None, description="Optional model to check")):
    # Resolve the selected model (query overrides default)
    (_, _, _), chosen = service._choose_model(model)
    return {
        "status": "ok",
        "default_model": service.default_model_name,
        "selected_model": chosen,
        "loaded_models": list(service.models.keys()),
    }

@app.get("/model/list", tags=["model"])
def list_models():
    return sorted(service.models.keys())  # preloaded + any that were requested

@app.get("/model/metadata", tags=["model"])
def model_metadata(model: Optional[str] = Query(default=None, description="baseline_knn | enhanced_auto"), 
                   x_model: Optional[str] = Header(default=None, convert_underscores=False)):
    name = x_model or model
    (model_obj, feat_order, version), chosen = service._choose_model(name)
    meta = service.get_metadata_for(chosen)  # add this method
    meta["selected_model"] = chosen
    return meta

@app.post("/predict/full", response_model=PredictionOut, response_model_exclude_none=True, tags=["predict"])
def predict_full(inp: FullInput = Body(...,
                                       example={"zipcode": "98103",
                                                "bedrooms": 3,
                                                "bathrooms": 2.5,
                                                "sqft_living": 2220,
                                                "sqft_lot": 6380,
                                                "floors": 1.5,
                                                "waterfront": 0,
                                                "view": 0,
                                                "condition": 4,
                                                "grade": 8,
                                                "sqft_above": 1660,
                                                "sqft_basement": 560,
                                                "yr_built": 1954,
                                                "yr_renovated": 0,
                                                "lat": 47.68,
                                                "long": -122.34,
                                                "sqft_living15": 1690,
                                                "sqft_lot15": 4000,}), 
                 model: Optional[str] = Query(default=None, description="baseline_knn | enhanced_auto"), 
                 x_model: Optional[str] = Header(default=None, convert_underscores=False)):
    name = x_model or model
    (model_obj, feat_order, version), chosen = service._choose_model(name)
    return service.predict_full_with_model(inp.model_dump(), chosen)

@app.post("/predict/minimal", response_model=PredictionOut, response_model_exclude_none=True, tags=["predict"])
def predict_minimal(payload: dict = Body(...,
                                         example={
                                             "zipcode": "98103",
                                             "bedrooms": 3,
                                             "bathrooms": 2.5,
                                             "sqft_living": 2220,
                                             "sqft_lot": 6380,
                                             "floors": 1.5,
                                             "sqft_above": 1660,
                                             "sqft_basement": 560,
                                             }
                                        ), 
                    model: Optional[str] = Query(default=None, description="baseline_knn | enhanced_auto"), 
                    x_model: Optional[str] = Header(default=None, convert_underscores=False)):
    name = x_model or model
    (model_obj, feat_order, version), chosen = service._choose_model(name)
    return service.predict_minimal_with_model(payload, chosen)
