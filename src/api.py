import os
import pickle
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Define directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'artifacts'))

# Path configurations for preprocessor and model binaries
PREPROCESSOR_PATH = os.path.join(ARTIFACTS_DIR, 'preprocessor.pkl')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'fraud_model.pkl')

def load_inference_artifacts():
    """
    Loads preprocessor and model binaries safely from artifacts.
    """
    if not os.path.exists(PREPROCESSOR_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError(
            "Required artifacts (preprocessor.pkl or fraud_model.pkl) are missing. "
            "Please ensure preprocessing and model training have been executed first."
        )
    
    print(f"Loading preprocessor from: {PREPROCESSOR_PATH}")
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
        
    print(f"Loading champion model from: {MODEL_PATH}")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    return preprocessor, model

# -------------------------------------------------------------------------
# Pydantic Schema: Input Validation
# -------------------------------------------------------------------------
# EDUCATIONAL NOTE ON PYDANTIC SCHEMAS:
# - In enterprise REST APIs, we must strictly validate the format, key structures, 
#   and datatypes of all incoming JSON payloads before running machine learning models.
# - Pydantic models automatically validate incoming requests. If a request is missing 
#   features or contains incorrect types, Pydantic immediately returns an HTTP 422 
#   Unprocessable Entity error, protecting the model from parsing corrupt data.
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the start of recording", examples=[42000.0])
    Amount: float = Field(..., description="Transaction amount in USD", examples=[150.0])
    # Define PCA anonymized components V1-V28 with default values of 0.0 (mean)
    V1: float = Field(0.0, description="PCA Anonymized Feature V1")
    V2: float = Field(0.0, description="PCA Anonymized Feature V2")
    V3: float = Field(0.0, description="PCA Anonymized Feature V3")
    V4: float = Field(0.0, description="PCA Anonymized Feature V4")
    V5: float = Field(0.0, description="PCA Anonymized Feature V5")
    V6: float = Field(0.0, description="PCA Anonymized Feature V6")
    V7: float = Field(0.0, description="PCA Anonymized Feature V7")
    V8: float = Field(0.0, description="PCA Anonymized Feature V8")
    V9: float = Field(0.0, description="PCA Anonymized Feature V9")
    V10: float = Field(0.0, description="PCA Anonymized Feature V10")
    V11: float = Field(0.0, description="PCA Anonymized Feature V11")
    V12: float = Field(0.0, description="PCA Anonymized Feature V12")
    V13: float = Field(0.0, description="PCA Anonymized Feature V13")
    V14: float = Field(0.0, description="PCA Anonymized Feature V14")
    V15: float = Field(0.0, description="PCA Anonymized Feature V15")
    V16: float = Field(0.0, description="PCA Anonymized Feature V16")
    V17: float = Field(0.0, description="PCA Anonymized Feature V17")
    V18: float = Field(0.0, description="PCA Anonymized Feature V18")
    V19: float = Field(0.0, description="PCA Anonymized Feature V19")
    V20: float = Field(0.0, description="PCA Anonymized Feature V20")
    V21: float = Field(0.0, description="PCA Anonymized Feature V21")
    V22: float = Field(0.0, description="PCA Anonymized Feature V22")
    V23: float = Field(0.0, description="PCA Anonymized Feature V23")
    V24: float = Field(0.0, description="PCA Anonymized Feature V24")
    V25: float = Field(0.0, description="PCA Anonymized Feature V25")
    V26: float = Field(0.0, description="PCA Anonymized Feature V26")
    V27: float = Field(0.0, description="PCA Anonymized Feature V27")
    V28: float = Field(0.0, description="PCA Anonymized Feature V28")

# -------------------------------------------------------------------------
# Pydantic Schema: Output Response
# -------------------------------------------------------------------------
# EDUCATIONAL NOTE:
# - Defining response models ensures consistent payload returns to clients. 
#   It serves as a contract between the data science engine and frontend services.
class PredictionResponse(BaseModel):
    is_fraud: int = Field(..., description="Prediction label: 1 if fraudulent, 0 if legitimate")
    fraud_probability: float = Field(..., description="Estimated probability of fraud (0.0 to 1.0)")
    status: str = Field(..., description="Human-readable decision status: 'Approved' or 'Flagged for Review'")

# -------------------------------------------------------------------------
# Application Lifespan: Startup & Shutdown Actions
# -------------------------------------------------------------------------
# EDUCATIONAL NOTE ON LIFESPAN CONTEXT MANAGERS:
# - Loading heavy machine learning models on every API request is a major anti-pattern. 
#   It causes high latency and exhausts memory.
# - FastAPI's modern 'lifespan' context manager allows us to load the scaler and XGBoost model 
#   ONCE when the server starts up, cache them in memory (via app.state), and reuse them 
#   instantly across all subsequent HTTP requests.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load ML artifacts on application startup
    preprocessor, model = load_inference_artifacts()
    app.state.preprocessor = preprocessor
    app.state.model = model
    print("Application Startup: preprocessor and model successfully loaded into memory cache.")
    yield
    # Shutdown clean up actions can go here
    print("Application Shutdown: releasing cached resources.")

# -------------------------------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------------------------------
app = FastAPI(
    title="🛡️ Financial Sentinel REST API",
    description=(
        "High-performance REST API serving an optimized XGBoost classifier "
        "and preprocessor pipelines for real-time transaction scoring."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------
@app.get("/", tags=["Health"])
async def root():
    """
    Health check endpoint to verify server is active.
    """
    return {
        "api_name": "Financial Sentinel REST API",
        "status": "Healthy",
        "version": "1.0.0"
    }

@app.post(
    "/predict", 
    response_model=PredictionResponse, 
    status_code=status.HTTP_200_OK,
    tags=["Inference"]
)
async def predict_transaction(payload: TransactionInput):
    """
    Scores an individual credit card transaction to determine fraud risk.
    """
    # -------------------------------------------------------------------------
    # 1. Retrieve Cached Models
    # -------------------------------------------------------------------------
    preprocessor = app.state.preprocessor
    model = app.state.model
    
    if not preprocessor or not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference artifacts are not fully loaded or initialized."
        )

    # -------------------------------------------------------------------------
    # 2. Convert Payload to DataFrame & Preprocess
    # -------------------------------------------------------------------------
    # EDUCATIONAL NOTE:
    # - Model pipelines built in pandas require column-aligned structures.
    # - We convert the validated Pydantic dictionary to a single-row DataFrame, 
    #   and run it through the preprocessor.
    try:
        input_data = payload.model_dump()
        df_input = pd.DataFrame([input_data])
        
        # Apply standard scaling dynamically
        X_processed = preprocessor.transform(df_input)
        
        # -------------------------------------------------------------------------
        # 3. Model Scoring & Inference
        # -------------------------------------------------------------------------
        prediction = int(model.predict(X_processed)[0])
        probability = float(model.predict_proba(X_processed)[0, 1])
        
        # Map statuses
        status_msg = "Flagged for Review" if prediction == 1 else "Approved"
        
        return PredictionResponse(
            is_fraud=prediction,
            fraud_probability=probability,
            status=status_msg
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution failure: {str(e)}"
        )
