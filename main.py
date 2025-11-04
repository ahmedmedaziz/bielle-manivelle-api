from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI
app = FastAPI(
    title="Bielle-Manivelle API",
    description="API pour prédire les paramètres optimaux r, L, et Ω du système bielle-manivelle",
    version="1.0.0"
)

# Enable CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class PredictionRequest(BaseModel):
    x_max: float
    x_min: float
    v_max: float
    v_min: float
    fixation_type: int

# Response model
class PredictionResponse(BaseModel):
    r: float
    L: float
    omega: float
    status: str
    message: str = ""

# Global variable for model
model = None

def load_model():
    """Load the trained model"""
    global model
    try:
        model = joblib.load("modele_inverse_bielle.pkl")
        print("✅ Modèle chargé avec succès")
        return True
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Load model when the API starts"""
    if not load_model():
        raise Exception("Échec du chargement du modèle au démarrage")

@app.get("/")
async def root():
    return {
        "message": "API Bielle-Manivelle - Prédiction des paramètres optimaux",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Modèle non chargé")
        
        # Handle different fixation types
        x_max = request.x_max
        x_min = request.x_min
        v_max = request.v_max
        v_min = request.v_min
        
        if request.fixation_type == 0:  # Déplacement
            v_max, v_min = 0, 0
        elif request.fixation_type == 1:  # Vitesse
            x_max, x_min = 0, 0
        
        # Prepare input features
        X_new = np.array([[x_max, x_min, v_max, v_min]])
        
        # Make prediction
        prediction = model.predict(X_new)[0]
        r_pred, L_pred, Omega_pred = prediction
        
        return PredictionResponse(
            r=float(r_pred),
            L=float(L_pred),
            omega=float(Omega_pred),
            status="success",
            message="Prédiction réalisée avec succès"
        )
        
    except Exception as e:
        error_msg = f"Erreur de prédiction: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)