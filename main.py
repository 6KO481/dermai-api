"""
API FastAPI pour DermAI - Classification de lésions cutanées
Optimisée pour Railway deployment
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# Import de vos modules (à adapter selon votre structure)
try:
    from predictor import EnsembleSkinLesionPredictor
    from config import (
        MODEL1_PATH, 
        MODEL2_PATH, 
        CLASSES_INFO,
        APP_METADATA
    )
except ImportError:
    # Pour le développement local si structure différente
    import sys
    sys.path.append(os.path.dirname(__file__))
    from predictor import EnsembleSkinLesionPredictor
    from config import MODEL1_PATH, MODEL2_PATH, CLASSES_INFO, APP_METADATA

# Initialisation de l'API
app = FastAPI(
    title="DermAI API",
    description="API de classification de lésions cutanées sur peau noire",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS pour permettre les appels depuis vos sites/apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez vos domaines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variable globale pour le prédicteur (chargé une seule fois)
predictor = None

# Modèles de réponse Pydantic
class PredictionResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    confidence_percentage: str
    detailed_class: str
    severity: str
    description: str
    recommendation: str
    color: str
    all_probabilities: Optional[Dict[str, float]] = None
    model1_prediction: str
    model1_confidence: float
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool
    error: str
    details: Optional[str] = None
    timestamp: str


@app.on_event("startup")
async def load_models():
    """Charge les modèles au démarrage de l'API"""
    global predictor
    try:
        print("🚀 Chargement des modèles...")
        predictor = EnsembleSkinLesionPredictor(MODEL1_PATH, MODEL2_PATH)
        print("✅ Modèles chargés avec succès")
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles: {e}")
        # L'API démarre quand même mais retournera des erreurs 503


@app.get("/", response_model=HealthResponse)
async def root():
    """Point d'entrée de l'API - Health check"""
    return {
        "status": "online",
        "models_loaded": predictor is not None,
        "version": APP_METADATA['version'],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Endpoint de santé pour Railway"""
    return {
        "status": "healthy" if predictor is not None else "unhealthy",
        "models_loaded": predictor is not None,
        "version": APP_METADATA['version'],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info")
async def app_info():
    """Informations sur l'application et les modèles"""
    return {
        "app": APP_METADATA,
        "endpoints": {
            "predict": "/predict - POST - Upload image for prediction",
            "health": "/health - GET - Health check",
            "info": "/info - GET - Application information",
            "classes": "/classes - GET - Available classification classes"
        }
    }


@app.get("/classes")
async def get_classes():
    """Liste toutes les classes de classification disponibles"""
    return {
        "classes": CLASSES_INFO,
        "total_classes": len(CLASSES_INFO)
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_lesion(file: UploadFile = File(...)):
    """
    Endpoint principal de prédiction
    
    Args:
        file: Image de la lésion cutanée (JPEG, PNG)
    
    Returns:
        Résultats de la classification avec recommandations
    """
    # Vérifier que les modèles sont chargés
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Modèles non chargés. Veuillez réessayer dans quelques instants."
        )
    
    # Valider le type de fichier
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Le fichier doit être une image (JPEG, PNG, etc.)"
        )
    
    try:
        # Lire l'image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convertir en RGB si nécessaire
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Faire la prédiction
        results = predictor.predict(image, show_results=False)
        
        if results is None:
            raise HTTPException(
                status_code=500,
                detail="Erreur lors de la prédiction"
            )
        
        # Récupérer les informations de la classe prédite
        final_class = results['final_prediction']
        class_info = CLASSES_INFO.get(final_class, {})
        
        # Construire la réponse
        response = {
            "success": True,
            "prediction": final_class,
            "confidence": results['final_confidence'],
            "confidence_percentage": results['confidence_percentage'],
            "detailed_class": results['detailed_class'],
            "severity": class_info.get('severity', 'unknown'),
            "description": class_info.get('description', 'Description non disponible'),
            "recommendation": class_info.get('recommendation', 'Consultez un professionnel'),
            "color": class_info.get('color', '#6b7280'),
            "all_probabilities": results.get('model1_results', {}).get('all_probabilities', {}),
            "model1_prediction": results['model1_prediction'],
            "model1_confidence": results['model1_confidence'],
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de l'image: {str(e)}"
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gestionnaire d'erreurs HTTP personnalisé"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'erreurs générales"""
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Erreur interne du serveur",
            "details": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Point d'entrée pour Railway
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False  # Désactivé en production
    )