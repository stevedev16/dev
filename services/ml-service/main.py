import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import numpy as np
import asyncio
from datetime import datetime

from models.cnn_model import CNNAnomalyDetector
from models.lstm_model import LSTMAnomalyDetector
from models.transformer_model import TransformerAnomalyDetector
from feature_extractor import FeatureExtractor
from inference_service import InferenceService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FinSecure Nexus ML Service",
    description="Machine Learning inference service for multi-domain security detection",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    model_type: str
    features: Dict[str, Any]
    event_type: str
    source_type: str

class PredictionResponse(BaseModel):
    is_anomaly: bool
    confidence: float
    anomaly_score: float
    model_type: str
    threat_type: Optional[str] = None
    risk_level: str
    explanation: List[str]
    timestamp: datetime

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    timestamp: datetime

class ModelMetrics(BaseModel):
    model_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float

# Global services
feature_extractor = FeatureExtractor()
inference_service = InferenceService()

# Initialize models
models = {
    'cnn': CNNAnomalyDetector(),
    'lstm': LSTMAnomalyDetector(),
    'transformer': TransformerAnomalyDetector()
}

@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    logger.info("Starting FinSecure Nexus ML Service...")
    
    # Initialize models
    for model_name, model in models.items():
        try:
            await model.initialize()
            logger.info(f"✅ {model_name.upper()} model initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize {model_name} model: {e}")
    
    # Initialize inference service
    await inference_service.initialize(models)
    logger.info("✅ Inference service initialized")
    
    logger.info("🚀 ML Service startup complete")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "FinSecure Nexus ML Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        timestamp=datetime.now()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    try:
        logger.info(f"Received prediction request for {request.model_type} model")
        
        # Validate model type
        if request.model_type not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model type: {request.model_type}. Available: {list(models.keys())}"
            )
        
        # Extract and enhance features
        enhanced_features = await feature_extractor.extract_features(
            request.features,
            request.event_type,
            request.source_type
        )
        
        # Run inference
        prediction = await inference_service.predict(
            request.model_type,
            enhanced_features,
            request.event_type,
            request.source_type
        )
        
        return PredictionResponse(
            is_anomaly=prediction['is_anomaly'],
            confidence=prediction['confidence'],
            anomaly_score=prediction['anomaly_score'],
            model_type=prediction['model_type'],
            threat_type=prediction.get('threat_type'),
            risk_level=prediction['risk_level'],
            explanation=prediction['explanation'],
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(requests: List[PredictionRequest]):
    """Batch prediction endpoint for multiple requests"""
    try:
        logger.info(f"Received batch prediction request with {len(requests)} items")
        
        # Process all requests concurrently
        tasks = []
        for request in requests:
            task = predict(request)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error in batch item {i}: {result}")
                responses.append({
                    "error": str(result),
                    "index": i
                })
            else:
                responses.append(result)
        
        return {
            "results": responses,
            "total_processed": len(requests),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get information about available models"""
    model_info = {}
    
    for model_name, model in models.items():
        model_info[model_name] = {
            "name": model.name,
            "type": model.model_type,
            "version": model.version,
            "is_loaded": model.is_loaded,
            "input_features": model.get_input_features(),
            "output_classes": model.get_output_classes()
        }
    
    return {
        "models": model_info,
        "total_models": len(models),
        "timestamp": datetime.now()
    }

@app.get("/models/{model_type}/metrics", response_model=ModelMetrics)
async def get_model_metrics(model_type: str):
    """Get performance metrics for a specific model"""
    if model_type not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_type} not found"
        )
    
    model = models[model_type]
    metrics = await model.get_metrics()
    
    return ModelMetrics(
        model_type=model_type,
        accuracy=metrics.get('accuracy', 0.0),
        precision=metrics.get('precision', 0.0),
        recall=metrics.get('recall', 0.0),
        f1_score=metrics.get('f1_score', 0.0),
        inference_time_ms=metrics.get('inference_time_ms', 0.0)
    )

@app.post("/models/{model_type}/retrain")
async def retrain_model(model_type: str, training_data: Dict[str, Any]):
    """Retrain a specific model with new data"""
    if model_type not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_type} not found"
        )
    
    try:
        logger.info(f"Starting retraining for {model_type} model")
        
        model = models[model_type]
        await model.retrain(training_data)
        
        logger.info(f"✅ {model_type} model retrained successfully")
        
        return {
            "message": f"Model {model_type} retrained successfully",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Retraining error for {model_type}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/extract")
async def extract_features_endpoint(
    event_type: str,
    source_type: str,
    raw_data: Dict[str, Any]
):
    """Extract features from raw event data"""
    try:
        features = await feature_extractor.extract_features(
            raw_data,
            event_type,
            source_type
        )
        
        return {
            "features": features,
            "feature_count": len(features),
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_service_metrics():
    """Get overall service metrics"""
    try:
        metrics = await inference_service.get_metrics()
        
        return {
            "service_metrics": metrics,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Error getting service metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )
