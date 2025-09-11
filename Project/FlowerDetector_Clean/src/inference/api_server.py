"""
REST API Server for Flower Detection
FastAPI-based production server for real-time inference.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import asyncio
from pathlib import Path
import tempfile
import json
from datetime import datetime

from .inference_engine import FlowerInferenceEngine

logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: float
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_time_ms: float
    images_processed: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    performance_stats: Dict[str, Any]
    timestamp: str

# Initialize FastAPI app
app = FastAPI(
    title="Flower Detection API",
    description="Production API for flower detection using computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine
inference_engine: Optional[FlowerInferenceEngine] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the inference engine on startup."""
    global inference_engine
    
    try:
        # Load model from default path
        model_path = "models/checkpoints/best_model.pth"
        if not Path(model_path).exists():
            # Try alternative paths
            alternative_paths = [
                "models/best_model.pth",
                "checkpoints/best_model.pth",
                "best_model.pth"
            ]
            for alt_path in alternative_paths:
                if Path(alt_path).exists():
                    model_path = alt_path
                    break
            else:
                raise FileNotFoundError("No trained model found")
        
        inference_engine = FlowerInferenceEngine(model_path, device="cpu")
        logger.info("Inference engine initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        inference_engine = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation."""
    return """
    <html>
        <head>
            <title>Flower Detection API</title>
        </head>
        <body>
            <h1>Flower Detection API</h1>
            <p>Computer vision API for flower detection</p>
            <h2>Endpoints:</h2>
            <ul>
                <li><strong>POST /predict</strong> - Single image prediction</li>
                <li><strong>POST /predict/batch</strong> - Batch image prediction</li>
                <li><strong>GET /health</strong> - Health check and performance stats</li>
                <li><strong>GET /docs</strong> - Interactive API documentation</li>
            </ul>
            <h2>Usage:</h2>
            <p>Upload images using multipart/form-data to /predict endpoint</p>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with performance statistics."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    performance_stats = inference_engine.get_performance_stats()
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        performance_stats=performance_stats,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(file: UploadFile = File(...)):
    """
    Predict flower presence in a single uploaded image.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
        
    Returns:
        Prediction result with confidence scores
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Run inference
        result = inference_engine.predict_single(tmp_file_path)
        
        # Clean up temporary file
        Path(tmp_file_path).unlink()
        
        # Check for errors
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in single prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict flower presence in multiple uploaded images.
    
    Args:
        files: List of image files
        
    Returns:
        Batch prediction results
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 images per batch")
    
    # Validate all files
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
    
    try:
        start_time = datetime.now()
        
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                temp_paths.append(tmp_file.name)
        
        # Run batch inference
        results = inference_engine.predict_batch(temp_paths)
        
        # Clean up temporary files
        for temp_path in temp_paths:
            Path(temp_path).unlink()
        
        # Check for errors
        if any('error' in result for result in results):
            error_results = [r for r in results if 'error' in r]
            raise HTTPException(status_code=500, detail=f"Errors in {len(error_results)} predictions")
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchPredictionResponse(
            predictions=[PredictionResponse(**result) for result in results],
            total_time_ms=total_time,
            images_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/url")
async def predict_from_url(url: str):
    """
    Predict flower presence from image URL.
    
    Args:
        url: Image URL
        
    Returns:
        Prediction result
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    try:
        import requests
        from PIL import Image
        from io import BytesIO
        
        # Download image
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Convert to PIL Image
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        # Run inference
        result = inference_engine.predict_single(image)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return PredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in URL prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get detailed performance statistics."""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Inference engine not initialized")
    
    return inference_engine.get_performance_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
