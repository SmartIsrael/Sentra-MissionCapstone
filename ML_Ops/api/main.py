"""
Professional FastAPI Server for Crop Disease Classification

This server provides REST API endpoints for IoT devices like Raspberry Pi
to upload images and receive crop disease predictions.

Features:
- Image upload and processing
- Real-time predictions with confidence scores
- Batch processing support
- Professional error handling
- Comprehensive logging
- Health checks and monitoring
"""

import io
import os
import time
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn


# ===== Configuration =====
class Config:
    """Application configuration"""
    # Model settings
    MODEL_PATH = "../models/resnet50_model_hf.pt"  # Path to trained model
    MODEL_TYPE = "resnet50"
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # API settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    MAX_BATCH_SIZE = 10
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "crop_api.log"


# ===== Logging Setup =====
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===== Data Models =====
class PredictionResponse(BaseModel):
    """Response model for single image prediction"""
    success: bool = True
    crop: str = Field(..., description="Predicted crop type")
    disease: str = Field(..., description="Predicted disease")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_info: Dict[str, Any] = Field(..., description="Model information")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    success: bool = True
    total_images: int = Field(..., description="Total number of images processed")
    successful_predictions: int = Field(..., description="Number of successful predictions")
    batch_results: List[Dict[str, Any]] = Field(..., description="Individual prediction results")
    total_processing_time: float = Field(..., description="Total processing time in seconds")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = "healthy"
    timestamp: str = Field(..., description="Current timestamp")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    device: str = Field(..., description="Device being used for inference")
    supported_formats: List[str] = Field(..., description="Supported image formats")


class ClassInfo(BaseModel):
    """Information about a single class"""
    index: int = Field(..., description="Class index")
    full_name: str = Field(..., description="Full class name")
    crop: str = Field(..., description="Crop type")
    disease: str = Field(..., description="Disease name")


class ClassesResponse(BaseModel):
    """Response model for available classes"""
    total_classes: int = Field(..., description="Total number of classes")
    classes: List[ClassInfo] = Field(..., description="List of all classes")


# ===== Model and Preprocessing =====
class CropDiseaseModel:
    """Crop disease classification model wrapper"""
    
    def __init__(self):
        self.model = None
        self.class_names = None
        self.transform = None
        self.device = Config.DEVICE
        self.model_loaded = False
        
        # Class names based on your African Common Crops dataset from Hugging Face
        self.class_names = [
            'Apple___Apple_scab',
            'Apple___Black_rot',
            'Apple___Cedar_apple_rust',
            'Apple___healthy',
            'Banana _Black_sigatoka_disease',
            'Banana _Yellow_sigatoka_disease',
            'Banana__Bract_mosaic_virus_disease',
            'Banana__Healthy Leaf',
            'Banana__Insect_pest_disease',
            'Banana__Moko_disease',
            'Banana__Panama_disease',
            'Banana__Pestalotiopsis',
            'Banana__cordana',
            'Beans_Angular_leaf_spot',
            'Beans_bean_rust',
            'Beans_healthy',
            'Blueberry___healthy',
            'Cacao__Fito',
            'Cacao__Monilia',
            'Cacao__Sana',
            'Cassava__cbb',
            'Cassava__cbsd',
            'Cassava__cgm',
            'Cassava__cmd',
            'Cassava__healthy',
            'Cherry_(including_sour)___Powdery_mildew',
            'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
            'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight',
            'Corn_(maize)___healthy',
            'Cowpea__Aphids',
            'Cowpea__Cercospora_leaf_spot',
            'Cowpea__Fusarium_wilt',
            'Cowpea__Maruca_pod_borer',
            'Cowpea__Mosaic_virus',
            'Cowpea__Thrips',
            'Cowpea__healthy',
            'Garlic__Downy_mildew',
            'Garlic__Fusarium_basal_rot',
            'Garlic__Nematodes',
            'Garlic__Onion_thrips',
            'Garlic__White_rot',
            'Garlic__healthy',
            'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)',
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            'Grape___healthy',
            'Groundnut__early_leaf_spot',
            'Groundnut__early_rust',
            'Orange___Haunglongbing_(Citrus_greening)',
            'Peach___Bacterial_spot',
            'Peach___healthy',
            'Pepper,_bell___Bacterial_spot',
            'Pepper,_bell___healthy',
            'Potato___Early_blight',
            'Potato___Late_blight',
            'Potato___healthy',
            'Raspberry___healthy',
            'Soybean___healthy',
            'Squash___Powdery_mildew',
            'Strawberry___Leaf_scorch',
            'Strawberry___healthy',
            'Tomato___Bacterial_spot',
            'Tomato___Early_blight',
            'Tomato___Late_blight',
            'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite',
            'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        
        self._setup_transforms()
        self._load_model()
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE + 32),
            transforms.CenterCrop(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the trained model"""
        try:
            # Create model architecture
            if Config.MODEL_TYPE == "resnet50":
                self.model = models.resnet50(pretrained=False)
                num_features = self.model.fc.in_features
                self.model.fc = nn.Linear(num_features, len(self.class_names))
            else:
                raise ValueError(f"Unsupported model type: {Config.MODEL_TYPE}")
            
            # Load trained weights if available
            model_path = Path(Config.MODEL_PATH)
            if model_path.exists():
                logger.info(f"Loading model from {model_path}")
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Model loaded with accuracy: {checkpoint.get('accuracy', 'N/A')}")
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model_loaded = True
            else:
                logger.warning(f"Model file not found at {model_path}. Using untrained model.")
                # For demo purposes, we'll continue without trained weights
                self.model_loaded = False
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {str(e)}"
            )
    
    def parse_class_name(self, class_name: str) -> tuple:
        """Parse class name to extract crop and disease information"""
        try:
            # Handle special cases for African crops
            if "Banana " in class_name or "Banana__" in class_name:
                # Handle Banana with various separators
                if "Banana " in class_name:
                    parts = class_name.split("_")
                    crop = "Banana"
                    disease = "_".join(parts[1:]).replace("_", " ").title()
                else:
                    parts = class_name.split("__")
                    crop = "Banana"
                    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
            elif "Cherry_(including_sour)" in class_name:
                parts = class_name.split("___")
                crop = "Cherry"
                disease = parts[1] if len(parts) > 1 else "Unknown"
            elif "Pepper,_bell" in class_name:
                parts = class_name.split("___")
                crop = "Bell Pepper"
                disease = parts[1] if len(parts) > 1 else "Unknown"
            elif "Corn_(maize)" in class_name:
                parts = class_name.split("___")
                crop = "Corn (Maize)"
                disease = parts[1] if len(parts) > 1 else "Unknown"
            elif "Spider_mites Two-spotted_spider_mite" in class_name:
                crop = "Tomato"
                disease = "Spider mites (Two-spotted spider mite)"
            elif "Beans_" in class_name:
                parts = class_name.split("_")
                crop = "Beans"
                disease = "_".join(parts[1:]).replace("_", " ").title()
            elif "Cacao__" in class_name:
                parts = class_name.split("__")
                crop = "Cacao"
                disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
            elif "Cassava__" in class_name:
                parts = class_name.split("__")
                crop = "Cassava"
                disease = parts[1].upper() if len(parts) > 1 else "Unknown"  # Keep abbreviations uppercase
            elif "Cowpea__" in class_name:
                parts = class_name.split("__")
                crop = "Cowpea"
                disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
            elif "Garlic__" in class_name:
                parts = class_name.split("__")
                crop = "Garlic"
                disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
            elif "Groundnut__" in class_name:
                parts = class_name.split("__")
                crop = "Groundnut"
                disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
            else:
                # Standard format: Crop___Disease
                if "___" in class_name:
                    parts = class_name.split("___")
                    crop = parts[0].replace("_", " ").title()
                    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
                elif "__" in class_name:
                    parts = class_name.split("__")
                    crop = parts[0].replace("_", " ").title()
                    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
                else:
                    parts = class_name.split("_")
                    crop = parts[0].title()
                    disease = "_".join(parts[1:]).replace("_", " ").title() if len(parts) > 1 else "Unknown"
            
            # Clean up disease names
            disease = disease.replace("_", " ")
            if disease.lower() == "healthy":
                disease = "Healthy"
            elif disease.lower() == "sana":
                disease = "Healthy"  # Cacao__Sana means healthy cacao
            
            return crop, disease
            
        except Exception as e:
            logger.error(f"Error parsing class name '{class_name}': {e}")
            return "Unknown", "Unknown"
    
    def predict_image(self, image: Image.Image) -> Dict[str, Any]:
        """Predict crop disease from a PIL Image"""
        start_time = time.time()
        
        try:
            # Preprocess image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_idx = torch.max(probabilities, 1)
            
            # Get prediction results
            predicted_class = self.class_names[predicted_idx.item()]
            confidence_score = confidence.item()
            crop, disease = self.parse_class_name(predicted_class)
            
            processing_time = time.time() - start_time
            
            return {
                "crop": crop,
                "disease": disease,
                "confidence": confidence_score,
                "predicted_class": predicted_class,
                "processing_time": processing_time,
                "model_info": {
                    "model_type": Config.MODEL_TYPE,
                    "device": str(self.device),
                    "image_size": Config.IMG_SIZE,
                    "processing_time_seconds": round(processing_time, 4)
                }
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )
    
    def get_class_info(self) -> List[ClassInfo]:
        """Get information about all available classes"""
        class_info = []
        for idx, class_name in enumerate(self.class_names):
            crop, disease = self.parse_class_name(class_name)
            class_info.append(ClassInfo(
                index=idx,
                full_name=class_name,
                crop=crop,
                disease=disease
            ))
        return class_info


# ===== FastAPI Application =====
app = FastAPI(
    title="Crop Disease Classification API",
    description="Professional API for crop disease prediction from images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
model_wrapper = CropDiseaseModel()


# ===== Utility Functions =====
def validate_image_file(file: UploadFile) -> None:
    """Validate uploaded image file"""
    # Check file size
    if hasattr(file, 'size') and file.size > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE / (1024*1024):.1f}MB"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )


def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load PIL Image from uploaded file"""
    try:
        # Read file content
        content = file.file.read()
        
        # Reset file pointer for potential re-use
        file.file.seek(0)
        
        # Load image
        image = Image.open(io.BytesIO(content))
        return image
        
    except Exception as e:
        logger.error(f"Error loading image from upload: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image file: {str(e)}"
        )


# ===== API Endpoints =====

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Crop Disease Classification API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        timestamp=datetime.now().isoformat(),
        model_loaded=model_wrapper.model_loaded,
        device=str(model_wrapper.device),
        supported_formats=list(Config.ALLOWED_EXTENSIONS)
    )


@app.get("/classes", response_model=ClassesResponse)
async def get_available_classes():
    """Get all available crop disease classes"""
    try:
        class_info = model_wrapper.get_class_info()
        return ClassesResponse(
            total_classes=len(class_info),
            classes=class_info
        )
    except Exception as e:
        logger.error(f"Error getting classes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve classes: {str(e)}"
        )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single_image(file: UploadFile = File(...)):
    """
    Predict crop disease from a single image
    
    Upload an image file and receive prediction results including:
    - Crop type
    - Disease identification  
    - Confidence score
    - Processing time
    """
    logger.info(f"Received prediction request for file: {file.filename}")
    
    try:
        # Validate file
        validate_image_file(file)
        
        # Load image
        image = load_image_from_upload(file)
        
        # Make prediction
        result = model_wrapper.predict_image(image)
        
        logger.info(f"Prediction completed: {result['crop']} - {result['disease']} ({result['confidence']:.3f})")
        
        return PredictionResponse(
            crop=result["crop"],
            disease=result["disease"],
            confidence=result["confidence"],
            processing_time=result["processing_time"],
            model_info=result["model_info"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prediction: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """
    Predict crop disease from multiple images
    
    Upload multiple image files and receive batch prediction results.
    Maximum batch size is configurable.
    """
    logger.info(f"Received batch prediction request for {len(files)} files")
    
    if len(files) > Config.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size too large. Maximum: {Config.MAX_BATCH_SIZE}"
        )
    
    start_time = time.time()
    results = []
    successful_predictions = 0
    
    for i, file in enumerate(files):
        try:
            logger.info(f"Processing file {i+1}/{len(files)}: {file.filename}")
            
            # Validate and process file
            validate_image_file(file)
            image = load_image_from_upload(file)
            
            # Make prediction
            prediction = model_wrapper.predict_image(image)
            
            results.append({
                "filename": file.filename,
                "index": i,
                "prediction": {
                    "crop": prediction["crop"],
                    "disease": prediction["disease"],
                    "confidence": prediction["confidence"],
                    "processing_time": prediction["processing_time"]
                }
            })
            
            successful_predictions += 1
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({
                "filename": file.filename,
                "index": i,
                "error": str(e)
            })
    
    total_time = time.time() - start_time
    
    logger.info(f"Batch prediction completed: {successful_predictions}/{len(files)} successful")
    
    return BatchPredictionResponse(
        total_images=len(files),
        successful_predictions=successful_predictions,
        batch_results=results,
        total_processing_time=total_time
    )


# ===== Application Startup =====
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Starting Crop Disease Classification API")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Model loaded: {model_wrapper.model_loaded}")
    logger.info(f"Available classes: {len(model_wrapper.class_names)}")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down Crop Disease Classification API")


# ===== Main Entry Point =====
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
