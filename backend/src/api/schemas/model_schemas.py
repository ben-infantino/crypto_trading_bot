from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class PredictionRequest(BaseModel):
    """
    Request schema for model predictions.
    """
    coin: str = Field(..., description="Cryptocurrency symbol (e.g., 'BTC')")
    pair: str = Field(..., description="Trading pair (e.g., 'USDT')")
    timeframe: str = Field("30m", description="Timeframe for prediction")
    features: Optional[Dict[str, Any]] = Field(None, description="Additional features for prediction")

class PredictionResponse(BaseModel):
    """
    Response schema for model predictions.
    """
    coin: str
    pair: str
    timestamp: str
    prediction: float = Field(..., description="Predicted price change percentage")
    confidence: float = Field(..., description="Confidence score (0-1)")
    
class ModelInfo(BaseModel):
    """
    Schema for model information.
    """
    name: str
    type: str
    coin: str
    pair: str
    timeframe: str
    last_trained: str
    metrics: Dict[str, float]
    
class TrainingRequest(BaseModel):
    """
    Request schema for model training.
    """
    coin: str
    pair: str
    timeframe: str = "30m"
    model_type: str = "xgboost"
    hyperparameters: Optional[Dict[str, Any]] = None 