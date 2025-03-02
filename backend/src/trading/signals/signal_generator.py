import pandas as pd
import numpy as np
from backend.src.shared.logging.logger import setup_logger

logger = setup_logger("signal_generator")

class SignalGenerator:
    """
    Generates trading signals based on model predictions and additional filters.
    """
    
    def __init__(self, threshold=0.02, confidence_threshold=0.6):
        """
        Initialize the signal generator.
        
        Args:
            threshold (float): Minimum price change threshold to generate a signal
            confidence_threshold (float): Minimum confidence level to generate a signal
        """
        self.threshold = threshold
        self.confidence_threshold = confidence_threshold
        logger.info(f"Signal generator initialized with threshold={threshold}, confidence_threshold={confidence_threshold}")
    
    def generate_signals(self, predictions):
        """
        Generate trading signals based on model predictions.
        
        Args:
            predictions (pd.DataFrame): DataFrame with predictions and confidence scores
            
        Returns:
            pd.DataFrame: DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.DataFrame(index=predictions.index)
        signals['signal'] = 0
        
        # Generate buy signals
        buy_condition = (
            (predictions['prediction'] > self.threshold) & 
            (predictions['confidence'] > self.confidence_threshold)
        )
        signals.loc[buy_condition, 'signal'] = 1
        
        # Generate sell signals
        sell_condition = (
            (predictions['prediction'] < -self.threshold) & 
            (predictions['confidence'] > self.confidence_threshold)
        )
        signals.loc[sell_condition, 'signal'] = -1
        
        # Log signal statistics
        buy_count = signals[signals['signal'] == 1].shape[0]
        sell_count = signals[signals['signal'] == -1].shape[0]
        hold_count = signals[signals['signal'] == 0].shape[0]
        
        logger.info(f"Generated signals: {buy_count} buy, {sell_count} sell, {hold_count} hold")
        
        return signals
    
    def apply_filters(self, signals, data):
        """
        Apply additional filters to refine signals.
        
        Args:
            signals (pd.DataFrame): DataFrame with initial signals
            data (pd.DataFrame): DataFrame with market data
            
        Returns:
            pd.DataFrame: DataFrame with filtered signals
        """
        # Example filter: Don't trade during high volatility
        if 'volatility' in data.columns:
            high_volatility = data['volatility'] > data['volatility'].quantile(0.9)
            signals.loc[high_volatility, 'signal'] = 0
            logger.info(f"Filtered out {high_volatility.sum()} signals due to high volatility")
        
        return signals 