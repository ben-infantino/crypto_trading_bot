import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from backend.src.shared.utils.data_processing.indicator_registry import INDICATOR_FUNCTIONS

@dataclass
class IndicatorConfig:
    """Configuration class for indicators"""
    name: str
    function: Callable
    params: Dict[str, Any]
    columns: List[str]
    
class IndicatorPipeline:
    """Pipeline for creating technical indicators"""
    
    def __init__(self, indicator_configs: List[IndicatorConfig], historical_data: bool = True):
        """
        Initialize the indicator pipeline with configurations
        
        Args:
            indicator_configs: List of indicator configurations
            historical_data: Whether to use historical data calculations
        """
        self.indicator_configs = indicator_configs
        self.historical_data = historical_data
        
    def run(self, df):
        """
        Apply each indicator in sequence, mutating the DataFrame.
        Returns the same DataFrame with new columns for each indicator.
        """
        for config in self.indicator_configs:
            ind_name = config['name']
            # get the function + default params from the registry
            if ind_name not in INDICATOR_FUNCTIONS:
                raise ValueError(f"Indicator '{ind_name}' not found in registry.")
            
            func = INDICATOR_FUNCTIONS[ind_name]['func']
            params = INDICATOR_FUNCTIONS[ind_name]['params'].copy()
            params['historical_data'] = self.historical_data
            # override defaults if provided
            if 'override_params' in config:
                for k, v in config['override_params'].items():
                    params[k] = v

            # apply the function
            df = func(df, **params)

        return df