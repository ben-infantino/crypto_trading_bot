from src.data_processing.utility.indicator_pipeline import IndicatorPipeline
import pandas as pd
from src.data_processing.configs import indicator_configs

def create_indicators(data:pd.DataFrame, timeframe:int, historical_data:bool=True):
    indicator_pipeline = IndicatorPipeline(indicator_configs[timeframe], historical_data)
    data = indicator_pipeline.run(data)
    data.dropna(inplace=True)
    return data