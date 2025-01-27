import pandas as pd
import numpy as np
from src.data_processing.utility.indicator_pipeline import IndicatorPipeline
from src.data_processing.configs import indicator_configs
from src.data_processing.create_indicators import create_indicators


def process_single_file(data:pd.DataFrame, timeframe:int, relative_returns:bool=True, historical_data:bool=True):
    data = create_indicators(data, timeframe, historical_data)
    if relative_returns:
        data = create_relative_returns(data, historical_data)
    data = create_targets(data, relative_returns)
    #drop open, high, low, volume, timestamp
    return data

def create_relative_returns(data, historical_data:bool=True):
    if historical_data:
        data['close_return'] = data['close'].pct_change().shift(1)
        data['open_return'] = data['open'].pct_change().shift(1)
        data['high_return'] = data['high'].pct_change().shift(1)
        data['low_return'] = data['low'].pct_change().shift(1)
        data['volume_return'] = data['volume'].pct_change().shift(1)
    else:
        data['close_return'] = data['close'].pct_change()
        data['open_return'] = data['open'].pct_change()
        data['high_return'] = data['high'].pct_change()
        data['low_return'] = data['low'].pct_change()
        data['volume_return'] = data['volume'].pct_change()
    return data

def create_targets(data, relative_returns:bool=True):
    if relative_returns:
        data['target'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        data.dropna(inplace=True)
        return data
    else:
        data['target'] = data['close']
        data.drop(columns=['close'], inplace=True)
        return data

def train_validate_test_split(data, train_size=0.75, validate_size=0.15, test_size=0.1):
    # Indices (assuming sums to 1.0):
    train_end = int(len(data) * train_size)
    validate_end = train_end + int(len(data) * validate_size)
    # test starts at validate_end automatically

    train_data = data.iloc[:train_end]
    validate_data = data.iloc[train_end:validate_end]
    test_data = data.iloc[validate_end:]

    train_data = {
        'features': train_data.drop(columns=['return_next', 'close', 'open', 'high', 'low', 'volume', 'timestamp']),
        'targets': train_data['target']
    }
    validate_data = {
        'features': validate_data.drop(columns=['target', 'close', 'open', 'high', 'low', 'volume', 'timestamp']),
        'targets': validate_data['target']
    }
    test_data = {
        'features': test_data.drop(columns=['target', 'close', 'open', 'high', 'low', 'volume', 'timestamp']),
        'targets': test_data['target']
    }

    return train_data, validate_data, test_data

# def generate_data_splits(coin, pair, timeframes):
#     file_structure = get_file_structure(data_folder)

#     # Process specific data
#     data = process_data(file_structure, coin=coin, pair=pair, timeframes=timeframes, format="pandas")
#     train_data, validate_data, test_data = train_validate_test_split(data)
#     return train_data, validate_data, test_data

# if __name__ == "__main__":

#     # Example: Get the file structure
#     file_structure = get_file_structure(data_folder)

#     # Process specific data
#     data = process_data(file_structure, coin="XRP", pair="XRPUSD", timeframes=[60], format="pandas")
#     targets = data['return_next']
#     features = data.drop(columns=['return_next', 'close', 'open', 'high', 'low', 'volume', 'timestamp'])
#     print(features.shape)
#     print(targets.shape)
