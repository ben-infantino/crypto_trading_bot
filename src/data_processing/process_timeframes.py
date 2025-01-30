import pandas as pd
from src.data_processing.create_indicators import create_indicators

def process_timeframe(data:pd.DataFrame, timeframe:int, target_timeframe:int, relative_returns:bool=False, historical_data:bool=True):
    data = create_indicators(data, timeframe, historical_data)
    # if relative_returns:
    #     data = create_relative_returns(data, historical_data)
    if target_timeframe == timeframe:
        data['previous_close'] = data['close'].shift(1)
        data = create_targets(data, relative_returns)
    #drop open, high, low, volume, timestamp
    data.drop(columns=['open', 'high', 'low','trades'], inplace=True)
    data.dropna(inplace=True)
    return data

def create_targets(data, relative_returns:bool=True):
    if relative_returns:
        data['target'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * 100
        data.drop(columns=['close'], inplace=True)
        return data
    else:
        data['target'] = data['close']
        data.drop(columns=['close'], inplace=True)
        return data

def create_relative_returns(data, historical_data:bool=True):
    if historical_data:
        data['close_return'] = data['close'].pct_change().shift(1) * 100
        # data['open_return'] = data['open'].pct_change().shift(1)
        # data['high_return'] = data['high'].pct_change().shift(1)
        # data['low_return'] = data['low'].pct_change().shift(1)
        data['volume_return'] = data['volume'].pct_change().shift(1) * 100
    else:
        data['close_return'] = data['close'].pct_change() * 100
        # data['open_return'] = data['open'].pct_change()
        # data['high_return'] = data['high'].pct_change()
        # data['low_return'] = data['low'].pct_change()
        data['volume_return'] = data['volume'].pct_change() * 100
    return data

# def create_lags(data, lags:int, historical_data:bool=True):
#     if historical_data:
#         for lag in range(1, lags+1):
#             data[f'close_return_{lag}'] = data['close'].pct_change(lag).shift(1)
#             data[f'volume_return_{lag}'] = data['volume'].pct_change(lag).shift(1)
#     else:
#         for lag in range(1, lags+1):
#             data[f'close_return_{lag}'] = data['close'].pct_change(lag)
#             data[f'volume_return_{lag}'] = data['volume'].pct_change(lag)
#     return data
