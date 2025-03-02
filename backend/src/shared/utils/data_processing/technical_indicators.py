import pandas as pd
import numpy as np

# For demonstration, we'll use the 'ta' library
import ta
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

################################################################################
# INDICATOR REGISTRY: each entry is an "indicator function" that knows how to
# compute certain columns from the DataFrame. We also define default parameters.
################################################################################

def compute_ema(df, close_col, window=20, col_name_prefix="ema", historical_data:bool=True):
    """Compute an EMA and return a new column name + the DataFrame."""
    ema_series = EMAIndicator(close=df[close_col], window=window).ema_indicator()
    new_col = f"{col_name_prefix}_{window}"
    if historical_data:
        df[new_col] = ema_series.shift(1)
    else:
        df[new_col] = ema_series
    return df

def compute_sma(df, close_col, window=20, col_name_prefix="sma", historical_data:bool=True):
    sma_series = SMAIndicator(close=df[close_col], window=window).sma_indicator()
    new_col = f"{col_name_prefix}_{window}"
    if historical_data:
        df[new_col] = sma_series.shift(1)
    else:
        df[new_col] = sma_series
    return df

def compute_rsi(df, close_col, window=14, col_name_prefix="rsi", historical_data:bool=True):
    rsi_series = RSIIndicator(close=df[close_col], window=window).rsi()
    new_col = f"{col_name_prefix}_{window}"
    if historical_data:
        df[new_col] = rsi_series.shift(1)
    else:
        df[new_col] = rsi_series
    return df

def compute_bollinger(df, close_col, window=20, std_dev=2, col_name_prefix="bb", historical_data:bool=True):
    bb = BollingerBands(close=df[close_col], window=window, window_dev=std_dev)
    mavg_col = f"{col_name_prefix}_mavg_{window}"
    high_col = f"{col_name_prefix}_high_{window}"
    low_col  = f"{col_name_prefix}_low_{window}"

    if historical_data:
        df[mavg_col] = bb.bollinger_mavg().shift(1)
        df[high_col] = bb.bollinger_hband().shift(1)
        df[low_col]  = bb.bollinger_lband().shift(1)
    else:
        df[mavg_col] = bb.bollinger_mavg()
        df[high_col] = bb.bollinger_hband()
        df[low_col]  = bb.bollinger_lband()
    return df

def compute_atr(df, high_col, low_col, close_col, window=14, col_name_prefix="atr", historical_data:bool=True):
    atr = AverageTrueRange(
        high=df[high_col], low=df[low_col], close=df[close_col], window=window
    ).average_true_range()
    new_col = f"{col_name_prefix}_{window}"
    if historical_data:
        df[new_col] = atr.shift(1)
    else:
        df[new_col] = atr
    return df
