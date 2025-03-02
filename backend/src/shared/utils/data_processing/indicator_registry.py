from backend.src.shared.utils.data_processing.technical_indicators import compute_ema, compute_sma, compute_rsi, compute_bollinger, compute_atr

# We can store references to these functions in a registry, with default parameter sets:
INDICATOR_FUNCTIONS = {
    "ema": {
        "func": compute_ema,
        "params": {"window": 20, "close_col": "close", "col_name_prefix": "ema"}
    },
    "sma": {
        "func": compute_sma,
        "params": {"window": 20, "close_col": "close", "col_name_prefix": "sma"}
    },
    "rsi": {
        "func": compute_rsi,
        "params": {"window": 14, "close_col": "close", "col_name_prefix": "rsi"}
    },
    "bollinger": {
        "func": compute_bollinger,
        "params": {"window": 20, "std_dev": 2, "close_col": "close", "col_name_prefix": "bb"}
    },
    "atr": {
        "func": compute_atr,
        "params": {"window": 14, "high_col": "high", "low_col": "low", "close_col": "close", "col_name_prefix": "atr"}
    }
}
