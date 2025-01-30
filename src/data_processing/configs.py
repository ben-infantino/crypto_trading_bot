coin_pairs = {
    # "X": "XRPUSD",
    "ETH": "ETHUSD",
    # "SOL": "SOLUSD",
}

timeframes = [5, 15, 30, 60, 240]
target_timeframe = 60

indicator_configs = {
    1:  [
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],

    5:  [
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],

    15: [
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],

    30: [
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],

    60: [
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],

    240:[
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],
        
    720:[
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ],
        
    1440:[
            {"name": "ema", "override_params": {"window": 14}},
            {"name": "ema", "override_params": {"window": 50}},
            {"name": "sma", "override_params": {"window": 14}},
            {"name": "sma", "override_params": {"window": 50}},
            {"name": "rsi", "override_params": {"window": 14}},
            {"name": "bollinger", "override_params": {"window": 20, "std_dev": 2}},
            {"name": "atr", "override_params": {"window": 14}}
        ]
}


