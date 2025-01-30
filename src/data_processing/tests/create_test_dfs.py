import pandas as pd

def create_and_save_timeframe_data():
    """
    Creates synthetic OHLCV-like DataFrames with:
     - Unix epoch timestamps (in seconds)
     - A 'color' column with a unique label per timeframe & row
    Saves each to a CSV in the current working directory.
    """
    
    # Define your timeframes (in minutes) and the "color" label for each
    timeframes = {
        1:    "red",
        5:    "orange",
        15:   "yellow",
        30:   "green",
        60:   "blue",
        240:  "purple",
        720:  "brown",
        1440: "black"
    }
    
    # We'll generate 1,000,000 rows for 1m,
    # and for other timeframes: 1,000,000 // timeframe
    BASE_ROWS = 100000
    
    for tf_in_minutes, color in timeframes.items():
        if tf_in_minutes == 1:
            n_rows = BASE_ROWS
        else:
            n_rows = BASE_ROWS // tf_in_minutes  # integer division
        
        # Each candle is (tf_in_minutes * 60) seconds wide
        # So row i has timestamp = i * tf_in_minutes * 60
        timestamps = [i * tf_in_minutes * 60 for i in range(n_rows)]
        
        # Create a color/timestamp string, e.g. "red_0", "red_60", ...
        color_vals = [f"{color}_{ts}" for ts in timestamps]
        
        # Build a simple DataFrame with 'timestamp' and 'color' columns
        df = pd.DataFrame({
            "timestamp": timestamps,
            "color": color_vals
        })
        
        # Save to CSV: e.g. df_1m.csv, df_5m.csv, ...
        csv_name = f"df_{tf_in_minutes}m.csv"
        df.to_csv(csv_name, index=False)
        
        print(f"Saved {csv_name} with {len(df)} rows (timeframe={tf_in_minutes}m).")

if __name__ == "__main__":
    create_and_save_timeframe_data()
