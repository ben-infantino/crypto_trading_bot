import os
import pandas as pd
from src.data_processing.merge_timeframes import merge_timeframes

def test_merge_timeframes():
    """
    - Reads CSVs from the 'data/' folder in the same directory as this script.
    - Constructs data_dict with some example 'lags' values.
    - Calls merge_timeframes with a chosen target timeframe.
    - Saves the merged DataFrame back into the 'data/' folder.
    """
    # Locate the folder where this script is, and build a path to the 'data/' subfolder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(script_dir, "data")

    # Define the timeframes and (optionally) their lag values
    # Adjust 'lags' as you see fit
    timeframes = [1, 5, 15, 30, 60, 240, 720, 1440]
    data_dict = {}

    for tf in timeframes:
        csv_filename = f"df_{tf}m.csv"
        csv_path = os.path.join(data_folder, csv_filename)
        df_tmp = pd.read_csv(csv_path)
        # Example: 3 lags for each timeframe
        data_dict[tf] = {"data": df_tmp, "lags": 3}
    
    # Pick a target timeframe, e.g. 30 minutes
    target_tf = 30

    # Merge
    merged_df = merge_timeframes(data_dict, target_timeframe=target_tf)

    # Print some info
    print("Merged DataFrame columns:")
    print(merged_df.columns.tolist())
    print("\nHead of merged DataFrame:")
    print(merged_df.tail(10))
    print(f"\nNumber of rows: {len(merged_df)}")

    # Save the merged DataFrame to the same 'data' folder
    output_csv = os.path.join(data_folder, f"merged_{target_tf}m.csv")
    merged_df.to_csv(output_csv, index=False)
    print(f"\nSaved merged DataFrame to: {output_csv}")


if __name__ == "__main__":
    test_merge_timeframes()