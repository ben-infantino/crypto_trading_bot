import pandas as pd
import numpy as np

def merge_timeframes(data_dict: dict, target_timeframe: int) -> pd.DataFrame:
    """
    Merges multiple timeframe datasets onto a target timeframe’s DataFrame.
    
    This function:
      - Finds, for each target timestamp T, the latest timestamp in each non-target timeframe
        that is <= T - tf_in_seconds (where tf_in_seconds is the timeframe's duration in seconds).
      - Merges the corresponding row's data into the target DataFrame.
      - Ensures no look-ahead bias by strictly using data that closed before the target's timestamp.
    
    Parameters
    ----------
    data_dict : dict
        A dictionary of the form:
            {
              timeframe_in_minutes: {
                "data": pd.DataFrame(...),  # must have 'timestamp' in epoch seconds
                "lags": int                  # No longer used; can be ignored or removed
              },
              ...
            }
        Example:
            {
              1:  {"data": df_1m,  },
              5:  {"data": df_5m,  },
              15: {"data": df_15m, },
              30: {"data": df_30m, },
              60: {"data": df_60m, }
            }
        Note: The "lags" key is ignored in this function.
    
    target_timeframe : int
        Which timeframe (in *minutes*) to use as our “base” DataFrame.
        The output DataFrame will have exactly the same rows as this timeframe’s DataFrame.
    
    Returns
    -------
    pd.DataFrame
        The target timeframe DataFrame, augmented with columns from each other timeframe
        according to their lag specifications.
    """
    
    if target_timeframe not in data_dict:
        raise ValueError(f"target_timeframe={target_timeframe} not found in data_dict.")
    
    # --- 1) Extract and prepare the target DataFrame
    target_info = data_dict[target_timeframe]
    target_df = target_info.copy()
    
    if "timestamp" not in target_df.columns:
        raise ValueError("Target DataFrame must have a 'timestamp' column.")
    
    # Sort target DataFrame by timestamp
    target_df.sort_values("timestamp", inplace=True)
    target_df.reset_index(drop=True, inplace=True)
    
    # Initialize the merged DataFrame with target data
    merged_df = target_df.copy()
    
    # --- 2) Iterate over each non-target timeframe and merge data
    for tf, tf_info in data_dict.items():
        if tf == target_timeframe:
            continue  # Skip merging the target timeframe with itself
        
        tf_df = tf_info.copy()
        
        if "timestamp" not in tf_df.columns:
            raise ValueError(f"Data for timeframe={tf} must have a 'timestamp' column.")
        
        # Sort the non-target timeframe DataFrame by timestamp
        tf_df.sort_values("timestamp", inplace=True)
        
        # Convert timeframe from minutes to seconds
        tf_in_seconds = tf * 60
        
        # Prepare the non-target DataFrame for merging
        # Rename columns to include the timeframe as a prefix, excluding 'timestamp'
        tf_cols = [col for col in tf_df.columns if col != "timestamp"]
        tf_df_renamed = tf_df.rename(columns={col: f"{tf}_{col}" for col in tf_cols})
        
        # Perform the asof merge:
        # For each target timestamp T, find the latest tf timestamp <= T - tf_in_seconds
        # This ensures the tf data closes before the target timeframe opens
        shifted_target = target_df[['timestamp']].copy()
        shifted_target['lookup_timestamp'] = shifted_target['timestamp'] - tf_in_seconds
        
        # Ensure no negative timestamps (optional, depending on your data)
        shifted_target['lookup_timestamp'] = shifted_target['lookup_timestamp'].apply(
            lambda x: x if x >= 0 else np.nan
        )
        
        # Merge using pandas.merge_asof
        merged = pd.merge_asof(
            shifted_target.sort_values('lookup_timestamp'),
            tf_df_renamed.sort_values('timestamp'),
            left_on='lookup_timestamp',
            right_on='timestamp',
            direction='backward'
        )
        
        # Drop the 'lookup_timestamp' and 'timestamp_right' columns
        merged.drop(['lookup_timestamp', 'timestamp_x', 'timestamp_y'], axis=1, inplace=True)
        
        # Merge the new columns into the main merged_df
        merged_df = pd.concat([merged_df, merged], axis=1)
    
    return merged_df
