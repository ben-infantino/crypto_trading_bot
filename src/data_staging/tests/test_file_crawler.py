import os
from pprint import pprint
from data_staging.get_csv_paths import get_file_structure
from src.shared.config import data_folder

def test_xrp_all_pairs_all_timeframes():
    """Test with XRP only (all pairs, all timeframes)."""
    print("\nRunning test: XRP - All Pairs, All Timeframes")
    result = get_file_structure(data_folder, coin="XRP")
    print("File Structure Returned:")
    pprint(result)

def test_xrp_with_specific_pair():
    """Test with XRP and a specific pair (e.g., XRPUSD)."""
    print("\nRunning test: XRP - Specific Pair (XRPUSD)")
    result = get_file_structure(data_folder, coin="XRP", pair="XRPUSD")
    print("File Structure Returned:")
    pprint(result)

def test_xrp_with_specific_pair_and_timeframe():
    """Test with XRP, a specific pair, and a specific timeframe."""
    print("\nRunning test: XRP - Specific Pair (XRPUSD) and Timeframe (1-minute)")
    result = get_file_structure(data_folder, coin="XRP", pair="XRPUSD", timeframes=[1])
    print("File Structure Returned:")
    pprint(result)

def test_xrp_with_timeframe_without_pair():
    """Test with XRP and a specific timeframe but no pair specified."""
    print("\nRunning test: XRP - All Pairs and Specific Timeframe (1440-minute)")
    result = get_file_structure(data_folder, coin="XRP", timeframes=[1440])
    print("File Structure Returned:")
    pprint(result)

if __name__ == "__main__":
    print("Starting tests...\n")
    test_xrp_all_pairs_all_timeframes()
    test_xrp_with_specific_pair()
    test_xrp_with_specific_pair_and_timeframe()
    test_xrp_with_timeframe_without_pair()
    print("\nAll tests completed.")
