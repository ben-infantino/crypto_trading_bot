from src.data_processing.process_coinpair import process_coinpair
from src.data_processing.utility.train_val_test_split import time_series_folds
from src.data_processing.configs import coin_pairs
from src.data_modeling.XGBoost.XGBoost import train_XGBoost
from src.backtesting.optimized_backtest import backtest_model as optimized_backtest_model
import time
import pandas as pd
if __name__ == "__main__":
    all_results = []  # List to accumulate all folds' backtest summaries

    for coin, pair in coin_pairs.items():
        print(f"Processing coin: {coin}")
        data = process_coinpair(coin, pair)

        folds = time_series_folds(data, n_folds=5, initial_train_frac=0.3, val_frac=0.2)



        for fold_idx, (train_data, val_data, test_data) in enumerate(folds):
            print(f"\nCoin: {coin}, processing fold {fold_idx+1}/{len(folds)}...")
            # --- Sklearn XGBoost Model ---
            start_time_sklearn_train = time.time()
            model_sklearn = train_XGBoost(train_data, val_data)
            end_time_sklearn_train = time.time()
            sklearn_train_time = end_time_sklearn_train - start_time_sklearn_train

            start_time_sklearn_bt = time.time()
            sklearn_summary = optimized_backtest_model(model_sklearn, test_data)
            end_time_sklearn_bt = time.time()
            sklearn_bt_time = end_time_sklearn_bt - start_time_sklearn_bt

            total_sklearn_time = sklearn_train_time + sklearn_bt_time

            print("\n[Sklearn XGBoost] Training Time: {:.4f} sec, Backtest Time: {:.4f} sec, Total: {:.4f} sec"
                  .format(sklearn_train_time, sklearn_bt_time, total_sklearn_time))
            print(f"Final Balance: {sklearn_summary.get('final_balance')}")
            print(f"Num Trades: {sklearn_summary.get('num_trades')}")
            print(f"Win Rate: {sklearn_summary.get('win_rate')}\n")


    # Trigger visualizations for key backtest results
    # from src.result_visualizations.visualizations import plot_backtest_results
    # plot_backtest_results(all_results)
  