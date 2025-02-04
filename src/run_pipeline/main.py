from src.data_processing.process_coinpair import process_coinpair
from src.data_processing.utility.train_val_test_split import time_series_folds
from src.data_processing.configs import coin_pairs
from src.data_modeling.XGBoost.XGBoost import train_XGBoost
# Import both backtest functions with aliases for clarity
from src.backtesting.simulate_backtest import backtest_model as simulate_backtest_model
from src.backtesting.optimized_backtest import backtest_model as optimized_backtest_model
import time

if __name__ == "__main__":
    all_results = []  # List to accumulate all folds' backtest summaries

    for coin, pair in coin_pairs.items():
        print(f"Processing coin: {coin}")
        data = process_coinpair(coin, pair)
        folds = time_series_folds(data, n_folds=3, initial_train_frac=0.3, val_frac=0.2)
        for fold_idx, (train_data, val_data, test_data) in enumerate(folds):
            print(f"\nCoin: {coin}, processing fold {fold_idx+1}/{len(folds)}...")
            # Train the model using training data and validation data for early stopping
            model = train_XGBoost(train_data, val_data)

            # --- Run the Simulate Backtest ---
            start_time_sim = time.time()
            sim_summary, sim_trade_logs_df = simulate_backtest_model(model, test_data)
            end_time_sim = time.time()
            sim_duration = end_time_sim - start_time_sim

            # --- Run the Optimized Backtest ---
            start_time_opt = time.time()
            opt_summary = optimized_backtest_model(model, test_data)
            end_time_opt = time.time()
            opt_duration = end_time_opt - start_time_opt

            # Print the Simulate Backtest Results
            print("\nSimulate Backtest Results:")
            print(f"Time Taken: {sim_duration:.4f} seconds")
            print(f"Final Balance: {sim_summary.get('final_balance')}")
            print(f"Num Trades: {sim_summary.get('num_trades')}")
            print(f"Win Rate: {sim_summary.get('win_rate')}")

            # Print the Optimized Backtest Results
            print("\nOptimized Backtest Results:")
            print(f"Time Taken: {opt_duration:.4f} seconds")
            print(f"Final Balance: {opt_summary.get('final_balance')}")
            print(f"Num Trades: {opt_summary.get('num_trades')}")
            print(f"Win Rate: {opt_summary.get('win_rate')}\n")

            # Aggregate results for further analysis if needed
            # all_results.append({
            #     'coin': coin,
            #     'fold': fold_idx + 1,
            #     'simulate': {
            #         'final_balance': sim_summary.get('final_balance'),
            #         'num_trades': sim_summary.get('num_trades'),
            #         'win_rate': sim_summary.get('win_rate'),
            #         'time': sim_duration
            #     },
            #     'optimized': {
            #         'final_balance': opt_summary.get('final_balance'),
            #         'num_trades': opt_summary.get('num_trades'),
            #         'win_rate': opt_summary.get('win_rate'),
            #         'time': opt_duration
            #     }
            # })

    # Trigger visualizations for key backtest results
    # from src.result_visualizations.visualizations import plot_backtest_results
    # plot_backtest_results(all_results)
  