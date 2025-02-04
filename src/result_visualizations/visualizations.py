import matplotlib.pyplot as plt
import pandas as pd

def plot_backtest_results(results):
    """
    Plots key backtesting metrics across folds for each coin.
    
    Expected keys in each result dict:
      - 'coin', 'fold', 'final_balance', 'total_profit', 'win_rate', 'max_drawdown', 'sharpe_ratio'
    """
    # Convert the list of result dictionaries to a DataFrame
    df = pd.DataFrame(results)
    
    coins = df['coin'].unique()
    metrics = ['final_balance', 'total_profit', 'win_rate', 'max_drawdown', 'sharpe_ratio']
    
    # Produce a plot for each coin
    for coin in coins:
        coin_df = df[df['coin'] == coin].sort_values(by='fold')
        plt.figure(figsize=(10, 6))
        for metric in metrics:
            plt.plot(coin_df['fold'], coin_df[metric], marker='o', label=metric)
        plt.title(f"Backtest Metrics Across Folds for {coin}")
        plt.xlabel("Fold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
