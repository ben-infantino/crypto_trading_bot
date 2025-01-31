import pandas as pd
import numpy as np

def backtest_model(model, data):
    """
    Backtests a simple long-only strategy using a trained model and provided data.

    Strategy:
      - For each row T (starting at T=1), use the features at time T to predict the coin price.
      - Use the previous row’s target (T-1 close) as the entry price.
      - If the predicted price > previous close, simulate a buy at T-1’s close and sell at T’s close.
      - Adjust the entry price upward and the exit price downward to account for slippage.
      - No fees are applied (fee percentage is 0%).
      - A fixed percentage of the current balance is used for each trade.
      - The balance is updated immediately after each trade.

    Parameters:
      - model: A trained model with a .predict() method.
      - data: A dictionary with keys:
            • 'features': pandas DataFrame containing the features (chronologically ordered)
            • 'targets': pandas Series containing the target (close) prices (chronologically ordered)

    Returns:
      - summary: Dictionary of aggregate metrics.
      - trade_logs_df: DataFrame containing per-trade details.
    """
    print("Backtesting model...")
    # --- Set Testing Variables ---
    INITIAL_BALANCE = 10000         # Starting balance
    PERCENT_TO_BUY = 0.10           # Invest 10% of current balance on each trade
    FEE_PERCENTAGE = 0       
    SLIPPAGE_PERCENTAGE = 0     



    # Unpack the data dictionary
    features = data["features"].reset_index(drop=True)
    targets = data["targets"].reset_index(drop=True)

    # Initialize tracking variables
    balance = INITIAL_BALANCE
    initial_balance = INITIAL_BALANCE
    balance_history = []  # Balance after each row
    trade_logs = []       # List to hold details for each trade

    print("Starting backtest...")
    # Loop through each row starting at index 1 since we need a previous row for the entry price
    for i in range(1, len(features)):
        #print every 200 rows
        if i % 200 == 0:
            print(f"Backtest progress: {i}/{len(features)}")

        # Predict the future price using the features at time T (current row)
        current_features = features.iloc[i]
        predicted_price = model.predict(current_features.values.reshape(1, -1))[0]


        # Entry price is the previous row's target (T-1 close)
        entry_price_raw = targets.iloc[i - 1]

        # If the predicted price is higher than the last close, execute a trade
        if predicted_price > entry_price_raw:
            # Determine the dollar amount to invest on this trade
            trade_value = balance * PERCENT_TO_BUY

            # --- Buy Side ---
            # In real markets, the effective entry price can be worse due to slippage.
            effective_entry_price = entry_price_raw * (1 + SLIPPAGE_PERCENTAGE)
            buy_fee = trade_value * FEE_PERCENTAGE  # will be 0.0 here
            net_trade_value = trade_value - buy_fee
            coin_qty = net_trade_value / effective_entry_price

            # --- Sell Side ---
            exit_price_raw = targets.iloc[i]
            # Slippage on selling typically reduces the exit price.
            effective_exit_price = exit_price_raw * (1 - SLIPPAGE_PERCENTAGE)
            gross_sell_value = coin_qty * effective_exit_price
            sell_fee = gross_sell_value * FEE_PERCENTAGE  # still 0.0
            net_sell_value = gross_sell_value - sell_fee

            # Calculate profit (or loss) for the trade
            profit = net_sell_value - trade_value
            balance += profit  # update balance immediately

            # Log the details of this trade
            trade_log = {
                'entry_index': i - 1,
                'exit_index': i,
                'entry_price_raw': entry_price_raw,
                'effective_entry_price': effective_entry_price,
                'exit_price_raw': exit_price_raw,
                'effective_exit_price': effective_exit_price,
                'trade_value': trade_value,
                'buy_fee': buy_fee,
                'sell_fee': sell_fee,
                'coin_qty': coin_qty,
                'gross_sell_value': gross_sell_value,
                'net_sell_value': net_sell_value,
                'profit': profit,
                'balance_after_trade': balance,
                'predicted_price': predicted_price
            }
            trade_logs.append(trade_log)

        # Record the balance regardless of whether a trade occurred
        balance_history.append(balance)

    # --- Calculate Aggregate Metrics ---
    final_balance = balance
    total_profit = final_balance - initial_balance
    num_trades = len(trade_logs)
    wins = sum(1 for trade in trade_logs if trade['profit'] > 0)
    win_rate = wins / num_trades if num_trades > 0 else 0
    avg_profit_per_trade = total_profit / num_trades if num_trades > 0 else 0

    # Maximum Drawdown
    balance_array = np.array(balance_history)
    running_max = np.maximum.accumulate(balance_array)
    drawdowns = (running_max - balance_array) / running_max
    max_drawdown = np.max(drawdowns)

    # Sharpe Ratio based on per-trade returns (profit divided by trade value)
    trade_returns = np.array([trade['profit'] / trade['trade_value'] for trade in trade_logs]) if num_trades > 0 else np.array([])
    if trade_returns.size > 0 and np.std(trade_returns) > 0:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
    else:
        sharpe_ratio = np.nan

    summary = {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_profit': total_profit,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'avg_profit_per_trade': avg_profit_per_trade,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

    # --- Output the Results ---
    print("Backtest Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

    trade_logs_df = pd.DataFrame(trade_logs)
    print("\nTrade Logs:")
    print(trade_logs_df)

    # Uncomment the following lines to write results to a CSV file instead of printing:
    # OUTPUT_LOG_FILE = "trade_logs.csv"
    # trade_logs_df.to_csv(OUTPUT_LOG_FILE, index=False)

    return summary, trade_logs_df