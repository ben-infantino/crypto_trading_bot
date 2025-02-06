import pandas as pd
import numpy as np
import xgboost as xgb  # Added for GPU based predictions using DMatrix
import numba
from numba import njit

@njit
def simulate_trades(predicted_prices, targets, percent_to_buy, fee_percentage, slippage_percentage, initial_balance):
    n = predicted_prices.shape[0]
    balance = initial_balance
    # Preallocate arrays for balance history & trade returns (max possible trades = n)
    balance_history = np.empty(n, dtype=np.float64)
    balance_history[0] = balance
    trade_returns = np.empty(n, dtype=np.float64)
    trade_count = 0
    win_count = 0
    loss_count = 0
    total_profit_win = 0.0
    total_profit_loss = 0.0

    for i in range(1, n):
        entry_price_raw = targets[i - 1]
        pred_price = predicted_prices[i]
        if pred_price > entry_price_raw:
            trade_value = balance * percent_to_buy
            effective_entry_price = entry_price_raw * (1.0 + slippage_percentage)
            buy_fee = trade_value * fee_percentage
            net_trade_value = trade_value - buy_fee
            coin_qty = net_trade_value / effective_entry_price

            exit_price_raw = targets[i]
            effective_exit_price = exit_price_raw * (1.0 - slippage_percentage)
            gross_sell_value = coin_qty * effective_exit_price
            sell_fee = gross_sell_value * fee_percentage
            net_sell_value = gross_sell_value - sell_fee

            profit = net_sell_value - trade_value
            balance += profit

            # Save trade return as profit percentage for Sharpe computation
            trade_returns[trade_count] = profit / trade_value

            if profit > 0:
                win_count += 1
                total_profit_win += profit
            elif profit < 0:
                loss_count += 1
                total_profit_loss += profit
            trade_count += 1
        balance_history[i] = balance

    return balance, trade_count, win_count, loss_count, total_profit_win, total_profit_loss, balance_history, trade_returns, initial_balance

def backtest_model(model, data):
    """
    Backtests a simple long-only strategy using a trained model and provided data.

    Strategy:
      - For each row T (starting at T=1), use the features at time T to predict the coin price.
      - Use the previous rows target (T-1 close) as the entry price.
      - If the predicted price > previous close, simulate a buy at T-1's close and sell at T's close.
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
    """
    # --- Set Testing Variables ---
    INITIAL_BALANCE = 10000         # Starting balance
    PERCENT_TO_BUY = 0.10           # Invest 10% of current balance on each trade
    FEE_PERCENTAGE = 0       
    SLIPPAGE_PERCENTAGE = 0     

    # Unpack the data dictionary
    features = data["features"].reset_index(drop=True)
    targets = data["targets"].reset_index(drop=True)
    
    # --- Batch Prediction Setup ---
    # Create a DMatrix with all features (this will leverage the GPU if model is configured for GPU)
    # Compute predictions for all rows at once
    predicted_prices = model.predict(features)
    
    # Convert both the predicted prices and targets to numpy arrays for faster access
    predicted_prices = np.asarray(predicted_prices)
    targets_arr = targets.to_numpy()

    # Call the optimized simulation function (uses Numba for speed)
    final_balance, num_trades, wins, losses, total_profit_win, total_profit_loss, balance_history, trade_returns_array, initial_balance = simulate_trades(
        predicted_prices,
        targets_arr,
        PERCENT_TO_BUY,
        FEE_PERCENTAGE,
        SLIPPAGE_PERCENTAGE,
        INITIAL_BALANCE,
    )

    # --- Calculate Aggregate Metrics ---
    total_profit = final_balance - initial_balance
    avg_profit_per_trade = total_profit / num_trades if num_trades > 0 else 0.0
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    loss_rate = losses / num_trades if num_trades > 0 else 0.0

    # Maximum Drawdown Calculation using the balance history array
    running_max = np.maximum.accumulate(balance_history)
    drawdowns = (running_max - balance_history) / running_max
    max_drawdown = np.max(drawdowns)

    # Sharpe Ratio based on per-trade returns (profit/trade_value)
    trade_returns = trade_returns_array[:num_trades]
    if num_trades > 0 and np.std(trade_returns) > 0:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
    else:
        sharpe_ratio = np.nan

    # --- New: Sortino Ratio Calculation with Penalty ---
    # Calculate downside deviation as the std of negative trade returns (assuming a target of 0)
    if num_trades > 0:
        negative_returns = trade_returns[trade_returns < 0]
        if negative_returns.size > 0 and np.std(negative_returns) > 0:
            sortino_ratio = np.mean(trade_returns) / np.std(negative_returns)
        else:
            sortino_ratio = -10 # Penalize if not calculable
    else:
        sortino_ratio = -10  # Penalize if no trades


    # Profit Factor (Total profit on winning trades divided by the absolute loss on losing trades)
    if total_profit_loss < 0:
        profit_factor = total_profit_win / abs(total_profit_loss)
    else:
        profit_factor = np.nan

    # --- Update Aggregate Metrics Summary ---
    summary = {
        'initial_balance': initial_balance,
        'final_balance': final_balance,
        'total_profit': final_balance - initial_balance,  # Kept for reference if needed
        'num_trades': num_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_profit_per_trade': avg_profit_per_trade,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'sortino_ratio': sortino_ratio  # Added sortino ratio to the summary
    }

    # --- Return the updated summary and sortino ratio instead of total profit ---
    return summary, sortino_ratio