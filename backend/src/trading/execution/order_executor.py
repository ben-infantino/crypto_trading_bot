import pandas as pd
from datetime import datetime
from src.shared.logging.logger import setup_logger

logger = setup_logger("order_executor")

class OrderExecutor:
    """
    Executes trading orders based on signals.
    Currently a simulation for manual execution, but can be extended for automated trading.
    """
    
    def __init__(self, initial_balance=10000.0, position_size=0.1, max_positions=5):
        """
        Initialize the order executor.
        
        Args:
            initial_balance (float): Initial account balance
            position_size (float): Size of each position as a fraction of balance
            max_positions (int): Maximum number of concurrent positions
        """
        self.balance = initial_balance
        self.position_size = position_size
        self.max_positions = max_positions
        self.positions = {}
        self.trade_history = []
        
        logger.info(f"Order executor initialized with balance=${initial_balance}, position_size={position_size}, max_positions={max_positions}")
    
    def execute_signal(self, signal, coin, pair, price, timestamp=None):
        """
        Execute a trading signal.
        
        Args:
            signal (int): Signal value (1 for buy, -1 for sell, 0 for hold)
            coin (str): Cryptocurrency symbol
            pair (str): Trading pair
            price (float): Current price
            timestamp (str, optional): Signal timestamp
            
        Returns:
            dict: Order details
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        symbol = f"{coin}{pair}"
        
        if signal == 0:
            logger.info(f"Hold signal for {symbol} at ${price}")
            return None
        
        # Check if we can open a new position
        if signal == 1 and len(self.positions) >= self.max_positions:
            logger.warning(f"Cannot open position for {symbol}: maximum positions reached")
            return None
        
        # Check if we have an existing position to sell
        if signal == -1 and symbol not in self.positions:
            logger.warning(f"Cannot sell {symbol}: no position exists")
            return None
        
        # Execute buy order
        if signal == 1:
            position_amount = self.balance * self.position_size
            quantity = position_amount / price
            
            self.positions[symbol] = {
                'entry_price': price,
                'quantity': quantity,
                'entry_time': timestamp,
                'cost': position_amount
            }
            
            self.balance -= position_amount
            
            order = {
                'type': 'buy',
                'symbol': symbol,
                'price': price,
                'quantity': quantity,
                'amount': position_amount,
                'timestamp': timestamp
            }
            
            logger.info(f"BUY {quantity} {symbol} at ${price} (${position_amount:.2f})")
        
        # Execute sell order
        elif signal == -1:
            position = self.positions[symbol]
            sell_amount = position['quantity'] * price
            profit = sell_amount - position['cost']
            profit_percent = (profit / position['cost']) * 100
            
            self.balance += sell_amount
            
            order = {
                'type': 'sell',
                'symbol': symbol,
                'price': price,
                'quantity': position['quantity'],
                'amount': sell_amount,
                'profit': profit,
                'profit_percent': profit_percent,
                'entry_price': position['entry_price'],
                'entry_time': position['entry_time'],
                'exit_time': timestamp
            }
            
            logger.info(f"SELL {position['quantity']} {symbol} at ${price} (${sell_amount:.2f}, profit: ${profit:.2f}, {profit_percent:.2f}%)")
            
            del self.positions[symbol]
        
        self.trade_history.append(order)
        return order
    
    def get_account_summary(self):
        """
        Get a summary of the account status.
        
        Returns:
            dict: Account summary
        """
        # Calculate unrealized profit/loss
        unrealized_pnl = 0
        for symbol, position in self.positions.items():
            # In a real implementation, we would get the current price from the market
            # For now, we'll use the entry price as a placeholder
            current_price = position['entry_price']  # Placeholder
            position_value = position['quantity'] * current_price
            unrealized_pnl += position_value - position['cost']
        
        # Calculate realized profit/loss
        realized_pnl = sum(trade.get('profit', 0) for trade in self.trade_history if trade['type'] == 'sell')
        
        # Calculate total profit/loss
        total_pnl = realized_pnl + unrealized_pnl
        
        # Calculate win rate
        winning_trades = sum(1 for trade in self.trade_history if trade['type'] == 'sell' and trade.get('profit', 0) > 0)
        total_closed_trades = sum(1 for trade in self.trade_history if trade['type'] == 'sell')
        win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0
        
        return {
            'balance': self.balance,
            'open_positions': len(self.positions),
            'realized_pnl': realized_pnl,
            'unrealized_pnl': unrealized_pnl,
            'total_pnl': total_pnl,
            'win_rate': win_rate,
            'total_trades': len(self.trade_history),
            'closed_trades': total_closed_trades
        } 