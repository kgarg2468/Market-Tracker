import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import streamlit as st
import os
from typing import Dict, List, Optional, Tuple

class PortfolioManager:
    def __init__(self, portfolio_file='portfolio.json'):
        self.portfolio_file = portfolio_file
        self.portfolio = self.load_portfolio()
        
    def load_portfolio(self) -> List[Dict]:
        """Load portfolio from JSON file"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def save_portfolio(self):
        """Save portfolio to JSON file"""
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2, default=str)
    
    def add_position(self, ticker: str, buy_price: float, buy_date: str, shares: float, 
                    target_return: float = None, stop_loss: float = None, hold_days: int = None):
        """Add a new position to the portfolio"""
        position = {
            'ticker': ticker.upper(),
            'buy_price': float(buy_price),
            'buy_date': buy_date,
            'shares': float(shares),
            'target_return': target_return,  # Target % return (e.g., 6.0 for 6%)
            'stop_loss': stop_loss,  # Stop loss % (e.g., -5.0 for -5%)
            'hold_days': hold_days,  # Maximum hold days
            'status': 'ACTIVE',
            'created_at': datetime.now().isoformat()
        }
        
        # Check if position already exists
        for i, pos in enumerate(self.portfolio):
            if pos['ticker'] == ticker and pos['buy_date'] == buy_date:
                # Update existing position
                self.portfolio[i] = position
                self.save_portfolio()
                return
        
        # Add new position
        self.portfolio.append(position)
        self.save_portfolio()
    
    def remove_position(self, ticker: str, buy_date: str):
        """Remove a position from the portfolio"""
        self.portfolio = [pos for pos in self.portfolio 
                         if not (pos['ticker'] == ticker and pos['buy_date'] == buy_date)]
        self.save_portfolio()
    
    def update_position_status(self, ticker: str, buy_date: str, status: str, sell_reason: str = None):
        """Update position status (ACTIVE, SOLD, etc.)"""
        for pos in self.portfolio:
            if pos['ticker'] == ticker and pos['buy_date'] == buy_date:
                pos['status'] = status
                if sell_reason:
                    pos['sell_reason'] = sell_reason
                    pos['sell_date'] = datetime.now().isoformat()
                break
        self.save_portfolio()
    
    def get_current_prices(self) -> Dict[str, float]:
        """Get current prices for all portfolio tickers"""
        tickers = list(set([pos['ticker'] for pos in self.portfolio if pos['status'] == 'ACTIVE']))
        
        if not tickers:
            return {}
        
        try:
            # Download current data for all tickers
            data = yf.download(tickers, period='1d', interval='1d', progress=False)
            
            if len(tickers) == 1:
                # Single ticker returns different structure
                return {tickers[0]: float(data['Close'].iloc[-1])}
            else:
                # Multiple tickers
                current_prices = {}
                for ticker in tickers:
                    try:
                        price = float(data['Close'][ticker].iloc[-1])
                        current_prices[ticker] = price
                    except (KeyError, IndexError):
                        # Try alternative data structure
                        try:
                            price = float(data[('Close', ticker)].iloc[-1])
                            current_prices[ticker] = price
                        except:
                            st.warning(f"Could not fetch current price for {ticker}")
                            continue
                return current_prices
                
        except Exception as e:
            st.error(f"Error fetching current prices: {str(e)}")
            return {}
    
    def calculate_position_metrics(self, position: Dict, current_price: float) -> Dict:
        """Calculate metrics for a single position"""
        buy_price = position['buy_price']
        shares = position['shares']
        buy_date = datetime.strptime(position['buy_date'], '%Y-%m-%d')
        days_held = (datetime.now() - buy_date).days
        
        # Calculate returns
        total_return = (current_price - buy_price) * shares
        percent_return = ((current_price - buy_price) / buy_price) * 100
        
        # Calculate position value
        initial_value = buy_price * shares
        current_value = current_price * shares
        
        return {
            'current_price': current_price,
            'total_return': total_return,
            'percent_return': percent_return,
            'initial_value': initial_value,
            'current_value': current_value,
            'days_held': days_held
        }
    
    def generate_sell_signals(self) -> List[Dict]:
        """Generate sell signals for all active positions"""
        signals = []
        current_prices = self.get_current_prices()
        
        for position in self.portfolio:
            if position['status'] != 'ACTIVE':
                continue
                
            ticker = position['ticker']
            if ticker not in current_prices:
                continue
                
            current_price = current_prices[ticker]
            metrics = self.calculate_position_metrics(position, current_price)
            
            # Check sell conditions
            sell_signal = self.check_sell_conditions(position, metrics)
            
            if sell_signal:
                signals.append({
                    'ticker': ticker,
                    'signal': sell_signal['action'],
                    'reason': sell_signal['reason'],
                    'current_price': current_price,
                    'percent_return': metrics['percent_return'],
                    'days_held': metrics['days_held'],
                    'position': position
                })
        
        return signals
    
    def check_sell_conditions(self, position: Dict, metrics: Dict) -> Optional[Dict]:
        """Check if position meets sell conditions"""
        percent_return = metrics['percent_return']
        days_held = metrics['days_held']
        
        # Check target return
        if position.get('target_return') and percent_return >= position['target_return']:
            return {
                'action': 'SELL',
                'reason': f'Target return reached ({percent_return:.1f}% >= {position["target_return"]:.1f}%)'
            }
        
        # Check stop loss
        if position.get('stop_loss') and percent_return <= position['stop_loss']:
            return {
                'action': 'SELL',
                'reason': f'Stop loss triggered ({percent_return:.1f}% <= {position["stop_loss"]:.1f}%)'
            }
        
        # Check time limit
        if position.get('hold_days') and days_held >= position['hold_days']:
            return {
                'action': 'SELL',
                'reason': f'Time limit reached ({days_held} >= {position["hold_days"]} days)'
            }
        
        # Technical analysis sell signals (basic implementation)
        if self.check_technical_sell_signal(position['ticker'], metrics):
            return {
                'action': 'SELL',
                'reason': 'Technical analysis suggests selling'
            }
        
        return None
    
    def check_technical_sell_signal(self, ticker: str, metrics: Dict) -> bool:
        """Check technical indicators for sell signals"""
        try:
            # Get recent data for technical analysis
            data = yf.download(ticker, period='30d', interval='1d', progress=False)
            
            if data.empty or len(data) < 20:
                return False
            
            # Calculate 20-day moving average
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            
            # Sell if price is below 20-day SMA
            current_price = data['Close'].iloc[-1]
            sma_20 = data['SMA_20'].iloc[-1]
            
            if current_price < sma_20:
                return True
                
            return False
            
        except Exception:
            return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get overall portfolio summary"""
        if not self.portfolio:
            return {
                'total_positions': 0,
                'active_positions': 0,
                'total_invested': 0,
                'current_value': 0,
                'total_return': 0,
                'percent_return': 0
            }
        
        current_prices = self.get_current_prices()
        active_positions = [pos for pos in self.portfolio if pos['status'] == 'ACTIVE']
        
        total_invested = 0
        current_value = 0
        total_return = 0
        
        for position in active_positions:
            ticker = position['ticker']
            if ticker in current_prices:
                metrics = self.calculate_position_metrics(position, current_prices[ticker])
                total_invested += metrics['initial_value']
                current_value += metrics['current_value']
                total_return += metrics['total_return']
        
        percent_return = ((current_value - total_invested) / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'total_positions': len(self.portfolio),
            'active_positions': len(active_positions),
            'total_invested': total_invested,
            'current_value': current_value,
            'total_return': total_return,
            'percent_return': percent_return
        }
    
    def get_portfolio_details(self) -> List[Dict]:
        """Get detailed information for all positions"""
        current_prices = self.get_current_prices()
        details = []
        
        for position in self.portfolio:
            ticker = position['ticker']
            
            if position['status'] == 'ACTIVE' and ticker in current_prices:
                metrics = self.calculate_position_metrics(position, current_prices[ticker])
                
                details.append({
                    'ticker': ticker,
                    'buy_date': position['buy_date'],
                    'buy_price': position['buy_price'],
                    'shares': position['shares'],
                    'current_price': metrics['current_price'],
                    'days_held': metrics['days_held'],
                    'percent_return': metrics['percent_return'],
                    'total_return': metrics['total_return'],
                    'current_value': metrics['current_value'],
                    'target_return': position.get('target_return'),
                    'stop_loss': position.get('stop_loss'),
                    'hold_days': position.get('hold_days'),
                    'status': position['status']
                })
            else:
                # Inactive positions
                details.append({
                    'ticker': ticker,
                    'buy_date': position['buy_date'],
                    'buy_price': position['buy_price'],
                    'shares': position['shares'],
                    'current_price': None,
                    'days_held': None,
                    'percent_return': None,
                    'total_return': None,
                    'current_value': None,
                    'target_return': position.get('target_return'),
                    'stop_loss': position.get('stop_loss'),
                    'hold_days': position.get('hold_days'),
                    'status': position['status']
                })
        
        return details
    
    def export_portfolio_csv(self) -> str:
        """Export portfolio to CSV format"""
        details = self.get_portfolio_details()
        df = pd.DataFrame(details)
        
        # Format currency columns
        currency_cols = ['buy_price', 'current_price', 'total_return', 'current_value']
        for col in currency_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        # Format percentage columns
        percent_cols = ['percent_return', 'target_return', 'stop_loss']
        for col in percent_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
        
        return df.to_csv(index=False)