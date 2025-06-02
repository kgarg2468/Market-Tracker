import pandas as pd
import numpy as np
import ta

def add_technical_indicators(df):
    """
    Add technical indicators to the stock data DataFrame
    
    Args:
        df (pandas.DataFrame): Stock data with OHLCV columns
    
    Returns:
        pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD (Moving Average Convergence Divergence)
    macd_indicator = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd_indicator.macd()
    df['MACD_Signal'] = macd_indicator.macd_signal()
    df['MACD_Histogram'] = macd_indicator.macd_diff()
    
    # Simple Moving Averages
    df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
    
    # Exponential Moving Averages
    df['EMA_12'] = ta.trend.EMAIndicator(close=df['Close'], window=12).ema_indicator()
    df['EMA_26'] = ta.trend.EMAIndicator(close=df['Close'], window=26).ema_indicator()
    
    # Bollinger Bands
    bb_indicator = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_Upper'] = bb_indicator.bollinger_hband()
    df['BB_Lower'] = bb_indicator.bollinger_lband()
    df['BB_Middle'] = bb_indicator.bollinger_mavg()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    
    # Stochastic Oscillator
    stoch_indicator = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
    df['Stoch_K'] = stoch_indicator.stoch()
    df['Stoch_D'] = stoch_indicator.stoch_signal()
    
    # Average True Range (ATR) for volatility
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
    
    # Williams %R
    df['Williams_R'] = ta.momentum.WilliamsRIndicator(high=df['High'], low=df['Low'], close=df['Close']).williams_r()
    
    # Price Rate of Change
    df['ROC'] = ta.momentum.ROCIndicator(close=df['Close'], window=10).roc()
    
    # Remove any rows with NaN values that might result from indicator calculations
    df = df.dropna()
    
    return df

def get_indicator_signals(df):
    """
    Generate individual signals from technical indicators
    
    Args:
        df (pandas.DataFrame): DataFrame with technical indicators
    
    Returns:
        dict: Dictionary containing individual indicator signals
    """
    latest = df.iloc[-1]
    signals = {}
    
    # RSI signals
    if latest['RSI'] < 30:
        signals['RSI'] = 'BUY'  # Oversold
    elif latest['RSI'] > 70:
        signals['RSI'] = 'SELL'  # Overbought
    else:
        signals['RSI'] = 'HOLD'
    
    # MACD signals
    if latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Histogram'] > 0:
        signals['MACD'] = 'BUY'
    elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Histogram'] < 0:
        signals['MACD'] = 'SELL'
    else:
        signals['MACD'] = 'HOLD'
    
    # Moving Average signals
    if latest['Close'] > latest['SMA_20'] > latest['SMA_50']:
        signals['SMA'] = 'BUY'  # Price above both SMAs, bullish trend
    elif latest['Close'] < latest['SMA_20'] < latest['SMA_50']:
        signals['SMA'] = 'SELL'  # Price below both SMAs, bearish trend
    else:
        signals['SMA'] = 'HOLD'
    
    # Bollinger Bands signals
    if latest['Close'] < latest['BB_Lower']:
        signals['BB'] = 'BUY'  # Price below lower band, potentially oversold
    elif latest['Close'] > latest['BB_Upper']:
        signals['BB'] = 'SELL'  # Price above upper band, potentially overbought
    else:
        signals['BB'] = 'HOLD'
    
    # Stochastic signals
    if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
        signals['Stochastic'] = 'BUY'  # Oversold
    elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
        signals['Stochastic'] = 'SELL'  # Overbought
    else:
        signals['Stochastic'] = 'HOLD'
    
    return signals

def calculate_trend_strength(df, window=10):
    """
    Calculate trend strength based on recent price movements
    
    Args:
        df (pandas.DataFrame): DataFrame with stock data
        window (int): Number of periods to analyze
    
    Returns:
        dict: Trend strength analysis
    """
    recent_data = df.tail(window)
    
    # Calculate price change over the window
    price_change = (recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[0]) / recent_data['Close'].iloc[0]
    
    # Calculate volatility (standard deviation of returns)
    returns = recent_data['Close'].pct_change().dropna()
    volatility = returns.std()
    
    # Determine trend direction and strength
    if abs(price_change) < 0.02:  # Less than 2% change
        trend_direction = 'SIDEWAYS'
        trend_strength = 'WEAK'
    elif price_change > 0:
        trend_direction = 'UPWARD'
        trend_strength = 'STRONG' if price_change > 0.05 else 'MODERATE'
    else:
        trend_direction = 'DOWNWARD'
        trend_strength = 'STRONG' if price_change < -0.05 else 'MODERATE'
    
    return {
        'direction': trend_direction,
        'strength': trend_strength,
        'price_change_pct': price_change * 100,
        'volatility': volatility
    }
