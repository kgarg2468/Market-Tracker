import pandas as pd
import numpy as np
from utils.indicators import get_indicator_signals, calculate_trend_strength

def generate_trading_signals(df):
    """
    Generate comprehensive trading signals based on technical indicators
    
    Args:
        df (pandas.DataFrame): DataFrame with stock data and technical indicators
    
    Returns:
        dict: Trading signals and analysis
    """
    # Get individual indicator signals
    indicator_signals = get_indicator_signals(df)
    
    # Get trend analysis
    trend_analysis = calculate_trend_strength(df)
    
    # Calculate signal strength and primary signal
    signal_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    # Weight the indicators (can be adjusted based on strategy)
    weights = {
        'RSI': 2.0,        # RSI gets higher weight for momentum
        'MACD': 2.5,       # MACD gets highest weight for trend confirmation
        'SMA': 2.0,        # Moving averages for trend direction
        'BB': 1.5,         # Bollinger Bands for volatility
        'Stochastic': 1.0  # Stochastic as supporting indicator
    }
    
    # Calculate weighted scores
    for indicator, signal in indicator_signals.items():
        weight = weights.get(indicator, 1.0)
        signal_scores[signal] += weight
    
    # Determine primary signal
    primary_signal = max(signal_scores, key=signal_scores.get)
    
    # Calculate signal strength (1-5 scale)
    max_score = signal_scores[primary_signal]
    total_possible_score = sum(weights.values())
    signal_strength = min(5, max(1, int((max_score / total_possible_score) * 5)))
    
    # Generate reasoning
    reasoning = generate_signal_reasoning(df, indicator_signals, trend_analysis)
    
    # Additional risk assessment
    risk_assessment = assess_risk(df)
    
    # Create comprehensive signal output
    signals = {
        'primary_signal': primary_signal,
        'signal_strength': signal_strength,
        'indicator_signals': indicator_signals,
        'signal_scores': signal_scores,
        'reasoning': reasoning,
        'trend_analysis': trend_analysis,
        'risk_assessment': risk_assessment,
        'confidence_level': calculate_confidence_level(signal_scores, signal_strength)
    }
    
    return signals

def generate_signal_reasoning(df, indicator_signals, trend_analysis):
    """
    Generate human-readable reasoning for the trading signals
    
    Args:
        df (pandas.DataFrame): DataFrame with stock data and indicators
        indicator_signals (dict): Individual indicator signals
        trend_analysis (dict): Trend strength analysis
    
    Returns:
        list: List of reasoning statements
    """
    latest = df.iloc[-1]
    reasoning = []
    
    # RSI reasoning
    rsi_value = latest['RSI']
    if rsi_value < 30:
        reasoning.append(f"RSI is oversold at {rsi_value:.1f}, indicating potential buying opportunity")
    elif rsi_value > 70:
        reasoning.append(f"RSI is overbought at {rsi_value:.1f}, suggesting potential selling pressure")
    else:
        reasoning.append(f"RSI is neutral at {rsi_value:.1f}, no strong momentum signal")
    
    # MACD reasoning
    macd_signal = indicator_signals['MACD']
    if macd_signal == 'BUY':
        reasoning.append("MACD line crossed above signal line, indicating bullish momentum")
    elif macd_signal == 'SELL':
        reasoning.append("MACD line crossed below signal line, indicating bearish momentum")
    else:
        reasoning.append("MACD shows neutral momentum with no clear directional bias")
    
    # Moving Average reasoning
    price = latest['Close']
    sma_20 = latest['SMA_20']
    sma_50 = latest['SMA_50']
    
    if price > sma_20 > sma_50:
        reasoning.append("Price is above both 20-day and 50-day moving averages, confirming uptrend")
    elif price < sma_20 < sma_50:
        reasoning.append("Price is below both moving averages, confirming downtrend")
    else:
        reasoning.append("Price relationship with moving averages suggests consolidation")
    
    # Trend reasoning
    trend_dir = trend_analysis['direction']
    trend_str = trend_analysis['strength']
    price_change = trend_analysis['price_change_pct']
    
    reasoning.append(f"Recent trend shows {trend_str.lower()} {trend_dir.lower()} movement ({price_change:+.1f}%)")
    
    # Volume analysis if available
    if 'Volume' in df.columns and 'Volume_SMA' in df.columns:
        volume_ratio = latest['Volume'] / latest['Volume_SMA']
        if volume_ratio > 1.5:
            reasoning.append("Above-average volume supports the current price movement")
        elif volume_ratio < 0.7:
            reasoning.append("Below-average volume suggests weak conviction in current move")
    
    return reasoning

def assess_risk(df):
    """
    Assess the risk level of the current trade setup
    
    Args:
        df (pandas.DataFrame): DataFrame with stock data and indicators
    
    Returns:
        dict: Risk assessment
    """
    latest = df.iloc[-1]
    
    # Calculate recent volatility
    recent_returns = df['Close'].pct_change().tail(20).dropna()
    volatility = recent_returns.std() * np.sqrt(252)  # Annualized volatility
    
    # ATR-based risk assessment
    atr = latest['ATR'] if 'ATR' in df.columns else 0
    price = latest['Close']
    atr_percentage = (atr / price) * 100 if price > 0 else 0
    
    # Risk level determination
    if volatility > 0.4 or atr_percentage > 3:
        risk_level = 'HIGH'
    elif volatility > 0.25 or atr_percentage > 2:
        risk_level = 'MEDIUM'
    else:
        risk_level = 'LOW'
    
    # Support and resistance levels
    recent_high = df['High'].tail(20).max()
    recent_low = df['Low'].tail(20).min()
    current_position = (price - recent_low) / (recent_high - recent_low) if recent_high > recent_low else 0.5
    
    return {
        'risk_level': risk_level,
        'volatility': volatility,
        'atr_percentage': atr_percentage,
        'support_level': recent_low,
        'resistance_level': recent_high,
        'position_in_range': current_position
    }

def calculate_confidence_level(signal_scores, signal_strength):
    """
    Calculate confidence level in the trading signal
    
    Args:
        signal_scores (dict): Signal scores for each action
        signal_strength (int): Signal strength (1-5)
    
    Returns:
        str: Confidence level (LOW, MEDIUM, HIGH)
    """
    # Calculate the difference between top two signals
    sorted_scores = sorted(signal_scores.values(), reverse=True)
    if len(sorted_scores) >= 2:
        score_difference = sorted_scores[0] - sorted_scores[1]
    else:
        score_difference = sorted_scores[0] if sorted_scores else 0
    
    # Determine confidence based on signal strength and score difference
    if signal_strength >= 4 and score_difference >= 2:
        return 'HIGH'
    elif signal_strength >= 3 and score_difference >= 1:
        return 'MEDIUM'
    else:
        return 'LOW'

def get_entry_exit_levels(df, signal):
    """
    Calculate suggested entry and exit levels based on technical analysis
    
    Args:
        df (pandas.DataFrame): DataFrame with stock data and indicators
        signal (str): Primary trading signal (BUY/SELL/HOLD)
    
    Returns:
        dict: Entry and exit level suggestions
    """
    latest = df.iloc[-1]
    current_price = latest['Close']
    atr = latest['ATR'] if 'ATR' in df.columns else current_price * 0.02
    
    levels = {}
    
    if signal == 'BUY':
        # Entry: current price or slight pullback
        levels['entry'] = current_price
        levels['stop_loss'] = current_price - (2 * atr)  # 2 ATR below entry
        levels['target_1'] = current_price + (1.5 * atr)  # 1.5:1 risk-reward
        levels['target_2'] = current_price + (3 * atr)    # 3:1 risk-reward
        
    elif signal == 'SELL':
        # Entry: current price or slight bounce
        levels['entry'] = current_price
        levels['stop_loss'] = current_price + (2 * atr)   # 2 ATR above entry
        levels['target_1'] = current_price - (1.5 * atr)  # 1.5:1 risk-reward
        levels['target_2'] = current_price - (3 * atr)    # 3:1 risk-reward
    
    else:  # HOLD
        levels['entry'] = None
        levels['stop_loss'] = None
        levels['target_1'] = None
        levels['target_2'] = None
    
    return levels
