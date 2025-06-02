import json
import pandas as pd
from datetime import datetime
import io

def generate_report(ticker, df, signals):
    """
    Generate comprehensive analysis report
    
    Args:
        ticker (str): Stock ticker symbol
        df (pandas.DataFrame): DataFrame with stock data and indicators
        signals (dict): Trading signals and analysis
    
    Returns:
        dict: Comprehensive analysis report
    """
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    
    # Calculate daily change
    daily_change = latest['Close'] - previous['Close']
    daily_change_pct = (daily_change / previous['Close']) * 100 if previous['Close'] > 0 else 0
    
    # Calculate additional metrics
    volume_change = ((latest['Volume'] - df['Volume'].tail(20).mean()) / df['Volume'].tail(20).mean()) * 100
    
    # Create comprehensive report
    report = {
        'metadata': {
            'ticker': ticker,
            'analysis_date': datetime.now().isoformat(),
            'data_period': f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            'total_trading_days': len(df)
        },
        
        'current_market_data': {
            'current_price': round(float(latest['Close']), 2),
            'daily_change': round(float(daily_change), 2),
            'daily_change_percent': round(float(daily_change_pct), 2),
            'volume': int(latest['Volume']),
            'volume_change_percent': round(float(volume_change), 2),
            'high_52_week': round(float(df['High'].max()), 2),
            'low_52_week': round(float(df['Low'].min()), 2)
        },
        
        'technical_indicators': {
            'rsi': {
                'value': round(float(latest['RSI']), 1),
                'interpretation': get_rsi_interpretation(latest['RSI']),
                'signal': signals['indicator_signals']['RSI']
            },
            'macd': {
                'value': round(float(latest['MACD']), 4),
                'signal_line': round(float(latest['MACD_Signal']), 4),
                'histogram': round(float(latest['MACD_Histogram']), 4),
                'interpretation': get_macd_interpretation(latest['MACD'], latest['MACD_Signal']),
                'signal': signals['indicator_signals']['MACD']
            },
            'moving_averages': {
                'sma_20': round(float(latest['SMA_20']), 2),
                'sma_50': round(float(latest['SMA_50']), 2),
                'price_vs_sma20_percent': round(((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100, 2),
                'price_vs_sma50_percent': round(((latest['Close'] - latest['SMA_50']) / latest['SMA_50']) * 100, 2),
                'signal': signals['indicator_signals']['SMA']
            },
            'bollinger_bands': {
                'upper_band': round(float(latest['BB_Upper']), 2),
                'middle_band': round(float(latest['BB_Middle']), 2),
                'lower_band': round(float(latest['BB_Lower']), 2),
                'position': get_bb_position(latest['Close'], latest['BB_Upper'], latest['BB_Lower']),
                'signal': signals['indicator_signals']['BB']
            },
            'stochastic': {
                'k_percent': round(float(latest['Stoch_K']), 1),
                'd_percent': round(float(latest['Stoch_D']), 1),
                'interpretation': get_stochastic_interpretation(latest['Stoch_K']),
                'signal': signals['indicator_signals']['Stochastic']
            }
        },
        
        'trading_recommendation': {
            'primary_signal': signals['primary_signal'],
            'signal_strength': signals['signal_strength'],
            'confidence_level': signals['confidence_level'],
            'reasoning': signals['reasoning']
        },
        
        'trend_analysis': {
            'direction': signals['trend_analysis']['direction'],
            'strength': signals['trend_analysis']['strength'],
            'price_change_percent': round(signals['trend_analysis']['price_change_pct'], 2),
            'volatility': round(signals['trend_analysis']['volatility'], 4)
        },
        
        'risk_assessment': {
            'risk_level': signals['risk_assessment']['risk_level'],
            'volatility_annualized': round(signals['risk_assessment']['volatility'], 4),
            'atr_percentage': round(signals['risk_assessment']['atr_percentage'], 2),
            'support_level': round(signals['risk_assessment']['support_level'], 2),
            'resistance_level': round(signals['risk_assessment']['resistance_level'], 2),
            'position_in_range': round(signals['risk_assessment']['position_in_range'], 2)
        },
        
        'summary_statistics': {
            'avg_volume_20d': int(df['Volume'].tail(20).mean()),
            'price_volatility_20d': round(df['Close'].tail(20).std(), 2),
            'highest_high_20d': round(df['High'].tail(20).max(), 2),
            'lowest_low_20d': round(df['Low'].tail(20).min(), 2),
            'avg_daily_return_20d': round(df['Close'].pct_change().tail(20).mean() * 100, 3)
        }
    }
    
    return report

def get_rsi_interpretation(rsi_value):
    """Get interpretation of RSI value"""
    if rsi_value >= 70:
        return "Overbought - Potential selling pressure"
    elif rsi_value <= 30:
        return "Oversold - Potential buying opportunity"
    elif rsi_value >= 50:
        return "Bullish momentum"
    else:
        return "Bearish momentum"

def get_macd_interpretation(macd, signal):
    """Get interpretation of MACD values"""
    if macd > signal:
        return "Bullish - MACD above signal line"
    else:
        return "Bearish - MACD below signal line"

def get_bb_position(price, upper, lower):
    """Get position within Bollinger Bands"""
    if price > upper:
        return "Above upper band (overbought)"
    elif price < lower:
        return "Below lower band (oversold)"
    else:
        position_pct = ((price - lower) / (upper - lower)) * 100
        return f"Within bands ({position_pct:.1f}% of range)"

def get_stochastic_interpretation(stoch_k):
    """Get interpretation of Stochastic value"""
    if stoch_k >= 80:
        return "Overbought territory"
    elif stoch_k <= 20:
        return "Oversold territory"
    else:
        return "Neutral territory"

def export_to_json(report):
    """
    Export report to JSON format
    
    Args:
        report (dict): Analysis report
    
    Returns:
        str: JSON formatted report
    """
    return json.dumps(report, indent=2, default=str)

def export_to_csv(df, report):
    """
    Export stock data and key metrics to CSV format
    
    Args:
        df (pandas.DataFrame): Stock data with indicators
        report (dict): Analysis report
    
    Returns:
        str: CSV formatted data
    """
    # Create a copy of the dataframe for export
    export_df = df.copy()
    
    # Add summary row with key metrics
    summary_data = {
        'Date': 'SUMMARY',
        'Open': report['trading_recommendation']['primary_signal'],
        'High': report['trading_recommendation']['signal_strength'],
        'Low': report['risk_assessment']['risk_level'],
        'Close': report['current_market_data']['current_price'],
        'Volume': report['current_market_data']['volume'],
        'RSI': report['technical_indicators']['rsi']['value'],
        'MACD': report['technical_indicators']['macd']['value'],
        'MACD_Signal': report['technical_indicators']['macd']['signal_line'],
        'SMA_20': report['technical_indicators']['moving_averages']['sma_20'],
        'SMA_50': report['technical_indicators']['moving_averages']['sma_50']
    }
    
    # Convert to CSV
    output = io.StringIO()
    export_df.to_csv(output, index=True)
    
    # Add summary information as comments
    csv_content = output.getvalue()
    
    # Add header with analysis summary
    header = f"""# Trading Analysis Report
# Ticker: {report['metadata']['ticker']}
# Analysis Date: {report['metadata']['analysis_date']}
# Recommendation: {report['trading_recommendation']['primary_signal']}
# Signal Strength: {report['trading_recommendation']['signal_strength']}/5
# Confidence: {report['trading_recommendation']['confidence_level']}
# Risk Level: {report['risk_assessment']['risk_level']}
#
"""
    
    return header + csv_content

def format_currency(value):
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value):
    """Format value as percentage"""
    return f"{value:+.2f}%"

def create_summary_text(report):
    """
    Create a text summary of the analysis
    
    Args:
        report (dict): Analysis report
    
    Returns:
        str: Formatted text summary
    """
    ticker = report['metadata']['ticker']
    price = report['current_market_data']['current_price']
    change_pct = report['current_market_data']['daily_change_percent']
    signal = report['trading_recommendation']['primary_signal']
    strength = report['trading_recommendation']['signal_strength']
    confidence = report['trading_recommendation']['confidence_level']
    
    summary = f"""
TRADING ANALYSIS SUMMARY
========================

Stock: {ticker}
Current Price: {format_currency(price)} ({format_percentage(change_pct)})

RECOMMENDATION: {signal}
Signal Strength: {strength}/5
Confidence Level: {confidence}

Key Technical Indicators:
• RSI: {report['technical_indicators']['rsi']['value']} ({report['technical_indicators']['rsi']['interpretation']})
• MACD: {report['technical_indicators']['macd']['value']} vs Signal {report['technical_indicators']['macd']['signal_line']}
• Price vs SMA20: {format_percentage(report['technical_indicators']['moving_averages']['price_vs_sma20_percent'])}
• Risk Level: {report['risk_assessment']['risk_level']}

Primary Reasoning:
"""
    
    for reason in report['trading_recommendation']['reasoning']:
        summary += f"• {reason}\n"
    
    summary += f"""
Support Level: {format_currency(report['risk_assessment']['support_level'])}
Resistance Level: {format_currency(report['risk_assessment']['resistance_level'])}

Analysis generated on: {report['metadata']['analysis_date']}
"""
    
    return summary
