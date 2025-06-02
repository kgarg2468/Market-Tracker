import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

def get_stock_data(ticker, days=30):
    """
    Fetch historical stock data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days of historical data to fetch
    
    Returns:
        pandas.DataFrame: Stock data with OHLCV columns
    """
    try:
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if df.empty:
            st.error(f"No data available for ticker '{ticker}'. Please verify the ticker symbol.")
            return None
        
        # Clean the data
        df = df.dropna()
        
        # Ensure we have enough data points for technical indicators
        if len(df) < 50:
            st.warning(f"Limited data available for {ticker}. Only {len(df)} days of data found. Some indicators may be less reliable.")
        
        return df
        
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def validate_ticker(ticker):
    """
    Validate if a ticker symbol exists
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return 'symbol' in info or 'shortName' in info
    except:
        return False

def get_stock_info(ticker):
    """
    Get basic information about a stock
    
    Args:
        ticker (str): Stock ticker symbol
    
    Returns:
        dict: Basic stock information
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            'name': info.get('shortName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'currency': info.get('currency', 'USD')
        }
    except Exception as e:
        return {
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'currency': 'USD'
        }
