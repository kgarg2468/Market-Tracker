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
        # Calculate date range - extend to account for weekends/holidays
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 2)  # Double the range to account for non-trading days
        
        # Download data from Yahoo Finance using explicit date range
        df = yf.download(
            ticker, 
            start=start_date.strftime('%Y-%m-%d'), 
            end=end_date.strftime('%Y-%m-%d'), 
            interval='1d',
            progress=False
        )
        
        if df.empty:
            st.error(f"No data available for ticker '{ticker}'. Please verify the ticker symbol.")
            return None
        
        # Handle MultiIndex columns (flatten if necessary)
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)
        
        # Ensure column names are standardized
        df.columns = [col.title() for col in df.columns]
        
        # Clean the data
        df = df.dropna()
        
        # Get the most recent trading days up to the requested amount
        df = df.tail(days) if len(df) > days else df
        
        # Check if we have sufficient data for technical indicators
        if len(df) < days:
            st.warning(f"Only {len(df)} trading days of data found for {ticker} (requested {days} days). Some indicators may be less reliable due to weekends, holidays, or limited trading history.")
        elif len(df) < 50:
            st.warning(f"Limited data available for {ticker}. Only {len(df)} days of data found. Some technical indicators may be less reliable with fewer than 50 data points.")
        
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
