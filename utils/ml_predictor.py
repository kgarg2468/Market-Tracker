import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta
import streamlit as st
from .indicators import add_technical_indicators

class StockMLPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        
    def get_sp500_tickers(self):
        """Get S&P 500 ticker symbols"""
        # Using a curated list of popular tickers for better reliability
        tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B', 'UNH', 'JNJ',
            'V', 'XOM', 'PG', 'JPM', 'HD', 'CVX', 'MA', 'ABBV', 'PFE', 'AVGO',
            'KO', 'COST', 'PEP', 'TMO', 'WMT', 'BAC', 'MRK', 'CSCO', 'ACN', 'LLY',
            'DHR', 'VZ', 'ABT', 'ADBE', 'CRM', 'TXN', 'CMCSA', 'NKE', 'NFLX', 'AMD',
            'QCOM', 'T', 'NEE', 'UPS', 'RTX', 'HON', 'SPGI', 'SBUX', 'LOW', 'INTU',
            'CAT', 'GS', 'AXP', 'DE', 'BKNG', 'IBM', 'GE', 'MDT', 'TJX', 'BLK',
            'MMC', 'LRCX', 'AMT', 'SYK', 'ISRG', 'CB', 'MU', 'C', 'NOW', 'ZTS',
            'REGN', 'PLD', 'PYPL', 'TGT', 'BMY', 'SO', 'BA', 'SCHW', 'AMAT', 'CVS',
            'LMT', 'ELV', 'GILD', 'FIS', 'MO', 'USB', 'DUK', 'BSX', 'SHW', 'AON',
            'CL', 'EQIX', 'NSC', 'ITW', 'HCA', 'FCX', 'APD', 'CSX', 'MCK', 'PNC'
        ]
        return tickers
    
    def create_features(self, df):
        """Create feature vectors from stock data"""
        if len(df) < 50:  # Need sufficient data for technical indicators
            return None
            
        # Add technical indicators
        df = add_technical_indicators(df)
        
        # Create additional features
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Change'] = df['Volume'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Moving averages ratios
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        df['Price_SMA50_Ratio'] = df['Close'] / df['SMA_50']
        df['SMA20_SMA50_Ratio'] = df['SMA_20'] / df['SMA_50']
        
        # Volatility features
        df['Price_Volatility_5d'] = df['Close'].rolling(5).std()
        df['Volume_Volatility_5d'] = df['Volume'].rolling(5).std()
        
        # Momentum features
        df['Momentum_3d'] = df['Close'] - df['Close'].shift(3)
        df['Momentum_7d'] = df['Close'] - df['Close'].shift(7)
        
        # Feature columns for ML
        feature_cols = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'BB_Upper', 'BB_Lower', 'BB_Middle',
            'Stoch_K', 'Stoch_D', 'ATR', 'Williams_R', 'ROC',
            'Price_Change', 'Volume_Change', 'High_Low_Ratio', 'Close_Open_Ratio',
            'Price_SMA20_Ratio', 'Price_SMA50_Ratio', 'SMA20_SMA50_Ratio',
            'Price_Volatility_5d', 'Volume_Volatility_5d',
            'Momentum_3d', 'Momentum_7d'
        ]
        
        # Clean data
        df = df.dropna()
        
        if len(df) == 0:
            return None
            
        return df[feature_cols]
    
    def create_target(self, df, prediction_days=3):
        """Create target variable: future return"""
        df = df.copy()
        future_prices = df['Close'].shift(-prediction_days)
        current_prices = df['Close']
        target = (future_prices - current_prices) / current_prices * 100
        return target.dropna()
    
    def fetch_training_data(self, tickers, days=90):
        """Fetch and prepare training data for multiple stocks"""
        all_features = []
        all_targets = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            try:
                status_text.text(f"Fetching data for {ticker}... ({i+1}/{len(tickers)})")
                progress_bar.progress((i + 1) / len(tickers))
                
                # Fetch extended data for training
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days * 2)
                
                stock_data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    progress=False
                )
                
                if stock_data.empty or len(stock_data) < 50:
                    continue
                
                # Handle MultiIndex columns
                if stock_data.columns.nlevels > 1:
                    stock_data.columns = stock_data.columns.droplevel(1)
                
                stock_data.columns = [col.title() for col in stock_data.columns]
                
                # Create features
                features = self.create_features(stock_data)
                if features is None or len(features) < 10:
                    continue
                
                # Create target
                targets = self.create_target(stock_data)
                if len(targets) == 0:
                    continue
                
                # Align features and targets
                min_len = min(len(features), len(targets))
                features = features.iloc[:min_len]
                targets = targets.iloc[:min_len]
                
                all_features.append(features)
                all_targets.append(targets)
                
            except Exception as e:
                st.warning(f"Error fetching data for {ticker}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_features:
            raise ValueError("No valid data collected for training")
        
        # Combine all data
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        return X, y
    
    def train_model(self, tickers=None, days=90):
        """Train the ML model"""
        if tickers is None:
            tickers = self.get_sp500_tickers()
        
        st.info(f"Training ML model on {len(tickers)} stocks with {days} days of data...")
        
        # Fetch training data
        X, y = self.fetch_training_data(tickers, days)
        
        # Remove outliers (returns beyond ±50%)
        valid_indices = (y >= -50) & (y <= 50)
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        st.success(f"Model trained successfully!")
        st.info(f"Model Performance - R² Score: {r2:.3f}, MSE: {mse:.3f}")
        
        return {
            'r2_score': r2,
            'mse': mse,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_stock_returns(self, tickers=None):
        """Predict returns for given tickers"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if tickers is None:
            tickers = self.get_sp500_tickers()
        
        predictions = {}
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, ticker in enumerate(tickers):
            try:
                status_text.text(f"Predicting {ticker}... ({i+1}/{len(tickers)})")
                progress_bar.progress((i + 1) / len(tickers))
                
                # Fetch recent data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=120)  # Extended for technical indicators
                
                stock_data = yf.download(
                    ticker,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    interval='1d',
                    progress=False
                )
                
                if stock_data.empty:
                    continue
                
                # Handle MultiIndex columns
                if stock_data.columns.nlevels > 1:
                    stock_data.columns = stock_data.columns.droplevel(1)
                
                stock_data.columns = [col.title() for col in stock_data.columns]
                
                # Create features
                features = self.create_features(stock_data)
                if features is None or len(features) == 0:
                    continue
                
                # Get latest features
                latest_features = features.iloc[-1:][self.feature_columns]
                
                # Scale features
                latest_features_scaled = self.scaler.transform(latest_features)
                
                # Predict
                predicted_return = self.model.predict(latest_features_scaled)[0]
                
                # Get additional info
                current_price = stock_data['Close'].iloc[-1]
                daily_change = ((current_price - stock_data['Close'].iloc[-2]) / stock_data['Close'].iloc[-2]) * 100
                
                predictions[ticker] = {
                    'predicted_return': predicted_return,
                    'current_price': current_price,
                    'daily_change': daily_change,
                    'volume': stock_data['Volume'].iloc[-1]
                }
                
            except Exception as e:
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        return predictions
    
    def get_top_stocks(self, n=50, min_return=0.5):
        """Get top N stocks based on predicted returns"""
        predictions = self.predict_stock_returns()
        
        # Filter by minimum return threshold
        filtered_predictions = {
            ticker: data for ticker, data in predictions.items()
            if data['predicted_return'] >= min_return
        }
        
        # Sort by predicted return
        sorted_predictions = sorted(
            filtered_predictions.items(),
            key=lambda x: x[1]['predicted_return'],
            reverse=True
        )
        
        return sorted_predictions[:n]
    
    def save_model(self, filepath='ml_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath='ml_model.pkl'):
        """Load trained model"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            
            return True
        except:
            return False