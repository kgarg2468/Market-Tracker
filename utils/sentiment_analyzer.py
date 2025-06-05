import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st
from typing import Dict, List, Optional, Tuple
import re
import json

class SentimentAnalyzer:
    def __init__(self, news_api_key: Optional[str] = None):
        self.news_api_key = news_api_key
        self.controversy_keywords = [
            'boycott', 'probe', 'investigation', 'sec', 'fraud', 'lawsuit', 
            'scandal', 'fine', 'penalty', 'violation', 'controversy', 
            'criminal', 'illegal', 'banned', 'suspended'
        ]
        self.positive_keywords = [
            'record revenue', 'partnership', 'approval', 'breakthrough', 
            'expansion', 'growth', 'profit', 'success', 'innovation',
            'acquisition', 'merger', 'deal', 'contract', 'earnings beat'
        ]
        self.negative_keywords = [
            'loss', 'decline', 'drop', 'fall', 'crash', 'plunge',
            'disappointing', 'missed', 'warning', 'concern', 'risk'
        ]
    
    def get_news_headlines(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """Get news headlines for a ticker using NewsAPI or alternative sources"""
        headlines = []
        
        # Try NewsAPI first
        if self.news_api_key:
            headlines.extend(self._get_newsapi_headlines(ticker, days_back))
        
        # Fallback to Yahoo Finance news
        headlines.extend(self._get_yahoo_news(ticker))
        
        # Remove duplicates
        seen_titles = set()
        unique_headlines = []
        for headline in headlines:
            if headline['title'] not in seen_titles:
                unique_headlines.append(headline)
                seen_titles.add(headline['title'])
        
        return unique_headlines[:20]  # Limit to 20 most recent
    
    def _get_newsapi_headlines(self, ticker: str, days_back: int) -> List[Dict]:
        """Get headlines from NewsAPI"""
        try:
            company_names = self._get_company_name(ticker)
            search_query = f"{ticker} OR {company_names}"
            
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': search_query,
                'from': from_date,
                'sortBy': 'publishedAt',
                'language': 'en',
                'apiKey': self.news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                headlines = []
                
                for article in data.get('articles', []):
                    headlines.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'published': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', 'NewsAPI')
                    })
                
                return headlines
            
        except Exception as e:
            st.warning(f"NewsAPI error: {str(e)}")
        
        return []
    
    def _get_yahoo_news(self, ticker: str) -> List[Dict]:
        """Get news from Yahoo Finance as fallback"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            headlines = []
            for article in news[:10]:  # Limit to 10 articles
                headlines.append({
                    'title': article.get('title', ''),
                    'description': article.get('summary', ''),
                    'published': datetime.fromtimestamp(article.get('providerPublishTime', 0)).isoformat(),
                    'source': article.get('publisher', 'Yahoo Finance')
                })
            
            return headlines
            
        except Exception as e:
            st.warning(f"Yahoo Finance news error: {str(e)}")
            return []
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name for better news search"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('longName', ticker)
        except:
            return ticker
    
    def analyze_sentiment(self, headlines: List[Dict]) -> Dict:
        """Analyze sentiment of headlines"""
        if not headlines:
            return {
                'overall_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'controversy_score': 0.0,
                'sentiment_summary': 'No news data available'
            }
        
        positive_score = 0
        negative_score = 0
        controversy_score = 0
        
        for headline in headlines:
            text = f"{headline['title']} {headline['description']}".lower()
            
            # Count positive keywords
            positive_matches = sum(1 for keyword in self.positive_keywords if keyword in text)
            positive_score += positive_matches
            
            # Count negative keywords
            negative_matches = sum(1 for keyword in self.negative_keywords if keyword in text)
            negative_score += negative_matches
            
            # Count controversy keywords
            controversy_matches = sum(1 for keyword in self.controversy_keywords if keyword in text)
            controversy_score += controversy_matches
        
        total_headlines = len(headlines)
        
        # Calculate overall sentiment (-1 to 1 scale)
        if positive_score + negative_score > 0:
            overall_sentiment = (positive_score - negative_score) / (positive_score + negative_score)
        else:
            overall_sentiment = 0.0
        
        # Calculate controversy score (0 to 100 scale)
        controversy_percentage = (controversy_score / total_headlines) * 100
        
        # Categorize sentiment
        if overall_sentiment > 0.3:
            sentiment_summary = "Positive"
        elif overall_sentiment < -0.3:
            sentiment_summary = "Negative"
        else:
            sentiment_summary = "Neutral"
        
        return {
            'overall_sentiment': overall_sentiment,
            'positive_count': positive_score,
            'negative_count': negative_score,
            'neutral_count': total_headlines - positive_score - negative_score,
            'controversy_score': controversy_percentage,
            'sentiment_summary': sentiment_summary,
            'total_headlines': total_headlines
        }
    
    def predict_recovery_potential(self, ticker: str, current_price: float, days_back: int = 30) -> Dict:
        """Predict if a stock is likely to recover from a drop"""
        try:
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=f"{days_back}d", interval="1d")
            
            if hist.empty or len(hist) < 20:
                return {'recovery_probability': 0.5, 'confidence': 'Low', 'reason': 'Insufficient data'}
            
            # Calculate technical indicators
            hist['RSI'] = self._calculate_rsi(hist['Close'])
            hist['SMA_20'] = hist['Close'].rolling(20).mean()
            hist['Volume_SMA'] = hist['Volume'].rolling(20).mean()
            
            latest = hist.iloc[-1]
            
            # Recovery factors
            recovery_factors = []
            recovery_score = 0
            
            # RSI oversold condition
            rsi = latest['RSI']
            if rsi < 30:
                recovery_factors.append(f"RSI oversold at {rsi:.1f}")
                recovery_score += 0.3
            elif rsi < 40:
                recovery_factors.append(f"RSI approaching oversold at {rsi:.1f}")
                recovery_score += 0.15
            
            # Price vs moving average
            sma_20 = latest['SMA_20']
            price_vs_sma = ((current_price - sma_20) / sma_20) * 100
            if price_vs_sma < -10:
                recovery_factors.append(f"Price {price_vs_sma:.1f}% below 20-day average")
                recovery_score += 0.2
            elif price_vs_sma < -5:
                recovery_factors.append(f"Price {price_vs_sma:.1f}% below 20-day average")
                recovery_score += 0.1
            
            # Volume analysis
            volume_ratio = latest['Volume'] / latest['Volume_SMA']
            if volume_ratio > 1.5:
                recovery_factors.append(f"High volume ({volume_ratio:.1f}x average) suggests capitulation")
                recovery_score += 0.2
            
            # Support level analysis
            recent_low = hist['Low'].tail(20).min()
            if current_price <= recent_low * 1.02:  # Within 2% of recent low
                recovery_factors.append("Near recent support level")
                recovery_score += 0.15
            
            # Historical recovery analysis
            recovery_patterns = self._analyze_historical_recoveries(hist)
            if recovery_patterns['success_rate'] > 0.6:
                recovery_factors.append(f"Historical recovery rate: {recovery_patterns['success_rate']:.1%}")
                recovery_score += 0.15
            
            # Determine confidence level
            if recovery_score > 0.7:
                confidence = 'High'
            elif recovery_score > 0.4:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            return {
                'recovery_probability': min(recovery_score, 1.0),
                'confidence': confidence,
                'factors': recovery_factors,
                'rsi': rsi,
                'price_vs_sma': price_vs_sma,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            return {
                'recovery_probability': 0.5, 
                'confidence': 'Low', 
                'reason': f'Analysis error: {str(e)}'
            }
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _analyze_historical_recoveries(self, hist: pd.DataFrame) -> Dict:
        """Analyze historical recovery patterns"""
        try:
            # Find periods where stock dropped significantly and then recovered
            hist['pct_change'] = hist['Close'].pct_change()
            hist['rolling_low'] = hist['Low'].rolling(5).min()
            hist['recovery'] = hist['Close'] > hist['rolling_low'] * 1.05  # 5% recovery
            
            drops = hist[hist['pct_change'] < -0.05]  # 5% drops
            
            if len(drops) == 0:
                return {'success_rate': 0.5, 'avg_recovery_days': 0}
            
            recoveries = 0
            total_drops = len(drops)
            
            for idx in drops.index:
                # Look for recovery in next 10 days
                future_data = hist.loc[idx:idx + pd.Timedelta(days=10)]
                if len(future_data) > 1:
                    if future_data['Close'].max() > hist.loc[idx]['Close'] * 1.03:  # 3% recovery
                        recoveries += 1
            
            success_rate = recoveries / total_drops if total_drops > 0 else 0.5
            
            return {
                'success_rate': success_rate,
                'total_drops': total_drops,
                'recoveries': recoveries
            }
            
        except:
            return {'success_rate': 0.5, 'avg_recovery_days': 0}
    
    def generate_enhanced_recommendation(self, ticker: str, technical_signal: str, 
                                       current_price: float, price_change_pct: float) -> Dict:
        """Generate recommendation combining technical analysis, sentiment, and recovery potential"""
        
        # Get news and sentiment
        headlines = self.get_news_headlines(ticker)
        sentiment = self.analyze_sentiment(headlines)
        
        # Get recovery potential if price dropped significantly
        recovery_analysis = None
        if price_change_pct < -3:  # If stock dropped more than 3%
            recovery_analysis = self.predict_recovery_potential(ticker, current_price)
        
        # Combine all factors for final recommendation
        recommendation = self._combine_factors(
            technical_signal, sentiment, recovery_analysis, price_change_pct
        )
        
        return {
            'final_recommendation': recommendation['action'],
            'confidence_level': recommendation['confidence'],
            'reasoning': recommendation['reasoning'],
            'sentiment_analysis': sentiment,
            'recovery_analysis': recovery_analysis,
            'risk_factors': self._identify_risk_factors(sentiment, recovery_analysis)
        }
    
    def _combine_factors(self, technical_signal: str, sentiment: Dict, 
                        recovery_analysis: Optional[Dict], price_change_pct: float) -> Dict:
        """Combine all factors for final recommendation"""
        
        reasoning = []
        confidence_score = 0.5
        
        # Technical signal weight
        if technical_signal == 'BUY':
            confidence_score += 0.2
            reasoning.append(f"Technical analysis suggests {technical_signal}")
        elif technical_signal == 'SELL':
            confidence_score -= 0.2
            reasoning.append(f"Technical analysis suggests {technical_signal}")
        else:
            reasoning.append(f"Technical analysis suggests {technical_signal}")
        
        # Sentiment weight
        if sentiment['controversy_score'] > 50:
            confidence_score -= 0.3
            reasoning.append(f"High controversy detected ({sentiment['controversy_score']:.1f}%)")
        elif sentiment['overall_sentiment'] > 0.3:
            confidence_score += 0.2
            reasoning.append(f"Positive news sentiment ({sentiment['sentiment_summary']})")
        elif sentiment['overall_sentiment'] < -0.3:
            confidence_score -= 0.2
            reasoning.append(f"Negative news sentiment ({sentiment['sentiment_summary']})")
        
        # Recovery analysis weight
        if recovery_analysis and price_change_pct < -3:
            if recovery_analysis['recovery_probability'] > 0.7:
                confidence_score += 0.25
                reasoning.append("Strong recovery potential detected")
            elif recovery_analysis['recovery_probability'] < 0.3:
                confidence_score -= 0.25
                reasoning.append("Low recovery probability")
        
        # Determine final action
        if confidence_score > 0.7:
            action = 'STRONG BUY'
            confidence = 'High'
        elif confidence_score > 0.55:
            action = 'BUY'
            confidence = 'Medium'
        elif confidence_score > 0.45:
            action = 'HOLD'
            confidence = 'Medium'
        elif confidence_score > 0.3:
            action = 'SELL'
            confidence = 'Medium'
        else:
            action = 'STRONG SELL'
            confidence = 'High'
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'confidence_score': confidence_score
        }
    
    def _identify_risk_factors(self, sentiment: Dict, recovery_analysis: Optional[Dict]) -> List[str]:
        """Identify specific risk factors"""
        risks = []
        
        if sentiment['controversy_score'] > 30:
            risks.append(f"Controversy risk: {sentiment['controversy_score']:.1f}% of news mentions controversies")
        
        if sentiment['overall_sentiment'] < -0.5:
            risks.append("Strongly negative news sentiment")
        
        if recovery_analysis and recovery_analysis.get('recovery_probability', 0) < 0.3:
            risks.append("Low probability of price recovery")
        
        if sentiment['total_headlines'] < 3:
            risks.append("Limited news coverage - low visibility stock")
        
        return risks