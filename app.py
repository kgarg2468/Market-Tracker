import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import io

# Import utility modules
from utils.data_loader import get_stock_data
from utils.indicators import add_technical_indicators
from utils.signals import generate_trading_signals
from utils.report_generator import generate_report, export_to_csv, export_to_json

# Configure page
st.set_page_config(
    page_title="Trading Algorithm Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("📈 Trading Algorithm Dashboard")
st.markdown("Analyze stock market data and get buy/sell/hold recommendations using technical indicators")

# Sidebar for inputs
st.sidebar.header("Configuration")

# Stock ticker input
ticker = st.sidebar.text_input(
    "Stock Ticker", 
    value="AAPL",
    help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
).upper()

# Date range selection
days_back = st.sidebar.selectbox(
    "Analysis Period",
    options=[30, 60, 90, 180, 365],
    index=1,
    help="Number of days of historical data to analyze"
)

# Analysis button
analyze_button = st.sidebar.button("🔍 Analyze Stock", type="primary")

# Initialize session state
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Main analysis logic
if analyze_button or st.session_state.analysis_data is not None:
    if analyze_button:
        # Show loading spinner
        with st.spinner(f"Fetching and analyzing data for {ticker}..."):
            try:
                # Fetch stock data
                df = get_stock_data(ticker, days_back)
                
                if df is None or df.empty:
                    st.error(f"❌ No data found for ticker '{ticker}'. Please check the ticker symbol and try again.")
                    st.session_state.analysis_data = None
                    st.stop()
                
                # Add technical indicators
                df = add_technical_indicators(df)
                
                # Generate trading signals
                signals = generate_trading_signals(df)
                
                # Generate report
                report = generate_report(ticker, df, signals)
                
                # Store in session state
                st.session_state.analysis_data = {
                    'df': df,
                    'signals': signals,
                    'report': report,
                    'ticker': ticker
                }
                
            except Exception as e:
                st.error(f"❌ Error analyzing {ticker}: {str(e)}")
                st.session_state.analysis_data = None
                st.stop()
    
    # Display results if data exists
    if st.session_state.analysis_data is not None:
        data = st.session_state.analysis_data
        df = data['df']
        signals = data['signals']
        report = data['report']
        ticker = data['ticker']
        
        # Main metrics
        st.header(f"📊 Analysis Results for {ticker}")
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest_data = df.iloc[-1]
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${latest_data['Close']:.2f}",
                delta=f"{((latest_data['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100):.2f}%"
            )
        
        with col2:
            rsi_value = latest_data['RSI']
            rsi_color = "🔴" if rsi_value > 70 else "🟢" if rsi_value < 30 else "🟡"
            st.metric(
                label="RSI",
                value=f"{rsi_color} {rsi_value:.1f}",
                help="Relative Strength Index: >70 overbought, <30 oversold"
            )
        
        with col3:
            macd_value = latest_data['MACD']
            macd_signal = latest_data['MACD_Signal']
            macd_diff = macd_value - macd_signal
            st.metric(
                label="MACD",
                value=f"{macd_value:.3f}",
                delta=f"{macd_diff:.3f}",
                help="MACD indicator with signal line difference"
            )
        
        with col4:
            sma_20 = latest_data['SMA_20']
            price_vs_sma20 = ((latest_data['Close'] - sma_20) / sma_20 * 100)
            st.metric(
                label="SMA 20",
                value=f"${sma_20:.2f}",
                delta=f"{price_vs_sma20:.1f}%",
                help="20-day Simple Moving Average"
            )
        
        with col5:
            sma_50 = latest_data['SMA_50']
            price_vs_sma50 = ((latest_data['Close'] - sma_50) / sma_50 * 100)
            st.metric(
                label="SMA 50",
                value=f"${sma_50:.2f}",
                delta=f"{price_vs_sma50:.1f}%",
                help="50-day Simple Moving Average"
            )
        
        # Trading Signal
        st.subheader("🎯 Trading Recommendation")
        
        signal_color = {
            'BUY': '🟢',
            'SELL': '🔴',
            'HOLD': '🟡'
        }
        
        primary_signal = signals['primary_signal']
        signal_strength = signals['signal_strength']
        reasoning = signals['reasoning']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"## {signal_color[primary_signal]} **{primary_signal}**")
            st.markdown(f"**Strength:** {signal_strength}/5")
        
        with col2:
            st.markdown("**Analysis:**")
            for reason in reasoning:
                st.markdown(f"• {reason}")
        
        # Price chart with indicators
        st.subheader("📈 Price Chart with Technical Indicators")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price and Moving Averages', 'RSI', 'MACD'),
            row_width=[0.7, 0.15, 0.15]
        )
        
        # Price and moving averages
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='orange')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='blue')),
            row=1, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=2, col=1
        )
        
        # Add RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='MACD Signal', line=dict(color='red')),
            row=3, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"{ticker} Technical Analysis"
        )
        
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed report
        st.subheader("📋 Detailed Analysis Report")
        
        with st.expander("View Full Report", expanded=False):
            st.json(report)
        
        # Export options
        st.subheader("💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export to JSON
            json_data = export_to_json(report)
            st.download_button(
                label="📄 Download JSON Report",
                data=json_data,
                file_name=f"{ticker}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Export to CSV
            csv_data = export_to_csv(df, report)
            st.download_button(
                label="📊 Download CSV Data",
                data=csv_data,
                file_name=f"{ticker}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This application is for educational and informational purposes only. 
    Trading recommendations are based on technical analysis and should not be considered as financial advice. 
    Always do your own research and consult with a financial advisor before making investment decisions.
    """
)
