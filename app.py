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
from utils.ml_predictor import StockMLPredictor
from utils.portfolio_manager import PortfolioManager
from utils.sentiment_analyzer import SentimentAnalyzer

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

# Create tabs for different features
tab1, tab2, tab3 = st.tabs(["Single Stock Analysis", "ML Top 50 Predictions", "Portfolio Tracker"])

# Initialize components in session state
if 'ml_predictor' not in st.session_state:
    st.session_state.ml_predictor = StockMLPredictor()

if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager()

if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer()

with tab1:
    # Create sidebar within tab
    with st.sidebar:
        st.header("Configuration")
        
        # Stock ticker input
        ticker = st.text_input(
            "Stock Ticker", 
            value="AAPL",
            help="Enter a valid stock ticker symbol (e.g., AAPL, GOOGL, MSFT)"
        ).upper()
        
        # Date range selection
        days_back = st.selectbox(
            "Analysis Period",
            options=[30, 60, 90, 180, 365],
            index=1,
            help="Number of days of historical data to analyze"
        )
        
        # Analysis button
        analyze_button = st.button("🔍 Analyze Stock", type="primary")
    
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
            
            # Enhanced Sentiment Analysis Section
            st.subheader("🧠 Enhanced AI Analysis")
            
            with st.spinner("Analyzing news sentiment and market conditions..."):
                try:
                    # Calculate daily price change
                    daily_change = ((latest_data['Close'] - df.iloc[-2]['Close']) / df.iloc[-2]['Close'] * 100) if len(df) > 1 else 0
                    
                    # Generate enhanced recommendation
                    enhanced_analysis = st.session_state.sentiment_analyzer.generate_enhanced_recommendation(
                        ticker=ticker,
                        technical_signal=primary_signal,
                        current_price=latest_data['Close'],
                        price_change_pct=daily_change
                    )
                    
                    # Display enhanced recommendation
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        # Final recommendation with confidence
                        rec_color = {
                            'STRONG BUY': '🟢',
                            'BUY': '🟢', 
                            'HOLD': '🟡',
                            'SELL': '🔴',
                            'STRONG SELL': '🔴'
                        }
                        
                        st.markdown(f"### {rec_color.get(enhanced_analysis['final_recommendation'], '🟡')} **{enhanced_analysis['final_recommendation']}**")
                        st.markdown(f"**AI Confidence:** {enhanced_analysis['confidence_level']}")
                        
                        # Risk factors
                        if enhanced_analysis['risk_factors']:
                            st.markdown("**⚠️ Risk Factors:**")
                            for risk in enhanced_analysis['risk_factors']:
                                st.markdown(f"• {risk}")
                    
                    with col2:
                        # Enhanced reasoning
                        st.markdown("**AI Reasoning:**")
                        for reason in enhanced_analysis['reasoning']:
                            st.markdown(f"• {reason}")
                    
                    # News Sentiment Analysis
                    sentiment = enhanced_analysis['sentiment_analysis']
                    if sentiment['total_headlines'] > 0:
                        st.subheader("📰 News Sentiment Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("News Sentiment", sentiment['sentiment_summary'])
                        
                        with col2:
                            st.metric("Headlines Analyzed", sentiment['total_headlines'])
                        
                        with col3:
                            sentiment_score = sentiment['overall_sentiment']
                            st.metric("Sentiment Score", f"{sentiment_score:+.2f}", 
                                    help="Range: -1.0 (very negative) to +1.0 (very positive)")
                        
                        with col4:
                            controversy = sentiment['controversy_score']
                            color = "normal" if controversy < 30 else "inverse"
                            st.metric("Controversy Level", f"{controversy:.1f}%", 
                                    help="Percentage of news mentioning controversies")
                        
                        # News breakdown
                        if sentiment['positive_count'] + sentiment['negative_count'] > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig = go.Figure(data=go.Bar(
                                    x=['Positive', 'Negative', 'Neutral'],
                                    y=[sentiment['positive_count'], sentiment['negative_count'], sentiment['neutral_count']],
                                    marker_color=['green', 'red', 'gray']
                                ))
                                fig.update_layout(title="News Sentiment Breakdown", yaxis_title="Number of Articles")
                                st.plotly_chart(fig, use_container_width=True)
                    
                    # Recovery Analysis (if price dropped significantly)
                    recovery = enhanced_analysis['recovery_analysis']
                    if recovery:
                        st.subheader("🔄 Recovery Potential Analysis")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            recovery_prob = recovery['recovery_probability']
                            color = "normal" if recovery_prob > 0.6 else "inverse"
                            st.metric("Recovery Probability", f"{recovery_prob:.1%}", 
                                    help="Likelihood of price recovery based on technical indicators")
                            
                            st.metric("Analysis Confidence", recovery['confidence'])
                        
                        with col2:
                            if 'factors' in recovery and recovery['factors']:
                                st.markdown("**Recovery Factors:**")
                                for factor in recovery['factors']:
                                    st.markdown(f"• {factor}")
                            
                            # Technical indicators supporting recovery
                            if 'rsi' in recovery:
                                rsi_val = recovery['rsi']
                                if rsi_val < 30:
                                    st.success(f"RSI Oversold: {rsi_val:.1f} (Strong buy signal)")
                                elif rsi_val < 40:
                                    st.info(f"RSI Low: {rsi_val:.1f} (Potential buy signal)")
                
                except Exception as e:
                    st.warning("Enhanced analysis temporarily unavailable. Using standard technical analysis.")
                    st.info("For full sentiment analysis including news and controversy detection, please provide a NewsAPI key in the settings.")
            
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

with tab2:
    st.header("🤖 ML-Powered Top 50 Stock Predictions")
    st.markdown("Get AI-powered predictions for the top 50 stocks with highest expected returns")
    
    # ML Controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**Model Status:**")
        if st.session_state.ml_predictor.is_trained:
            st.success("✅ Model is trained and ready for predictions")
        else:
            st.warning("⚠️ Model needs to be trained before making predictions")
    
    with col2:
        train_button = st.button("🎯 Train Model", type="primary", disabled=st.session_state.ml_predictor.is_trained)
    
    with col3:
        predict_button = st.button("🔮 Get Predictions", disabled=not st.session_state.ml_predictor.is_trained)
    
    # Model Training
    if train_button:
        with st.spinner("Training ML model on historical data..."):
            try:
                performance = st.session_state.ml_predictor.train_model()
                st.success("Model training completed successfully!")
                
                # Display performance metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("R² Score", f"{performance['r2_score']:.3f}")
                with col2:
                    st.metric("Training Samples", f"{performance['training_samples']:,}")
                    
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
    
    # Predictions
    if predict_button:
        with st.spinner("Generating predictions for top stocks..."):
            try:
                top_stocks = st.session_state.ml_predictor.get_top_stocks(n=50, min_return=0.5)
                
                if not top_stocks:
                    st.warning("No stocks meet the minimum return criteria. Try lowering the threshold.")
                else:
                    st.success(f"Found {len(top_stocks)} stocks with positive predicted returns!")
                    
                    # Display predictions table
                    predictions_data = []
                    for i, (ticker, data) in enumerate(top_stocks, 1):
                        predictions_data.append({
                            'Rank': i,
                            'Ticker': ticker,
                            'Predicted Return (%)': f"{data['predicted_return']:.2f}%",
                            'Current Price ($)': f"${data['current_price']:.2f}",
                            'Daily Change (%)': f"{data['daily_change']:+.2f}%",
                            'Volume': f"{data['volume']:,}"
                        })
                    
                    predictions_df = pd.DataFrame(predictions_data)
                    
                    # Top 10 highlights
                    st.subheader("🏆 Top 10 Predicted Winners")
                    top_10_df = predictions_df.head(10)
                    st.dataframe(top_10_df, use_container_width=True)
                    
                    # Full results
                    st.subheader("📊 All Predictions")
                    st.dataframe(predictions_df, use_container_width=True)
                    
                    # Download predictions
                    csv_predictions = predictions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv_predictions,
                        file_name=f"ml_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Prediction distribution chart
                    st.subheader("📈 Prediction Distribution")
                    returns = [data['predicted_return'] for _, data in top_stocks]
                    
                    fig = go.Figure(data=go.Histogram(x=returns, nbinsx=20))
                    fig.update_layout(
                        title="Distribution of Predicted Returns",
                        xaxis_title="Predicted Return (%)",
                        yaxis_title="Number of Stocks"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
    
    # Model Information
    with st.expander("ℹ️ About the ML Model", expanded=False):
        st.markdown("""
        **Model Details:**
        - **Algorithm:** Random Forest Regressor with 100 trees
        - **Features:** 27 technical indicators including RSI, MACD, Bollinger Bands, moving averages, and momentum indicators
        - **Prediction Target:** 3-day future return percentage
        - **Training Data:** Historical data from 100 popular stocks
        - **Feature Engineering:** Price ratios, volatility measures, and momentum indicators
        
        **How it Works:**
        1. The model analyzes historical patterns in technical indicators
        2. It learns relationships between current market conditions and future price movements
        3. For each stock, it calculates current technical indicators
        4. The model predicts the expected return over the next 3 days
        5. Stocks are ranked by predicted return to identify the best opportunities
        
        **Disclaimer:** Predictions are based on historical patterns and should not be considered as financial advice.
        """)

with tab3:
    st.header("📊 Portfolio Tracker & Sell Alerts")
    st.markdown("Track your positions and get automated sell signals based on targets, stop losses, and technical analysis")
    
    # Portfolio summary at the top
    portfolio_summary = st.session_state.portfolio_manager.get_portfolio_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Positions", 
            portfolio_summary['total_positions'],
            delta=f"Active: {portfolio_summary['active_positions']}"
        )
    
    with col2:
        st.metric(
            "Total Invested", 
            f"${portfolio_summary['total_invested']:,.2f}"
        )
    
    with col3:
        st.metric(
            "Current Value", 
            f"${portfolio_summary['current_value']:,.2f}",
            delta=f"${portfolio_summary['total_return']:+,.2f}"
        )
    
    with col4:
        color = "normal" if portfolio_summary['percent_return'] >= 0 else "inverse"
        st.metric(
            "Total Return", 
            f"{portfolio_summary['percent_return']:+.2f}%",
            delta=None
        )
    
    # Sell Alerts Section
    st.subheader("🚨 Sell Alerts")
    sell_signals = st.session_state.portfolio_manager.generate_sell_signals()
    
    if sell_signals:
        st.warning(f"You have {len(sell_signals)} sell alert(s)!")
        
        for signal in sell_signals:
            with st.expander(f"🔔 {signal['ticker']} - {signal['signal']}", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Reason:** {signal['reason']}")
                    st.write(f"**Current Price:** ${signal['current_price']:.2f}")
                
                with col2:
                    st.write(f"**Return:** {signal['percent_return']:+.2f}%")
                    st.write(f"**Days Held:** {signal['days_held']}")
                
                with col3:
                    if st.button(f"Mark as Sold", key=f"sell_{signal['ticker']}_{signal['position']['buy_date']}"):
                        st.session_state.portfolio_manager.update_position_status(
                            signal['ticker'], 
                            signal['position']['buy_date'], 
                            'SOLD', 
                            signal['reason']
                        )
                        st.success(f"Marked {signal['ticker']} as sold!")
                        st.rerun()
    else:
        st.success("No sell alerts at this time")
    
    # Add new position section
    st.subheader("➕ Add New Position")
    
    with st.form("add_position_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_ticker = st.text_input("Stock Ticker", placeholder="e.g., AAPL").upper()
            new_buy_price = st.number_input("Buy Price ($)", min_value=0.01, step=0.01)
            new_shares = st.number_input("Number of Shares", min_value=0.01, step=0.01)
        
        with col2:
            new_buy_date = st.date_input("Buy Date", value=datetime.now().date())
            new_target_return = st.number_input("Target Return (%)", value=10.0, step=0.1)
            new_stop_loss = st.number_input("Stop Loss (%)", value=-5.0, step=0.1)
        
        with col3:
            new_hold_days = st.number_input("Max Hold Days", value=30, min_value=1, step=1)
            st.write("")  # Spacing
            add_position_button = st.form_submit_button("Add Position", type="primary")
        
        if add_position_button and new_ticker and new_buy_price > 0 and new_shares > 0:
            st.session_state.portfolio_manager.add_position(
                ticker=new_ticker,
                buy_price=new_buy_price,
                buy_date=str(new_buy_date),
                shares=new_shares,
                target_return=new_target_return,
                stop_loss=new_stop_loss,
                hold_days=new_hold_days
            )
            st.success(f"Added {new_shares} shares of {new_ticker} to portfolio!")
            st.rerun()
    
    # Current positions table
    st.subheader("📋 Current Positions")
    
    portfolio_details = st.session_state.portfolio_manager.get_portfolio_details()
    
    if portfolio_details:
        # Filter controls
        col1, col2 = st.columns(2)
        with col1:
            show_inactive = st.checkbox("Show inactive positions", value=False)
        with col2:
            refresh_button = st.button("🔄 Refresh Prices")
        
        # Filter data
        if not show_inactive:
            portfolio_details = [pos for pos in portfolio_details if pos['status'] == 'ACTIVE']
        
        if portfolio_details:
            # Create DataFrame for display
            display_data = []
            for pos in portfolio_details:
                display_data.append({
                    'Ticker': pos['ticker'],
                    'Buy Date': pos['buy_date'],
                    'Buy Price': f"${pos['buy_price']:.2f}",
                    'Shares': f"{pos['shares']:.2f}",
                    'Current Price': f"${pos['current_price']:.2f}" if pos['current_price'] else "N/A",
                    'Days Held': pos['days_held'] if pos['days_held'] else "N/A",
                    'Return (%)': f"{pos['percent_return']:+.2f}%" if pos['percent_return'] else "N/A",
                    'Total Return': f"${pos['total_return']:+,.2f}" if pos['total_return'] else "N/A",
                    'Current Value': f"${pos['current_value']:,.2f}" if pos['current_value'] else "N/A",
                    'Target (%)': f"{pos['target_return']:.1f}%" if pos['target_return'] else "N/A",
                    'Stop Loss (%)': f"{pos['stop_loss']:.1f}%" if pos['stop_loss'] else "N/A",
                    'Max Days': pos['hold_days'] if pos['hold_days'] else "N/A",
                    'Status': pos['status']
                })
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
            
            # Position management
            st.subheader("🔧 Manage Positions")
            
            # Remove position
            with st.expander("Remove Position"):
                remove_col1, remove_col2, remove_col3 = st.columns(3)
                
                with remove_col1:
                    remove_ticker = st.selectbox(
                        "Select Ticker to Remove",
                        options=[pos['ticker'] for pos in portfolio_details]
                    )
                
                with remove_col2:
                    # Filter buy dates for selected ticker
                    ticker_positions = [pos for pos in portfolio_details if pos['ticker'] == remove_ticker]
                    remove_buy_date = st.selectbox(
                        "Select Buy Date",
                        options=[pos['buy_date'] for pos in ticker_positions]
                    )
                
                with remove_col3:
                    st.write("")  # Spacing
                    if st.button("Remove Position", type="secondary"):
                        st.session_state.portfolio_manager.remove_position(remove_ticker, remove_buy_date)
                        st.success(f"Removed {remove_ticker} position from {remove_buy_date}")
                        st.rerun()
            
            # Export options
            st.subheader("💾 Export Portfolio")
            
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_csv = st.session_state.portfolio_manager.export_portfolio_csv()
                st.download_button(
                    label="📊 Download Portfolio CSV",
                    data=portfolio_csv,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                portfolio_json = json.dumps(portfolio_details, indent=2, default=str)
                st.download_button(
                    label="📄 Download Portfolio JSON",
                    data=portfolio_json,
                    file_name=f"portfolio_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        else:
            st.info("No positions match the current filter criteria")
    
    else:
        st.info("No positions in portfolio. Add your first position above!")
    
    # Portfolio analytics
    if portfolio_details:
        st.subheader("📈 Portfolio Analytics")
        
        active_positions = [pos for pos in portfolio_details if pos['status'] == 'ACTIVE' and pos['percent_return'] is not None]
        
        if active_positions:
            # Performance distribution
            returns = [pos['percent_return'] for pos in active_positions]
            tickers = [pos['ticker'] for pos in active_positions]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Returns histogram
                fig = go.Figure(data=go.Histogram(x=returns, nbinsx=10))
                fig.update_layout(
                    title="Portfolio Returns Distribution",
                    xaxis_title="Return (%)",
                    yaxis_title="Number of Positions"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Individual position returns
                fig = go.Figure(data=go.Bar(x=tickers, y=returns))
                fig.update_layout(
                    title="Individual Position Returns",
                    xaxis_title="Ticker",
                    yaxis_title="Return (%)"
                )
                st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    **Disclaimer:** This application is for educational and informational purposes only. 
    Trading recommendations are based on technical analysis and machine learning predictions and should not be considered as financial advice. 
    Always do your own research and consult with a financial advisor before making investment decisions.
    """
)