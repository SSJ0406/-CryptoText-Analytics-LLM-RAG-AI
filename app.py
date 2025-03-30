# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from dotenv import load_dotenv

# Import custom modules
from src.data_loader import CryptoDataLoader
from src.analysis import CryptoAnalysis
from src.visualization import FinanceViz
from src.llm_integration import LLMClient
from src.event_calendar import EventCalendar
from src.rag_page import render_rag_page

# Load environment variables from .env file
load_dotenv()

# Define theme colors for consistent UI styling
THEME = {
    'primary': '#1E88E5',       # Main blue
    'secondary': '#FFC107',     # Amber accent
    'positive': '#4CAF50',      # Green for positive values
    'negative': '#F44336',      # Red for negative values
    'neutral': '#9E9E9E',       # Gray for neutral elements
    'background': '#F8F9FA',    # Light background
    'text': '#212121',          # Dark text
    'light_text': '#757575'     # Light text for subtitles
}

# Application constants
MAX_CRYPTOS = 5  # Maximum number of cryptocurrencies to analyze (API rate limit)
DEFAULT_CRYPTOS = ["bitcoin", "ethereum", "solana", "cardano", "dogecoin"]  # Default selection

def setup_page_config():
    """
    Configure the Streamlit page settings.
    Sets the page title, icon, layout, and initial sidebar state.
    """
    st.set_page_config(
        page_title="CryptoText Analytics",
        page_icon="🪙",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
def load_custom_css():
    """
    Load custom CSS styles for the application.
    Improves visual appearance and consistency throughout the app.
    """
    st.markdown("""
    <style>
        /* Base container styling */
        .reportview-container {
            background-color: #F8F9FA;
        }
        .sidebar .sidebar-content {
            background-color: #FFFFFF;
        }
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        /* Typography */
        h1, h2, h3 {
            color: #1E88E5;
            font-weight: 600;
        }
        
        /* Metric Card Component */
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            text-align: center;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .positive {
            color: #4CAF50;
        }
        .negative {
            color: #F44336;
        }
        .metric-label {
            font-size: 14px;
            color: #757575;
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            border-radius: 5px;
            background-color: #F0F2F6;
            color: #424242;
            font-weight: 500;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1E88E5 !important;
            color: white !important;
        }
        
        /* News Item Styling */
        .crypto-news-item {
            padding: 15px;
            border-radius: 5px;
            background-color: white;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Layout Adjustments */
        div.block-container {
            padding-top: 1rem;
        }
        
        /* DataFrames */
        div.stDataFrame {
            width: 100%;
        }
        div.stDataFrame [data-testid="stTable"] {
            width: 100%;
        }
        
        /* Sentiment colors */
        .sentiment-positive {
            color: #4CAF50;
            font-weight: 500;
        }
        .sentiment-negative {
            color: #F44336;
            font-weight: 500;
        }
        .sentiment-neutral {
            color: #9E9E9E;
            font-weight: 500;
        }
        
        /* Link styling */
        .news-title a {
            color: #1E88E5 !important;
            text-decoration: none !important;
            border-bottom: 1px solid #1E88E5;
            transition: color 0.2s, border-bottom 0.2s;
        }
        .news-title a:hover {
            color: #0D47A1 !important;
            border-bottom: 2px solid #0D47A1;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """
    Render the application header with logo and title.
    Creates a consistent branding at the top of the application.
    """
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("# 🪙")
    with col2:
        st.title("CryptoText Analytics")
        st.markdown("<p style='font-size: 1.2em; color: #757575;'>Cryptocurrency market analysis powered by data and AI</p>", unsafe_allow_html=True)
    
    # Add horizontal divider for visual separation
    st.markdown("<hr style='margin: 1em 0;'>", unsafe_allow_html=True)

def render_footer():
    """
    Render the application footer with disclaimers and attribution.
    """
    st.markdown("<hr style='margin: 2em 0 1em 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: #757575; font-size: 0.8em;">
        <p>CryptoText Analytics | Data provided by CoinGecko | Created with Streamlit</p>
        <p>© 2025 | For educational purposes only. Not financial advice.</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """
    Render the sidebar with user input controls.
    
    This function creates all the user-configurable options in the sidebar:
    - Cryptocurrency selection
    - Time period and interval
    - Data source selection
    - LLM settings
    - Primary cryptocurrency selection
    - Data refresh option
    
    Returns:
        dict: A dictionary containing all user settings
    """
    with st.sidebar:
        st.markdown("## 🔍 Settings")
        
        # Cryptocurrency selection
        st.markdown("### Cryptocurrencies")
        cryptos_input = st.text_input("Enter comma separated crypto IDs", ",".join(DEFAULT_CRYPTOS))
        
        # Parse and validate crypto input
        cryptos = [crypto.strip().lower() for crypto in cryptos_input.split(",") if crypto.strip()]
        
        # Ensure we always have at least one cryptocurrency
        if not cryptos:
            cryptos = DEFAULT_CRYPTOS
            st.warning("Using default cryptocurrencies because no valid ones were provided.")
        
        # Limit number of cryptocurrencies for API rate limits
        if len(cryptos) > MAX_CRYPTOS:
            st.warning(f"Too many cryptocurrencies selected. Limiting to first {MAX_CRYPTOS} to avoid API limits.")
            cryptos = cryptos[:MAX_CRYPTOS]
        
        # Time period selection
        st.markdown("### Time Period & Interval")
        period_options = {
            "24 Hours": "1",
            "7 Days": "7",
            "14 Days": "14",
            "30 Days": "30",
            "90 Days": "90",
            "180 Days": "180",
            "1 Year": "365",
            "Max": "max"
        }
        selected_period = st.select_slider("Analysis Period", options=list(period_options.keys()))
        period = period_options[selected_period]
        
        # Data interval selection (daily or hourly)
        interval_options = {
            "Daily": "daily",
            "Hourly": "hourly"  # Only available for certain time ranges
        }
        col1, col2 = st.columns(2)
        with col1:
            selected_interval = st.radio("Interval", list(interval_options.keys()))
        interval = interval_options[selected_interval]
        
        # Check if hourly is compatible with selected period
        if interval == "hourly" and period not in ["1", "7", "14", "30"]:
            st.warning("⚠️ Hourly data is only available for periods up to 30 days. Switching to daily.")
            interval = "daily"
        
        # Data source selection
        st.markdown("### Data Source")
        source_options = {
            "CoinGecko": "coingecko",
            "Mock Data (Demo)": "mock"
        }
        selected_source = st.radio("Select data source", list(source_options.keys()), index=0)
        source = source_options[selected_source]
        
        # Add warning for API limits
        if source == "coingecko":
            st.info("ℹ️ CoinGecko free API is limited to 50 calls per minute.")
        if source == "mock":
            st.info("ℹ️ Using mock data for demonstration purposes.")
        
        # LLM feature settings
        st.markdown("### Advanced Features")
        use_llm = st.checkbox("Use LLM for enhanced analysis", value=True)
        use_mock_llm = st.checkbox("Use mock LLM (for demo)", value=False)
        
        # Check for API key if real LLM is selected
        if use_llm and not os.getenv("OPENAI_API_KEY") and not use_mock_llm:
            st.warning("⚠️ OpenAI API key not found. Please add it to your .env file to use LLM features, or enable 'Use mock LLM'.")
            use_llm = False
        
        # Primary cryptocurrency for detailed analysis
        st.markdown("### Primary Analysis")
        primary_crypto = st.selectbox("Primary crypto for detailed analysis", cryptos)
        
        # Refresh data button
        st.markdown("### Update Data")
        refresh = st.button("🔄 Refresh Data", help="Force refresh data from API")
        
        # Return all settings as a dictionary
        return {
            'cryptos': cryptos,
            'period': period,
            'interval': interval,
            'source': source,
            'use_llm': use_llm,
            'use_mock_llm': use_mock_llm,
            'primary_crypto': primary_crypto,
            'refresh': refresh
        }

def load_data(data_loader, settings, analyzer, llm_client):
    """
    Load data for all cryptocurrencies based on user settings.
    
    This function handles:
    - Loading price data for selected cryptocurrencies
    - Fetching cryptocurrency information
    - Getting news data
    - Calculating metrics and trends
    - Adding sentiment analysis to news if LLM is enabled
    - Getting trending cryptocurrencies
    
    Args:
        data_loader: An instance of CryptoDataLoader
        settings: Dictionary containing user settings from sidebar
        analyzer: An instance of CryptoAnalysis
        llm_client: An instance of LLMClient for sentiment analysis
        
    Returns:
        dict: A dictionary containing all loaded data or None if an error occurs
    """
    # Create a loader spinner to indicate data loading
    load_placeholder = st.empty()
    with load_placeholder.container():
        loading = st.markdown("### 🔄 Loading data... Please wait...")
    
    # Initialize data containers
    crypto_data = {}
    crypto_info = {}
    news_data = {}
    metrics = {}
    
    try:
        # Load data for each selected cryptocurrency
        for crypto in settings['cryptos']:
            # Get price data
            crypto_data[crypto] = data_loader.get_crypto_data(
                crypto, 
                days=settings['period'], 
                interval=settings['interval'], 
                force_refresh=settings['refresh'], 
                source=settings['source']
            )
            
            # Get general information about the cryptocurrency
            crypto_info[crypto] = data_loader.get_crypto_info(crypto)
            
            # Get latest news about the cryptocurrency
            news_data[crypto] = data_loader.get_crypto_news(crypto, limit=5)
            
            # Process the data if it was successfully loaded
            if not crypto_data[crypto].empty:
                # Calculate performance metrics
                metrics[crypto] = analyzer.calculate_metrics(crypto_data[crypto])
                
                # Add technical indicators and trend analysis
                crypto_data[crypto] = analyzer.analyze_price_trends(crypto_data[crypto])
                
                # Add sentiment to news using LLM if enabled
                if settings['use_llm']:
                    if settings['use_mock_llm']:
                        # Use mock sentiment analysis for demo purposes
                        for item in news_data[crypto]:
                            title = item.get('title', '')
                            item['sentiment'] = llm_client.mock_sentiment_analysis(title)
                    else:
                        # Use real LLM for sentiment analysis
                        news_data[crypto] = analyzer.analyze_news_sentiment(news_data[crypto], llm_client)
                else:
                    # Use basic sentiment analysis without LLM
                    news_data[crypto] = analyzer.analyze_news_sentiment(news_data[crypto])

        # Fetch trending cryptocurrencies for the market overview
        trending_cryptos = data_loader.get_trending_cryptos()
        
        # Clear the loading message
        load_placeholder.empty()
        
        # Return all the loaded data
        return {
            'crypto_data': crypto_data,
            'crypto_info': crypto_info,
            'news_data': news_data,
            'metrics': metrics,
            'trending_cryptos': trending_cryptos
        }
        
    except Exception as e:
        # Handle any errors during data loading
        load_placeholder.empty()
        st.error(f"Error loading data: {str(e)}")
        return None

def render_price_analysis_tab(tab, data, primary_crypto, viz):
    """
    Render content for the Price Analysis tab.
    
    This function creates the main price analysis dashboard for the primary cryptocurrency, including:
    - Price metrics
    - Market overview
    - Price chart
    - Technical indicators
    - Returns distribution
    - Trending cryptocurrencies
    
    Args:
        tab: The Streamlit tab object to render content in
        data: Dictionary containing all cryptocurrency data
        primary_crypto: The selected primary cryptocurrency for detailed analysis
        viz: An instance of FinanceViz for chart creation
    """
    with tab:
        st.header(f"Price Analysis: {data['crypto_info'][primary_crypto].get('name', primary_crypto)}")
        
        # Check if we have valid data for the primary cryptocurrency
        if primary_crypto in data['crypto_data'] and not data['crypto_data'][primary_crypto].empty:
            # Display key price metrics (current price, 24h change, market cap)
            render_price_metrics(data['crypto_info'], primary_crypto)
            
            # Market overview section with financial metrics and coin info
            st.markdown("### 📊 Market Overview")
            col1, col2 = st.columns([2, 3])
            
            with col1:
                render_key_metrics(data['metrics'], primary_crypto)
            
            with col2:
                render_crypto_info(data['crypto_info'], primary_crypto)
            
            # Price chart section
            st.markdown("### 📈 Price Chart")
            symbol = data['crypto_info'][primary_crypto].get('symbol', '').upper()
            price_chart = viz.plot_stock_price(data['crypto_data'][primary_crypto], symbol)
            price_chart = format_price_chart(price_chart, symbol)
            st.plotly_chart(price_chart, use_container_width=True, config={'displayModeBar': False})
            
            # Technical analysis indicators
            render_technical_indicators(data['crypto_data'], primary_crypto, symbol, THEME)
            
            # Returns distribution
            render_returns_distribution(data['crypto_data'], primary_crypto, symbol, viz)
            
            # Trending cryptocurrencies section
            render_trending_cryptos(data['trending_cryptos'])
        else:
            st.warning(f"No data available for {primary_crypto}. Please check the cryptocurrency ID or try a different time period or data source.")

def render_price_metrics(crypto_info, primary_crypto):
    """
    Render the key price metrics cards at the top of the price analysis tab.
    
    Args:
        crypto_info: Dictionary containing cryptocurrency information
        primary_crypto: The selected primary cryptocurrency
    """
    # Current price and price change
    current_price = crypto_info[primary_crypto].get('current_price', 0)
    price_change_24h = crypto_info[primary_crypto].get('price_change_percentage_24h', 0)
    
    # Display metrics in styled cards
    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            value=f"${current_price:,.2f}",
            label="Current Price"
        )
    
    with col2:
        color_class = 'positive' if price_change_24h > 0 else 'negative'
        render_metric_card(
            value=f"{price_change_24h:+.2f}%",
            label="24h Change",
            color_class=color_class
        )
    
    with col3:
        market_cap = crypto_info[primary_crypto].get('market_cap', 0)
        # Format market cap in billions or millions for better readability
        market_cap_formatted = f"${market_cap/1_000_000_000:.2f}B" if market_cap >= 1_000_000_000 else f"${market_cap/1_000_000:.2f}M"
        render_metric_card(
            value=market_cap_formatted,
            label="Market Cap"
        )

def render_metric_card(value, label, color_class=""):
    """
    Render a metric card with consistent styling.
    
    Args:
        value: The metric value to display
        label: The label describing the metric
        color_class: Optional CSS class for styling (e.g., 'positive', 'negative')
    """
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value {color_class}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

def render_key_metrics(metrics, primary_crypto):
    """
    Render key financial metrics for the primary cryptocurrency.
    
    Args:
        metrics: Dictionary containing calculated metrics for cryptocurrencies
        primary_crypto: The selected primary cryptocurrency
    """
    # Only show if metrics are available
    if primary_crypto in metrics:
        # Create metrics cards for key performance indicators
        m_col1, m_col2 = st.columns(2)
        
        # Total Return
        total_return = metrics[primary_crypto].get('total_return', 0)
        total_return_color = 'positive' if total_return > 0 else 'negative'
        m_col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {total_return_color}">{total_return:.2f}%</div>
            <div class="metric-label">Total Return</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Volatility
        volatility = metrics[primary_crypto].get('annual_volatility', 0)
        m_col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{volatility:.2f}%</div>
            <div class="metric-label">Annual Volatility</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Max Drawdown
        max_drawdown = metrics[primary_crypto].get('max_drawdown', 0)
        m_col1.markdown(f"""
        <div class="metric-card">
            <div class="metric-value negative">{max_drawdown:.2f}%</div>
            <div class="metric-label">Max Drawdown</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sharpe Ratio
        sharpe_ratio = metrics[primary_crypto].get('sharpe_ratio', 0)
        sharpe_color = 'positive' if sharpe_ratio > 1 else 'neutral'
        m_col2.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {sharpe_color}">{sharpe_ratio:.2f}</div>
            <div class="metric-label">Sharpe Ratio</div>
        </div>
        """, unsafe_allow_html=True)

def render_crypto_info(crypto_info, primary_crypto):
    """
    Render general information about the primary cryptocurrency.
    
    Args:
        crypto_info: Dictionary containing cryptocurrency information
        primary_crypto: The selected primary cryptocurrency
    """
    # Crypto description with character limit
    st.markdown("### About")
    description = crypto_info[primary_crypto].get('description', '')
    # Truncate long descriptions for better UI
    if len(description) > 300:
        description = description[:300] + "..."
    st.markdown(f"{description}")
    
    # Add some key information and links
    col1, col2 = st.columns(2)
    col1.markdown(f"**Symbol:** {crypto_info[primary_crypto].get('symbol', '').upper()}")
    col1.markdown(f"**Rank:** #{crypto_info[primary_crypto].get('market_cap_rank', 'N/A')}")
    
    homepage = crypto_info[primary_crypto].get('homepage', '')
    reddit = crypto_info[primary_crypto].get('reddit', '')
    
    if homepage:
        col2.markdown(f"[Official Website]({homepage})")
    if reddit:
        col2.markdown(f"[Reddit Community]({reddit})")

def format_price_chart(price_chart, symbol):
    """
    Format the price chart with consistent styling.
    
    Args:
        price_chart: The Plotly chart object
        symbol: The cryptocurrency symbol for the title
        
    Returns:
        The formatted Plotly chart object
    """
    price_chart.update_layout(
        template="plotly_white",
        title=f"{symbol} - Price and Volume Analysis",
        height=500,
        margin=dict(l=20, r=20, t=30, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            title="Price ($)",
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        )
    )
    return price_chart

def render_trending_cryptos(trending_cryptos):
    """
    Render trending cryptocurrencies section.
    
    Args:
        trending_cryptos: DataFrame containing trending cryptocurrency data
    """
    st.markdown("### 🔥 Trending Cryptocurrencies")
    if not trending_cryptos.empty:
        # Format trending cryptos as cards in a grid
        trending_cols = st.columns(3)
        for i, (_, row) in enumerate(trending_cryptos.iterrows()):
            if i < 6:  # Display top 6 only
                col_idx = i % 3
                with trending_cols[col_idx]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-weight: 600; font-size: 18px;">{row['name']} ({row['symbol']})</div>
                        <div class="metric-label">Rank #{row['market_cap_rank'] if pd.notna(row['market_cap_rank']) else 'N/A'}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No trending cryptocurrency data available at the moment.")

def render_technical_indicators(crypto_data, primary_crypto, symbol, theme):
    """
    Render technical indicators for the primary cryptocurrency.
    
    Creates a dual-panel chart with price and RSI (Relative Strength Index),
    including overbought/oversold regions and explanations.
    
    Args:
        crypto_data: Dictionary containing price data for cryptocurrencies
        primary_crypto: The selected primary cryptocurrency
        symbol: The cryptocurrency symbol for the chart title
        theme: Dictionary containing theme colors
    """
    st.markdown("### 🔎 Technical Indicators")
    
    # RSI Chart if available
    if 'RSI' in crypto_data[primary_crypto].columns:
        # Create RSI figure with styling - using subplots for price and RSI
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.1, 
            row_heights=[0.7, 0.3],
            subplot_titles=[f"{symbol} Price Chart", "Relative Strength Index (RSI)"]
        )
        
        # Add price to top subplot
        fig.add_trace(
            go.Scatter(
                x=crypto_data[primary_crypto].index, 
                y=crypto_data[primary_crypto]['Close'], 
                name='Price',
                line=dict(color=theme['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add SMA (Simple Moving Average) lines if available
        if 'SMA_short' in crypto_data[primary_crypto].columns:
            fig.add_trace(
                go.Scatter(
                    x=crypto_data[primary_crypto].index, 
                    y=crypto_data[primary_crypto]['SMA_short'], 
                    name='7-day MA',
                    line=dict(color='rgba(255, 193, 7, 0.8)', width=1.5, dash='dash')
                ),
                row=1, col=1
            )
        
        if 'SMA_long' in crypto_data[primary_crypto].columns:
            fig.add_trace(
                go.Scatter(
                    x=crypto_data[primary_crypto].index, 
                    y=crypto_data[primary_crypto]['SMA_long'], 
                    name='30-day MA',
                    line=dict(color='rgba(233, 30, 99, 0.8)', width=1.5, dash='dash')
                ),
                row=1, col=1
            )
        
        # Add RSI to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=crypto_data[primary_crypto].index, 
                y=crypto_data[primary_crypto]['RSI'], 
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Add overbought line (RSI = 70)
        fig.add_shape(
            type="line", 
            line=dict(dash='dash', width=1, color=theme['negative']),
            y0=70, y1=70, 
            x0=crypto_data[primary_crypto].index[0], 
            x1=crypto_data[primary_crypto].index[-1],
            row=2, col=1
        )
        
        # Add oversold line (RSI = 30)
        fig.add_shape(
            type="line", 
            line=dict(dash='dash', width=1, color=theme['positive']),
            y0=30, y1=30, 
            x0=crypto_data[primary_crypto].index[0], 
            x1=crypto_data[primary_crypto].index[-1],
            row=2, col=1
        )
        
        # Add reference line at RSI = 50 (neutral)
        fig.add_shape(
            type="line", 
            line=dict(dash='dot', width=0.5, color='rgba(0, 0, 0, 0.2)'),
            y0=50, y1=50, 
            x0=crypto_data[primary_crypto].index[0], 
            x1=crypto_data[primary_crypto].index[-1],
            row=2, col=1
        )
        
        # Add background color zones to RSI for better visualization
        # Overbought region (RSI > 70)
        fig.add_shape(
            type="rect", 
            x0=crypto_data[primary_crypto].index[0], 
            x1=crypto_data[primary_crypto].index[-1],
            y0=70, y1=100,
            fillcolor="rgba(244, 67, 54, 0.1)", 
            line=dict(width=0),
            row=2, col=1
        )
        
        # Oversold region (RSI < 30)
        fig.add_shape(
            type="rect", 
            x0=crypto_data[primary_crypto].index[0], 
            x1=crypto_data[primary_crypto].index[-1],
            y0=0, y1=30,
            fillcolor="rgba(76, 175, 80, 0.1)", 
            line=dict(width=0),
            row=2, col=1
        )
        
        # Update layout with improved styling
        fig.update_layout(
            template="plotly_white",
            title=f"{symbol} - Technical Analysis",
            height=600,
            margin=dict(l=20, r=20, t=30, b=60),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            xaxis_rangeslider_visible=False,
            xaxis=dict(
                title="Date",
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.8)'
            ),
            yaxis=dict(
                title="Price ($)",
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.8)'
            )
        )
        
        # Add axis titles
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='RSI Value', row=2, col=1, range=[0, 100])
        
        # Adjust annotation font size
        fig.update_annotations(font_size=12)
        
        # Display the chart
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Add explanation for RSI to help users understand the indicator
        with st.expander("What is RSI?"):
            st.markdown("""
            **Relative Strength Index (RSI)** is a momentum oscillator that measures the speed and change of price movements. RSI oscillates between 0 and 100.

            - **RSI > 70**: The asset may be **overbought** (potential sell signal)
            - **RSI < 30**: The asset may be **oversold** (potential buy signal)
            - **RSI = 50**: Neutral indication

            RSI can form chart patterns that may not be visible on the underlying price chart, such as divergences and trend lines.
            """)

def render_returns_distribution(crypto_data, primary_crypto, symbol, viz):
    """
    Render returns distribution chart for the primary cryptocurrency.
    
    This shows the distribution of daily returns, which helps visualize
    volatility and potential risk characteristics.
    
    Args:
        crypto_data: Dictionary containing price data for cryptocurrencies
        primary_crypto: The selected primary cryptocurrency
        symbol: The cryptocurrency symbol for the chart title
        viz: An instance of FinanceViz for chart creation
    """
    st.markdown("### 📊 Returns Distribution")
    
    # Generate the returns distribution chart using the visualization module
    returns_chart = viz.plot_returns_distribution(crypto_data[primary_crypto])
    
    # Format the chart with consistent styling
    returns_chart.update_layout(
        template="plotly_white",
        title=f"{symbol} - Daily Returns Analysis",
        height=400,
        margin=dict(l=20, r=20, t=30, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis=dict(
            title="Daily Return (%)",
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        ),
        yaxis=dict(
            title="Frequency",
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)'
        )
    )
    
    # Display the chart
    st.plotly_chart(returns_chart, use_container_width=True, config={'displayModeBar': False})

def render_comparison_tab(tab, data, viz):
    """
    Render content for the Crypto Comparison tab.
    
    This function creates the comparison dashboard for multiple cryptocurrencies, including:
    - Relative performance chart
    - Key metrics comparison
    - Current market data comparison
    - Trading volume comparison
    
    Args:
        tab: The Streamlit tab object to render content in
        data: Dictionary containing all cryptocurrency data
        viz: An instance of FinanceViz for chart creation
    """
    with tab:
        st.header("Cryptocurrency Comparison")
        
        # Check if we have any valid data for comparison
        valid_data = {crypto: data_item for crypto, data_item in data['crypto_data'].items() 
                     if not data_item.empty and 'Close' in data_item.columns and len(data_item['Close']) > 0}
        
        if valid_data:
            # Comparative performance chart with improved styling
            st.markdown("### 📈 Performance Comparison")
            comparison_chart = viz.plot_comparison(valid_data, title='Relative Performance')
            comparison_chart.update_layout(
                template="plotly_white",
                title="Cryptocurrency Performance Comparison",
                height=500,
                margin=dict(l=20, r=20, t=30, b=60),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                xaxis=dict(
                    title="Date",
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                ),
                yaxis=dict(
                    title="Normalized Value (First Day = 100)",
                    showgrid=True,
                    gridcolor='rgba(230, 230, 230, 0.8)'
                )
            )
            st.plotly_chart(comparison_chart, use_container_width=True, config={'displayModeBar': False})
            
            # Interactive comparison tools
            st.markdown("### 🔍 Key Metrics Comparison")
            
            # Current market data comparison in cards
            render_market_data_comparison(valid_data, data['crypto_info'])
            
            # Performance metrics comparison table
            render_metrics_comparison_table(valid_data, data['metrics'], data['crypto_info'])
            
            # Volume comparison chart
            render_volume_comparison(valid_data, data['crypto_info'])
        else:
            st.warning("No valid data available for comparison. Try a different data source.")

def render_market_data_comparison(valid_data, crypto_info):
    """
    Render current market data comparison for multiple cryptocurrencies.
    
    Args:
        valid_data: Dictionary containing validated price data
        crypto_info: Dictionary containing cryptocurrency information
    """
    st.markdown("#### Current Market Data")
    market_cols = st.columns(len(valid_data))
    
    for i, crypto in enumerate(valid_data.keys()):
        if crypto in crypto_info:
            with market_cols[i]:
                price = crypto_info[crypto].get('current_price', 0)
                price_change = crypto_info[crypto].get('price_change_percentage_24h', 0)
                price_color = 'positive' if price_change > 0 else 'negative'
                
                st.markdown(f"""
                <div class="metric-card" style="height: 180px;">
                    <div style="font-weight: 600; font-size: 20px;">{crypto_info[crypto].get('name', crypto)}</div>
                    <div style="font-size: 14px; color: #757575;">{crypto_info[crypto].get('symbol', '').upper()}</div>
                    <div style="font-size: 24px; font-weight: 600; margin: 10px 0;">${price:,.2f}</div>
                    <div class="{price_color}" style="font-weight: 500;">{price_change:+.2f}%</div>
                    <div class="metric-label">24h Change</div>
                </div>
                """, unsafe_allow_html=True)

def render_metrics_comparison_table(valid_data, metrics, crypto_info):
    """
    Render a comparison table of performance metrics for multiple cryptocurrencies.
    
    Args:
        valid_data: Dictionary containing validated price data
        metrics: Dictionary containing calculated metrics for cryptocurrencies
        crypto_info: Dictionary containing cryptocurrency information
    """
    st.markdown("#### Performance Metrics")
    
    # Create a DataFrame of metrics for easy display
    if metrics:
        # Filter out cryptos without metrics
        valid_metrics = {crypto: metric for crypto, metric in metrics.items() if crypto in valid_data}
        if valid_metrics:
            metrics_df = pd.DataFrame.from_dict(valid_metrics, orient='index')
            
            # Format for better display
            if not metrics_df.empty and all(col in metrics_df.columns for col in ['total_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']):
                formatted_metrics = metrics_df[['total_return', 'annual_volatility', 'sharpe_ratio', 'max_drawdown']].copy()
                formatted_metrics.columns = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
                formatted_metrics = formatted_metrics.round(2)
                
                # Add crypto name for better display
                formatted_metrics.index = [crypto_info[idx].get('name', idx) for idx in formatted_metrics.index]
                
                # Display the styled DataFrame
                st.dataframe(formatted_metrics, use_container_width=True)
            else:
                st.warning("Incomplete metrics data available.")
        else:
            st.warning("No metrics data available for comparison.")
    else:
        st.warning("No metrics data available for comparison.")

def render_volume_comparison(valid_data, crypto_info):
    """
    Render a trading volume comparison chart for multiple cryptocurrencies.
    
    Args:
        valid_data: Dictionary containing validated price data
        crypto_info: Dictionary containing cryptocurrency information
    """
    st.markdown("#### Trading Volume Comparison")

    # Create volume comparison chart
    volume_data = []
    for crypto, data in valid_data.items():
        if 'Volume' in data.columns and not data['Volume'].empty:
            latest_volume = data['Volume'].iloc[-1]
            symbol = crypto_info[crypto].get('symbol', '').upper()
            name = crypto_info[crypto].get('name', crypto)
            volume_data.append({
                'Cryptocurrency': name,
                'Symbol': symbol,
                'Volume': latest_volume
            })

    if volume_data:
        volume_df = pd.DataFrame(volume_data)
        fig = px.bar(
            volume_df, 
            x='Cryptocurrency', 
            y='Volume',
            color='Cryptocurrency',
            text='Symbol',
            labels={'Cryptocurrency': 'Cryptocurrency', 'Volume': 'Trading Volume (USD)'},
            title="Daily Trading Volume Comparison"
        )
        
        fig.update_layout(
            template="plotly_white",
            title="Daily Trading Volume by Cryptocurrency",
            height=400,
            margin=dict(l=40, r=40, t=50, b=60),
            xaxis=dict(
                title="Cryptocurrency",
                showgrid=False,
                tickangle=-45,  # Angled labels for better readability
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title="Volume (USD)",
                type='log',  # Log scale for better visualization when volumes vary greatly
                showgrid=True,
                gridcolor='rgba(230, 230, 230, 0.8)'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Format volume values in the bars
        fig.update_traces(
            texttemplate='%{text}',
            textposition='inside',
            textfont=dict(size=14),
            insidetextanchor='middle',
            hovertemplate='%{x}<br>Volume: $%{y:,.0f}'  # Formatted tooltip
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    else:
        st.info("No volume data available for comparison.")

def render_event_calendar_tab(tab, event_calendar, cryptos, refresh, viz):
    """
    Render content for the Event Calendar tab.
    
    This function creates the event calendar dashboard, including:
    - Filtering options
    - Event timeline visualization
    - Upcoming events list
    
    Args:
        tab: The Streamlit tab object to render content in
        event_calendar: An instance of EventCalendar for fetching event data
        cryptos: List of selected cryptocurrencies
        refresh: Boolean indicating whether to force refresh data
        viz: An instance of FinanceViz for chart creation
    """
    with tab:
        st.header("Cryptocurrency Event Calendar")
        
        # Simplified filter options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📅 Upcoming Events")
            filter_project = st.multiselect(
                "Filter by project", 
                options=["All"] + cryptos,
                default=["All"]
            )
        
        with col2:
            st.markdown("### ⚙️ Settings")
            days_ahead = st.slider("Days to look ahead", min_value=7, max_value=180, value=60, step=7)
        
        # Fetch event data based on filters
        events_data = fetch_filtered_events(filter_project, days_ahead, event_calendar, refresh)
        
        # Visualization of event calendar
        if not events_data.empty:
            # Event Timeline visualization
            st.markdown("### 📊 Event Timeline")
            event_timeline = viz.plot_event_timeline(events_data)
            if event_timeline:
                st.plotly_chart(event_timeline, use_container_width=True)
            
            # Upcoming events list with formatted cards
            st.markdown("### 📋 Upcoming Events List")
            render_events_list(events_data)
        else:
            st.warning("No events data available for the selected criteria. Try adjusting your filters or refreshing the data.")

def fetch_filtered_events(filter_project, days_ahead, event_calendar, refresh):
    """
    Fetch event data based on user filter selections.
    
    Args:
        filter_project: List of selected projects to filter events by
        days_ahead: Number of days to look ahead for events
        event_calendar: An instance of EventCalendar
        refresh: Boolean indicating whether to force refresh data
        
    Returns:
        DataFrame containing the filtered event data
    """
    if "All" in filter_project:
        # Fetch all events if "All" is selected
        events_data = event_calendar.get_crypto_events(days_ahead=days_ahead, force_refresh=refresh)
    else:
        # Fetch events for selected cryptocurrencies
        all_events = []
        for crypto in filter_project:
            crypto_events = event_calendar.get_crypto_events(crypto_id=crypto, days_ahead=days_ahead, force_refresh=refresh)
            all_events.append(crypto_events)
        
        if all_events:
            events_data = pd.concat(all_events, ignore_index=True)
        else:
            events_data = pd.DataFrame()
    
    return events_data

def render_events_list(events_data):
    """
    Render a formatted list of upcoming events.
    
    Args:
        events_data: DataFrame containing event data
    """
    # Add column for days until event
    events_data['days_until'] = (events_data['date'] - datetime.now()).dt.days
    
    # Sort by date (nearest first)
    events_list = events_data.sort_values('date')
    
    # Create cards for each event
    for _, event in events_list.iterrows():
        days_until = event['days_until']
        
        # Format the time text based on proximity
        if days_until == 0:
            time_text = "Today"
        elif days_until == 1:
            time_text = "Tomorrow"
        else:
            time_text = f"In {days_until} days"
        
        # Determine color for importance level
        if event['importance'] == 'high':
            importance_bg = 'rgba(30, 136, 229, 0.1)'  # Light blue for all importance levels
            importance_icon = "🔵"
        elif event['importance'] == 'medium':
            importance_bg = 'rgba(30, 136, 229, 0.05)'
            importance_icon = "🔵"
        else:
            importance_bg = 'rgba(30, 136, 229, 0.02)'
            importance_icon = "🔵"
        
        # Create event card with consistent styling
        st.markdown(f"""
        <div style="background-color: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1); padding: 15px; margin-bottom: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="font-weight: 600; font-size: 18px;">{event['project_name']} ({event['symbol']})</div>
                    <div style="color: #757575; font-size: 14px;">{event['category']} | {event['event_type']}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-weight: 500;">{event['date'].strftime('%Y-%m-%d')}</div>
                    <div style="color: #757575; font-size: 14px;">{time_text}</div>
                </div>
            </div>
            <div style="margin: 10px 0;">
                {event['description']}
            </div>
            <div style="display: flex; justify-content: flex-start; margin-top: 10px; gap: 10px;">
                <div style="background-color: {importance_bg}; color: #424242; font-weight: 500; font-size: 12px; padding: 3px 8px; border-radius: 12px; white-space: nowrap;">
                    {importance_icon} {event['importance'].capitalize()} Importance
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_ai_insights_tab(tab, data, settings, llm_client, event_calendar, theme):
    """
    Render content for the AI Insights tab.
    
    This function creates the AI-powered insights dashboard, including:
    - AI analysis of the primary cryptocurrency
    - Interactive Q&A feature
    
    Args:
        tab: The Streamlit tab object to render content in
        data: Dictionary containing all cryptocurrency data
        settings: Dictionary containing user settings
        llm_client: An instance of LLMClient for AI analysis
        event_calendar: An instance of EventCalendar
        theme: Dictionary containing theme colors
    """
    with tab:
        st.header("AI-Powered Crypto Insights")
        
        primary_crypto = settings['primary_crypto']
        use_llm = settings['use_llm']
        use_mock_llm = settings['use_mock_llm']
        
        if use_llm or use_mock_llm:
            if primary_crypto in data['crypto_data'] and not data['crypto_data'][primary_crypto].empty:
                # Prepare data context for the LLM
                context = prepare_llm_context(data, primary_crypto, event_calendar)
                
                # Generate AI analysis
                st.markdown(f"### 🤖 AI Analysis for {data['crypto_info'][primary_crypto].get('name', primary_crypto)}")
                
                with st.spinner("Generating AI analysis..."):
                    analysis = generate_financial_analysis(
                        primary_crypto, 
                        context, 
                        llm_client, 
                        use_mock_llm
                    )
                    
                    # Add styled container for the analysis
                    st.markdown(f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                        {analysis}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Interactive Q&A section
                render_interactive_qa(data, primary_crypto, context, llm_client, use_mock_llm, theme)
            else:
                st.warning(f"No data available for {primary_crypto}. Please check the cryptocurrency ID or try a different time period.")
        else:
            # Styled info card for disabled LLM features
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <h3 style="color: {theme['primary']};">Enable AI Features</h3>
                <p>Enable LLM features in the sidebar to use AI-powered insights.</p>
                <p><strong>AI-powered features include:</strong></p>
                <ul>
                    <li>Advanced sentiment analysis of news</li>
                    <li>Comprehensive cryptocurrency analysis and insights</li>
                    <li>Interactive Q&A about specific cryptocurrencies</li>
                </ul>
                <p>You can use either OpenAI's API (requires API key) or the mock LLM option for demonstrations.</p>
            </div>
            """, unsafe_allow_html=True)

def prepare_llm_context(data, primary_crypto, event_calendar):
    """
    Prepare context data for LLM analysis.
    
    Args:
        data: Dictionary containing all cryptocurrency data
        primary_crypto: The selected primary cryptocurrency
        event_calendar: An instance of EventCalendar
        
    Returns:
        dict: A dictionary containing structured context for LLM
    """
    # Get metrics text
    metrics_text = "\n".join([f"{k}: {v}" for k, v in data['metrics'].get(primary_crypto, {}).items()])
    
    # Get price trend description
    price_data = data['crypto_data'][primary_crypto]
    start_price = price_data['Close'].iloc[0] if not price_data.empty and len(price_data['Close']) > 0 else 0
    end_price = price_data['Close'].iloc[-1] if not price_data.empty and len(price_data['Close']) > 0 else 0
    price_change = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
    trend_text = f"Starting price: ${start_price:.2f}, Current price: ${end_price:.2f}, Change: {price_change:.2f}%"
    
    # Add RSI information if available
    if 'RSI' in price_data.columns:
        latest_rsi = price_data['RSI'].iloc[-1] if not price_data.empty and len(price_data['RSI']) > 0 else None
        if latest_rsi is not None:
            trend_text += f"\nCurrent RSI: {latest_rsi:.2f}"
            if latest_rsi > 70:
                trend_text += " (Overbought territory)"
            elif latest_rsi < 30:
                trend_text += " (Oversold territory)"
    
    # Add market cap info
    if data['crypto_info'][primary_crypto].get('market_cap') is not None:
        market_cap = data['crypto_info'][primary_crypto].get('market_cap')
        trend_text += f"\nMarket Cap: ${market_cap:,.0f}"
    
    # Get upcoming events for this crypto
    upcoming_events = event_calendar.get_crypto_events(crypto_id=primary_crypto, days_ahead=30)
    events_text = ""
    
    if not upcoming_events.empty:
        events_text = "Upcoming Events:\n"
        for _, event in upcoming_events.iterrows():
            event_date = event['date'].strftime('%Y-%m-%d')
            events_text += f"- {event_date}: {event['event_type']} - {event['description']} (Potential impact: {event['potential_impact']})\n"
    
    # Get news summary from the news data
    news_text = ""
    if primary_crypto in data['news_data'] and data['news_data'][primary_crypto]:
        news_text = "Recent News:\n"
        news_text += "\n".join([f"- {item.get('title', 'No title')} ({item.get('sentiment', 'neutral')})" 
                              for item in data['news_data'][primary_crypto][:5]])
    
    # Combine all context
    combined_context = {
        'metrics': metrics_text,
        'trend': trend_text,
        'events': events_text,
        'news': news_text
    }
    
    return combined_context

def generate_financial_analysis(primary_crypto, context, llm_client, use_mock_llm):
    """
    Generate a comprehensive financial analysis using the LLM.
    
    Args:
        primary_crypto: The selected primary cryptocurrency
        context: Dictionary containing context data for analysis
        llm_client: An instance of LLMClient
        use_mock_llm: Boolean indicating whether to use mock LLM
        
    Returns:
        str: Generated analysis text
    """
    # Use either mock or real LLM based on settings
    if use_mock_llm:
        analysis = llm_client.mock_financial_analysis(
            primary_crypto, 
            context['metrics'], 
            f"Events:\n{context['events']}\n\nNews:\n{context['news']}", 
            context['trend']
        )
    else:
        analysis = llm_client.generate_financial_analysis(
            primary_crypto, 
            context['metrics'], 
            f"Events:\n{context['events']}\n\nNews:\n{context['news']}", 
            context['trend']
        )
    
    return analysis

def render_interactive_qa(data, primary_crypto, context, llm_client, use_mock_llm, theme):
    """
    Render an interactive Q&A section for cryptocurrency insights.
    
    Args:
        data: Dictionary containing all cryptocurrency data
        primary_crypto: The selected primary cryptocurrency
        context: Dictionary containing context data for analysis
        llm_client: An instance of LLMClient
        use_mock_llm: Boolean indicating whether to use mock LLM
        theme: Dictionary containing theme colors
    """
    st.markdown("### 💬 Ask about this cryptocurrency")
    question = st.text_input("Enter your question here:")
    
    if question:
        # Prepare comprehensive context for the question
        qa_context = f"""
        Cryptocurrency: {data['crypto_info'][primary_crypto].get('name', primary_crypto)} ({data['crypto_info'][primary_crypto].get('symbol', '').upper()})
        
        Financial Metrics:
        {context['metrics']}
        
        Price Trend:
        {context['trend']}
        
        {context['events']}
        
        {context['news']}
        """
        
        with st.spinner("Generating answer..."):
            # Generate answer using either mock or real LLM
            if use_mock_llm:
                answer = llm_client.mock_answer_question(
                    question, 
                    data['crypto_info'][primary_crypto].get('name', primary_crypto), 
                    qa_context
                )
            else:
                answer = llm_client.answer_financial_question(
                    question, 
                    data['crypto_info'][primary_crypto].get('name', primary_crypto), 
                    qa_context
                )
            
            # Display answer in a styled container
            st.markdown(f"""
            <div style="background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
                <p style="font-weight: 600; color: {theme['primary']};">Question:</p>
                <p>{question}</p>
                <hr style="margin: 10px 0;">
                <p style="font-weight: 600; color: {theme['primary']};">Answer:</p>
                {answer}
            </div>
            """, unsafe_allow_html=True)

def main():
    """
    Main application function that orchestrates the entire CryptoText Analytics app.
    
    This function:
    1. Sets up the page configuration
    2. Loads custom CSS styling
    3. Renders the application header
    4. Initializes components and sidebar
    5. Loads cryptocurrency data
    6. Creates and populates all tabs
    7. Renders the footer
    """
    # Setup page configuration
    setup_page_config()
    
    # Load custom CSS for styling
    load_custom_css()
    
    # Render app header
    render_header()
    
    # Initialize components
    data_loader = CryptoDataLoader()
    analyzer = CryptoAnalysis()
    viz = FinanceViz()
    llm_client = LLMClient()
    event_calendar = EventCalendar()
    
    # Render sidebar and get user settings
    settings = render_sidebar()
    
    # Load data based on user settings
    data = load_data(data_loader, settings, analyzer, llm_client)
    
    # Only proceed if data was successfully loaded
    if data:
        # Create tabs with custom styling
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Price Analysis", 
            "🔍 Crypto Comparison", 
            "📅 Event Calendar", 
            "🤖 AI Insights", 
            "📚 Research & Documents"
        ])
        
        # Render content for each tab
        render_price_analysis_tab(tab1, data, settings['primary_crypto'], viz)
        render_comparison_tab(tab2, data, viz)
        render_event_calendar_tab(tab3, event_calendar, settings['cryptos'], settings['refresh'], viz)
        render_ai_insights_tab(tab4, data, settings, llm_client, event_calendar, THEME)
        
        # Render the Research tab using the imported function
        with tab5:
            render_rag_page(crypto_info=data['crypto_info'], primary_crypto=settings['primary_crypto'], theme=THEME)
    
    # Add footer with attribution and disclaimers
    render_footer()

# Run the application when script is executed directly
if __name__ == "__main__":
    main()