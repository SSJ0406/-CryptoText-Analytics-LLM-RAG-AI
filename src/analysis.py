import pandas as pd
import numpy as np

class CryptoAnalysis:
    def __init__(self):
        # Dictionaries for sentiment analysis
        self.positive_keywords = [
            "bullish", "surge", "rally", "growth", "gain", "positive", "rise", "soar", "jump",
            "upgrade", "higher", "all-time high", "profit", "outperform", "beat", "skyrocket",
            "breakthrough", "uptrend", "record high", "opportunity", "strong", "boost", "recovery",
            "partnership", "adoption", "success", "green", "potential", "support", "progress"
        ]
        
        self.negative_keywords = [
            "bearish", "crash", "plunge", "slump", "decline", "drop", "fall", "negative", "decrease",
            "downgrade", "lower", "loss", "weaker", "correction", "underperform", "tumble", "struggle",
            "sell-off", "downtrend", "risk", "concern", "warning", "volatile", "problem", "bubble",
            "fraud", "hack", "regulation", "ban", "attack", "red", "liquidation", "fear", "resistance"
        ]
    
    def calculate_metrics(self, data):
        """
        Calculates basic metrics from historical price data.
        
        Args:
            data (DataFrame): DataFrame containing price data with at least a 'Close' column
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        if data.empty:
            return {}
        
        # Ensure index is datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Calculate daily returns
        data['daily_return'] = data['Close'].pct_change()
        
        # Basic price metrics
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        max_price = data['Close'].max()
        min_price = data['Close'].min()
        
        # Volatility metrics - annualized
        volatility = data['daily_return'].std() * np.sqrt(365)  # Annual volatility for crypto (365 days)
        
        # Performance metrics
        total_return = (end_price / start_price - 1) * 100
        
        # Risk metrics - drawdown calculation
        peak = data['Close'].cummax()
        drawdown = (data['Close'] / peak - 1) * 100
        max_drawdown = drawdown.min()
        
        # Risk-adjusted return metrics
        sharpe_ratio = (data['daily_return'].mean() * 365) / (data['daily_return'].std() * np.sqrt(365))
        
        # Crypto-specific metrics
        market_cap_latest = None
        volume_latest = None
        
        if 'MarketCap' in data.columns and not data['MarketCap'].empty:
            market_cap_latest = data['MarketCap'].iloc[-1]
        
        if 'Volume' in data.columns and not data['Volume'].empty:
            volume_latest = data['Volume'].iloc[-1]
        
        metrics = {
            'start_price': start_price,
            'end_price': end_price,
            'max_price': max_price,
            'min_price': min_price,
            'total_return': total_return,
            'annual_volatility': volatility * 100,  # as percent
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'market_cap': market_cap_latest,
            'volume': volume_latest
        }
        
        return metrics
    
    def analyze_price_trends(self, data, window_short=7, window_long=30):
        """
        Analyzes price trends using moving averages - crypto uses shorter windows typically.
        
        Args:
            data (DataFrame): DataFrame containing price data with at least a 'Close' column
            window_short (int): Short-term moving average window size
            window_long (int): Long-term moving average window size
            
        Returns:
            DataFrame: Original data with additional trend analysis columns
        """
        if data.empty:
            return data
        
        # Calculate moving averages
        data['SMA_short'] = data['Close'].rolling(window=window_short).mean()
        data['SMA_long'] = data['Close'].rolling(window=window_long).mean()
        
        # Generate trend signals based on SMA crossovers
        data['trend_signal'] = 0
        data.loc[data['SMA_short'] > data['SMA_long'], 'trend_signal'] = 1  # Bullish
        data.loc[data['SMA_short'] < data['SMA_long'], 'trend_signal'] = -1  # Bearish
        
        # Detect trend changes for potential entry/exit points
        data['trend_change'] = data['trend_signal'].diff().fillna(0)
        
        # Calculate RSI - common in crypto analysis
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        return data
    
    def _keyword_sentiment_analysis(self, title):
        """
        Analyzes the sentiment of a news title based on keywords.
        
        Args:
            title (str): News title to analyze
            
        Returns:
            str: 'positive', 'negative', or 'neutral' sentiment
        """
        if not title:
            return 'neutral'
            
        title = title.lower()
        
        # Count positive and negative words
        positive_count = 0
        negative_count = 0
        
        # Check for positive words
        for keyword in self.positive_keywords:
            if keyword in title:
                positive_count += 1
                
        # Check for negative words
        for keyword in self.negative_keywords:
            if keyword in title:
                negative_count += 1
        
        # Determine sentiment based on word count
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_news_sentiment(self, news_data, llm_client=None):
        """
        Analyzes sentiment of crypto news articles.
        
        Args:
            news_data (list): List of news items, each containing at least a 'title' key
            llm_client (object, optional): Client for LLM-based sentiment analysis
            
        Returns:
            list: Original news_data with added 'sentiment' field for each item
        """
        if not news_data or len(news_data) == 0:
            return []
        
        # If LLM client is provided, use it for sentiment analysis
        if llm_client:
            for item in news_data:
                title = item.get('title', '')
                sentiment = llm_client.analyze_sentiment(title)
                if sentiment:
                    item['sentiment'] = sentiment
                else:
                    # Fallback to simple keyword-based analysis
                    item['sentiment'] = self._keyword_sentiment_analysis(title)
            return news_data
        
        # Simple keyword-based sentiment analysis as fallback
        for item in news_data:
            if 'sentiment' not in item:  # Skip if sentiment already exists
                title = item.get('title', '').lower()
                item['sentiment'] = self._keyword_sentiment_analysis(title)
        
        return news_data