import pandas as pd
import numpy as np
import os
import time
import requests
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv

class CryptoDataLoader:
    def __init__(self):
        # Set up cache directory for storing data
        self.cache_dir = os.path.join('data', 'crypto_data')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Base URL for CoinGecko API
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # API call tracking for rate limiting
        self.api_calls = 0
        self.api_reset_time = datetime.now()
        self.api_limit = 50  # CoinGecko allows ~50 calls/minute for free tier
    
    def get_crypto_data(self, crypto_id, days="30", interval="daily", force_refresh=False, source="coingecko"):
        """Fetches historical data for a given cryptocurrency ID"""
        cache_file = os.path.join(self.cache_dir, f"{crypto_id}_{days}_{interval}_{source}.csv")
        
        # Check if we can use cached data
        if os.path.exists(cache_file) and not force_refresh:
            # Check if data is less than 24 hours old
            file_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - file_time < 86400:  # 24h in seconds
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        
        # Check API rate limit
        current_time = datetime.now()
        # Reset counter every minute
        if (current_time - self.api_reset_time).total_seconds() >= 60:
            self.api_calls = 0
            self.api_reset_time = current_time
            
        # If we're approaching the limit, wait a bit
        if self.api_calls >= self.api_limit:
            print(f"API call limit reached, using cached data or mock data for {crypto_id}")
            if os.path.exists(cache_file):
                return pd.read_csv(cache_file, index_col=0, parse_dates=True)
            else:
                return self.get_mock_data(crypto_id, days)
                
        # Increment API call counter
        self.api_calls += 1
        
        # CoinGecko API
        if source == "coingecko" or source == "auto":
            try:
                # Construct API URL
                url = f"{self.base_url}/coins/{crypto_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': interval
                }
                
                # Make API request
                response = requests.get(url, params=params)
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    
                    # Transform data into pandas DataFrame
                    prices = data.get('prices', [])
                    market_caps = data.get('market_caps', [])
                    total_volumes = data.get('total_volumes', [])
                    
                    if not prices:
                        print(f"No price data available for {crypto_id}")
                        if source == "auto":
                            return self.get_mock_data(crypto_id, days)
                        return pd.DataFrame()
                    
                    # Create DataFrame
                    df = pd.DataFrame()
                    df['timestamp'] = [datetime.fromtimestamp(entry[0]/1000) for entry in prices]
                    df['price'] = [entry[1] for entry in prices]
                    
                    # Calculate OHLC from daily prices if interval is daily
                    if interval == "daily":
                        # Group by date and calculate OHLC
                        df['date'] = df['timestamp'].dt.date
                        ohlc = df.groupby('date').agg(
                            Open=('price', 'first'),
                            High=('price', 'max'),
                            Low=('price', 'min'),
                            Close=('price', 'last'),
                        ).reset_index()
                        
                        # Add volume and market cap
                        volumes = {datetime.fromtimestamp(entry[0]/1000).date(): entry[1] for entry in total_volumes}
                        market_caps_dict = {datetime.fromtimestamp(entry[0]/1000).date(): entry[1] for entry in market_caps}
                        
                        ohlc['Volume'] = ohlc['date'].map(volumes)
                        ohlc['MarketCap'] = ohlc['date'].map(market_caps_dict)
                        
                        # Set date as index
                        ohlc['date'] = pd.to_datetime(ohlc['date'])
                        ohlc.set_index('date', inplace=True)
                        
                        # Save to cache
                        ohlc.to_csv(cache_file)
                        return ohlc
                    else:
                        # For hourly data, just return price and volume
                        df.set_index('timestamp', inplace=True)
                        df.rename(columns={'price': 'Close'}, inplace=True)
                        
                        # Add volume and market cap if available
                        if total_volumes:
                            volume_df = pd.DataFrame({
                                'timestamp': [datetime.fromtimestamp(entry[0]/1000) for entry in total_volumes],
                                'Volume': [entry[1] for entry in total_volumes]
                            })
                            volume_df.set_index('timestamp', inplace=True)
                            df = df.join(volume_df)
                        
                        if market_caps:
                            mcap_df = pd.DataFrame({
                                'timestamp': [datetime.fromtimestamp(entry[0]/1000) for entry in market_caps],
                                'MarketCap': [entry[1] for entry in market_caps]
                            })
                            mcap_df.set_index('timestamp', inplace=True)
                            df = df.join(mcap_df)
                        
                        # Save to cache
                        df.to_csv(cache_file)
                        return df
                else:
                    print(f"Error fetching data: {response.status_code} - {response.text}")
                    if source == "auto":
                        return self.get_mock_data(crypto_id, days)
                    elif os.path.exists(cache_file):
                        return pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    return pd.DataFrame()
                    
            except Exception as e:
                print(f"Error fetching data from CoinGecko for {crypto_id}: {e}")
                if source == "auto":
                    return self.get_mock_data(crypto_id, days)
                elif os.path.exists(cache_file):
                    return pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return pd.DataFrame()
        
        # Mock data as fallback
        elif source == "mock":
            return self.get_mock_data(crypto_id, days)
        
        else:
            print(f"Unknown data source: {source}")
            return pd.DataFrame()
    
    def get_mock_data(self, crypto_id, days):
        """Generate mock data for demonstrations"""
        print(f"Generating mock data for {crypto_id}")
        
        # Convert days to integer
        try:
            num_days = int(days)
        except:
            num_days = 30  # Default to 30 days
            
        # Generate date range
        dates = pd.date_range(end=datetime.now(), periods=num_days)
        
        # Generate mock price data with some volatility based on crypto_id
        # Use crypto_id as seed for consistency
        np.random.seed(hash(crypto_id) % 10000)
        
        # Base price depends on crypto (to make them look different)
        if crypto_id == "bitcoin":
            base_price = 50000
            volatility = 0.03
        elif crypto_id == "ethereum":
            base_price = 3000
            volatility = 0.04
        elif crypto_id == "solana":
            base_price = 100
            volatility = 0.06
        elif crypto_id == "cardano":
            base_price = 2
            volatility = 0.05
        elif crypto_id == "dogecoin":
            base_price = 0.1
            volatility = 0.08
        else:
            base_price = 10
            volatility = 0.05
            
        # Generate price series with random walk
        returns = np.random.normal(0, volatility, num_days)
        price_series = base_price * (1 + np.cumsum(returns))
        
        # Create dataframe with OHLC data
        data = pd.DataFrame({
            'Open': price_series * np.random.uniform(0.98, 1.0, num_days),
            'High': price_series * np.random.uniform(1.01, 1.05, num_days),
            'Low': price_series * np.random.uniform(0.95, 0.99, num_days),
            'Close': price_series,
            'Volume': np.random.randint(1000000, 100000000, num_days),
            'MarketCap': price_series * np.random.randint(10000000, 1000000000, num_days)
        }, index=dates)
        
        return data
    
    def get_crypto_info(self, crypto_id):
        """Fetches information about a cryptocurrency"""
        cache_file = os.path.join(self.cache_dir, f"{crypto_id}_info.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            # Check if data is less than 24 hours old
            file_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - file_time < 86400:  # 24h in seconds
                try:
                    return pd.read_json(cache_file, typ='series')
                except:
                    pass
        
        # Check API rate limit
        current_time = datetime.now()
        if (current_time - self.api_reset_time).total_seconds() >= 60:
            self.api_calls = 0
            self.api_reset_time = current_time
            
        # If we're approaching the limit, use mock data
        if self.api_calls >= self.api_limit:
            return self.get_mock_crypto_info(crypto_id)
                
        # Increment API call counter
        self.api_calls += 1
        
        try:
            # Make API request
            url = f"{self.base_url}/coins/{crypto_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'market_data': 'true',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                info = {
                    'id': data.get('id', ''),
                    'symbol': data.get('symbol', '').upper(),
                    'name': data.get('name', ''),
                    'description': data.get('description', {}).get('en', ''),
                    'homepage': data.get('links', {}).get('homepage', [''])[0],
                    'github': data.get('links', {}).get('repos_url', {}).get('github', ['']),
                    'reddit': data.get('links', {}).get('subreddit_url', ''),
                    'twitter': data.get('links', {}).get('twitter_screen_name', ''),
                    'market_cap_rank': data.get('market_cap_rank', 0),
                    'current_price': data.get('market_data', {}).get('current_price', {}).get('usd', 0),
                    'market_cap': data.get('market_data', {}).get('market_cap', {}).get('usd', 0),
                    'total_volume': data.get('market_data', {}).get('total_volume', {}).get('usd', 0),
                    'high_24h': data.get('market_data', {}).get('high_24h', {}).get('usd', 0),
                    'low_24h': data.get('market_data', {}).get('low_24h', {}).get('usd', 0),
                    'price_change_24h': data.get('market_data', {}).get('price_change_24h', 0),
                    'price_change_percentage_24h': data.get('market_data', {}).get('price_change_percentage_24h', 0),
                    'circulating_supply': data.get('market_data', {}).get('circulating_supply', 0),
                    'total_supply': data.get('market_data', {}).get('total_supply', 0),
                    'max_supply': data.get('market_data', {}).get('max_supply', 0),
                    'image': data.get('image', {}).get('large', '')
                }
                
                # Save to cache
                pd.Series(info).to_json(cache_file)
                
                return pd.Series(info)
            else:
                print(f"Error fetching info: {response.status_code} - {response.text}")
                return self.get_mock_crypto_info(crypto_id)
                
        except Exception as e:
            print(f"Error fetching info for {crypto_id}: {e}")
            return self.get_mock_crypto_info(crypto_id)
    
    def get_mock_crypto_info(self, crypto_id):
        """Generate mock cryptocurrency information"""
        if crypto_id == "bitcoin":
            return pd.Series({
                'id': 'bitcoin',
                'symbol': 'BTC',
                'name': 'Bitcoin',
                'description': 'Bitcoin is the first and most well-known cryptocurrency, created in 2009 by an unknown person using the pseudonym Satoshi Nakamoto.',
                'homepage': 'https://bitcoin.org/',
                'market_cap_rank': 1,
                'current_price': 50000,
                'market_cap': 950000000000.0,
                'total_volume': 30000000000.0,
                'high_24h': 51000,
                'low_24h': 49000,
                'price_change_24h': 1000,
                'price_change_percentage_24h': 2.0,
                'circulating_supply': 19000000,
                'total_supply': 21000000,
                'max_supply': 21000000
            })
        elif crypto_id == "ethereum":
            return pd.Series({
                'id': 'ethereum',
                'symbol': 'ETH',
                'name': 'Ethereum',
                'description': 'Ethereum is a decentralized, open-source blockchain with smart contract functionality.',
                'homepage': 'https://ethereum.org/',
                'market_cap_rank': 2,
                'current_price': 3000,
                'market_cap': 360000000000.0,
                'total_volume': 15000000000.0,
                'high_24h': 3100,
                'low_24h': 2900,
                'price_change_24h': 100,
                'price_change_percentage_24h': 3.3,
                'circulating_supply': 120000000,
                'total_supply': None,
                'max_supply': None
            })
        elif crypto_id == "solana":
            return pd.Series({
                'id': 'solana',
                'symbol': 'SOL',
                'name': 'Solana',
                'description': 'Solana is a high-performance blockchain supporting builders around the world creating crypto apps.',
                'homepage': 'https://solana.com/',
                'market_cap_rank': 5,
                'current_price': 100,
                'market_cap': 40000000000.0,
                'total_volume': 2000000000.0,
                'high_24h': 105,
                'low_24h': 95,
                'price_change_24h': 5,
                'price_change_percentage_24h': 5.0,
                'circulating_supply': 400000000,
                'total_supply': 500000000,
                'max_supply': None
            })
        elif crypto_id == "cardano":
            return pd.Series({
                'id': 'cardano',
                'symbol': 'ADA',
                'name': 'Cardano',
                'description': 'Cardano is a proof-of-stake blockchain platform that says its goal is to allow "changemakers, innovators and visionaries" to bring about positive global change.',
                'homepage': 'https://cardano.org/',
                'market_cap_rank': 8,
                'current_price': 1.2,
                'market_cap': 40000000000.0,
                'total_volume': 1500000000.0,
                'high_24h': 1.25,
                'low_24h': 1.15,
                'price_change_24h': 0.05,
                'price_change_percentage_24h': 4.3,
                'circulating_supply': 33000000000,
                'total_supply': 45000000000,
                'max_supply': 45000000000
            })
        elif crypto_id == "dogecoin":
            return pd.Series({
                'id': 'dogecoin',
                'symbol': 'DOGE',
                'name': 'Dogecoin',
                'description': 'Dogecoin is a cryptocurrency that was created as a joke in 2013, based on a popular meme featuring a Shiba Inu dog.',
                'homepage': 'https://dogecoin.com/',
                'market_cap_rank': 10,
                'current_price': 0.15,
                'market_cap': 20000000000.0,
                'total_volume': 1000000000.0,
                'high_24h': 0.16,
                'low_24h': 0.14,
                'price_change_24h': 0.01,
                'price_change_percentage_24h': 7.1,
                'circulating_supply': 132000000000,
                'total_supply': None,
                'max_supply': None
            })
        else:
            # Use smaller values for random data and convert to float to avoid int32 bounds issues
            return pd.Series({
                'id': crypto_id,
                'symbol': crypto_id.upper()[:4],
                'name': crypto_id.capitalize(),
                'description': f'This is a mock description for {crypto_id.capitalize()}.',
                'homepage': f'https://{crypto_id}.org',
                'market_cap_rank': int(np.random.randint(10, 100)),
                'current_price': float(np.random.uniform(1, 1000)),
                'market_cap': float(np.random.randint(10000000, 999999999)),
                'total_volume': float(np.random.randint(1000000, 99999999)),
                'high_24h': float(np.random.uniform(1, 1000) * 1.05),
                'low_24h': float(np.random.uniform(1, 1000) * 0.95),
                'price_change_24h': float(np.random.uniform(-100, 100)),
                'price_change_percentage_24h': float(np.random.uniform(-10, 10)),
                'circulating_supply': float(np.random.randint(1000000, 99999999)),
                'total_supply': float(np.random.randint(1000000, 99999999)),
                'max_supply': float(np.random.randint(1000000, 99999999))
            })
    
    def get_trending_cryptos(self):
        """Get top trending cryptocurrencies"""
        cache_file = os.path.join(self.cache_dir, "trending.json")
        
        # Check cache first
        if os.path.exists(cache_file):
            # Check if data is less than 1 hour old
            file_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - file_time < 3600:  # 1h in seconds
                try:
                    return pd.read_json(cache_file)
                except:
                    pass
        
        # Check API rate limit
        current_time = datetime.now()
        if (current_time - self.api_reset_time).total_seconds() >= 60:
            self.api_calls = 0
            self.api_reset_time = current_time
            
        # If we're approaching the limit, use mock data
        if self.api_calls >= self.api_limit:
            return self.get_mock_trending()
                
        # Increment API call counter
        self.api_calls += 1
        
        try:
            # Make API request
            url = f"{self.base_url}/search/trending"
            
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract coins from data
                coins = data.get('coins', [])
                
                if not coins:
                    return self.get_mock_trending()
                
                # Transform to dataframe
                trending = []
                for coin in coins:
                    item = coin.get('item', {})
                    trending.append({
                        'id': item.get('id', ''),
                        'name': item.get('name', ''),
                        'symbol': item.get('symbol', ''),
                        'market_cap_rank': item.get('market_cap_rank', 0),
                        'price_btc': item.get('price_btc', 0),
                        'score': item.get('score', 0),
                        'thumb': item.get('thumb', '')
                    })
                
                trending_df = pd.DataFrame(trending)
                
                # Save to cache
                trending_df.to_json(cache_file)
                
                return trending_df
            else:
                print(f"Error fetching trending: {response.status_code} - {response.text}")
                return self.get_mock_trending()
                
        except Exception as e:
            print(f"Error fetching trending cryptocurrencies: {e}")
            return self.get_mock_trending()
    
    def get_mock_trending(self):
        """Generate mock trending cryptocurrencies"""
        trending = [
            {'id': 'bitcoin', 'name': 'Bitcoin', 'symbol': 'BTC', 'market_cap_rank': 1, 'price_btc': 1.0, 'score': 0},
            {'id': 'ethereum', 'name': 'Ethereum', 'symbol': 'ETH', 'market_cap_rank': 2, 'price_btc': 0.06, 'score': 1},
            {'id': 'solana', 'name': 'Solana', 'symbol': 'SOL', 'market_cap_rank': 5, 'price_btc': 0.002, 'score': 2},
            {'id': 'cardano', 'name': 'Cardano', 'symbol': 'ADA', 'market_cap_rank': 7, 'price_btc': 0.0001, 'score': 3},
            {'id': 'dogecoin', 'name': 'Dogecoin', 'symbol': 'DOGE', 'market_cap_rank': 10, 'price_btc': 0.00001, 'score': 4},
            {'id': 'polkadot', 'name': 'Polkadot', 'symbol': 'DOT', 'market_cap_rank': 13, 'price_btc': 0.0004, 'score': 5},
            {'id': 'shiba-inu', 'name': 'Shiba Inu', 'symbol': 'SHIB', 'market_cap_rank': 15, 'price_btc': 0.0000001, 'score': 6}
        ]
        
        return pd.DataFrame(trending)
    
    def get_crypto_news(self, crypto_id, limit=5):
        """
        Return mock news about cryptocurrency since CoinGecko doesn't provide news API
        """
        news_items = []
        
        # Generate pseudo-random news based on crypto id
        np.random.seed(hash(crypto_id) % 10000)
        
        # Common news templates
        templates = [
            "{0} Price Surges {1}% as Market Shows Bullish Momentum",
            "{0} Announces Major Partnership with {2}",
            "Analysts Predict {0} Could Reach ${3} by End of Year",
            "{0} Network Upgrade Scheduled for Next Month",
            "Major Exchange Lists {0}, Price Jumps {1}%",
            "{0} Foundation Launches $100M Development Fund",
            "Is {0} the Next Big Thing in Crypto? Experts Weigh In",
            "{0} Trading Volume Hits All-Time High of ${4}M",
            "New {0} Use Case Emerges in {5} Industry",
            "Whale Moves ${4}M Worth of {0}, Market Reacts"
        ]
        
        companies = ["Microsoft", "Amazon", "Google", "Meta", "Apple", "IBM", "Oracle", "Samsung", "Visa", "MasterCard"]
        industries = ["Finance", "Healthcare", "Gaming", "Supply Chain", "Social Media", "Entertainment", "Banking", "Insurance"]
        
        # Actual crypto news domains
        news_domains = [
            "coindesk.com",
            "cointelegraph.com",
            "cryptonews.com",
            "decrypt.co",
            "bitcoinmagazine.com",
            "newsbtc.com",
            "cryptoslate.com",
            "cryptobriefing.com"
        ]
        
        for i in range(limit):
            template = templates[np.random.randint(0, len(templates))]
            company = companies[np.random.randint(0, len(companies))]
            industry = industries[np.random.randint(0, len(industries))]
            price_change = np.random.randint(5, 30)
            price_target = np.random.randint(10, 100000)
            volume = np.random.randint(10, 500)
            
            # Format the news title
            if crypto_id == "bitcoin":
                title = template.format("Bitcoin", price_change, company, price_target, volume, industry)
                symbol = "BTC"
            elif crypto_id == "ethereum":
                title = template.format("Ethereum", price_change, company, price_target, volume, industry)
                symbol = "ETH"
            elif crypto_id == "solana":
                title = template.format("Solana", price_change, company, price_target, volume, industry)
                symbol = "SOL"
            elif crypto_id == "cardano":
                title = template.format("Cardano", price_change, company, price_target, volume, industry)
                symbol = "ADA"
            elif crypto_id == "dogecoin":
                title = template.format("Dogecoin", price_change, company, price_target, volume, industry)
                symbol = "DOGE"
            else:
                title = template.format(crypto_id.capitalize(), price_change, company, price_target, volume, industry)
                symbol = crypto_id.upper()[:4]
            
            # Generate publish time (within last 7 days)
            days_ago = np.random.randint(0, 7)
            hours_ago = np.random.randint(0, 24)
            publish_time = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
            
            # Format date for URL
            date_str = publish_time.strftime("%Y/%m/%d")
            
            # Generate a more realistic URL with date path and slug
            domain = news_domains[np.random.randint(0, len(news_domains))]
            slug = title.lower().replace(" ", "-").replace("$", "").replace("%", "percent")
            # Limit slug length and remove special characters
            slug = re.sub(r'[^a-z0-9-]', '', slug)[:50]
            
            # Create a realistic-looking URL that will work in href
            url = f"https://www.{domain}/news/{date_str}/{slug}-{i}"
            
            # Determine sentiment
            if "Surges" in title or "Jumps" in title or "Next Big Thing" in title or "All-Time High" in title:
                sentiment = "positive"
            elif "Could Reach" in title or "Partnership" in title or "Upgrade" in title or "Launches" in title:
                sentiment = "positive" if np.random.random() > 0.3 else "neutral"
            else:
                sentiment = np.random.choice(["positive", "neutral", "negative"], p=[0.5, 0.3, 0.2])
            
            # Create news item
            news_items.append({
                'title': title,
                'link': url,
                'publisher': domain.split('.')[0].capitalize(),
                'providerPublishTime': int(publish_time.timestamp()),
                'type': "STORY",
                'symbol': symbol,
                'sentiment': sentiment
            })
        
        # Sort by publish time (newest first)
        news_items.sort(key=lambda x: x['providerPublishTime'], reverse=True)
        
        return news_items