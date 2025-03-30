import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os
import time

class EventCalendar:
    def __init__(self):
        # Set up cache directory for storing event data
        self.cache_dir = os.path.join('data', 'crypto_events')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Base URL for CoinGecko API
        self.base_url = "https://api.coingecko.com/api/v3"
        
        # Track API calls for rate limiting
        self.api_calls = 0
        self.api_reset_time = datetime.now()
        self.api_limit = 50  # CoinGecko allows ~50 calls/minute in the free plan
        
    def get_crypto_events(self, crypto_id=None, days_ahead=60, force_refresh=False):
        """
        Fetches upcoming cryptocurrency events from CoinGecko or generates simulated events.
        
        Args:
            crypto_id (str, optional): Cryptocurrency ID (e.g. 'bitcoin', 'ethereum')
            days_ahead (int): Number of days ahead for which events are fetched
            force_refresh (bool): Whether to force a refresh of data from the API
            
        Returns:
            pandas.DataFrame: DataFrame containing event data
        """
        # Define cache filename based on input parameters
        cache_filename = f"events_{crypto_id or 'all'}_{days_ahead}.csv"
        cache_file = os.path.join(self.cache_dir, cache_filename)
        
        # Check if we can use cached data
        if os.path.exists(cache_file) and not force_refresh:
            # Check if data is older than 24 hours
            file_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - file_time < 86400:  # 24h in seconds
                return pd.read_csv(cache_file, parse_dates=['date'])
        
        try:
            # Check API limit
            current_time = datetime.now()
            # Reset counter every minute
            if (current_time - self.api_reset_time).total_seconds() >= 60:
                self.api_calls = 0
                self.api_reset_time = current_time
                
            # If we're approaching the limit, wait a moment
            if self.api_calls >= self.api_limit:
                print(f"API call limit reached, using cached or simulated data for events")
                if os.path.exists(cache_file):
                    return pd.read_csv(cache_file, parse_dates=['date'])
                else:
                    return self.generate_mock_events(crypto_id, days_ahead)
            
            # Prepare list of all events
            all_events = []
            
            # Get events for a specific cryptocurrency or for all major cryptos
            if crypto_id:
                coin_list = [crypto_id]
            else:
                # Get list of most important cryptocurrencies
                self.api_calls += 1
                response = requests.get(f"{self.base_url}/coins/markets", params={
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': 25,
                    'page': 1
                })
                
                if response.status_code == 200:
                    top_coins = response.json()
                    coin_list = [coin['id'] for coin in top_coins]
                else:
                    print(f"Error fetching coin list: {response.status_code}")
                    coin_list = ['bitcoin', 'ethereum', 'solana', 'cardano', 'dogecoin']
            
            # For each cryptocurrency, get details
            for coin in coin_list:
                # Check API limit
                if self.api_calls >= self.api_limit:
                    break
                    
                self.api_calls += 1
                
                # Get coin data
                response = requests.get(f"{self.base_url}/coins/{coin}", params={
                    'localization': 'false',
                    'tickers': 'false',
                    'market_data': 'false',
                    'community_data': 'false',
                    'developer_data': 'true'  # Contains development information
                })
                
                if response.status_code == 200:
                    coin_data = response.json()
                    
                    # Extract name and symbol
                    coin_name = coin_data.get('name', '')
                    coin_symbol = coin_data.get('symbol', '').upper()
                    
                    # Check upcoming release dates
                    if 'developer_data' in coin_data and 'upcoming_release' in coin_data['developer_data']:
                        release_date = coin_data['developer_data'].get('upcoming_release')
                        if release_date:
                            try:
                                event_date = datetime.strptime(release_date, '%Y-%m-%d')
                                # Check if the date is within range
                                if (event_date - datetime.now()).days <= days_ahead:
                                    event = {
                                        "project_id": coin,
                                        "project_name": coin_name,
                                        "symbol": coin_symbol,
                                        "category": "Development",
                                        "date": event_date,
                                        "event_type": "Release",
                                        "description": f"Scheduled release for {coin_name}",
                                        "importance": "high",
                                        "potential_impact": "positive",
                                        "source": "CoinGecko"
                                    }
                                    all_events.append(event)
                            except:
                                pass
                    
                    # Get project and currency statuses
                    statuses = coin_data.get('status_updates', [])
                    for status in statuses:
                        try:
                            update_time = datetime.fromtimestamp(status.get('created_at', 0) / 1000)
                            # Check if the status is current
                            if (datetime.now() - update_time).days <= 30:  # Last 30 days
                                category = status.get('category', '')
                                description = status.get('description', '')
                                
                                # Create an event for important updates
                                if category in ['general', 'milestone', 'partnership', 'exchange', 'release']:
                                    importance = 'high' if category in ['milestone', 'release'] else 'medium'
                                    event = {
                                        "project_id": coin,
                                        "project_name": coin_name,
                                        "symbol": coin_symbol,
                                        "category": category.capitalize(),
                                        "date": update_time + timedelta(days=np.random.randint(1, days_ahead)),  # Simulate future date
                                        "event_type": category.capitalize(),
                                        "description": description,
                                        "importance": importance,
                                        "potential_impact": "positive" if category in ['milestone', 'partnership', 'release'] else "neutral",
                                        "source": "CoinGecko"
                                    }
                                    all_events.append(event)
                        except:
                            continue
            
            # If we don't have enough events, supplement with simulated ones
            if len(all_events) < 10:
                mock_events = self.generate_mock_events(crypto_id, days_ahead).to_dict('records')
                all_events.extend(mock_events)
                
            # Create DataFrame and sort by date
            events_df = pd.DataFrame(all_events)
            events_df = events_df.sort_values(by="date")
            
            # Save to cache
            events_df.to_csv(cache_file, index=False)
            
            return events_df
                
        except Exception as e:
            print(f"Error while fetching events: {str(e)}")
            # In case of error, use simulated data
            events = self.generate_mock_events(crypto_id, days_ahead)
            events.to_csv(cache_file, index=False)
            return events
    
    def generate_mock_events(self, crypto_id=None, days_ahead=60):
        """
        Generates realistic simulated events for the cryptocurrency calendar.
        
        Args:
            crypto_id (str, optional): Specific cryptocurrency to generate events for
            days_ahead (int): Number of days ahead to generate events
            
        Returns:
            pandas.DataFrame: DataFrame containing simulated event data
        """
        # Common crypto projects
        projects = {
            "bitcoin": {"name": "Bitcoin", "symbol": "BTC", "category": "Layer 1"},
            "ethereum": {"name": "Ethereum", "symbol": "ETH", "category": "Layer 1"},
            "solana": {"name": "Solana", "symbol": "SOL", "category": "Layer 1"},
            "cardano": {"name": "Cardano", "symbol": "ADA", "category": "Layer 1"},
            "dogecoin": {"name": "Dogecoin", "symbol": "DOGE", "category": "Meme Coin"},
            "ripple": {"name": "Ripple", "symbol": "XRP", "category": "Payment"},
            "polkadot": {"name": "Polkadot", "symbol": "DOT", "category": "Interoperability"},
            "avalanche": {"name": "Avalanche", "symbol": "AVAX", "category": "Layer 1"},
            "chainlink": {"name": "Chainlink", "symbol": "LINK", "category": "Oracle"},
            "uniswap": {"name": "Uniswap", "symbol": "UNI", "category": "DEX"}
        }
        
        # Common event types for crypto
        event_types = [
            {"type": "Hard Fork", "importance": "high", "description_template": "{name} network upgrade to improve {feature}"},
            {"type": "Mainnet Launch", "importance": "high", "description_template": "{name} Mainnet launch after successful testnet phase"},
            {"type": "Halving", "importance": "high", "description_template": "{name} mining reward halving event"},
            {"type": "Airdrop", "importance": "medium", "description_template": "{name} token airdrop to community members"},
            {"type": "Token Burn", "importance": "medium", "description_template": "Scheduled {name} token burn event"},
            {"type": "Conference", "importance": "medium", "description_template": "{name} annual developer conference"},
            {"type": "AMA Session", "importance": "low", "description_template": "Ask Me Anything session with {name} core team"},
            {"type": "Listing", "importance": "medium", "description_template": "{name} listing on major exchange"},
            {"type": "Partnership", "importance": "medium", "description_template": "{name} announces strategic partnership"},
            {"type": "Protocol Update", "importance": "medium", "description_template": "{name} protocol update to version {version}"},
            {"type": "Testnet Launch", "importance": "medium", "description_template": "{name} Testnet launch for new features"},
            {"type": "Whitepaper Release", "importance": "medium", "description_template": "{name} whitepaper 2.0 release"}
        ]
        
        # Features for project upgrades
        features = ["scalability", "security", "interoperability", "governance", "user experience", 
                   "smart contracts", "privacy", "token economics", "consensus mechanism"]
        
        # Conference locations
        locations = ["San Francisco", "New York", "London", "Singapore", "Tokyo", "Dubai", 
                    "Berlin", "Seoul", "Hong Kong", "Amsterdam", "Paris", "Miami"]
        
        # Generate events
        events = []
        
        # Set the seed for random number generation to ensure deterministic results
        if crypto_id:
            np.random.seed(hash(crypto_id) % 10000)
        else:
            np.random.seed(42)  # Fixed seed for deterministic output
        
        # Current date
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Filter projects if crypto_id is specified
        if crypto_id and crypto_id in projects:
            project_list = {crypto_id: projects[crypto_id]}
        else:
            project_list = projects
        
        # Number of events to generate (more for 'all' than for specific crypto)
        num_events = 15 if crypto_id else 50
        
        for _ in range(num_events):
            # Random project
            project_id = np.random.choice(list(project_list.keys()))
            project = project_list[project_id]
            
            # Random event type
            event_type_info = event_types[np.random.randint(0, len(event_types))]
            event_type = event_type_info["type"]
            importance = event_type_info["importance"]
            
            # Random date within range
            days_forward = np.random.randint(1, days_ahead)
            event_date = today + timedelta(days=days_forward)
            
            # Format description based on template
            description_template = event_type_info["description_template"]
            
            # Generate version number for protocol updates
            version = f"{np.random.randint(1, 5)}.{np.random.randint(0, 10)}.{np.random.randint(0, 10)}"
            
            # Select random feature for upgrades
            feature = np.random.choice(features)
            
            # Select random location for conferences
            location = np.random.choice(locations)
            
            # Format description
            description = description_template.format(
                name=project["name"], 
                version=version,
                feature=feature
            )
            
            # Add location for conferences
            if event_type == "Conference":
                description += f" in {location}"
            
            # Potential price impact
            if importance == "high":
                price_impact = np.random.choice(["positive", "neutral", "negative"], p=[0.7, 0.2, 0.1])
            elif importance == "medium":
                price_impact = np.random.choice(["positive", "neutral", "negative"], p=[0.5, 0.4, 0.1])
            else:
                price_impact = np.random.choice(["positive", "neutral", "negative"], p=[0.3, 0.6, 0.1])
            
            # Create event
            event = {
                "project_id": project_id,
                "project_name": project["name"],
                "symbol": project["symbol"],
                "category": project["category"],
                "date": event_date,
                "event_type": event_type,
                "description": description,
                "importance": importance,
                "potential_impact": price_impact,
                "source": "CryptoText Analytics (Mock Data)"
            }
            
            events.append(event)
        
        # Sort by date
        events_df = pd.DataFrame(events)
        events_df = events_df.sort_values(by="date")
        
        return events_df
        
    def get_notable_past_events(self, days_back=30):
        """
        Fetches notable past events from recent history.
        
        Args:
            days_back (int): Number of days back to fetch events for
            
        Returns:
            pandas.DataFrame: DataFrame containing past event data
        """
        # Define cache filename
        cache_file = os.path.join(self.cache_dir, f"past_events_{days_back}.csv")
        
        # Check if we can use cached data
        if os.path.exists(cache_file):
            # Check if data is less than 24 hours old
            file_time = os.path.getmtime(cache_file)
            if datetime.now().timestamp() - file_time < 86400:  # 24h in seconds
                return pd.read_csv(cache_file, parse_dates=['date'])
        
        # For now, we'll use mock data
        events = self.generate_mock_past_events(days_back)
        
        # Save to cache
        events.to_csv(cache_file, index=False)
        
        return events
        
    def generate_mock_past_events(self, days_back=30):
        """
        Generates realistic mock past events for cryptocurrency calendar.
        
        Args:
            days_back (int): Number of days back to generate events
             
        Returns:
            pandas.DataFrame: DataFrame containing mock past event data
        """
        # Use the same structure as generate_mock_events but with past dates
        # and add actual_impact field based on price changes
        
        # Set the seed for random number generation
        np.random.seed(123)  # Different seed from future events
        
        # Get mock future events as a base and modify them
        mock_events = self.generate_mock_events(None, days_back)
        
        # Current date
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Modify dates to be in the past
        past_events = []
        
        for _, event in mock_events.iterrows():
            # Random date within range (past)
            days_backward = np.random.randint(1, days_back)
            event_date = today - timedelta(days=days_backward)
            
            # Copy the event and modify
            past_event = event.to_dict()
            past_event["date"] = event_date
            
            # Add actual impact based on potential impact (with some randomness)
            potential = past_event["potential_impact"]
            
            if potential == "positive":
                actual = np.random.choice(["positive", "neutral", "negative"], p=[0.7, 0.2, 0.1])
            elif potential == "neutral":
                actual = np.random.choice(["positive", "neutral", "negative"], p=[0.3, 0.4, 0.3])
            else:
                actual = np.random.choice(["positive", "neutral", "negative"], p=[0.1, 0.2, 0.7])
                
            past_event["actual_impact"] = actual
            
            # Add price change percentage based on actual impact
            if actual == "positive":
                price_change = np.random.uniform(2.0, 15.0)
            elif actual == "neutral":
                price_change = np.random.uniform(-2.0, 2.0)
            else:
                price_change = np.random.uniform(-15.0, -2.0)
                
            past_event["price_change"] = price_change
            
            past_events.append(past_event)
        
        # Sort by date
        past_events_df = pd.DataFrame(past_events)
        past_events_df = past_events_df.sort_values(by="date")
        
        return past_events_df