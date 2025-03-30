import os
from openai import OpenAI
from dotenv import load_dotenv
import json
from typing import List, Dict, Any, Optional
import hashlib
import random

class LLMClientRAG:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            self.client = None
            print("Warning: OPENAI_API_KEY not found in environment variables")
    
    def generate_response_with_context(self, 
                                     query: str, 
                                     context_documents: List[Dict[str, Any]], 
                                     crypto_info: Dict[str, Any] = None,
                                     max_tokens: int = 800,
                                     temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generates a response to the user's question using context from documents
        
        Args:
            query: User's question
            context_documents: List of contextual documents from search
            crypto_info: Optional cryptocurrency information
            max_tokens: Maximum number of tokens in the response
            temperature: Generation temperature (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Dictionary with generated response and sources
        """
        if not self.client:
            return self._mock_rag_response(query, context_documents, crypto_info)
        
        try:
            # Prepare context from documents
            context_text = ""
            sources = []
            
            for i, doc in enumerate(context_documents):
                # Add document to context
                context_snippet = f"\n--- Document {i+1} ---\n"
                
                if 'segment_text' in doc:
                    context_snippet += doc['segment_text']
                else:
                    context_snippet += "Content not available"
                
                context_text += context_snippet
                
                # Add source
                source = {
                    'doc_id': doc['doc_id'],
                    'score': doc.get('score', 0.0),
                }
                
                # Add source metadata
                if 'metadata' in doc:
                    if 'source' in doc['metadata']:
                        source['source'] = doc['metadata']['source']
                    if 'title' in doc['metadata']:
                        source['title'] = doc['metadata']['title']
                    if 'type' in doc['metadata']:
                        source['type'] = doc['metadata']['type']
                    if 'processed_date' in doc['metadata']:
                        source['date'] = doc['metadata']['processed_date']
                
                sources.append(source)
            
            # Add cryptocurrency information
            crypto_context = ""
            if crypto_info:
                crypto_context = f"\n--- Cryptocurrency Information ---\n"
                for key, value in crypto_info.items():
                    crypto_context += f"{key}: {value}\n"
            
            # Prepare prompt
            prompt = f"""
            You are an advanced cryptocurrency expert and financial analyst. 
            Answer the following question based on the provided context documents and cryptocurrency information.
            
            Context documents:
            {context_text}
            
            {crypto_context}
            
            Question: {query}
            
            Please provide a comprehensive and factual answer using only the information in the provided context.
            If the information is not available in the context, say so.
            Format your response in markdown for better readability.
            """
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency expert with access to document information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            
            # Return response with sources
            return {
                'answer': answer,
                'sources': sources
            }
                
        except Exception as e:
            print(f"Error generating RAG response: {e}")
            return {
                'answer': f"Error generating response: {str(e)}",
                'sources': []
            }
    
    def analyze_document_sentiment(self, document_text: str, crypto_symbol: str = None) -> Dict[str, Any]:
        """
        Analyzes the sentiment of a document text for cryptocurrency
        
        Args:
            document_text: Document text to analyze
            crypto_symbol: Cryptocurrency symbol (optional)
            
        Returns:
            Dict[str, Any]: Sentiment analysis result
        """
        if not self.client:
            return self._mock_document_sentiment(document_text, crypto_symbol)
        
        try:
            # Limit document length
            max_length = 4000
            if len(document_text) > max_length:
                document_text = document_text[:max_length] + "... [truncated]"
            
            # Prepare prompt
            prompt = f"""
            Analyze the sentiment of the following document related to cryptocurrency.
            
            Document:
            {document_text}
            
            {f'Focus on the implications for {crypto_symbol}.' if crypto_symbol else ''}
            
            Analyze and provide:
            1. Overall sentiment (positive, negative, or neutral)
            2. Confidence level of your assessment (0-100%)
            3. Key sentiment drivers (main points affecting sentiment)
            4. A one-line summary of the document
            
            Format your response as a JSON object with the following structure:
            {{
                "sentiment": "positive|negative|neutral",
                "confidence": 85,
                "key_drivers": ["point 1", "point 2", "point 3"],
                "summary": "One-line summary of the document."
            }}
            """
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency sentiment analyzer that responds in JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Remove any characters before and after JSON
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                return result
            except Exception as json_error:
                print(f"Error parsing JSON: {json_error}")
                # Fallback - return as plain text
                return {
                    'sentiment': 'neutral' if 'positive' not in result_text.lower() and 'negative' not in result_text.lower() else
                               'positive' if 'positive' in result_text.lower() else 'negative',
                    'confidence': 50,
                    'key_drivers': ["Could not parse response"],
                    'summary': result_text[:100] + "..."
                }
                
        except Exception as e:
            print(f"Error analyzing document sentiment: {e}")
            return self._mock_document_sentiment(document_text, crypto_symbol)
    
    def summarize_document(self, document_text: str, crypto_symbol: str = None) -> str:
        """
        Generates a document summary with cryptocurrency context
        
        Args:
            document_text: Document text to summarize
            crypto_symbol: Cryptocurrency symbol (optional)
            
        Returns:
            str: Document summary
        """
        if not self.client:
            return self._mock_document_summary(document_text, crypto_symbol)
        
        try:
            # Limit document length
            max_length = 6000
            if len(document_text) > max_length:
                document_text = document_text[:max_length] + "... [truncated]"
            
            # Prepare prompt
            prompt = f"""
            Summarize the following document about cryptocurrency.
            
            Document:
            {document_text}
            
            {f'Focus on the information related to {crypto_symbol}.' if crypto_symbol else ''}
            
            Provide a comprehensive but concise summary that captures the main points.
            Format your response in markdown for better readability.
            """
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a cryptocurrency expert that provides concise document summaries."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error summarizing document: {e}")
            return self._mock_document_summary(document_text, crypto_symbol)
    
    def extract_entities(self, document_text: str) -> Dict[str, List[str]]:
        """
        Extracts entities from a document (cryptocurrencies, people, organizations, etc.)
        
        Args:
            document_text: Document text to analyze
            
        Returns:
            Dict[str, List[str]]: Dictionary of entities grouped by types
        """
        if not self.client:
            return self._mock_entities_extraction(document_text)
        
        try:
            # Limit document length
            max_length = 4000
            if len(document_text) > max_length:
                document_text = document_text[:max_length] + "... [truncated]"
            
            # Prepare prompt
            prompt = f"""
            Extract named entities from the following document about cryptocurrency.
            
            Document:
            {document_text}
            
            Extract and categorize entities into the following types:
            1. Cryptocurrencies (e.g., Bitcoin, Ethereum)
            2. People (e.g., Satoshi Nakamoto, Vitalik Buterin)
            3. Organizations (e.g., Binance, Coinbase, Bitcoin Foundation)
            4. Technologies (e.g., blockchain, smart contracts, DeFi)
            5. Events (e.g., Bitcoin halving, crypto conferences)
            
            Format your response as a JSON object with the following structure:
            {{
                "cryptocurrencies": ["Bitcoin", "Ethereum", ...],
                "people": ["Name1", "Name2", ...],
                "organizations": ["Org1", "Org2", ...],
                "technologies": ["Tech1", "Tech2", ...],
                "events": ["Event1", "Event2", ...]
            }}
            Include only the entities that are actually mentioned in the document.
            """
            
            # Generate response
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an entity extraction system that responds in JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Remove any characters before and after JSON
                json_start = result_text.find('{')
                json_end = result_text.rfind('}') + 1
                if json_start >= 0 and json_end > 0:
                    json_str = result_text[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                
                return result
            except Exception as json_error:
                print(f"Error parsing JSON: {json_error}")
                # Fallback - return empty dictionary
                return {
                    'cryptocurrencies': [],
                    'people': [],
                    'organizations': [],
                    'technologies': [],
                    'events': []
                }
                
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return self._mock_entities_extraction(document_text)
    
    # Methods for generating simulated responses (mock)
    def _mock_rag_response(self, query: str, context_documents: List[Dict[str, Any]], crypto_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generates a simulated RAG response when API access is unavailable
        
        Args:
            query: User's question
            context_documents: List of contextual documents
            crypto_info: Cryptocurrency information
            
        Returns:
            Dict[str, Any]: Simulated RAG response
        """
        # Generate deterministic response based on query
        hash_value = int(hashlib.md5(query.encode()).hexdigest(), 16) % 1000
        random.seed(hash_value)
        
        crypto_name = "Bitcoin"
        if crypto_info and 'name' in crypto_info:
            crypto_name = crypto_info['name']
        
        templates = [
            f"# {crypto_name} Analysis\n\nBased on the provided documents, {crypto_name} shows a promising growth trajectory in the coming months. The technical indicators suggest a bullish trend, with resistance levels at key price points. However, market sentiment remains mixed due to regulatory concerns.\n\n## Key Points\n\n- Trading volume has increased by 20% in the last week\n- Social media sentiment is predominantly positive\n- Major institutional investors have shown increased interest\n- Upcoming protocol upgrades may impact price stability",
            
            f"# Market Insights: {crypto_name}\n\nThe analysis of recent market data for {crypto_name} indicates a period of consolidation after recent volatility. Support levels have been tested multiple times but have held firm, suggesting strong market confidence.\n\n## Technical Analysis\n\n- RSI indicates slightly overbought conditions at 72\n- Moving averages show a potential golden cross forming\n- Volume profile confirms the strength of current support levels\n- Fibonacci retracement levels align with historical resistance zones",
            
            f"# {crypto_name} Investment Outlook\n\nCurrently, {crypto_name} presents a moderate investment opportunity with both upside potential and significant risks. The on-chain metrics show healthy network activity, but macroeconomic factors could influence price action in the short term.\n\n## Risk Assessment\n\n- Volatility remains higher than traditional markets but has decreased compared to previous quarters\n- Regulatory developments in key markets pose uncertainty\n- Technology adoption continues at a steady pace\n- Correlation with traditional markets has been decreasing, suggesting more maturity as an asset class"
        ]
        
        answer = random.choice(templates)
        
        # Prepare simulated sources
        sources = []
        for i, doc in enumerate(context_documents[:3]):  # Use maximum of 3 documents
            source = {
                'doc_id': doc.get('doc_id', f"mock_doc_{i}"),
                'score': round(random.uniform(0.7, 0.95), 2),
                'title': doc.get('metadata', {}).get('title', f"Sample Document {i+1}"),
                'type': doc.get('metadata', {}).get('type', 'text'),
                'date': doc.get('metadata', {}).get('processed_date', '2023-01-01')
            }
            sources.append(source)
        
        return {
            'answer': answer,
            'sources': sources
        }
    
    def _mock_document_sentiment(self, document_text: str, crypto_symbol: str = None) -> Dict[str, Any]:
        """
        Generates simulated document sentiment analysis
        
        Args:
            document_text: Document text
            crypto_symbol: Cryptocurrency symbol
            
        Returns:
            Dict[str, Any]: Simulated sentiment analysis
        """
        # Generate deterministic response based on text
        hash_value = int(hashlib.md5(document_text[:100].encode()).hexdigest(), 16) % 1000
        random.seed(hash_value)
        
        # Positive/negative keywords
        positive_words = ["bullish", "surge", "rally", "growth", "gain", "positive", "rise", "soar", "jump", "success"]
        negative_words = ["bearish", "crash", "plunge", "slump", "decline", "drop", "fall", "negative", "decrease", "risk"]
        
        # Check occurrences of positive and negative words
        positive_count = sum(1 for word in positive_words if word in document_text.lower())
        negative_count = sum(1 for word in negative_words if word in document_text.lower())
        
        # Determine sentiment
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = random.randint(65, 95)
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = random.randint(65, 95)
        else:
            sentiment = "neutral"
            confidence = random.randint(55, 75)
        
        # Generate key factors
        all_factors = [
            "Price momentum shows bullish signals",
            "Trading volume has increased significantly",
            "Market sentiment on social media is positive",
            "Technical indicators suggest upward trend",
            "Recent regulatory news affects market negatively",
            "Institutional adoption is growing",
            "Correlation with traditional markets remains high",
            "Volatility has decreased in recent weeks",
            "On-chain metrics indicate healthy network activity",
            "Competitive pressures from alternative cryptocurrencies"
        ]
        
        if sentiment == "positive":
            factors = random.sample(all_factors[:6], k=3)
        elif sentiment == "negative":
            factors = random.sample(all_factors[4:], k=3)
        else:
            factors = random.sample(all_factors, k=3)
        
        # Generate summary
        if crypto_symbol:
            summary_templates = [
                f"{crypto_symbol} shows {sentiment} indicators with {confidence}% confidence based on market metrics and sentiment analysis.",
                f"Analysis indicates {sentiment} outlook for {crypto_symbol} driven by market trends and technical factors.",
                f"Document suggests {sentiment} perspective on {crypto_symbol} with moderate confidence level."
            ]
        else:
            summary_templates = [
                f"Cryptocurrency market displays {sentiment} signals with {confidence}% confidence based on recent trends.",
                f"Analysis indicates {sentiment} market outlook driven by multiple factors and technical indicators.",
                f"Document suggests {sentiment} perspective on cryptocurrency markets with moderate confidence level."
            ]
        
        summary = random.choice(summary_templates)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'key_drivers': factors,
            'summary': summary
        }
    
    def _mock_document_summary(self, document_text: str, crypto_symbol: str = None) -> str:
        """
        Generates a simulated document summary
        
        Args:
            document_text: Document text
            crypto_symbol: Cryptocurrency symbol
            
        Returns:
            str: Simulated summary
        """
        # Generate deterministic response based on text
        hash_value = int(hashlib.md5(document_text[:100].encode()).hexdigest(), 16) % 1000
        random.seed(hash_value)
        
        crypto = crypto_symbol if crypto_symbol else "the cryptocurrency market"
        
        templates = [
            f"""# Summary: Analysis of {crypto.upper()}

## Key Points

1. **Market Performance**: {crypto.upper()} has shown considerable volatility in recent periods, with price fluctuations reflecting broader market sentiment and macroeconomic conditions.

2. **Technical Analysis**: Charts indicate potential support levels forming, with key resistance zones that will be critical for future price movement.

3. **On-Chain Metrics**: Network activity remains robust with increasing adoption rates and transaction volumes suggesting healthy ecosystem development.

4. **Market Sentiment**: Social media and community sentiment appears cautiously optimistic, though regulatory concerns continue to impact investor confidence.

5. **Future Outlook**: Several upcoming developments including protocol upgrades and potential institutional adoption could significantly influence price action in the coming months.

This document provides a comprehensive analysis of current market conditions while acknowledging the inherent uncertainties in cryptocurrency investments.
""",
            f"""# {crypto.upper()} Market Analysis Summary

The document provides an in-depth analysis of {crypto}'s current market position and potential future developments. Several key insights emerge:

- **Price Action**: Recent price movements have established a pattern of higher lows, potentially indicating the formation of an uptrend.
- **Volume Analysis**: Trading volume has shown notable increases during price rallies, confirming buyer interest at key levels.
- **Market Context**: When compared to broader cryptocurrency market trends, {crypto} has demonstrated {random.choice(["stronger", "similar", "slightly weaker"])} correlation with major assets.
- **Risk Factors**: Regulatory uncertainty and macroeconomic pressures remain the primary concerns for investors.
- **Technological Developments**: Upcoming protocol updates promise improved scalability and functionality, which could drive adoption.

Overall, the document presents a {random.choice(["balanced", "cautiously optimistic", "data-driven"])} view of {crypto}'s prospects, emphasizing the importance of risk management while highlighting growth potential.
""",
            f"""# Market Insights: {crypto.upper()}

## Executive Summary

This document analyzes the current state of {crypto} from multiple perspectives including technical analysis, fundamental factors, and market sentiment. The analysis reveals:

### Market Dynamics
- Price action has established a {random.choice(["trading range", "descending channel", "potential accumulation pattern"])} between key support and resistance levels
- Trading volume patterns suggest {random.choice(["accumulation by larger players", "distribution phases concluding", "retail interest decreasing while institutional interest grows"])}

### Fundamental Analysis
- Development activity on the network remains {random.choice(["strong", "consistent", "above average"])}
- Adoption metrics show {random.choice(["steady growth", "increasing interest from new user segments", "potential for expanded use cases"])}

### Sentiment Analysis
- Social media sentiment is trending {random.choice(["positively", "cautiously", "with mixed signals"])}
- Institutional commentary reflects {random.choice(["growing interest", "cautious optimism", "strategic positioning"])}

The document concludes that while short-term volatility remains likely, the longer-term outlook presents compelling opportunities for investors who understand the technological value proposition and can tolerate market fluctuations.
"""
        ]
        
        return random.choice(templates)
    
    def _mock_entities_extraction(self, document_text: str) -> Dict[str, List[str]]:
        """
        Generates simulated entity extraction from a document
        
        Args:
            document_text: Document text
            
        Returns:
            Dict[str, List[str]]: Simulated list of entities
        """
        # Generate deterministic response based on text
        hash_value = int(hashlib.md5(document_text[:100].encode()).hexdigest(), 16) % 1000
        random.seed(hash_value)
        
        # Lists of possible entities
        all_cryptos = ["Bitcoin", "Ethereum", "Solana", "Cardano", "Polkadot", "Binance Coin", "Ripple", "Dogecoin", "Shiba Inu", "Avalanche"]
        all_people = ["Satoshi Nakamoto", "Vitalik Buterin", "Charles Hoskinson", "Changpeng Zhao", "Brian Armstrong", "Michael Saylor", "Elon Musk", "Sam Bankman-Fried", "Gavin Wood", "Elizabeth Stark"]
        all_orgs = ["Binance", "Coinbase", "FTX", "Kraken", "Bitcoin Foundation", "Ethereum Foundation", "Ripple Labs", "Cardano Foundation", "Digital Currency Group", "Grayscale"]
        all_techs = ["blockchain", "smart contracts", "DeFi", "NFT", "proof of stake", "proof of work", "layer 2 scaling", "sidechains", "cryptography", "consensus mechanisms"]
        all_events = ["Bitcoin halving", "Ethereum Merge", "Consensus Conference", "Bitcoin Miami", "ETH Denver", "Token2049", "DeFi Summer", "Crypto Winter", "Bull Market", "Market Correction"]
        
        # Random selection of entities
        return {
            'cryptocurrencies': random.sample(all_cryptos, k=random.randint(2, 5)),
            'people': random.sample(all_people, k=random.randint(1, 3)),
            'organizations': random.sample(all_orgs, k=random.randint(1, 4)),
            'technologies': random.sample(all_techs, k=random.randint(2, 4)),
            'events': random.sample(all_events, k=random.randint(0, 3))
        }