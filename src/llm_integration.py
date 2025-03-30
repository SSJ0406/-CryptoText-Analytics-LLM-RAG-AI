# src/llm_integration.py
import os
from openai import OpenAI
from dotenv import load_dotenv

class LLMClient:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize OpenAI client with API key from environment
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
        else:
            # If no API key is found, set client to None and show warning
            self.client = None
            print("Warning: OPENAI_API_KEY not found in environment variables")
    
    def analyze_sentiment(self, title, content=None):
        """
        Analyze sentiment of news article using LLM (Language Model)
        
        Args:
            title (str): The title of the article to analyze
            content (str, optional): The content/body of the article
            
        Returns:
            str: Sentiment classification - 'positive', 'negative', or 'neutral'
                 Returns None if API key is not configured or if an error occurs
        """
        if not self.client:
            return None
            
        try:
            # Construct prompt for sentiment analysis
            prompt = f"""
            Analyze the sentiment of this financial news article title (and content if provided) related to stock market.
            Classify it as 'positive', 'negative', or 'neutral' from an investor's perspective.
            
            Title: {title}
            """
            
            # Add content to prompt if provided
            if content:
                prompt += f"\nContent: {content}"
                
            # Make API call to OpenAI
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in sentiment analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1  # Low temperature for more deterministic responses
            )
            
            # Extract and classify sentiment from response
            sentiment_text = response.choices[0].message.content.lower()
            
            if 'positive' in sentiment_text:
                return 'positive'
            elif 'negative' in sentiment_text:
                return 'negative'
            else:
                return 'neutral'
                
        except Exception as e:
            print(f"Error analyzing sentiment with LLM: {e}")
            return None
    
    def generate_financial_analysis(self, ticker, metrics, recent_news, price_trend):
        """
        Generate a comprehensive financial analysis for a stock using LLM
        
        Args:
            ticker (str): Stock ticker symbol
            metrics (str): Financial metrics data
            recent_news (str): Recent news about the stock
            price_trend (str): Description of recent price movements
            
        Returns:
            str: Comprehensive financial analysis or error message
        """
        if not self.client:
            return "API key not configured. Please add your OpenAI API key to use this feature."
            
        try:
            # Construct detailed prompt for financial analysis
            prompt = f"""
            Generate a comprehensive analysis for {ticker} stock based on the following information:
            
            Financial Metrics:
            {metrics}
            
            Recent Price Trend:
            {price_trend}
            
            Recent News:
            {recent_news}
            
            Provide analysis including:
            1. Overall market sentiment
            2. Key factors affecting the stock
            3. Potential risks and opportunities
            4. A brief outlook
            """
            
            # Make API call to OpenAI with higher temperature for more creative analysis
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst with expertise in stock market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7  # Higher temperature for more varied responses
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error generating analysis with LLM: {e}")
            return f"Error generating analysis: {str(e)}"
    
    def answer_financial_question(self, question, ticker, context):
        """
        Answer a specific financial question about a stock using LLM
        
        Args:
            question (str): The financial question to answer
            ticker (str): Stock ticker symbol
            context (str): Contextual information to help answer the question
            
        Returns:
            str: Answer to the financial question or error message
        """
        if not self.client:
            return "API key not configured. Please add your OpenAI API key to use this feature."
            
        try:
            # Construct prompt for Q&A
            prompt = f"""
            Answer the following question about {ticker} stock based on the provided context information.
            
            Question: {question}
            
            Context Information:
            {context}
            
            Provide a concise, accurate answer based only on the information provided in the context.
            If the information is not available in the context, acknowledge this limitation.
            """
            
            # Make API call to OpenAI with balanced temperature
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial advisor with expertise in stock market analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.5  # Balanced temperature for informative but natural responses
            )
            
            return response.choices[0].message.content
                
        except Exception as e:
            print(f"Error answering question with LLM: {e}")
            return f"Error answering question: {str(e)}"
    
    def mock_sentiment_analysis(self, title):
        """
        Generate deterministic mock sentiment analysis when OpenAI API is not available
        
        Args:
            title (str): Article title to analyze
            
        Returns:
            str: Mock sentiment classification ('positive', 'negative', or 'neutral')
        """
        import hashlib
        
        # Use hash of title to generate consistent sentiment
        # This ensures the same title always gets the same sentiment
        hash_value = int(hashlib.md5(title.encode()).hexdigest(), 16)
        sentiment_value = hash_value % 3
        
        if sentiment_value == 0:
            return 'positive'
        elif sentiment_value == 1:
            return 'negative'
        else:
            return 'neutral'
    
    def mock_financial_analysis(self, ticker, metrics, recent_news, price_trend):
        """
        Generate mock financial analysis when OpenAI API is not available
        
        Args:
            ticker (str): Stock ticker symbol
            metrics (str): Financial metrics data (not used in mock)
            recent_news (str): Recent news about the stock (not used in mock)
            price_trend (str): Description of recent price movements (not used in mock)
            
        Returns:
            str: Generic financial analysis with markdown formatting
        """
        # Return a pre-formatted markdown string with generic financial analysis
        return f"""
        # Financial Analysis for {ticker}
        
        ## Market Sentiment
        Based on the available data, the market sentiment for {ticker} appears to be mixed. The price action shows some volatility, but there are potential signs of stabilization.
        
        ## Key Factors
        - Technical indicators suggest a cautious approach
        - Recent news has had varying impact on stock performance
        - Overall market conditions continue to influence this stock
        
        ## Risks and Opportunities
        
        **Risks:**
        - Market volatility could lead to unexpected price movements
        - Competitive pressures in the industry
        - Economic uncertainty
        
        **Opportunities:**
        - Long-term growth potential in the sector
        - Possible undervaluation based on some metrics
        - Strategic initiatives may improve performance
        
        ## Outlook
        The short-term outlook remains uncertain, but long-term prospects depend on both company execution and broader market trends. Investors should monitor key financial metrics and news developments.
        
        *Note: This is a mock analysis for demonstration purposes. For actual investment decisions, please consult a financial advisor.*
        """
    
    def mock_answer_question(self, question, ticker, context):
        """
        Generate mock answer to financial question when OpenAI API is not available
        
        Args:
            question (str): The financial question (not used in mock)
            ticker (str): Stock ticker symbol
            context (str): Contextual information (not used in mock)
            
        Returns:
            str: Generic answer with disclaimer
        """
        # Return a pre-formatted string with generic answer and disclaimer
        return f"""
        Based on the available information about {ticker}, I can provide a general response to your question.
        
        The data shows mixed performance metrics, with some positive indicators alongside areas of concern. Without more specific financial data, it's difficult to make precise forecasts or recommendations.
        
        For detailed analysis of {ticker}, I'd recommend reviewing the most recent quarterly reports and analyst recommendations from financial services providers.
        
        *Note: This is a mock answer for demonstration purposes. For actual investment advice, please consult a financial advisor.*
        """