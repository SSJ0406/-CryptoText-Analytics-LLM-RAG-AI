import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

class FinanceViz:
    def __init__(self):
        # Define color scheme for consistent visualizations
        self.colors = {
            'primary': '#1f77b4',    # Main chart color
            'secondary': '#ff7f0e',  # Secondary elements
            'positive': '#2ca02c',   # For positive values/returns
            'negative': '#d62728',   # For negative values/returns
            'neutral': '#7f7f7f'     # For neutral values
        }

    def plot_stock_price(self, data, ticker, include_volume=True):
        """Creates an interactive stock/crypto price chart with optional volume panel"""
        if include_volume and 'Volume' in data.columns:
            # Create a figure with two vertically stacked subplots
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.2,  # Increased from 0.1 to 0.2 for better spacing
                               row_heights=[0.7, 0.3],
                               subplot_titles=[f"{ticker} Price", "Trading Volume"])  # Explicit subplot titles
            
            # Add price line to top subplot
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['Close'], 
                    name='Closing Price',
                    line=dict(color=self.colors['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Add volume bars to bottom subplot with color coded by price direction
            colors = [self.colors['positive'] if row['Close'] >= row['Open'] else self.colors['negative'] 
                     for i, row in data.iterrows()]
            
            fig.add_trace(
                go.Bar(
                    x=data.index, 
                    y=data['Volume'], 
                    name='Volume',
                    marker_color=colors
                ),
                row=2, col=1
            )
            
            # Configure layout
            fig.update_layout(
                title=f'{ticker} - Price and Volume Chart',
                height=600,
                showlegend=False,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=30, b=80)  # Increased bottom margin from 60 to 80
            )
            
            fig.update_yaxes(title_text='Price ($)', row=1, col=1)
            fig.update_yaxes(title_text='Volume', row=2, col=1)
            
            # Adjust annotation font size and position
            fig.update_annotations(font_size=12)
            
            # Adjust position of Trading Volume label
            if len(fig.layout.annotations) > 1:
                fig.layout.annotations[1].y = fig.layout.annotations[1].y - 0.05  # Move down
            
        else:
            # Simple price chart without volume
            fig = px.line(
                data, 
                x=data.index, 
                y='Close', 
                title=f'{ticker} - Price Chart',
                labels={'Close': 'Price ($)', 'index': 'Date'}
            )
            
            fig.update_traces(line=dict(color=self.colors['primary'], width=2))
            fig.update_layout(margin=dict(l=20, r=20, t=30, b=60))  # Increased bottom margin
            
        return fig

    def plot_returns_distribution(self, data):
        """Creates a histogram of daily returns with normal distribution overlay"""
        if 'daily_return' not in data.columns:
            data['daily_return'] = data['Close'].pct_change()
        
        # Remove NaN values (first row will have NaN return)
        returns = data['daily_return'].dropna()
        
        # Create histogram
        fig = px.histogram(
            returns, 
            nbins=50,
            title='Daily Returns Distribution',
            labels={'value': 'Daily Return (%)', 'count': 'Number of Days'},
            color_discrete_sequence=[self.colors['primary']]
        )
        
        # Add normal distribution curve for comparison
        x = np.linspace(returns.min(), returns.max(), 100)
        y = np.exp(-(x - returns.mean())**2 / (2 * returns.std()**2)) / (returns.std() * np.sqrt(2 * np.pi))
        # Scale the normal curve to match histogram height
        y = y * (len(returns) * (returns.max() - returns.min()) / 50)
        
        fig.add_trace(
            go.Scatter(
                x=x, 
                y=y, 
                mode='lines', 
                name='Normal Distribution',
                line=dict(color=self.colors['secondary'])
            )
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            legend_title='Distribution'
        )
        
        return fig

    def plot_comparison(self, data_dict, metric='Close', title='Crypto Comparison'):
        """Creates a comparison chart for multiple cryptocurrencies with normalized values"""
        fig = go.Figure()
        
        for ticker, data in data_dict.items():
            # Skip empty dataframes or dataframes without the required metric
            if data.empty or metric not in data.columns or len(data[metric]) == 0:
                continue
                
            # Normalize data to make cryptos comparable (first day = 100)
            normalized = data[metric] / data[metric].iloc[0] * 100
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized,
                    name=ticker,
                    mode='lines'
                )
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Normalized Value (First Day = 100)',
            height=500
        )
        
        return fig

    def plot_sentiment_timeline(self, news_data):
        """Creates a timeline visualization of news sentiment"""
        if not news_data:
            return None
        
        # Prepare data
        data = []
        for ticker, news_list in news_data.items():
            for item in news_list:
                if 'sentiment' in item and 'providerPublishTime' in item:
                    date = datetime.fromtimestamp(item['providerPublishTime'])
                    data.append({
                        'ticker': ticker,
                        'date': date,
                        'title': item['title'],
                        'sentiment': item['sentiment'],
                        'link': item.get('link', '')
                    })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Sort by date so newest appear on top
        df = df.sort_values('date')
        
        # Map sentiments to colors
        color_map = {
            'positive': self.colors['positive'],
            'negative': self.colors['negative'],
            'neutral': self.colors['neutral']
        }
        
        # Create scatter plot with news items
        fig = px.scatter(
            df, 
            x='date', 
            y='ticker',
            color='sentiment',
            hover_name='title',
            color_discrete_map=color_map,
            size=[15] * len(df),  # Larger point size
            opacity=0.8,  # Add some transparency
            title='News Sentiment Timeline',
            labels={'date': 'Publication Date', 'ticker': 'Cryptocurrency', 'sentiment': 'Sentiment'}
        )
        
        # Customize the hover template
        hovertemplate = '<b>%{hovertext}</b><br>Date: %{x}<br>Sentiment: %{marker.color}<extra></extra>'
        fig.update_traces(hovertemplate=hovertemplate)
        
        # Update ticker labels to be more readable
        ticker_labels = {}
        for ticker in df['ticker'].unique():
            ticker_labels[ticker] = ticker.capitalize()
        
        fig.update_layout(
            xaxis_title='Publication Date',
            yaxis_title='Cryptocurrency',
            height=400,
            yaxis={'categoryorder':'category ascending', 'categoryarray': sorted(df['ticker'].unique())},
            xaxis={'tickangle': -45},  # Rotate x-axis labels for better readability
            plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Lighter background
            margin=dict(l=20, r=20, t=50, b=60)  # More space for x-axis labels
        )
        
        # Add vertical grid lines for better time reference
        fig.update_xaxes(
            showgrid=True,
            gridcolor='rgba(230, 230, 230, 0.8)',
            tickformat='%Y-%m-%d'  # Format date labels
        )
        
        # Remove horizontal grid
        fig.update_yaxes(
            showgrid=False
        )
        
        return fig

    def plot_rsi(self, data, ticker):
        """Creates an RSI chart with price - specific for crypto"""
        if 'RSI' not in data.columns:
            return None
            
        # Create RSI figure
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.2,  # Increased from 0.1 to 0.2
                          row_heights=[0.7, 0.3],
                          subplot_titles=[f"{ticker} Price", "Relative Strength Index (RSI)"])  # Explicit subplot titles
        
        # Add price to top subplot
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['Close'], 
                name='Price',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Add RSI to bottom subplot
        fig.add_trace(
            go.Scatter(
                x=data.index, 
                y=data['RSI'], 
                name='RSI',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line", line=dict(dash='dash', width=1, color='red'),
            y0=70, y1=70, x0=data.index[0], x1=data.index[-1],
            row=2, col=1
        )
        fig.add_shape(
            type="line", line=dict(dash='dash', width=1, color='green'),
            y0=30, y1=30, x0=data.index[0], x1=data.index[-1],
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f'{ticker} - Price and RSI Analysis',
            height=600,
            xaxis_rangeslider_visible=False,
            margin=dict(l=20, r=20, t=30, b=80)  # Increased bottom margin
        )
        
        fig.update_yaxes(title_text='Price ($)', row=1, col=1)
        fig.update_yaxes(title_text='RSI Value', row=2, col=1)
        
        # Adjust annotation font size
        fig.update_annotations(font_size=12)
        
        # Adjust position of RSI label
        if len(fig.layout.annotations) > 1:
            fig.layout.annotations[1].y = fig.layout.annotations[1].y - 0.05  # Move down
        
        return fig

    def plot_event_timeline(self, events_df):
        """
        Creates an interactive timeline visualization of cryptocurrency events.
        
        Args:
            events_df (pandas.DataFrame): DataFrame containing event data
            
        Returns:
            plotly.graph_objects.Figure: Interactive timeline visualization
        """
        if events_df.empty:
            return None
        
        # Map importance to size for better visibility
        importance_size = {
            'high': 14,
            'medium': 10,
            'low': 8
        }
        
        # Create sizes based on importance
        sizes = events_df['importance'].map(importance_size).fillna(10).tolist()
        
        # Create hover text
        hover_texts = []
        for _, row in events_df.iterrows():
            text = f"<b>{row['project_name']} ({row['symbol']})</b><br>"
            text += f"Event: {row['event_type']}<br>"
            text += f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
            text += f"Description: {row['description']}<br>"
            text += f"Importance: {row['importance'].capitalize()}"
            hover_texts.append(text)
        
        # Create figure
        fig = go.Figure()
        
        # Add scatter plot for events - using a single color for simplicity
        fig.add_trace(
            go.Scatter(
                x=events_df['date'],
                y=events_df['project_name'],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color='#1E88E5',  # Primary blue color for all events
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                text=hover_texts,
                hoverinfo='text'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Cryptocurrency Events Timeline",
            xaxis_title="Date",
            yaxis_title="Project",
            height=500,
            hovermode='closest',
            xaxis=dict(
                tickangle=-45,
                tickformat='%Y-%m-%d'
            ),
            margin=dict(l=20, r=20, t=50, b=50)
        )
        
        # Remove horizontal grid
        fig.update_yaxes(showgrid=False)
        
        return fig

    def plot_events_by_type(self, events_df):
        """
        Creates a bar chart showing distribution of events by type.
        
        Args:
            events_df (pandas.DataFrame): DataFrame containing event data
            
        Returns:
            plotly.graph_objects.Figure: Bar chart visualization
        """
        if events_df.empty:
            return None
        
        # Count events by type
        event_counts = events_df['event_type'].value_counts().reset_index()
        event_counts.columns = ['Event Type', 'Count']
        
        # Sort by count
        event_counts = event_counts.sort_values('Count', ascending=False)
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=event_counts['Event Type'],
                y=event_counts['Count'],
                marker_color=self.colors['primary']
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Distribution of Events by Type",
            xaxis_title="Event Type",
            yaxis_title="Number of Events",
            height=400,
            margin=dict(l=20, r=20, t=50, b=50)
        )
        
        return fig
    
    def plot_events_by_impact(self, events_df):
        """
        Creates a pie chart showing distribution of events by potential impact.
        
        Args:
            events_df (pandas.DataFrame): DataFrame containing event data
            
        Returns:
            plotly.graph_objects.Figure: Pie chart visualization
        """
        if events_df.empty:
            return None
        
        # Count events by impact
        impact_counts = events_df['potential_impact'].value_counts().reset_index()
        impact_counts.columns = ['Impact', 'Count']
        
        # Define colors
        colors = [self.colors['positive'], self.colors['neutral'], self.colors['negative']]
        
        # Create pie chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Pie(
                labels=impact_counts['Impact'],
                values=impact_counts['Count'],
                marker=dict(colors=colors),
                textinfo='percent+label',
                insidetextorientation='radial'
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Distribution of Events by Potential Impact",
            height=400,
            margin=dict(l=20, r=20, t=50, b=50)
        )
        
        return fig

    def plot_past_events_analysis(self, past_events_df):
        """
        Creates a comparative visualization of potential vs actual impact for past events.
        
        Args:
            past_events_df (pandas.DataFrame): DataFrame containing past event data
            
        Returns:
            plotly.graph_objects.Figure: Comparative visualization
        """
        if past_events_df.empty or 'actual_impact' not in past_events_df.columns:
            return None
        
        # Group by event type and compare potential vs actual impact
        impact_comparison = pd.crosstab(
            index=past_events_df['event_type'], 
            columns=[past_events_df['potential_impact'], past_events_df['actual_impact']]
        ).reset_index()
        
        # Prepare data for stacked bar chart
        event_types = impact_comparison['event_type'].tolist()
        
        # Create figure with subplots
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("Potential Impact by Event Type", "Actual Impact by Event Type"),
            shared_yaxes=True
        )
        
        # Add potential impact
        for impact in ['positive', 'neutral', 'negative']:
            try:
                values = [impact_comparison.loc[impact_comparison['event_type'] == event_type, 
                                              ('potential_impact', impact)].values[0] 
                        if ('potential_impact', impact) in impact_comparison.columns 
                        else 0 
                        for event_type in event_types]
                
                fig.add_trace(
                    go.Bar(
                        y=event_types,
                        x=values,
                        name=f"Potential: {impact.capitalize()}",
                        orientation='h',
                        marker_color=self.colors[impact],
                        legendgroup=f"potential_{impact}"
                    ),
                    row=1, col=1
                )
            except:
                continue
        
        # Add actual impact
        for impact in ['positive', 'neutral', 'negative']:
            try:
                values = [impact_comparison.loc[impact_comparison['event_type'] == event_type, 
                                              ('actual_impact', impact)].values[0] 
                        if ('actual_impact', impact) in impact_comparison.columns 
                        else 0 
                        for event_type in event_types]
                
                fig.add_trace(
                    go.Bar(
                        y=event_types,
                        x=values,
                        name=f"Actual: {impact.capitalize()}",
                        orientation='h',
                        marker_color=self.colors[impact],
                        legendgroup=f"actual_{impact}"
                    ),
                    row=1, col=2
                )
            except:
                continue
        
        # Update layout
        fig.update_layout(
            title="Comparison of Potential vs Actual Impact of Events",
            height=500,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=50, b=50)
        )
        
        return fig