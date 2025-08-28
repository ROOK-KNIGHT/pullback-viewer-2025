#!/usr/bin/env python3
"""
Interactive Trading Desk Viewer
Based on TradingView's Trading Desk indicator

This script creates an interactive chart that identifies market stages based on breaks of structure (BOS).
It determines Accumulation, Distribution, Reaccumulation, Redistribution, or Neutral market stages.
Includes pullback dots for bullish and bearish MB points.

Features:
- Real-time data fetching using Schwab API
- Interactive plotly chart with zoom, pan, and hover
- BOS high and low step lines
- Bullish and bearish MB dots
- Market stage display
- Configurable parameters
- Support for different timeframes
"""

import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import argparse
import numpy as np
import requests
import json
import base64

# Standalone data handler (same as before)
class StandaloneDataHandler:
    def __init__(self):
        self.token_file = "cs_tokens.json"
        self.keys_path = "/Users/isaac/Desktop/Projects/CS_KEYS/KEYS.json"
        
    def load_api_keys(self):
        """Load API keys from the external KEYS.json file"""
        try:
            with open(self.keys_path, 'r') as f:
                keys = json.load(f)
            return keys["APP_KEY"], keys["APP_SECRET"]
        except Exception as e:
            print(f"Error loading API keys from {self.keys_path}: {e}")
            raise
    
    def load_tokens(self):
        if os.path.exists(self.token_file):
            with open(self.token_file, 'r') as f:
                return json.load(f)
        return None
    
    def ensure_valid_tokens(self, refresh=False):
        """Enhanced token validation with refresh capability"""
        tokens = self.load_tokens()
        if tokens:
            expires_at = tokens.get('expires_at')
            
            if expires_at:
                try:
                    expires_at = datetime.fromisoformat(expires_at)
                except ValueError:
                    print("Invalid 'expires_at' format in tokens. Re-authentication required.")
                    tokens = None
            else:
                print("'expires_at' missing from tokens. Re-authentication required.")
                tokens = None

            if tokens:
                refresh_token = tokens.get("refresh_token")
                if refresh or datetime.now() >= expires_at - timedelta(minutes=2):
                    print("Access token expired or refresh requested, attempting to refresh...")
                    new_tokens = self.refresh_tokens(refresh_token)
                    if new_tokens:
                        return new_tokens
                    else:
                        print("Failed to refresh tokens. Please re-authenticate.")
                        return None
                else:
                    return tokens

        print("No valid tokens found. Please run your authentication script first.")
        return None
    
    def save_tokens(self, tokens):
        """Save tokens with expiration time"""
        expires_at = datetime.now() + timedelta(seconds=int(tokens['expires_in']))
        tokens['expires_at'] = expires_at.isoformat()
        
        with open(self.token_file, 'w') as f:
            json.dump(tokens, f)
    
    def refresh_tokens(self, refresh_token):
        """Refresh access token using refresh token"""
        print("Refreshing access token...")
        app_key, app_secret = self.load_api_keys()
        credentials = f"{app_key}:{app_secret}"
        base64_credentials = base64.b64encode(credentials.encode()).decode("utf-8")

        headers = {
            "Authorization": f"Basic {base64_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token
        }

        token_url = "https://api.schwabapi.com/v1/oauth/token"
        refresh_response = requests.post(token_url, headers=headers, data=payload)
        
        if refresh_response.status_code == 200:
            new_tokens = refresh_response.json()
            self.save_tokens(new_tokens)
            print("Tokens refreshed successfully!")
            return new_tokens
        else:
            print("Failed to refresh tokens")
            print("Status Code:", refresh_response.status_code)
            print("Response:", refresh_response.text)
            return None
    
    def fetch_historical_data(self, symbol, periodType, period, frequencyType, freq, startDate=None, endDate=None, needExtendedHoursData=True):
        """Fetch historical data from Schwab API"""
        tokens = self.ensure_valid_tokens()
        if not tokens:
            return None
            
        access_token = tokens["access_token"]
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        url = f"https://api.schwabapi.com/marketdata/v1/pricehistory?symbol={symbol}&periodType={periodType}&period={period}&frequencyType={frequencyType}&frequency={freq}&needExtendedHoursData={str(needExtendedHoursData).lower()}"
        
        if startDate:
            url += f"&startDate={startDate}"
        if endDate:
            url += f"&endDate={endDate}"
        
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("empty", True):
                candles = [
                    {
                        "datetime": self.convert_timestamp(bar["datetime"]),
                        "open": bar.get("open"),
                        "high": bar.get("high"),
                        "low": bar.get("low"),
                        "close": bar.get("close"),
                        "volume": bar.get("volume")
                    }
                    for bar in data["candles"]
                ]
                return {
                    "symbol": symbol,
                    "candles": candles,
                    "previousClose": data.get("previousClose"),
                    "previousCloseDate": self.convert_timestamp(data.get("previousCloseDate"))
                }
            else:
                print("No data returned from API.")
                return None
                
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred: {http_err}")
            return None
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return None
    
    def convert_timestamp(self, timestamp):
        """Convert timestamp to datetime string"""
        if timestamp is not None:
            return datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        return None

class MarketStructureAnalyzer:
    def __init__(self):
        self.data_handler = StandaloneDataHandler()
        self.df = None
        self.bos_high = None
        self.bos_low = None
        self.market_stages = None
        self.mb_points = None
        
    def fetch_data(self, symbol, period_days=30, frequency_minutes=5, force_refresh=False):
        """
        Fetch historical data for the specified symbol
        
        Args:
            symbol (str): Stock symbol
            period_days (int): Number of days of data to fetch
            frequency_minutes (int): Frequency in minutes (1, 5, 15, 30, 60)
            force_refresh (bool): Force refresh of authentication tokens
        """
        print(f"Fetching data for {symbol}...")
        
        if force_refresh:
            print("Force refreshing authentication tokens...")
            self.data_handler.ensure_valid_tokens(refresh=True)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        start_date_ms = int(start_date.timestamp() * 1000)
        end_date_ms = int(end_date.timestamp() * 1000)
        
        try:
            data = self.data_handler.fetch_historical_data(
                symbol=symbol,
                periodType="day",
                period=1,
                frequencyType="minute",
                freq=frequency_minutes,
                startDate=start_date_ms,
                endDate=end_date_ms,
                needExtendedHoursData=False
            )
            
            if not data or not data.get('candles'):
                print(f"No data received for {symbol}")
                return False
                
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            self.df = df
            print(f"Successfully loaded {len(df)} candles for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def compute_market_stages(self):
        """
        Compute BOS levels, MB points, and market stages based on the PineScript logic
        """
        if self.df is None or len(self.df) < 2:
            print("Insufficient data for analysis")
            return None
            
        df = self.df
        n = len(df)
        
        self.bos_high = [np.nan] * n
        self.bos_low = [np.nan] * n
        self.market_stages = ["Neutral"] * n
        mb_points = []
        
        mb_array = []
        bullish_sequence = []
        bearish_sequence = []
        
        # Initial values
        self.bos_high[0] = df['high'].iloc[0]
        self.bos_low[0] = df['low'].iloc[0]
        
        for bar in range(1, n):
            self.bos_high[bar] = self.bos_high[bar-1]
            self.bos_low[bar] = self.bos_low[bar-1]
            market_stage = "Neutral"
            
            close = df['close'].iloc[bar]
            high = df['high'].iloc[bar]
            low = df['low'].iloc[bar]
            open_ = df['open'].iloc[bar]
            
            # Logic for BOS High (Potential Bullish MB)
            prev_bos_high = self.bos_high[bar-1]
            prev_bos_low = self.bos_low[bar-1]
            is_bullish_mb = False
            if close > self.bos_high[bar-1]:
                self.bos_high[bar] = high
                self.bos_low[bar] = low
                i = 1
                wave_found = True
                while wave_found and (bar - i) >= 0 and i < 2000:
                    prev_bar = bar - i
                    close_i = df['close'].iloc[prev_bar]
                    open_i = df['open'].iloc[prev_bar]
                    low_i = df['low'].iloc[prev_bar]
                    if close_i < open_i:
                        if (bar - (i + 1)) >= 0:
                            low_i_plus1 = df['low'].iloc[bar - (i + 1)]
                            if close_i < low_i_plus1:
                                wave_found = False
                    if low_i < self.bos_low[bar]:
                        self.bos_low[bar] = low_i
                        if self.bos_low[bar] != prev_bos_low and not is_bullish_mb:
                            is_bullish_mb = True
                            mb_array.append(1.0)
                            bullish_sequence.append(1)
                            bearish_sequence = []
                            mb_points.append({
                                'index': bar,
                                'datetime': df['datetime'].iloc[bar],
                                'type': 'bullish',
                                'price': close,
                                'previous_high': prev_bos_high
                            })
                    i += 1
            
            # Logic for BOS Low (Potential Bearish MB)
            prev_bos_high = self.bos_high[bar]
            prev_bos_low = self.bos_low[bar]
            is_bearish_mb = False
            if close < self.bos_low[bar]:
                self.bos_low[bar] = low
                self.bos_high[bar] = high
                i = 1
                wave_found = True
                while wave_found and (bar - i) >= 0 and i < 2000:
                    prev_bar = bar - i
                    close_i = df['close'].iloc[prev_bar]
                    open_i = df['open'].iloc[prev_bar]
                    high_i = df['high'].iloc[prev_bar]
                    if close_i > open_i:
                        if (bar - (i + 1)) >= 0:
                            high_i_plus1 = df['high'].iloc[bar - (i + 1)]
                            if close_i > high_i_plus1:
                                wave_found = False
                    if high_i > self.bos_high[bar]:
                        self.bos_high[bar] = high_i
                        if self.bos_high[bar] != prev_bos_high and not is_bearish_mb:
                            is_bearish_mb = True
                            mb_array.append(-1.0)
                            bearish_sequence.append(1)
                            bullish_sequence = []
                            mb_points.append({
                                'index': bar,
                                'datetime': df['datetime'].iloc[bar],
                                'type': 'bearish',
                                'price': close,
                                'previous_low': prev_bos_low
                            })
                    i += 1
            
            # Trim mb_array to last 8
            if len(mb_array) > 8:
                mb_array = mb_array[-8:]
            
            # Determine market stage
            if len(mb_array) >= 2:
                last_bullish_count = len(bullish_sequence)
                last_bearish_count = len(bearish_sequence)
                last_mb = mb_array[-1] if mb_array else 0
                
                if last_bullish_count >= 2 and 0 < last_bearish_count <= 3:
                    market_stage = "Reaccumulation"
                elif last_bearish_count >= 2 and 0 < last_bullish_count <= 3:
                    market_stage = "Redistribution"
                elif last_bearish_count > 3:
                    market_stage = "Distribution"
                    bearish_sequence = []
                elif last_bullish_count > 3:
                    market_stage = "Accumulation"
                    bullish_sequence = []
                elif last_mb == 1.0:
                    market_stage = "Accumulation"
                elif last_mb == -1.0:
                    market_stage = "Distribution"
                else:
                    market_stage = "Neutral"
            else:
                market_stage = "Neutral"
            
            self.market_stages[bar] = market_stage
        
        self.mb_points = pd.DataFrame(mb_points) if mb_points else pd.DataFrame()
        print(f"Identified {len(self.mb_points)} MB points")
        print("Market stages computed")
        
    def create_interactive_chart(self, symbol, show_volume=True, line_width=1, line_color='blue', output_dir="examples"):
        """
        Create an interactive plotly chart with BOS lines, MB dots, and market stage
        
        Args:
            symbol (str): Stock symbol for chart title
            show_volume (bool): Whether to show volume subplot
            line_width (int): Width of BOS lines
            line_color (str): Color of BOS lines
            output_dir (str): Directory to save the chart file
        """
        if self.df is None:
            print("No data available for charting")
            return None
            
        # Create subplots
        if show_volume:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f'{symbol} Price with BOS Lines and MB Points', 'Volume'),
                row_width=[0.3, 0.7]
            )
        else:
            fig = go.Figure()
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=self.df['datetime'],
            open=self.df['open'],
            high=self.df['high'],
            low=self.df['low'],
            close=self.df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444',
            increasing_fillcolor='#00ff88',
            decreasing_fillcolor='#ff4444'
        )
        
        if show_volume:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)
        
        # Add BOS High as step line
        if self.bos_high:
            stepped_x_high = []
            stepped_y_high = []
            for i in range(len(self.df)):
                if i > 0:
                    stepped_x_high.append(self.df['datetime'].iloc[i])
                    stepped_y_high.append(self.bos_high[i-1])
                stepped_x_high.append(self.df['datetime'].iloc[i])
                stepped_y_high.append(self.bos_high[i])
            fig.add_trace(go.Scatter(
                x=stepped_x_high,
                y=stepped_y_high,
                mode='lines',
                line=dict(color=line_color, width=abs(line_width)),
                name='BOS High'
            ), row=1 if show_volume else None, col=1 if show_volume else None)
        
        # Add BOS Low as step line
        if self.bos_low:
            stepped_x_low = []
            stepped_y_low = []
            for i in range(len(self.df)):
                if i > 0:
                    stepped_x_low.append(self.df['datetime'].iloc[i])
                    stepped_y_low.append(self.bos_low[i-1])
                stepped_x_low.append(self.df['datetime'].iloc[i])
                stepped_y_low.append(self.bos_low[i])
            fig.add_trace(go.Scatter(
                x=stepped_x_low,
                y=stepped_y_low,
                mode='lines',
                line=dict(color=line_color, width=abs(line_width)),
                name='BOS Low'
            ), row=1 if show_volume else None, col=1 if show_volume else None)
        
        # Add MB points if available
        if not self.mb_points.empty:
            # Bearish MB (red dots) - positioned below low of candle
            bearish = self.mb_points[self.mb_points['type'] == 'bearish']
            if not bearish.empty:
                bearish_y_positions = []
                for idx in bearish['index']:
                    candle = self.df.iloc[idx]
                    bearish_y_positions.append(candle['low'] - (candle['high'] - candle['low']) * 0.1)
                
                fig.add_trace(go.Scatter(
                    x=bearish['datetime'],
                    y=bearish_y_positions,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    name='Bearish MB',
                    text=[f"Bearish MB<br>Close: ${price:.2f}<br>Prev Low: ${prev_low:.2f}" 
                          for price, prev_low in zip(bearish['price'], bearish['previous_low'])],
                    hovertemplate='%{text}<extra></extra>'
                ), row=1 if show_volume else None, col=1 if show_volume else None)
            
            # Bullish MB (green dots) - positioned above high of candle
            bullish = self.mb_points[self.mb_points['type'] == 'bullish']
            if not bullish.empty:
                bullish_y_positions = []
                for idx in bullish['index']:
                    candle = self.df.iloc[idx]
                    bullish_y_positions.append(candle['high'] + (candle['high'] - candle['low']) * 0.1)
                
                fig.add_trace(go.Scatter(
                    x=bullish['datetime'],
                    y=bullish_y_positions,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    name='Bullish MB',
                    text=[f"Bullish MB<br>Close: ${price:.2f}<br>Prev High: ${prev_high:.2f}" 
                          for price, prev_high in zip(bullish['price'], bullish['previous_high'])],
                    hovertemplate='%{text}<extra></extra>'
                ), row=1 if show_volume else None, col=1 if show_volume else None)
        
        # Add volume bars if requested
        if show_volume:
            colors = ['green' if close >= open_ else 'red' 
                      for close, open_ in zip(self.df['close'], self.df['open'])]
            
            fig.add_trace(go.Bar(
                x=self.df['datetime'],
                y=self.df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
        
        # Add market stage annotation
        if self.market_stages:
            last_stage = self.market_stages[-1]
            bg_color = 'green' if last_stage in ["Accumulation", "Reaccumulation"] else 'red' if last_stage in ["Distribution", "Redistribution"] else 'gray'
            fig.add_annotation(
                x=1,
                y=1,
                xref='paper',
                yref='paper',
                text=last_stage,
                showarrow=False,
                font=dict(size=14, color='white'),
                bgcolor=bg_color,
                opacity=0.8,
                align='center',
                bordercolor='white',
                borderwidth=1,
                borderpad=4
            )
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Market Structure Analyzer',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            template='plotly_dark',
            showlegend=True,
            height=800 if show_volume else 600,
            hovermode='x unified'
        )
        
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def analyze_market_stage(self):
        """
        Print analysis of the current market stage
        """
        if self.market_stages is None or len(self.market_stages) == 0:
            print("No market stages computed")
            return
            
        print("\n" + "="*60)
        print("MARKET STAGE ANALYSIS")
        print("="*60)
        
        current_stage = self.market_stages[-1]
        print(f"Current Market Stage: {current_stage}")
        
        # Optional: Count occurrences of each stage
        from collections import Counter
        stage_counts = Counter(self.market_stages)
        print("\nStage Distribution:")
        for stage, count in stage_counts.items():
            print(f"{stage}: {count} bars")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Market Structure Analyzer - Identify market stages based on BOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pullback_viewer_interactive.py AAPL
  python3 pullback_viewer_interactive.py NVDA --days 7 --frequency 1
  python3 pullback_viewer_interactive.py TSLA --no-volume --line-width 2 --line-color red
        """
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA, TSLA)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data (default: 7)')
    parser.add_argument('--frequency', type=int, default=5, choices=[1, 5, 15, 30, 60], 
                        help='Frequency in minutes (default: 5)')
    parser.add_argument('--no-volume', action='store_true', help='Hide volume chart')
    parser.add_argument('--force-refresh', action='store_true', 
                        help='Force refresh authentication tokens')
    parser.add_argument('--line-width', type=int, default=1, help='Line width for BOS lines (default: 1)')
    parser.add_argument('--line-color', type=str, default='blue', help='Line color for BOS lines (default: blue)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MarketStructureAnalyzer()
    
    print("="*60)
    print("MARKET STRUCTURE ANALYZER")
    print("="*60)
    print(f"Symbol: {args.symbol.upper()}")
    print(f"Period: {args.days} days")
    print(f"Frequency: {args.frequency} minutes")
    print(f"Line Width: {args.line_width}")
    print(f"Line Color: {args.line_color}")
    print(f"Force Refresh: {args.force_refresh}")
    print("="*60)
    
    # Fetch data
    if not analyzer.fetch_data(args.symbol.upper(), args.days, args.frequency, args.force_refresh):
        print("Failed to fetch data. Exiting.")
        return
    
    # Compute market stages
    analyzer.compute_market_stages()
    
    # Analyze market stage
    analyzer.analyze_market_stage()
    
    # Create and show interactive chart
    print("\nGenerating interactive chart...")
    fig = analyzer.create_interactive_chart(
        args.symbol.upper(), 
        show_volume=not args.no_volume,
        line_width=args.line_width,
        line_color=args.line_color
    )
    
    if fig:
        print("Opening interactive chart in browser...")
        fig.show()
        
        # Create examples directory if it doesn't exist
        output_dir = "examples"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as HTML file in examples directory
        filename = f"{args.symbol.upper()}_market_structure_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Chart saved as: {filepath}")
    else:
        print("Failed to create chart")

if __name__ == "__main__":
    main()
