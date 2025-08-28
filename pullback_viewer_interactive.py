#!/usr/bin/env python3
"""
Interactive Range-Based Market Analysis Viewer
Advanced implementation with dynamic bullish/bearish range tracking

This script creates an interactive chart that identifies and tracks dynamic market ranges
with sophisticated high/low updating rules and structural pivot identification.

Features:
- Dynamic bullish/bearish range state management
- Structural high/low identification (pivots with confirmation on both sides)
- Body-based validation for conservative range updates
- Dynamic high/low tracking with specific update rules
- Range break detection and momentum assessment
- Real-time data fetching using existing handlers
- Interactive plotly chart with zoom, pan, and hover
"""

import sys
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import argparse
import numpy as np

# Import required modules for data handling
import requests
import json
import base64
import webbrowser
import urllib.parse
from datetime import timedelta

# Standalone data handler (simplified version of your handlers)
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
            
            # Check if 'expires_at' exists and is a valid string
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
                # Check if access token is expired or about to expire (within 2 minutes) or force refresh
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

class RangeBasedAnalyzer:
    def __init__(self):
        self.data_handler = StandaloneDataHandler()
        self.df = None
        self.range_data = None
        
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
        
        # Force refresh tokens if requested
        if force_refresh:
            print("Force refreshing authentication tokens...")
            self.data_handler.ensure_valid_tokens(refresh=True)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days)
        
        # Convert to milliseconds
        start_date_ms = int(start_date.timestamp() * 1000)
        end_date_ms = int(end_date.timestamp() * 1000)
        
        try:
            # Fetch data using existing handler
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
                
            # Convert to DataFrame
            df = pd.DataFrame(data['candles'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            self.df = df
            print(f"Successfully loaded {len(df)} candles for {symbol}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def identify_structural_pivots(self, lookback=3):
        """
        Identify structural highs and lows (pivots with confirmation on both sides)
        
        Args:
            lookback (int): Number of candles to look back/forward for confirmation
        """
        if self.df is None or len(self.df) < (lookback * 2 + 1):
            return [], []
            
        df = self.df.copy()
        structural_highs = []
        structural_lows = []
        
        # Check each candle for structural pivots (excluding edges)
        for i in range(lookback, len(df) - lookback):
            current_high = df.iloc[i]['high']
            current_low = df.iloc[i]['low']
            
            # Check for structural high (higher than all surrounding highs)
            is_structural_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['high'] >= current_high:
                    is_structural_high = False
                    break
            
            if is_structural_high:
                structural_highs.append({
                    'index': i,
                    'datetime': df.iloc[i]['datetime'],
                    'price': current_high,
                    'type': 'structural_high'
                })
            
            # Check for structural low (lower than all surrounding lows)
            is_structural_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_structural_low = False
                    break
            
            if is_structural_low:
                structural_lows.append({
                    'index': i,
                    'datetime': df.iloc[i]['datetime'],
                    'price': current_low,
                    'type': 'structural_low'
                })
        
        return structural_highs, structural_lows
    
    def analyze_dynamic_ranges(self, lookback=3):
        """
        Implement the complex range-based algorithm with dynamic high/low tracking
        """
        if self.df is None or len(self.df) < 10:
            print("Insufficient data for range analysis")
            return
            
        df = self.df.copy()
        
        # Get structural pivots
        structural_highs, structural_lows = self.identify_structural_pivots(lookback)
        
        # Initialize range tracking variables
        range_states = []
        current_state = None  # 'bullish' or 'bearish'
        current_high = None
        current_low = None
        range_high_locked = False
        range_low_locked = False
        
        # Track last major movements for range determination
        last_sell_before_buy_index = None
        last_buy_before_sell_index = None
        
        print(f"Found {len(structural_highs)} structural highs and {len(structural_lows)} structural lows")
        
        # Process each candle
        for i in range(1, len(df)):
            current = df.iloc[i]
            previous = df.iloc[i-1]
            
            # Determine if we have a directional break (body-based)
            body_above_prev_high = (current['close'] > previous['high'] and 
                                  current['open'] > previous['high'])
            body_below_prev_low = (current['close'] < previous['low'] and 
                                 current['open'] < previous['low'])
            
            # Check for range state changes
            if body_above_prev_high and current_state != 'bullish':
                # Switch to bullish range
                current_state = 'bullish'
                range_high_locked = False
                range_low_locked = False
                
                # Find the last "sell before buy" for low determination
                if last_sell_before_buy_index is not None:
                    # Find lowest low after the last sell signal
                    low_search_start = max(0, last_sell_before_buy_index)
                    lowest_low = df.iloc[low_search_start:i+1]['low'].min()
                    current_low = lowest_low
                else:
                    current_low = current['low']
                
                current_high = current['high']
                
            elif body_below_prev_low and current_state != 'bearish':
                # Switch to bearish range
                current_state = 'bearish'
                range_high_locked = False
                range_low_locked = False
                
                # Find the last "buy before sell" for high determination
                if last_buy_before_sell_index is not None:
                    # Find highest high after the last buy signal
                    high_search_start = max(0, last_buy_before_sell_index)
                    highest_high = df.iloc[high_search_start:i+1]['high'].max()
                    current_high = highest_high
                else:
                    current_high = current['high']
                
                current_low = current['low']
            
            # Update highs and lows based on current range state
            if current_state == 'bullish' and current_high is not None:
                # In bullish range: update high only when candle body exceeds current high
                if (current['close'] > current_high or current['open'] > current_high):
                    current_high = current['high']
                    # Reset low tracking when high is updated
                    if not range_low_locked:
                        # Find new low based on last sell before this new high
                        for j in range(i-1, -1, -1):
                            if df.iloc[j]['close'] < df.iloc[j-1]['low'] if j > 0 else False:
                                low_search_start = j
                                current_low = df.iloc[low_search_start:i+1]['low'].min()
                                break
                
            elif current_state == 'bearish' and current_low is not None:
                # In bearish range: update low only when candle body goes below current low
                if (current['close'] < current_low or current['open'] < current_low):
                    current_low = current['low']
                    # Reset high tracking when low is updated
                    if not range_high_locked:
                        # Find new high based on last buy before this new low
                        for j in range(i-1, -1, -1):
                            if df.iloc[j]['close'] > df.iloc[j-1]['high'] if j > 0 else False:
                                high_search_start = j
                                current_high = df.iloc[high_search_start:i+1]['high'].max()
                                break
            
            # Track major movements for future range calculations
            if current['close'] < previous['low']:
                last_sell_before_buy_index = i
            elif current['close'] > previous['high']:
                last_buy_before_sell_index = i
            
            # Store range state for this candle
            range_states.append({
                'index': i,
                'datetime': current['datetime'],
                'state': current_state,
                'range_high': current_high,
                'range_low': current_low,
                'price': current['close']
            })
        
        # Convert to DataFrame
        self.range_data = pd.DataFrame(range_states)
        
        # Add structural pivots to the data
        self.structural_highs = pd.DataFrame(structural_highs)
        self.structural_lows = pd.DataFrame(structural_lows)
        
        print(f"Analyzed {len(range_states)} range states")
        if len(range_states) > 0:
            bullish_count = len([r for r in range_states if r['state'] == 'bullish'])
            bearish_count = len([r for r in range_states if r['state'] == 'bearish'])
            print(f"Bullish periods: {bullish_count}")
            print(f"Bearish periods: {bearish_count}")
        
        return self.range_data
    
    def create_interactive_chart(self, symbol, show_volume=True, output_dir="examples"):
        """
        Create an interactive plotly chart with dynamic range visualization
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
                subplot_titles=(f'{symbol} - Dynamic Range Analysis', 'Volume'),
                row_width=[0.7, 0.3]
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
        
        # Add structural pivots
        if hasattr(self, 'structural_highs') and len(self.structural_highs) > 0:
            fig.add_trace(go.Scatter(
                x=self.structural_highs['datetime'],
                y=self.structural_highs['price'],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='orange',
                    line=dict(width=2, color='darkorange')
                ),
                name='Structural High',
                text=[f"Structural High<br>${price:.2f}" for price in self.structural_highs['price']],
                hovertemplate='%{text}<extra></extra>'
            ), row=1 if show_volume else None, col=1 if show_volume else None)
        
        if hasattr(self, 'structural_lows') and len(self.structural_lows) > 0:
            fig.add_trace(go.Scatter(
                x=self.structural_lows['datetime'],
                y=self.structural_lows['price'],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='purple',
                    line=dict(width=2, color='darkviolet')
                ),
                name='Structural Low',
                text=[f"Structural Low<br>${price:.2f}" for price in self.structural_lows['price']],
                hovertemplate='%{text}<extra></extra>'
            ), row=1 if show_volume else None, col=1 if show_volume else None)
        
        # Add dynamic range lines
        if self.range_data is not None and len(self.range_data) > 0:
            # Create step-like range high line
            range_high_times = []
            range_high_values = []
            range_high_colors = []
            
            # Create step-like range low line
            range_low_times = []
            range_low_values = []
            range_low_colors = []
            
            for _, row in self.range_data.iterrows():
                range_high_times.append(row['datetime'])
                range_high_values.append(row['range_high'])
                
                range_low_times.append(row['datetime'])
                range_low_values.append(row['range_low'])
                
                # Color based on range state
                if row['state'] == 'bullish':
                    range_high_colors.append('green')
                    range_low_colors.append('green')
                else:
                    range_high_colors.append('red')
                    range_low_colors.append('red')
            
            # Add range high line
            if range_high_values and any(v is not None for v in range_high_values):
                fig.add_trace(go.Scatter(
                    x=range_high_times,
                    y=range_high_values,
                    mode='lines',
                    line=dict(color='blue', dash='solid', width=3),
                    name='Dynamic Range High',
                    showlegend=True,
                    hovertemplate='Range High: $%{y:.2f}<extra></extra>'
                ), row=1 if show_volume else None, col=1 if show_volume else None)
            
            # Add range low line
            if range_low_values and any(v is not None for v in range_low_values):
                fig.add_trace(go.Scatter(
                    x=range_low_times,
                    y=range_low_values,
                    mode='lines',
                    line=dict(color='red', dash='solid', width=3),
                    name='Dynamic Range Low',
                    showlegend=True,
                    hovertemplate='Range Low: $%{y:.2f}<extra></extra>'
                ), row=1 if show_volume else None, col=1 if show_volume else None)
            
            # Add range state indicators (optimized for performance)
            # Instead of many rectangles, use scatter points to show state changes
            state_changes = []
            prev_state = None
            
            for _, row in self.range_data.iterrows():
                if row['state'] != prev_state:
                    state_changes.append({
                        'datetime': row['datetime'],
                        'price': row['price'],
                        'state': row['state']
                    })
                    prev_state = row['state']
            
            if state_changes:
                state_df = pd.DataFrame(state_changes)
                bullish_changes = state_df[state_df['state'] == 'bullish']
                bearish_changes = state_df[state_df['state'] == 'bearish']
                
                if len(bullish_changes) > 0:
                    fig.add_trace(go.Scatter(
                        x=bullish_changes['datetime'],
                        y=bullish_changes['price'],
                        mode='markers',
                        marker=dict(
                            symbol='arrow-up',
                            size=8,
                            color='lightgreen',
                            line=dict(width=1, color='green')
                        ),
                        name='Bullish Range Start',
                        text=[f"Bullish Range Start<br>${price:.2f}" for price in bullish_changes['price']],
                        hovertemplate='%{text}<extra></extra>'
                    ), row=1 if show_volume else None, col=1 if show_volume else None)
                
                if len(bearish_changes) > 0:
                    fig.add_trace(go.Scatter(
                        x=bearish_changes['datetime'],
                        y=bearish_changes['price'],
                        mode='markers',
                        marker=dict(
                            symbol='arrow-down',
                            size=8,
                            color='lightcoral',
                            line=dict(width=1, color='red')
                        ),
                        name='Bearish Range Start',
                        text=[f"Bearish Range Start<br>${price:.2f}" for price in bearish_changes['price']],
                        hovertemplate='%{text}<extra></extra>'
                    ), row=1 if show_volume else None, col=1 if show_volume else None)
        
        # Add volume bars if requested
        if show_volume:
            colors = ['green' if close >= open else 'red' 
                     for close, open in zip(self.df['close'], self.df['open'])]
            
            fig.add_trace(go.Bar(
                x=self.df['datetime'],
                y=self.df['volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ), row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Dynamic Range-Based Analysis',
            xaxis_title='Time',
            yaxis_title='Price ($)',
            template='plotly_dark',
            showlegend=True,
            height=800 if show_volume else 600,
            hovermode='x unified'
        )
        
        # Remove rangeslider for cleaner look
        fig.update_layout(xaxis_rangeslider_visible=False)
        
        if show_volume:
            fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig
    
    def analyze_ranges(self):
        """
        Provide analysis of identified ranges and states
        """
        if self.range_data is None or len(self.range_data) == 0:
            print("No range data available")
            return
            
        print("\n" + "="*60)
        print("DYNAMIC RANGE ANALYSIS")
        print("="*60)
        
        total_periods = len(self.range_data)
        bullish_periods = len(self.range_data[self.range_data['state'] == 'bullish'])
        bearish_periods = len(self.range_data[self.range_data['state'] == 'bearish'])
        
        print(f"Total Analysis Periods: {total_periods}")
        print(f"Bullish Range Periods: {bullish_periods}")
        print(f"Bearish Range Periods: {bearish_periods}")
        
        if total_periods > 0:
            print(f"Bullish Time Ratio: {bullish_periods/total_periods:.1%}")
            print(f"Bearish Time Ratio: {bearish_periods/total_periods:.1%}")
        
        # Show current range state
        if len(self.range_data) > 0:
            current_state = self.range_data.iloc[-1]
            print(f"\nCurrent Range State: {current_state['state'].upper()}")
            if current_state['range_high'] is not None:
                print(f"Current Range High: ${current_state['range_high']:.2f}")
            if current_state['range_low'] is not None:
                print(f"Current Range Low: ${current_state['range_low']:.2f}")
        
        # Show structural pivots
        if hasattr(self, 'structural_highs') and len(self.structural_highs) > 0:
            print(f"\nStructural Highs Found: {len(self.structural_highs)}")
        if hasattr(self, 'structural_lows') and len(self.structural_lows) > 0:
            print(f"Structural Lows Found: {len(self.structural_lows)}")
        
        print("\nRecent Range States:")
        print("-" * 40)
        
        # Show last 5 range changes
        recent = self.range_data.tail(5)
        for _, state in recent.iterrows():
            state_type = state['state'].upper()
            price = state['price']
            time = state['datetime'].strftime('%Y-%m-%d %H:%M')
            range_high = f"${state['range_high']:.2f}" if state['range_high'] else "N/A"
            range_low = f"${state['range_low']:.2f}" if state['range_low'] else "N/A"
            print(f"{time} | {state_type:8} | Price: ${price:.2f} | High: {range_high} | Low: {range_low}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Dynamic Range-Based Market Analysis - Advanced range tracking with structural pivots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 pullback_viewer_interactive.py AAPL
  python3 pullback_viewer_interactive.py NVDA --days 7 --frequency 1
  python3 pullback_viewer_interactive.py TSLA --lookback 5 --no-volume
        """
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, NVDA, TSLA)')
    parser.add_argument('--days', type=int, default=7, help='Number of days of data (default: 7)')
    parser.add_argument('--frequency', type=int, default=5, choices=[1, 5, 15, 30, 60], 
                       help='Frequency in minutes (default: 5)')
    parser.add_argument('--lookback', type=int, default=3, 
                       help='Lookback period for structural pivot identification (default: 3)')
    parser.add_argument('--no-volume', action='store_true', help='Hide volume chart')
    parser.add_argument('--force-refresh', action='store_true', 
                       help='Force refresh authentication tokens')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RangeBasedAnalyzer()
    
    print("="*60)
    print("DYNAMIC RANGE-BASED MARKET ANALYZER")
    print("="*60)
    print(f"Symbol: {args.symbol.upper()}")
    print(f"Period: {args.days} days")
    print(f"Frequency: {args.frequency} minutes")
    print(f"Structural Pivot Lookback: {args.lookback}")
    print(f"Force Refresh: {args.force_refresh}")
    print("="*60)
    
    # Fetch data
    if not analyzer.fetch_data(args.symbol.upper(), args.days, args.frequency, args.force_refresh):
        print("Failed to fetch data. Exiting.")
        return
    
    # Analyze dynamic ranges
    range_data = analyzer.analyze_dynamic_ranges(lookback=args.lookback)
    
    # Analyze ranges
    analyzer.analyze_ranges()
    
    # Create and show interactive chart
    print("\nGenerating interactive chart...")
    fig = analyzer.create_interactive_chart(
        args.symbol.upper(), 
        show_volume=not args.no_volume
    )
    
    if fig:
        print("Opening interactive chart in browser...")
        fig.show()
        
        # Create examples directory if it doesn't exist
        output_dir = "examples"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as HTML file in examples directory
        filename = f"{args.symbol.upper()}_range_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(output_dir, filename)
        fig.write_html(filepath)
        print(f"Chart saved as: {filepath}")
    else:
        print("Failed to create chart")

if __name__ == "__main__":
    main()
