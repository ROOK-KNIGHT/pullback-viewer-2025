# Market Structure Analyzer

Based on TradingView's Trading Desk indicator, this interactive chart identifies market stages based on breaks of structure (BOS). It determines Accumulation, Distribution, Reaccumulation, Redistribution, or Neutral market stages using sophisticated wave analysis.

## Features

- **Real-time data fetching** using Schwab API integration
- **Interactive plotly charts** with zoom, pan, and hover functionality
- **Market structure analysis** with BOS High/Low tracking
- **Market Break (MB) identification** with bullish/bearish dots
- **Market stage classification** (Accumulation, Distribution, etc.)
- **Step-line visualization** matching TradingView behavior
- **Wave analysis** with up to 2000-candle lookback validation
- **Volume analysis** with optional volume subplot
- **Configurable parameters** for different timeframes

## What is Market Structure Analysis?

The Market Structure Analyzer implements sophisticated Break of Structure (BOS) logic:

- **BOS High/Low Tracking**: Dynamic high and low levels that update based on market breaks
- **Market Breaks (MB)**: Validated breaks of structure with wave analysis confirmation
- **Market Stages**: Classification into Accumulation, Distribution, Reaccumulation, Redistribution, or Neutral
- **Wave Validation**: Up to 2000-candle lookback to confirm valid market breaks
- **Step-line Visualization**: Clean step-line representation of BOS levels

## Market Stages Explained

- **Accumulation**: Bullish market bias with upward momentum
- **Distribution**: Bearish market bias with downward momentum  
- **Reaccumulation**: Continuation of bullish trend after consolidation
- **Redistribution**: Continuation of bearish trend after consolidation
- **Neutral**: No clear directional bias

## Usage

### Basic Usage
```bash
python3 pullback_viewer_interactive.py AAPL
```

### Advanced Options
```bash
# 1 day of 1-minute data (high frequency analysis)
python3 pullback_viewer_interactive.py NVDA --days 1 --frequency 1

# 7 days of 5-minute data
python3 pullback_viewer_interactive.py TSLA --days 7 --frequency 5

# Hide volume chart with custom line styling
python3 pullback_viewer_interactive.py MSFT --no-volume --line-width 2 --line-color red

# Force token refresh
python3 pullback_viewer_interactive.py GOOGL --force-refresh
```

### Command Line Options

- `symbol`: Stock symbol (required)
- `--days`: Number of days of data (default: 7)
- `--frequency`: Frequency in minutes - choices: 1, 5, 15, 30, 60 (default: 5)
- `--no-volume`: Hide volume chart
- `--force-refresh`: Force refresh authentication tokens
- `--line-width`: Width of BOS lines (default: 1)
- `--line-color`: Color of BOS lines (default: blue)

## Output

The script provides:

1. **Console Analysis**: 
   - Current market stage
   - Stage distribution across timeframe
   - Market Break (MB) point count
   - Data processing statistics

2. **Interactive Chart**: 
   - Candlestick price chart with BOS step-lines
   - Green dots (●) for bullish MB points
   - Red dots (●) for bearish MB points
   - Market stage indicator badge
   - Volume bars (optional)
   - Hover tooltips with detailed information

3. **HTML File**: 
   - Saved automatically with timestamp
   - Format: `{SYMBOL}_market_structure_{YYYYMMDD_HHMMSS}.html`
   - Can be shared or viewed later

## Example Output

### Console Analysis
```
============================================================
MARKET STRUCTURE ANALYZER
============================================================
Symbol: AAPL
Period: 1 days
Frequency: 1 minutes
============================================================
Successfully loaded 780 candles for AAPL
Identified 45 MB points
Market stages computed

============================================================
MARKET STAGE ANALYSIS
============================================================
Current Market Stage: Accumulation

Stage Distribution:
Neutral: 16 bars
Accumulation: 490 bars (62.8%)
Distribution: 274 bars (35.1%)
```

## Algorithm Details

### BOS Logic Implementation
```python
# Bullish MB: When close > BOS High
if close > self.bos_high[bar-1]:
    self.bos_high[bar] = high  # Use actual high
    self.bos_low[bar] = low    # Use actual low
    # Wave analysis validation...
    
# Bearish MB: When close < BOS Low  
if close < self.bos_low[bar]:
    self.bos_low[bar] = low    # Use actual low
    self.bos_high[bar] = high  # Use actual high
    # Wave analysis validation...
```

### Market Stage Classification
- **Sequence Tracking**: Monitors bullish/bearish MB sequences
- **Threshold Logic**: Uses sequence counts to determine stages
- **Dynamic Updates**: Real-time stage classification as new data arrives
- **Historical Context**: Maintains last 8 MB points for analysis

## Platform Versions

### PineScript Version (TradingView)

For TradingView users, a complete PineScript v6 implementation is available (`market_structure_analyzer.pine`) that provides the full Market Structure Analyzer functionality.

#### Installation in TradingView:
1. **Download** the `market_structure_analyzer.pine` file from the repository
2. **Open TradingView** and go to Pine Editor
3. **Copy and paste** the PineScript code into the editor
4. **Click "Add to Chart"** to apply the indicator
5. **Configure** line width and color in the indicator settings

#### Features:
- **Complete BOS Logic**: Full implementation matching the Python version
- **2000-Candle Wave Analysis**: Complete wave validation for market breaks
- **Market Stage Display**: Real-time market stage table in top-right corner
- **Step-line Visualization**: Perfect step-line BOS High/Low tracking
- **Configurable Parameters**: Adjustable line width (-10 to 10) and color
- **Market Stage Classification**: Accumulation, Distribution, Reaccumulation, Redistribution, Neutral
- **Visual Market Stage Table**: Color-coded stage indicator (green for bullish, red for bearish)

### ThinkScript Version (ThinkorSwim)

For ThinkorSwim users, a ThinkScript version is available (`pullback_viewer.tos`) that provides Market Structure Analysis with some limitations due to platform constraints.

### Installation in ThinkorSwim:
1. **Download** the `pullback_viewer.tos` file from the repository
2. **Open ThinkorSwim** and go to Charts
3. **Click Studies** → **Edit Studies** → **Import**
4. **Select** the downloaded `.tos` file
5. **Apply** the study to your chart

## Use Cases

Market Structure Analysis can be used to:

- **Identify market bias** and directional momentum
- **Spot key structural levels** for support and resistance
- **Time entries and exits** based on market stage transitions
- **Understand market context** for trading decisions
- **Validate breakouts** with proper wave analysis
- **Track institutional accumulation/distribution** patterns

## Requirements

- Valid Schwab API tokens (automatically refreshed)
- Python packages: plotly, pandas, numpy, requests
- API keys configured in `/Users/isaac/Desktop/Projects/CS_KEYS/KEYS.json`

## Authentication

The script automatically handles token refresh. If you get authentication errors, ensure your Schwab API credentials are properly configured and you have valid refresh tokens.

## Chart Features

The interactive chart includes:
- **Zoom and Pan**: Mouse wheel and drag functionality
- **Hover Details**: Price, MB points, and market stage information
- **Legend Toggle**: Click to show/hide data series
- **Dark Theme**: Professional trading interface
- **Step-line BOS Levels**: Clean visualization matching TradingView
- **MB Point Markers**: Clear bullish (green) and bearish (red) indicators
- **Market Stage Badge**: Real-time stage display
- **Volume Correlation**: Optional volume bars with price-matched colors

## Repository Structure

```
pullback-viewer/
├── README.md                           # This documentation
├── pullback_viewer_interactive.py     # Main Python script (MarketStructureAnalyzer)
├── market_structure_analyzer.pine     # PineScript version for TradingView
├── pullback_viewer.tos                # ThinkScript version for ThinkorSwim
├── examples/                          # Interactive chart examples
│   └── AAPL_market_structure_*.html  # Current market structure charts
├── cs_tokens.json                    # API authentication tokens
└── KEYS.json                         # API credentials (external)
```

## Files Generated

Each run creates an HTML file with format:
`{SYMBOL}_market_structure_{YYYYMMDD_HHMMSS}.html`

Example: `AAPL_market_structure_20250827_223554.html`

All generated charts are automatically saved to the `examples/` directory.

## How to View Interactive Charts

### 📥 Download Charts from GitHub (Recommended for Viewers)

If you want to view the pre-generated charts without running the Python script:

1. **Go to the GitHub repository**: [pullback-viewer-2025](https://github.com/ROOK-KNIGHT/pullback-viewer-2025)
2. **Click on any HTML file** in the examples directory
3. **Click the "Download raw file" button** (download icon in the top-right of the file view)
4. **Save the file** to your computer
5. **Double-click the downloaded HTML file** to open it in your browser

**Available Charts:**
- 📱 **AAPL**: Apple Inc. market structure analysis with BOS levels and MB points

### 🖥️ Open Local HTML Charts

If you've generated charts locally using the Python script:

#### Method 1: Double-click (Easiest)
Simply double-click on any generated HTML file to open it in your default web browser.

#### Method 2: Command Line
```bash
# Open specific chart
open AAPL_market_structure_20250827_223554.html

# On Windows
start AAPL_market_structure_20250827_223554.html

# On Linux
xdg-open AAPL_market_structure_20250827_223554.html
```

#### Method 3: From Browser
1. Open your web browser
2. Press `Ctrl+O` (or `Cmd+O` on Mac)
3. Navigate to the HTML file and select it
4. The interactive chart will load with full functionality

#### Method 4: Drag and Drop
Drag the HTML file directly into any open browser window or tab.

### Chart Interaction Tips
Once opened, you can:
- **Zoom**: Mouse wheel or click-drag to select area
- **Pan**: Click and drag to move around
- **Hover**: Mouse over points for detailed MB information
- **Toggle**: Click legend items to show/hide BOS lines, MB points, etc.
- **Reset**: Double-click to reset zoom
- **Analyze**: Observe market stage transitions and BOS level behavior
- **Download**: Use browser's save/print functions to export

## Technical Implementation

### Key Classes
- **MarketStructureAnalyzer**: Main analysis engine
- **StandaloneDataHandler**: Schwab API integration
- **Wave Analysis**: 2000-candle lookback validation
- **Step-line Visualization**: TradingView-matching behavior

### Performance Optimizations
- Efficient wave analysis algorithms
- Optimized for high-frequency 1-minute data
- Memory-efficient BOS level tracking
- Fast MB point identification and validation

### Data Processing
- Real-time Schwab API data fetching
- Automatic token refresh handling
- Robust error handling and validation
- Support for multiple timeframes (1m, 5m, 15m, 30m, 60m)
