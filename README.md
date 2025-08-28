# Interactive Pullback Viewer

Based on TradingView's Pullback Viewer by emka, this interactive chart identifies valid pullback points in trending markets using live data from your Schwab API handlers.

## Features

- **Real-time data fetching** using existing Schwab API handlers
- **Interactive plotly charts** with zoom, pan, and hover functionality
- **Pullback identification** for both bullish and bearish trends
- **Configurable parameters** for different timeframes and analysis types
- **Volume analysis** with optional volume subplot
- **Structure-only mode** to show only the most recent pullbacks of each type

## What is a Valid Pullback?

According to the TradingView indicator:

- **Bearish Pullback**: Current candle body closes above the previous candle's high (bullish candle breaking above resistance)
- **Bullish Pullback**: Current candle body closes below the previous candle's low (bearish candle breaking below support)
- Must be a **clean body close**, not just a wick touching the level

## Usage

### Basic Usage
```bash
python3 pullback_viewer_interactive.py AAPL
```

### Advanced Options
```bash
# 7 days of 1-minute data
python3 pullback_viewer_interactive.py NVDA --days 7 --frequency 1

# Structure-only mode (shows only most recent pullbacks)
python3 pullback_viewer_interactive.py TSLA --structure-only

# Hide volume chart
python3 pullback_viewer_interactive.py MSFT --no-volume

# Combine options
python3 pullback_viewer_interactive.py GOOGL --days 5 --frequency 15 --structure-only --no-volume
```

### Command Line Options

- `symbol`: Stock symbol (required)
- `--days`: Number of days of data (default: 7)
- `--frequency`: Frequency in minutes - choices: 1, 5, 15, 30, 60 (default: 5)
- `--structure-only`: Show only structural pullbacks (last of each type)
- `--no-volume`: Hide volume chart

## Output

The script provides:

1. **Console Analysis**: 
   - Total pullback count
   - Bullish vs bearish ratio
   - Recent pullback details

2. **Interactive Chart**: 
   - Candlestick price chart
   - Red dots (●) for bearish pullbacks
   - Green dots (●) for bullish pullbacks
   - Volume bars (optional)
   - Hover tooltips with details

3. **HTML File**: 
   - Saved automatically with timestamp
   - Can be shared or viewed later

## Examples

### Regular Mode
Shows all pullback points identified in the data:
```bash
python3 pullback_viewer_interactive.py AAPL --days 3 --frequency 5
```
Output: `Identified 76 pullback points`

### Structure-Only Mode
Shows only the most recent pullback of each type:
```bash
python3 pullback_viewer_interactive.py NVDA --days 2 --frequency 15 --structure-only
```
Output: `Identified 2 pullback points`

## ThinkScript Version

For ThinkorSwim users, a ThinkScript version is available (`pullback_viewer.tos`) that provides the same pullback identification logic directly in your ThinkorSwim charts.

### Installation in ThinkorSwim:
1. **Download** the `pullback_viewer.tos` file from the repository
2. **Open ThinkorSwim** and go to Charts
3. **Click Studies** → **Edit Studies** → **Import**
4. **Select** the downloaded `.tos` file
5. **Apply** the study to your chart

### Features:
- **Green dots** above candles for bullish pullbacks
- **Red dots** below candles for bearish pullbacks
- **Same logic** as the Python version: clean body closes outside previous candle range
- **Optional chart bubbles** (uncomment the AddChartBubble lines for labels)
- **Customizable** colors and positioning

### ThinkScript Logic:
```thinkscript
# Bullish pullback: current close > previous high AND bullish candle
def bullishPB = c > h[1] and c > o;

# Bearish pullback: current close < previous low AND bearish candle  
def bearishPB = c < l[1] and c < o;
```

## Use Cases

As mentioned in the original TradingView indicator, pullbacks can be used to:

- **Identify supply and demand zones**
- **Spot key levels for support and resistance**
- **Use as anchor points for trendlines**
- **Find potential reaction points in trending markets**

## Requirements

- Valid Schwab API tokens (automatically refreshed)
- Python packages: plotly, pandas, numpy, requests
- API keys configured in `/Users/isaac/Desktop/Projects/CS_KEYS/KEYS.json`

## Authentication

The script automatically handles token refresh. If you get authentication errors, ensure your Schwab API credentials are properly configured and you have valid refresh tokens.

## Chart Features

The interactive chart includes:
- **Zoom and Pan**: Mouse wheel and drag
- **Hover Details**: Price and pullback information
- **Legend Toggle**: Click to show/hide data series
- **Dark Theme**: Professional trading interface
- **Time Navigation**: Click and drag on time axis
- **Volume Correlation**: Optional volume bars with price-matched colors

## Repository Structure

```
pullback-viewer/
├── README.md                           # This documentation
├── pullback_viewer_interactive.py     # Main Python script
├── pullback_viewer.tos                # ThinkScript version for ThinkorSwim
├── examples/                          # Interactive chart examples
│   ├── AAPL_pullback_viewer_*.html   # Apple charts (multiple versions)
│   ├── NVDA_pullback_viewer_*.html   # NVIDIA chart (structure-only)
│   ├── MSFT_pullback_viewer_*.html   # Microsoft chart (15min frequency)
│   ├── GOOGL_pullback_viewer_*.html  # Google chart (5min frequency)
│   ├── SPY_pullback_viewer_*.html    # S&P 500 ETF charts (multiple versions)
│   ├── QQQ_pullback_viewer_*.html    # NASDAQ ETF (structure-only)
│   └── AMD_pullback_viewer_*.html    # AMD chart (15min, no-volume)
├── charts/                            # Legacy chart directory
├── docs/                             # Additional documentation
└── cs_tokens.json                    # API authentication tokens
```

## Files Generated

Each run creates an HTML file with format:
`{SYMBOL}_pullback_viewer_{YYYYMMDD_HHMMSS}.html`

Example: `AAPL_pullback_viewer_20250826_210331.html`

All generated charts are automatically saved to the `examples/` directory.

## How to View Interactive Charts

### 📥 Download Charts from GitHub (Recommended for Viewers)

If you want to view the pre-generated charts without running the Python script:

1. **Go to the GitHub repository**: [pullback-viewer-2025](https://github.com/ROOK-KNIGHT/pullback-viewer-2025)
2. **Click on any HTML file** (e.g., `NVDA_pullback_viewer_20250826_210913.html`)
3. **Click the "Download raw file" button** (download icon in the top-right of the file view)
4. **Save the file** to your computer
5. **Double-click the downloaded HTML file** to open it in your browser

**Available Charts:**
- 📱 **AAPL**: Apple Inc. pullback analysis (multiple timeframes available)
- 🎮 **NVDA**: NVIDIA Corporation (2 days, 15min, structure-only - 2 pullbacks, 50/50 split)
- 💻 **MSFT**: Microsoft Corporation (2 days, 15min - 25 pullbacks, 28% bullish, 72% bearish)
- 🔍 **GOOGL**: Google/Alphabet (3 days, 5min - 76 pullbacks, 51.3% bullish, 48.7% bearish)
- 📊 **SPY**: S&P 500 ETF (1 day, 5min - 38 pullbacks, 60.5% bullish, 39.5% bearish)
- 🏛️ **QQQ**: NASDAQ ETF (2 days, 15min, structure-only - 2 pullbacks, 50/50 split)
- 🔥 **AMD**: Advanced Micro Devices (1 day, 15min, no-volume - 12 pullbacks, 66.7% bullish, 33.3% bearish)

### 🖥️ Open Local HTML Charts

If you've generated charts locally using the Python script:

#### Method 1: Double-click (Easiest)
Simply double-click on any generated HTML file to open it in your default web browser.

#### Method 2: Command Line
```bash
# Open specific chart
open AAPL_pullback_viewer_20250826_210331.html

# On Windows
start AAPL_pullback_viewer_20250826_210331.html

# On Linux
xdg-open AAPL_pullback_viewer_20250826_210331.html
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
- **Hover**: Mouse over points for detailed information
- **Toggle**: Click legend items to show/hide data series
- **Reset**: Double-click to reset zoom
- **Download**: Use browser's save/print functions to export
