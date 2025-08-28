# Dynamic Range-Based Market Analysis Viewer

Advanced implementation with dynamic bullish/bearish range tracking, structural pivot identification, and sophisticated high/low updating rules.

## Overview

This tool implements a complex range-based algorithm that goes far beyond simple pullback identification. It creates a sophisticated dynamic range tracking system with intelligent state management and conservative body-based validation.

## Key Features

### 🎯 Dynamic Range State Management
- **Bullish/Bearish Range Assignment**: Market momentum assessment assigns persistent range states
- **State Persistence**: Ranges continue until broken by significant directional moves
- **Body-Based Validation**: Uses candle bodies (not wicks) for conservative sentiment analysis

### 📊 Structural Pivot Identification
- **Confirmed Highs/Lows**: Identifies pivots with confirmation on both sides
- **Configurable Lookback**: Adjustable confirmation period (default: 3 candles)
- **Visual Markers**: Orange triangles for structural highs, purple for structural lows

### 🔄 Dynamic High/Low Tracking
- **Intelligent Updates**: Range highs/lows update only under specific conditions
- **Last Major Movement Logic**: Tracks "sell before buy" and "buy before sell" patterns
- **Conservative Approach**: Body-based validation prevents false signals

### 📈 Advanced Visualization
- **Dynamic Range Lines**: Blue line for range highs, red line for range lows
- **State Change Indicators**: Arrows showing bullish/bearish range transitions
- **Structural Pivots**: Clear marking of confirmed pivot points
- **Performance Optimized**: Handles high-frequency data efficiently

## Algorithm Details

### Range State Logic

**Bullish Range Assignment:**
- Triggered when candle body closes above previous candle high
- Continues updating high until body exceeds current high
- Low determined by finding lowest point after last "sell before buy"
- Range persists until bearish break occurs

**Bearish Range Assignment:**
- Triggered when candle body closes below previous candle low  
- Continues updating low until body goes below current low
- High determined by finding highest point after last "buy before sell"
- Range persists until bullish break occurs

### High/Low Update Rules

**In Bullish Ranges:**
- High updates only when candle body (open or close) exceeds current high
- When high updates, low recalculates based on last major sell signal
- Provides dynamic support/resistance levels

**In Bearish Ranges:**
- Low updates only when candle body (open or close) goes below current low
- When low updates, high recalculates based on last major buy signal
- Maintains relevant range boundaries

### Structural Pivot Detection

**Structural High:**
- Higher than all surrounding highs within lookback period
- Confirmed on both left and right sides
- Represents significant resistance levels

**Structural Low:**
- Lower than all surrounding lows within lookback period
- Confirmed on both left and right sides
- Represents significant support levels

## Usage

### Basic Usage
```bash
python3 pullback_viewer_interactive.py AAPL
```

### Advanced Options
```bash
# High-frequency analysis
python3 pullback_viewer_interactive.py AAPL --days 1 --frequency 1

# Custom structural pivot detection
python3 pullback_viewer_interactive.py NVDA --lookback 5

# Without volume chart for cleaner view
python3 pullback_viewer_interactive.py TSLA --no-volume

# Force token refresh
python3 pullback_viewer_interactive.py MSFT --force-refresh
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `symbol` | Stock symbol (required) | - |
| `--days` | Number of days of data | 7 |
| `--frequency` | Frequency in minutes (1,5,15,30,60) | 5 |
| `--lookback` | Structural pivot confirmation period | 3 |
| `--no-volume` | Hide volume chart | False |
| `--force-refresh` | Force refresh authentication tokens | False |

## Example Results

### AAPL 1-Day 1-Minute Analysis
- **Total Analysis Periods**: 779
- **Bullish Range Periods**: 531 (68.2%)
- **Bearish Range Periods**: 230 (29.5%)
- **Structural Highs Found**: 64
- **Structural Lows Found**: 57
- **Current Range State**: BULLISH
- **Current Range High**: $230.90
- **Current Range Low**: $230.45

## Chart Elements

### Visual Components
- **Candlestick Chart**: Standard OHLC price data
- **Dynamic Range High**: Blue solid line tracking range resistance
- **Dynamic Range Low**: Red solid line tracking range support
- **Structural Highs**: Orange downward triangles
- **Structural Lows**: Purple upward triangles
- **Bullish Range Start**: Green upward arrows
- **Bearish Range Start**: Red downward arrows
- **Volume**: Optional volume bars (green/red)

### Interactive Features
- **Zoom & Pan**: Full chart navigation
- **Hover Details**: Detailed information on hover
- **Legend Toggle**: Show/hide specific elements
- **Time Navigation**: Precise time-based analysis

## Technical Implementation

### Core Classes
- **`RangeBasedAnalyzer`**: Main analysis engine
- **`StandaloneDataHandler`**: Schwab API integration
- **Dynamic State Tracking**: Real-time range management
- **Structural Analysis**: Pivot identification system

### Performance Optimizations
- **Efficient Visualization**: Optimized for high-frequency data
- **State Change Detection**: Minimal computational overhead
- **Memory Management**: Handles large datasets efficiently

## File Structure

```
pullback-viewer/
├── pullback_viewer_interactive.py    # Main analysis script
├── pullback_viewer.tos              # ThinkScript version
├── README.md                        # This documentation
├── examples/                        # Generated chart files
│   ├── AAPL_range_analysis_*.html
│   └── [other generated charts]
└── cs_tokens.json                   # API tokens (auto-generated)
```

## Requirements

### Python Dependencies
- pandas
- plotly
- requests
- numpy
- argparse

### API Requirements
- Schwab API credentials
- Valid authentication tokens
- Market data access

### Installation
```bash
pip install pandas plotly requests numpy
```

## Advanced Features

### Body-Based Validation
Unlike traditional technical analysis that uses wicks, this implementation focuses on candle bodies for more conservative and accurate sentiment analysis.

### Dynamic Range Recalculation
Ranges aren't static - they dynamically adjust based on market behavior, providing always-current support and resistance levels.

### Momentum Assessment
The algorithm assesses market momentum to assign range states, ensuring alignment with actual market direction.

### Conservative Approach
By requiring body-based validation and confirmed structural pivots, the system reduces false signals and provides more reliable analysis.

## Comparison with Simple Pullback Analysis

| Feature | Simple Pullbacks | Dynamic Range Analysis |
|---------|------------------|----------------------|
| State Management | None | Bullish/Bearish ranges |
| High/Low Logic | Static points | Dynamic updating rules |
| Validation | Wick-based | Body-based conservative |
| Structural Analysis | Basic | Confirmed pivots |
| Range Tracking | None | Continuous range management |
| Market Context | Limited | Full momentum assessment |

## Use Cases

### Day Trading
- Identify current range state for directional bias
- Use dynamic support/resistance for entries/exits
- Monitor structural pivots for key levels

### Swing Trading
- Assess longer-term range states
- Identify range breakouts and continuations
- Use structural pivots for position sizing

### Market Analysis
- Understand market structure and momentum
- Identify key support/resistance zones
- Analyze range-bound vs trending behavior

## Future Enhancements

- Multi-timeframe analysis
- Alert system for range state changes
- Statistical analysis of range performance
- Integration with additional data sources
- Machine learning pattern recognition

---

*This implementation represents a significant advancement over traditional pullback analysis, providing sophisticated range-based market structure analysis with dynamic state management and conservative validation.*
