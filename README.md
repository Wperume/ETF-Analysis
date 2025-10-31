# ETF Holdings Analyzer

A Python project for analyzing ETF (Exchange-Traded Fund) CSV holdings files using pandas.

## Features

- Load and parse ETF holdings data from CSV files
- Generate summary statistics
- Identify top holdings by weight
- Analyze holdings by sector or other categories
- Visualize holdings with charts
- Compare multiple ETFs side by side
- Export analysis results

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Analyze All ETFs in a Directory (Recommended)

```python
from etf_analyzer import ETFPortfolioAnalyzer

# Load all ETF holdings from a directory
portfolio = ETFPortfolioAnalyzer('data')
portfolio.load_all_etfs()

# Get summary of all ETFs
print(portfolio.get_etf_summary())

# Get list of loaded ETFs
print(portfolio.get_etf_list())

# Get underlying assets for a specific ETF
spy_assets = portfolio.get_etf_assets('SPY')
print(spy_assets)  # Returns: Symbol, Name, Weight, Shares

# Find which ETFs contain a specific asset
asset_mapping = portfolio.get_asset_to_etf_mapping()
print(asset_mapping['AAPL'])  # ['SPY', 'VOO', 'QQQ', ...]

# Get most common assets across all ETFs
common_assets = portfolio.get_assets_by_etf_count()
print(common_assets.head(10))  # Top 10 most held assets

# Filter holdings by specific ETF
spy_holdings = portfolio.filter_by_etf('SPY')
print(spy_holdings)

# Access the combined DataFrame
print(portfolio.df.head())
```

### Analyze a Single ETF

```python
from etf_analyzer import ETFAnalyzer

# Load a single ETF holdings CSV file
analyzer = ETFAnalyzer('data/SPY-etf-holdings.csv')
analyzer.load_data()

# Get summary statistics
stats = analyzer.get_summary_stats()
print(stats)

# Get top 10 holdings
top_holdings = analyzer.get_top_holdings(n=10, weight_column='weight')
print(top_holdings)

# Analyze by sector
sector_analysis = analyzer.analyze_by_sector(sector_column='sector')
print(sector_analysis)

# Plot top holdings
analyzer.plot_top_holdings(n=10, weight_column='weight', name_column='name')
```

## CSV File Format

### File Naming Convention
Files must follow this pattern: `{SYMBOL}-etf-holdings.csv`

Examples:
- `SPY-etf-holdings.csv` (S&P 500 ETF)
- `QQQ-etf-holdings.csv` (Nasdaq-100 ETF)
- `VOO-etf-holdings.csv` (Vanguard S&P 500 ETF)

### CSV Structure
Your CSV files should contain ETF holdings data with columns such as:
- `name` or `ticker`: Security name or ticker symbol
- `weight` or `percentage`: Percentage of the ETF portfolio
- `sector`: Industry sector (optional)
- `shares`: Number of shares held (optional)
- `market_value`: Market value of the holding (optional)

Example CSV structure:
```csv
ticker,name,weight,sector,shares,market_value
AAPL,Apple Inc.,5.2,Technology,1000000,175000000
MSFT,Microsoft Corporation,4.8,Technology,900000,310000000
```

**Note:** When loaded, an `etf_symbol` column is automatically added to identify which ETF each holding belongs to.

## Project Structure

```
ETF-Analysis/
├── data/              # Place your CSV files here
├── etf_analyzer.py    # Main analyzer class
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical operations
- matplotlib: Data visualization
- seaborn: Statistical data visualization
- openpyxl: Excel file support

## License

MIT
