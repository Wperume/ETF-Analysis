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

### Command-Line Interface (Recommended)

The analyzer includes a command-line interface for quick operations:

```bash
# Display help
python etf_analyzer.py --help

# Load CSV files from directory and display summary
python etf_analyzer.py -d data

# Load CSV files and export DataFrame to parquet
python etf_analyzer.py -d data -f export -o etf_data.parquet

# Load previously saved parquet file and display summary
python etf_analyzer.py -i etf_data.parquet

# Load and export as CSV
python etf_analyzer.py -i etf_data.parquet -f export -o etf_data.csv

# Show list of ETFs
python etf_analyzer.py -d data -f list

# Export list to file
python etf_analyzer.py -d data -f list -o etf_list.txt

# Show all assets with ETF associations
python etf_analyzer.py -d data -f assets

# Export assets to CSV
python etf_analyzer.py -d data -f assets -o assets.csv

# Show asset-to-ETF mapping
python etf_analyzer.py -d data -f mapping

# Export mapping to CSV
python etf_analyzer.py -d data -f mapping -o mapping.csv

# Export summary to CSV
python etf_analyzer.py -d data -f summary -o summary.csv
```

**Command-Line Options:**
- `-d DIR` or `--data DIR`: Directory containing ETF CSV files
- `-i FILE` or `--import FILE`: Import previously exported DataFrame
- `-f FUNCTION` or `--function FUNCTION`: Operation to perform
  - `summary` (default): Display ETF portfolio summary
  - `list`: List all ETF symbols
  - `assets`: Show all assets with ETF associations
  - `mapping`: Show asset-to-ETF mapping
  - `export`: Export DataFrame to file (requires `-o`)
- `-o FILE` or `--output FILE`: Output file (if not specified, print to stdout)

**Note:** Either `-d` or `-i` must be specified. The `-o` option is required for `-f export`.

### Python API - Analyze All ETFs in a Directory

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

# Get list of all unique asset symbols
unique_symbols = portfolio.get_unique_assets()
print(f"Total assets: {len(unique_symbols)}")  # e.g., 500 unique stocks

# Find which ETFs contain a specific asset
asset_mapping = portfolio.get_asset_to_etf_mapping()
print(asset_mapping['AAPL'])  # ['SPY', 'VOO', 'QQQ', ...]

# Get most common assets across all ETFs (sorted by ETF count)
common_assets = portfolio.get_assets_by_etf_count()
print(common_assets.head(10))  # Top 10 most held assets

# Get all assets with their ETF associations (sorted alphabetically)
assets_alphabetical = portfolio.get_assets_with_etf_list()
print(assets_alphabetical.head(10))  # First 10 assets alphabetically

# Export asset analysis to CSV file
portfolio.export_asset_analysis('assets_by_symbol.csv', sort_by='symbol')
portfolio.export_asset_analysis('assets_by_etf_count.csv', sort_by='etf_count')

# Export the full DataFrame for later use
portfolio.export_dataframe('etf_data.parquet')  # Default format
portfolio.export_dataframe('etf_data.csv')      # CSV format
portfolio.export_dataframe('etf_data.pkl')      # Pickle format
portfolio.export_dataframe('etf_data')          # Uses .parquet by default

# Load a previously saved DataFrame
portfolio2 = ETFPortfolioAnalyzer('data')
portfolio2.load_dataframe('etf_data.parquet')
print(portfolio2.df.head())  # DataFrame is ready to use

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
- pyarrow: Parquet file support for efficient DataFrame storage

## License

MIT
