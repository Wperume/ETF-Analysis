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

### Basic Analysis

```python
from etf_analyzer import ETFAnalyzer

# Load an ETF holdings CSV file
analyzer = ETFAnalyzer('data/your_etf.csv')
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

### Compare Multiple ETFs

```python
from etf_analyzer import compare_etfs

etf_files = ['data/etf1.csv', 'data/etf2.csv', 'data/etf3.csv']
comparison = compare_etfs(etf_files)
```

## CSV File Format

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
