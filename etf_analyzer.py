"""
ETF Holdings Analyzer
Analyze ETF holdings data from CSV files using pandas
"""

from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import matplotlib.pyplot as plt


class ETFAnalyzer:
    """Analyze ETF holdings from CSV files"""

    def __init__(self, csv_path: str):
        """
        Initialize the analyzer with a CSV file path

        Args:
            csv_path: Path to the ETF holdings CSV file
                Expected format: {SYMBOL}-etf-holdings.csv
        """
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        # Extract ETF symbol from filename (e.g., "SPY" from
        # "SPY-etf-holdings.csv")
        filename = self.csv_path.stem
        self.etf_name = filename.split("-etf-holdings")[0].upper()

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load CSV data into a pandas DataFrame

        Args:
            **kwargs: Additional arguments to pass to pd.read_csv()

        Returns:
            Loaded DataFrame with ETF symbol added as a column
        """
        df = pd.read_csv(self.csv_path, **kwargs)
        df.insert(0, "etf_symbol", self.etf_name)
        self.df = df
        print(f"Loaded {len(df)} holdings from {self.csv_path.name}")
        return df

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the holdings

        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        memory_kb = self.df.memory_usage(deep=True).sum() / 1024
        return {
            "total_holdings": len(self.df),
            "columns": list(self.df.columns),
            "memory_usage": f"{memory_kb:.2f} KB",
        }

    def get_top_holdings(
        self, n: int = 10, weight_column: str = "weight"
    ) -> pd.DataFrame:
        """
        Get the top N holdings by weight

        Args:
            n: Number of top holdings to return
            weight_column: Name of the weight/percentage column

        Returns:
            DataFrame with top holdings
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if weight_column not in self.df.columns:
            raise ValueError(
                f"Column '{weight_column}' not found in DataFrame"
            )

        return self.df.nlargest(n, weight_column)

    def analyze_by_sector(self, sector_column: str = "sector") -> pd.DataFrame:
        """
        Analyze holdings grouped by sector

        Args:
            sector_column: Name of the sector column

        Returns:
            DataFrame with sector analysis
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if sector_column not in self.df.columns:
            raise ValueError(
                f"Column '{sector_column}' not found in DataFrame"
            )

        sector_analysis = (
            self.df.groupby(sector_column)
            .agg({sector_column: "count"})
            .rename(columns={sector_column: "count"})
        )

        return sector_analysis.sort_values("count", ascending=False)

    def plot_top_holdings(
        self,
        n: int = 10,
        weight_column: str = "weight",
        name_column: str = "name",
    ):
        """
        Plot a bar chart of top N holdings

        Args:
            n: Number of top holdings to plot
            weight_column: Name of the weight/percentage column
            name_column: Name of the holding name column
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        top_holdings = self.get_top_holdings(n, weight_column)

        plt.figure(figsize=(12, 6))
        plt.barh(range(len(top_holdings)), top_holdings[weight_column])
        plt.yticks(
            range(len(top_holdings)), top_holdings[name_column].tolist()
        )
        plt.xlabel("Weight (%)")
        plt.title(f"Top {n} Holdings - {self.etf_name}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def export_analysis(
        self, output_path: str, analysis_type: str = "summary"
    ):
        """
        Export analysis results to a file

        Args:
            output_path: Path to save the output file
            analysis_type: Type of analysis to export
                ('summary', 'top_holdings', etc.)
        """
        output_file = Path(output_path)

        if analysis_type == "summary":
            stats = self.get_summary_stats()
            with open(output_file, "w") as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")

        print(f"Analysis exported to {output_file}")


class ETFPortfolioAnalyzer:
    """Analyze multiple ETF holdings from a directory"""

    def __init__(self, data_dir: str):
        """
        Initialize the portfolio analyzer with a data directory

        Args:
            data_dir: Path to directory containing ETF CSV files
                Expected format: {SYMBOL}-etf-holdings.csv
        """
        self.data_dir = Path(data_dir)
        self.df: Optional[pd.DataFrame] = None
        self.etf_analyzers: Dict[str, ETFAnalyzer] = {}

    def load_all_etfs(
        self, pattern: str = "*-etf-holdings.csv", **kwargs
    ) -> pd.DataFrame:
        """
        Load all ETF CSV files from the directory into a single DataFrame

        Args:
            pattern: Glob pattern for ETF files (default: *-etf-holdings.csv)
            **kwargs: Additional arguments to pass to pd.read_csv()

        Returns:
            Combined DataFrame with all ETF holdings
        """
        csv_files = list(self.data_dir.glob(pattern))

        if not csv_files:
            raise FileNotFoundError(
                f"No files matching '{pattern}' found in {self.data_dir}"
            )

        all_dfs = []
        for csv_file in sorted(csv_files):
            analyzer = ETFAnalyzer(str(csv_file))
            df = analyzer.load_data(**kwargs)
            all_dfs.append(df)
            self.etf_analyzers[analyzer.etf_name] = analyzer

        self.df = pd.concat(all_dfs, ignore_index=True)
        total_holdings = len(self.df)
        print(
            f"\nCombined {len(csv_files)} ETFs with "
            f"{total_holdings} total holdings"
        )
        return self.df

    def get_etf_list(self) -> List[str]:
        """Get list of ETF symbols that have been loaded"""
        return list(self.etf_analyzers.keys())

    def get_etf_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each ETF

        Returns:
            DataFrame with ETF summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        summary = self.df.groupby("etf_symbol").agg(
            holdings_count=("etf_symbol", "count")
        )
        return summary

    def filter_by_etf(self, etf_symbol: str) -> pd.DataFrame:
        """
        Filter holdings by ETF symbol

        Args:
            etf_symbol: ETF symbol to filter by

        Returns:
            DataFrame with holdings for specified ETF
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        return self.df[self.df["etf_symbol"] == etf_symbol.upper()]

    def get_columns(self, etf_symbol: Optional[str] = None) -> List[str]:
        """
        Get list of columns available in the data

        Args:
            etf_symbol: Optional ETF symbol to get columns for a specific ETF.
                If None, returns columns from combined DataFrame.

        Returns:
            List of column names
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        if etf_symbol:
            etf_data = self.filter_by_etf(etf_symbol)
            return list(etf_data.columns)
        return list(self.df.columns)

    def get_etf_assets(
        self,
        etf_symbol: str,
        symbol_col: str = "Symbol",
        name_col: str = "Name",
        weight_col: str = "% Weight",
        shares_col: str = "Shares",
    ) -> pd.DataFrame:
        """
        Extract underlying assets for a specific ETF

        Args:
            etf_symbol: ETF symbol to get assets for
            symbol_col: Name of the ticker/symbol column
            name_col: Name of the security name column
            weight_col: Name of the weight/percentage column
            shares_col: Name of the shares column

        Returns:
            DataFrame with Symbol, Name, Weight, and Shares columns

        Raises:
            ValueError: If no matching columns found or ETF doesn't exist
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        # Filter by ETF
        etf_data = self.filter_by_etf(etf_symbol)

        if len(etf_data) == 0:
            raise ValueError(f"No data found for ETF: {etf_symbol}")

        # Build column list - only include columns that exist
        columns_to_include = []
        column_mapping = {}

        if symbol_col in etf_data.columns:
            columns_to_include.append(symbol_col)
            column_mapping[symbol_col] = "Symbol"

        if name_col in etf_data.columns:
            columns_to_include.append(name_col)
            column_mapping[name_col] = "Name"

        if weight_col in etf_data.columns:
            columns_to_include.append(weight_col)
            column_mapping[weight_col] = "Weight"

        if shares_col in etf_data.columns:
            columns_to_include.append(shares_col)
            column_mapping[shares_col] = "Shares"

        # Warn if no columns found
        if not columns_to_include:
            available_cols = list(etf_data.columns)
            raise ValueError(
                f"None of the specified columns found in ETF data.\n"
                f"Looking for: {symbol_col}, {name_col}, "
                f"{weight_col}, {shares_col}\n"
                f"Available columns: {available_cols}"
            )

        # Select and rename columns
        result = etf_data[columns_to_include].copy()
        result = result.rename(columns=column_mapping)

        return result

    def get_asset_to_etf_mapping(
        self, symbol_col: str = "Symbol"
    ) -> Dict[str, List[str]]:
        """
        Create a mapping of assets to ETFs that contain them

        Args:
            symbol_col: Name of the asset symbol column

        Returns:
            Dictionary where keys are asset symbols and values are
            lists of ETF symbols that contain that asset

        Raises:
            ValueError: If data not loaded or symbol column not found
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        if symbol_col not in self.df.columns:
            available_cols = list(self.df.columns)
            raise ValueError(
                f"Column '{symbol_col}' not found in data.\n"
                f"Available columns: {available_cols}"
            )

        # Group by asset symbol and collect ETF symbols
        asset_mapping = {}
        for asset_symbol in self.df[symbol_col].unique():
            # Skip NaN values
            if pd.isna(asset_symbol):
                continue

            # Get all ETFs that contain this asset
            etfs_with_asset = self.df[self.df[symbol_col] == asset_symbol][
                "etf_symbol"
            ].unique()
            asset_mapping[asset_symbol] = sorted(list(etfs_with_asset))

        return asset_mapping

    def get_assets_by_etf_count(
        self, symbol_col: str = "Symbol", name_col: str = "Name"
    ) -> pd.DataFrame:
        """
        Get assets ranked by how many ETFs contain them

        Args:
            symbol_col: Name of the asset symbol column
            name_col: Name of the asset name column

        Returns:
            DataFrame with columns: Symbol, Name, ETF_Count, ETFs
            Sorted by ETF_Count descending
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        # Get the mapping
        mapping = self.get_asset_to_etf_mapping(symbol_col)

        # Build result DataFrame
        results = []
        for symbol, etfs in mapping.items():
            # Get the asset name (use first occurrence)
            asset_data = self.df[self.df[symbol_col] == symbol].iloc[0]
            name = asset_data.get(name_col, "N/A")

            results.append(
                {
                    "Symbol": symbol,
                    "Name": name,
                    "ETF_Count": len(etfs),
                    "ETFs": ", ".join(etfs),
                }
            )

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("ETF_Count", ascending=False)
        result_df = result_df.reset_index(drop=True)

        return result_df


def compare_etfs(
    etf_paths: List[str], weight_column: str = "weight"
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Compare multiple ETFs side by side

    Args:
        etf_paths: List of paths to ETF CSV files
        weight_column: Name of the weight column

    Returns:
        Comparison DataFrame
    """
    etf_data = {}

    for path in etf_paths:
        analyzer = ETFAnalyzer(path)
        analyzer.load_data()
        etf_data[analyzer.etf_name] = analyzer.df

    print(f"Loaded {len(etf_data)} ETFs for comparison")
    return etf_data


if __name__ == "__main__":
    # Example usage
    print("ETF Analyzer - Example Usage")
    print("-" * 50)

    print("\nOption 1: Analyze all ETFs in a directory")
    print("=" * 50)
    print("1. Place your ETF CSV files in the 'data/' directory")
    print("   Format: {SYMBOL}-etf-holdings.csv")
    print("   Examples: SPY-etf-holdings.csv, QQQ-etf-holdings.csv\n")
    print("2. Run the following code:\n")
    print("portfolio = ETFPortfolioAnalyzer('data')")
    print("portfolio.load_all_etfs()")
    print("print(portfolio.get_etf_summary())")
    print("print(portfolio.get_etf_list())")

    print("\n\nOption 2: Analyze a single ETF file")
    print("=" * 50)
    print("analyzer = ETFAnalyzer('data/SPY-etf-holdings.csv')")
    print("analyzer.load_data()")
    print("print(analyzer.get_summary_stats())")
    print("print(analyzer.get_top_holdings(10))")

    portfolio = ETFPortfolioAnalyzer("data")
    portfolio.load_all_etfs()
    print(portfolio.get_etf_summary())
    print(portfolio.get_etf_list())

    # Check available columns first
    print("\nAvailable columns in VOT:")
    print(portfolio.get_columns("VOT"))

    # Get assets using default column names
    print("\nVOT Assets (first 5):")
    print(portfolio.get_etf_assets("VOT").head())

    # Get asset to ETF mapping
    print("\nAsset to ETF mapping (showing AAPL as example):")
    mapping = portfolio.get_asset_to_etf_mapping()
    if "AAPL" in mapping:
        print(f"AAPL is in: {mapping['AAPL']}")

    # Get most common assets across all ETFs
    print("\nMost common assets (top 10):")
    print(portfolio.get_assets_by_etf_count().head(10))
