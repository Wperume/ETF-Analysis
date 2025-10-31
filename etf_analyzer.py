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
        self.etf_analyzers: Dict[str, Optional[ETFAnalyzer]] = {}

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

    def get_unique_assets(self, symbol_col: str = "Symbol") -> List[str]:
        """
        Get list of all unique asset symbols across all ETFs

        Args:
            symbol_col: Name of the asset symbol column

        Returns:
            Sorted list of unique asset symbols, excluding NaN,
            empty strings, and "n/a" values

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

        # Get unique symbols
        unique_symbols = self.df[symbol_col].unique()

        # Filter out invalid values
        valid_symbols = []
        for symbol in unique_symbols:
            # Skip NaN/None values
            if pd.isna(symbol):
                continue

            # Convert to string and strip whitespace
            symbol_str = str(symbol).strip()

            # Skip empty strings and "n/a" (case insensitive)
            if symbol_str == "" or symbol_str.lower() == "n/a":
                continue

            valid_symbols.append(symbol_str)

        return sorted(valid_symbols)

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

            # Convert to string and check for invalid values
            symbol_str = str(asset_symbol).strip()
            if symbol_str == "" or symbol_str.lower() == "n/a":
                continue

            # Get all ETFs that contain this asset
            etfs_with_asset = self.df[self.df[symbol_col] == asset_symbol][
                "etf_symbol"
            ].unique()
            asset_mapping[symbol_str] = sorted(list(etfs_with_asset))

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

    def get_assets_with_etf_list(
        self, symbol_col: str = "Symbol", name_col: str = "Name"
    ) -> pd.DataFrame:
        """
        Get all assets with their ETF associations, sorted by symbol

        Args:
            symbol_col: Name of the asset symbol column
            name_col: Name of the asset name column

        Returns:
            DataFrame with columns: Symbol, Name, ETF_Count, ETFs
            Sorted alphabetically by Symbol
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
        result_df = result_df.sort_values("Symbol", ascending=True)
        result_df = result_df.reset_index(drop=True)

        return result_df

    def export_asset_analysis(
        self,
        output_path: str,
        sort_by: str = "symbol",
        symbol_col: str = "Symbol",
        name_col: str = "Name",
    ) -> None:
        """
        Export asset analysis to CSV file

        Args:
            output_path: Path to output CSV file
            sort_by: Sort order - "symbol" (alphabetical) or
                "etf_count" (by number of ETFs)
            symbol_col: Name of the asset symbol column
            name_col: Name of the asset name column

        Raises:
            ValueError: If sort_by is not "symbol" or "etf_count"
        """
        if sort_by == "symbol":
            df = self.get_assets_with_etf_list(symbol_col, name_col)
        elif sort_by == "etf_count":
            df = self.get_assets_by_etf_count(symbol_col, name_col)
        else:
            raise ValueError(
                f"sort_by must be 'symbol' or 'etf_count', " f"got: {sort_by}"
            )

        # Export to CSV with proper quoting for comma-separated ETF lists
        output_file = Path(output_path)
        df.to_csv(output_file, index=False, quoting=1)  # QUOTE_ALL

        print(f"Asset analysis exported to {output_file}")
        print(f"Total assets: {len(df)}")
        print(f"Sort order: {sort_by}")

    def export_dataframe(self, output_path: str) -> None:
        """
        Export the full combined DataFrame to a file

        Supports multiple formats based on file extension:
        - .parquet (default): Efficient binary format, preserves types
        - .csv: Human-readable text format
        - .pkl/.pickle: Python pickle format

        Args:
            output_path: Path to output file. If no extension provided,
                .parquet is used as default

        Raises:
            ValueError: If data not loaded or unsupported file format
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        output_file = Path(output_path)

        # Add default extension if none provided
        if not output_file.suffix:
            output_file = output_file.with_suffix(".parquet")

        # Export based on file extension
        extension = output_file.suffix.lower()

        if extension == ".parquet":
            self.df.to_parquet(output_file, index=False)
        elif extension == ".csv":
            self.df.to_csv(output_file, index=False)
        elif extension in [".pkl", ".pickle"]:
            self.df.to_pickle(output_file)
        else:
            raise ValueError(
                f"Unsupported format: {extension}. "
                f"Use .parquet, .csv, .pkl, or .pickle"
            )

        print(f"DataFrame exported to {output_file}")
        print(f"Format: {extension}")
        print(f"Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")

    def load_dataframe(self, input_path: str) -> pd.DataFrame:
        """
        Load a previously exported DataFrame from a file

        Supports multiple formats based on file extension:
        - .parquet (default): Efficient binary format
        - .csv: Human-readable text format
        - .pkl/.pickle: Python pickle format

        Args:
            input_path: Path to input file. If no extension provided,
                .parquet is assumed

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If unsupported file format
        """
        input_file = Path(input_path)

        # Add default extension if none provided
        if not input_file.suffix:
            input_file = input_file.with_suffix(".parquet")

        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")

        # Load based on file extension
        extension = input_file.suffix.lower()

        if extension == ".parquet":
            df = pd.read_parquet(input_file)
        elif extension == ".csv":
            df = pd.read_csv(input_file)
        elif extension in [".pkl", ".pickle"]:
            df = pd.read_pickle(input_file)
        else:
            raise ValueError(
                f"Unsupported format: {extension}. "
                f"Use .parquet, .csv, .pkl, or .pickle"
            )

        self.df = df
        print(f"DataFrame loaded from {input_file}")
        print(f"Format: {extension}")
        print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

        # Rebuild etf_analyzers dictionary
        self.etf_analyzers = {}
        if "etf_symbol" in df.columns:
            for etf_symbol in df["etf_symbol"].unique():
                self.etf_analyzers[etf_symbol] = None

        return df


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


def load_config() -> dict:
    """
    Load configuration from .etfrc file in current directory or home
    directory. Returns a dictionary of configuration values.

    Config file format (INI):
    [defaults]
    data_dir = data
    function = summary
    symbol_col = Symbol
    name_col = Name
    weight_col = % Weight
    shares_col = Shares
    """
    import configparser
    from pathlib import Path

    # Set built-in defaults first
    defaults = {
        "function": "summary",
        "symbol_col": "Symbol",
        "name_col": "Name",
        "weight_col": "% Weight",
        "shares_col": "Shares",
    }

    config = configparser.ConfigParser()
    config_paths = [
        Path.cwd() / ".etfrc",  # Current directory
        Path.home() / ".etfrc",  # Home directory
    ]

    # Try to load config file
    for config_path in config_paths:
        if config_path.exists():
            config.read(config_path)
            break

    # Override with config file values if they exist
    if "defaults" in config:
        section = config["defaults"]
        if "data_dir" in section:
            defaults["data"] = section["data_dir"]
        if "function" in section:
            defaults["function"] = section["function"]
        if "symbol_col" in section:
            defaults["symbol_col"] = section["symbol_col"]
        if "name_col" in section:
            defaults["name_col"] = section["name_col"]
        if "weight_col" in section:
            defaults["weight_col"] = section["weight_col"]
        if "shares_col" in section:
            defaults["shares_col"] = section["shares_col"]

    return defaults


def main():
    """
    Command-line interface for ETF Analyzer
    """
    import argparse
    import sys

    # Load configuration file defaults
    config_defaults = load_config()

    parser = argparse.ArgumentParser(
        description="ETF Holdings Analyzer - Load, analyze, and "
        "export ETF portfolio data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load CSV files from directory and display summary
  python etf_analyzer.py -d data

  # Show assets unique to one ETF only
  python etf_analyzer.py -d data -f unique

  # Show assets that overlap across multiple ETFs
  python etf_analyzer.py -d data -f overlap

  # Export unique assets to CSV
  python etf_analyzer.py -d data -f unique -o unique_assets.csv

  # Load CSV files and export to parquet
  python etf_analyzer.py -d data -f export -o etf_data.parquet

  # Load previously saved parquet file and display summary
  python etf_analyzer.py -i etf_data.parquet

  # Load parquet and export as CSV
  python etf_analyzer.py -i etf_data.parquet -f export -o etf_data.csv

Configuration File:
  Create a .etfrc file in your current or home directory with:
  [defaults]
  data_dir = data
  function = summary
  symbol_col = Symbol
  name_col = Name
  weight_col = %% Weight
  shares_col = Shares

  Note: Use %% to escape percent signs in .etfrc file
        """,
    )

    # Apply config file defaults
    parser.set_defaults(**config_defaults)

    # Input options (mutually exclusive)
    # Note: required=True unless config provides data_dir
    input_required = "data" not in config_defaults
    input_group = parser.add_mutually_exclusive_group(
        required=input_required
    )
    input_group.add_argument(
        "-d",
        "--data",
        metavar="DIR",
        help="Directory containing ETF CSV files (*-etf-holdings.csv)",
    )
    input_group.add_argument(
        "-i",
        "--import",
        dest="import_file",
        metavar="FILE",
        help="Import previously exported DataFrame file",
    )

    # Function to perform
    parser.add_argument(
        "-f",
        "--function",
        choices=[
            "export",
            "summary",
            "list",
            "assets",
            "mapping",
            "unique",
            "overlap",
        ],
        help="Operation to perform (default: summary)",
    )

    # Output file
    parser.add_argument(
        "-o",
        "--output",
        metavar="FILE",
        help="Output file (if not specified, print to stdout)",
    )

    # Column name overrides
    parser.add_argument(
        "--symbol-col",
        metavar="COLUMN",
        help="Column name for asset symbol (default: Symbol)",
    )
    parser.add_argument(
        "--name-col",
        metavar="COLUMN",
        help="Column name for asset name (default: Name)",
    )
    parser.add_argument(
        "--weight-col",
        metavar="COLUMN",
        help="Column name for weight/percentage (default: %% Weight)",
    )
    parser.add_argument(
        "--shares-col",
        metavar="COLUMN",
        help="Column name for shares (default: Shares)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.function == "export" and not args.output:
        parser.error("-f export requires -o OUTPUT to be specified")

    try:
        # Initialize portfolio analyzer
        portfolio = ETFPortfolioAnalyzer(".")

        # Load data
        if args.data:
            print(f"Loading ETF data from directory: {args.data}")
            portfolio.data_dir = Path(args.data)
            portfolio.load_all_etfs()
        elif args.import_file:
            print(f"Importing DataFrame from: {args.import_file}")
            portfolio.load_dataframe(args.import_file)

        print()  # Blank line after loading messages

        # Execute requested function
        if args.function == "export":
            portfolio.export_dataframe(args.output)

        elif args.function == "summary":
            summary = portfolio.get_etf_summary()
            if args.output:
                summary.to_csv(args.output, index=True)
                print(f"Summary exported to: {args.output}")
            else:
                print("ETF Portfolio Summary")
                print("=" * 60)
                print(summary)
                print()
                print(f"Total ETFs: {len(summary)}")
                total_holdings = summary["holdings_count"].sum()
                print(f"Total holdings across all ETFs: {total_holdings}")

        elif args.function == "list":
            etf_list = portfolio.get_etf_list()
            if args.output:
                with open(args.output, "w") as f:
                    for etf in etf_list:
                        f.write(f"{etf}\n")
                print(f"ETF list exported to: {args.output}")
            else:
                print("ETF List")
                print("=" * 60)
                for etf in etf_list:
                    print(etf)
                print()
                print(f"Total ETFs: {len(etf_list)}")

        elif args.function == "assets":
            assets = portfolio.get_assets_with_etf_list(
                symbol_col=args.symbol_col, name_col=args.name_col
            )
            if args.output:
                assets.to_csv(args.output, index=False)
                print(f"Asset list exported to: {args.output}")
            else:
                print("Assets Across All ETFs (sorted by symbol)")
                print("=" * 60)
                print(assets.to_string())
                print()
                print(f"Total unique assets: {len(assets)}")

        elif args.function == "mapping":
            mapping = portfolio.get_asset_to_etf_mapping(
                symbol_col=args.symbol_col
            )
            if args.output:
                # Export mapping as CSV with proper formatting
                rows = []
                for symbol, etfs in sorted(mapping.items()):
                    rows.append(
                        {"Symbol": symbol, "ETFs": ", ".join(etfs)}
                    )
                df = pd.DataFrame(rows)
                df.to_csv(args.output, index=False)
                print(f"Asset-to-ETF mapping exported to: {args.output}")
            else:
                print("Asset to ETF Mapping")
                print("=" * 60)
                for symbol, etfs in sorted(mapping.items()):
                    print(f"{symbol}: {', '.join(etfs)}")
                print()
                print(f"Total unique assets: {len(mapping)}")

        elif args.function == "unique":
            # Get assets that appear in only one ETF (ETF_Count = 1)
            assets = portfolio.get_assets_with_etf_list(
                symbol_col=args.symbol_col, name_col=args.name_col
            )
            unique_assets = assets[assets["ETF_Count"] == 1]
            if args.output:
                unique_assets.to_csv(args.output, index=False)
                print(f"Unique assets exported to: {args.output}")
            else:
                print("Assets Unique to One ETF (ETF_Count = 1)")
                print("=" * 60)
                print(unique_assets.to_string())
                print()
                print(f"Total unique assets: {len(unique_assets)}")

        elif args.function == "overlap":
            # Get assets that appear in more than one ETF (ETF_Count > 1)
            assets = portfolio.get_assets_with_etf_list(
                symbol_col=args.symbol_col, name_col=args.name_col
            )
            overlap_assets = assets[assets["ETF_Count"] > 1]
            # Sort by ETF_Count descending for better readability
            overlap_assets = overlap_assets.sort_values(
                "ETF_Count", ascending=False
            )
            overlap_assets = overlap_assets.reset_index(drop=True)
            if args.output:
                overlap_assets.to_csv(args.output, index=False)
                print(f"Overlapping assets exported to: {args.output}")
            else:
                print("Assets in Multiple ETFs (ETF_Count > 1)")
                print("=" * 60)
                print(overlap_assets.to_string())
                print()
                print(f"Total overlapping assets: {len(overlap_assets)}")
                if len(overlap_assets) > 0:
                    max_count = overlap_assets["ETF_Count"].max()
                    print(
                        f"Maximum overlap: {max_count} ETFs "
                        f"({overlap_assets.iloc[0]['Symbol']})"
                    )

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
