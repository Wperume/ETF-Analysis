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

    def load_data(
        self, symbol_col: str = "Symbol", no_col: str = "No.", **kwargs
    ) -> pd.DataFrame:
        """
        Load CSV data into a pandas DataFrame

        Args:
            symbol_col: Column name containing asset symbols
            no_col: Column name containing row numbers/indices
            **kwargs: Additional arguments to pass to pd.read_csv()

        Returns:
            Loaded DataFrame with ETF symbol added as a column
        """
        # Preserve "n/a" as a literal string, not NaN
        # Only treat empty strings and actual NaN as missing values
        df = pd.read_csv(
            self.csv_path, keep_default_na=False, na_values=[""], **kwargs
        )

        # Generate synthetic symbols for empty or invalid Symbol values
        # Treat empty strings, NaN, and ":" as missing symbols
        # Format: ETF_SYMBOL + No. (e.g., CANE1, CANE2)
        if symbol_col in df.columns and no_col in df.columns:
            empty_mask = (
                (df[symbol_col] == "")
                | (df[symbol_col].isna())
                | (df[symbol_col] == ":")
            )
            df.loc[empty_mask, symbol_col] = (
                self.etf_name + df.loc[empty_mask, no_col].astype(str)
            )

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

    def get_etf_summary(
        self,
        symbol_col: str = "Symbol",
        weight_col: str = "% Weight",
        include_assets: bool = True,
        sort_by: str = "weight",
    ) -> pd.DataFrame:
        """
        Get summary statistics for each ETF

        Args:
            symbol_col: Column name containing asset symbols
            weight_col: Column name containing asset weights/percentages
            include_assets: If True, include comma-separated list of
                assets in the summary
            sort_by: Sort assets by "weight" (descending) or "alpha"
                (alphabetically)

        Returns:
            DataFrame with ETF summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        summary = self.df.groupby("etf_symbol").agg(
            holdings_count=("etf_symbol", "count")
        )

        # Add assets column if requested
        if include_assets and symbol_col in self.df.columns:
            if sort_by == "alpha":
                # Alphabetical sorting
                assets_by_etf = (
                    self.df.groupby("etf_symbol")[symbol_col]
                    .apply(
                        lambda x: ", ".join(
                            sorted(x.astype(str).unique())
                        )
                    )
                    .rename("assets")
                )
            else:
                # Weight-based sorting (default)
                def format_assets_by_weight(group):
                    # Get access to both columns via full DataFrame
                    if (
                        weight_col in group.index.names
                        or weight_col in self.df.columns
                    ):
                        # Get the full rows for this ETF
                        etf_symbol = group.name
                        etf_rows = self.df[
                            self.df["etf_symbol"] == etf_symbol
                        ]

                        # Convert weight to numeric, handling %
                        weights = (
                            etf_rows[weight_col]
                            .astype(str)
                            .str.replace("%", "")
                            .str.replace(",", "")
                        )
                        weights = pd.to_numeric(
                            weights, errors="coerce"
                        ).fillna(0)

                        # Sort by weight descending
                        sorted_indices = weights.argsort()[::-1]
                        sorted_symbols = (
                            etf_rows.iloc[sorted_indices][symbol_col]
                            .astype(str)
                            .unique()
                        )

                        return ", ".join(sorted_symbols)
                    else:
                        # Fallback to alphabetical if no weight
                        return ", ".join(
                            sorted(group.astype(str).unique())
                        )

                assets_by_etf = (
                    self.df.groupby("etf_symbol")[symbol_col]
                    .apply(format_assets_by_weight)
                    .rename("assets")
                )

            summary = summary.join(assets_by_etf)

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
        self,
        symbol_col: str = "Symbol",
        name_col: str = "Name",
        sort_etfs_by: str = "alpha",
        weight_col: str = "% Weight",
    ) -> pd.DataFrame:
        """
        Get all assets with their ETF associations, sorted by symbol

        Args:
            symbol_col: Name of the asset symbol column
            name_col: Name of the asset name column
            sort_etfs_by: How to sort ETFs for each asset -
                "alpha" for alphabetical (default),
                "weight" for by asset weight in each ETF
            weight_col: Name of the weight column (for weight sorting)

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

            # Sort ETFs based on the chosen method
            if sort_etfs_by == "alpha":
                # Alphabetical sorting (current behavior)
                sorted_etfs = sorted(etfs)
            else:
                # Weight-based sorting - sort by weight descending
                # Get weight for this asset in each ETF
                etf_weights = []
                for etf in etfs:
                    # Find this asset's weight in this ETF
                    asset_in_etf = self.df[
                        (self.df[symbol_col] == symbol)
                        & (self.df["etf_symbol"] == etf)
                    ]
                    has_weight = (
                        not asset_in_etf.empty
                        and weight_col in self.df.columns
                    )
                    if has_weight:
                        weight_str = asset_in_etf.iloc[0].get(
                            weight_col, "0%"
                        )
                        # Parse percentage string to float
                        try:
                            weight_val = float(
                                str(weight_str).replace("%", "").strip()
                            )
                        except (ValueError, AttributeError):
                            weight_val = 0.0
                    else:
                        weight_val = 0.0

                    etf_weights.append((etf, weight_val))

                # Sort by weight descending, then alphabetically
                sorted_etfs = [
                    etf
                    for etf, _ in sorted(
                        etf_weights, key=lambda x: (-x[1], x[0])
                    )
                ]

            results.append(
                {
                    "Symbol": symbol,
                    "Name": name,
                    "ETF_Count": len(etfs),
                    "ETFs": ", ".join(sorted_etfs),
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
            # Preserve "n/a" as literal string, not NaN
            df = pd.read_csv(
                input_file, keep_default_na=False, na_values=[""]
            )
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

    def get_etf_comparison(
        self,
        etf_symbols: List[str],
        symbol_col: str = "Symbol",
        weight_col: str = "% Weight",
        sort_by: str = "weight",
    ) -> pd.DataFrame:
        """
        Compare multiple ETFs side by side

        Args:
            etf_symbols: List of ETF symbols to compare
            symbol_col: Name of the asset symbol column
            weight_col: Name of the weight column
            sort_by: How to sort the comparison -
                "weight" for multi-column weight sorting (default),
                "alpha" for alphabetical by asset symbol

        Returns:
            DataFrame with Asset as first column, followed by one
            column per ETF showing the weight of that asset in that ETF.
            Missing assets show "N/A"

        Raises:
            ValueError: If data not loaded or ETF symbols not found
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_all_etfs() first.")

        # Validate that all requested ETFs exist
        available_etfs = set(self.df["etf_symbol"].unique())
        requested_etfs = set(etf_symbols)
        missing_etfs = requested_etfs - available_etfs
        if missing_etfs:
            raise ValueError(
                f"ETF symbols not found: {', '.join(missing_etfs)}\n"
                f"Available ETFs: {', '.join(sorted(available_etfs))}"
            )

        # Get all unique assets across the selected ETFs
        selected_df = self.df[self.df["etf_symbol"].isin(etf_symbols)]
        all_assets = selected_df[symbol_col].unique()

        # Build comparison DataFrame
        comparison_data = {symbol_col: all_assets}

        # For each ETF, get the weight for each asset
        for etf in etf_symbols:
            etf_df = selected_df[selected_df["etf_symbol"] == etf]
            # Create a dict mapping asset to weight
            asset_weights = {}
            for _, row in etf_df.iterrows():
                asset = row[symbol_col]
                weight = row.get(weight_col, "N/A")
                asset_weights[asset] = weight

            # Add column for this ETF
            comparison_data[etf] = [
                asset_weights.get(asset, "N/A") for asset in all_assets
            ]

        # Create DataFrame
        result = pd.DataFrame(comparison_data)

        # Sort the DataFrame
        if sort_by == "alpha":
            # Alphabetical by asset symbol
            result = result.sort_values(symbol_col)
        else:
            # Multi-column weight sorting
            # Need to convert weight strings to numeric for sorting
            sort_cols = []
            for etf in etf_symbols:
                # Create temporary numeric column for sorting
                temp_col = f"_sort_{etf}"
                result[temp_col] = result[etf].apply(
                    lambda x: (
                        float(str(x).replace("%", "").strip())
                        if x != "N/A" and str(x).strip()
                        else -999999.0
                    )
                )
                sort_cols.append(temp_col)

            # Sort by the temp columns (descending, with N/A at bottom)
            result = result.sort_values(
                sort_cols, ascending=False
            )

            # Drop the temporary sorting columns
            result = result.drop(columns=sort_cols)

        # Reset index
        result = result.reset_index(drop=True)

        return result


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


def check_file_overwrite(output_path: str, force: bool = False) -> bool:
    """
    Check if output file exists and prompt user for overwrite permission.

    Args:
        output_path: Path to output file
        force: If True, skip prompt and allow overwrite

    Returns:
        True if file should be written, False if operation should be
        cancelled
    """
    from pathlib import Path

    if not Path(output_path).exists():
        return True

    if force:
        return True

    # Prompt user
    while True:
        response = input(
            f"File '{output_path}' already exists. "
            f"Overwrite? (y/n): "
        ).lower()
        if response in ["y", "yes"]:
            return True
        elif response in ["n", "no"]:
            return False
        else:
            print("Please enter 'y' or 'n'")


def add_default_extension(
    output_path: Optional[str], function: str
) -> Optional[str]:
    """
    Add default file extension based on function if not already present.

    Args:
        output_path: Output file path (may be None)
        function: Function being executed

    Returns:
        Output path with appropriate extension, or None if output_path
        is None
    """
    if output_path is None:
        return None

    from pathlib import Path

    path = Path(output_path)

    # If path already has an extension, return as-is
    if path.suffix:
        return output_path

    # Add extension based on function
    extension_map = {
        "export": ".parquet",  # Default for DataFrame export
        "summary": ".csv",
        "list": ".txt",
        "assets": ".csv",
        "mapping": ".csv",
        "unique": ".csv",
        "overlap": ".csv",
        "compare": ".csv",
    }

    default_ext = extension_map.get(function, ".txt")
    return str(path) + default_ext


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
            "compare",
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

    # Asset sorting option
    parser.add_argument(
        "--sort-assets",
        choices=["weight", "alpha"],
        default="weight",
        help="Sort assets by weight (default) or alphabetically",
    )

    # ETF sorting option for overlap function
    parser.add_argument(
        "--sort-etfs",
        choices=["weight", "alpha"],
        default="weight",
        help="Sort ETFs in overlap by weight (default) or alphabetically",
    )

    # ETF selection for compare function
    parser.add_argument(
        "--etfs",
        metavar="ETF1,ETF2,...",
        help=(
            "Comma-separated list of ETF symbols to compare "
            "(for compare function)"
        ),
    )

    # Force overwrite flag
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite of existing output files without prompting",
    )

    args = parser.parse_args()

    # Add default extension to output file if needed
    if args.output:
        args.output = add_default_extension(args.output, args.function)

    # Validate arguments
    if args.function == "export" and not args.output:
        parser.error("-f export requires -o OUTPUT to be specified")

    # Check if output file exists and prompt for overwrite
    if args.output and not check_file_overwrite(args.output, args.force):
        print("Operation cancelled.")
        sys.exit(0)

    try:
        # Initialize portfolio analyzer
        portfolio = ETFPortfolioAnalyzer(".")

        # Load data
        if args.data:
            print(f"Loading ETF data from directory: {args.data}")
            portfolio.data_dir = Path(args.data)
            portfolio.load_all_etfs(symbol_col=args.symbol_col)
        elif args.import_file:
            print(f"Importing DataFrame from: {args.import_file}")
            portfolio.load_dataframe(args.import_file)

        print()  # Blank line after loading messages

        # Execute requested function
        if args.function == "export":
            # Type assertion: output is guaranteed to be str by validation
            assert args.output is not None
            portfolio.export_dataframe(args.output)

        elif args.function == "summary":
            summary = portfolio.get_etf_summary(
                symbol_col=args.symbol_col,
                weight_col=args.weight_col,
                include_assets=args.output is not None,
                sort_by=args.sort_assets,
            )
            if args.output:
                summary.to_csv(args.output, index=True)
                print(f"Summary exported to: {args.output}")
            else:
                # For stdout, don't include assets (too verbose)
                summary_no_assets = portfolio.get_etf_summary(
                    symbol_col=args.symbol_col,
                    weight_col=args.weight_col,
                    include_assets=False,
                    sort_by=args.sort_assets,
                )
                print("ETF Portfolio Summary")
                print("=" * 60)
                print(summary_no_assets)
                print()
                print(f"Total ETFs: {len(summary_no_assets)}")
                total_holdings = summary_no_assets["holdings_count"].sum()
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
                symbol_col=args.symbol_col,
                name_col=args.name_col,
                sort_etfs_by=args.sort_etfs,
                weight_col=args.weight_col,
            )
            overlap_assets = assets[assets["ETF_Count"] > 1]
            # Sort by ETF_Count descending, then by Symbol ascending
            overlap_assets = overlap_assets.sort_values(
                ["ETF_Count", args.symbol_col], ascending=[False, True]
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

        elif args.function == "compare":
            # Validate --etfs parameter
            if not args.etfs:
                parser.error(
                    "-f compare requires --etfs ETF1,ETF2,... "
                    "to specify which ETFs to compare"
                )

            # Parse comma-separated ETF list
            etf_list = [etf.strip().upper() for etf in args.etfs.split(",")]

            if len(etf_list) < 2:
                parser.error(
                    "At least 2 ETFs required for comparison. "
                    f"Provided: {len(etf_list)}"
                )

            # Get comparison DataFrame
            comparison = portfolio.get_etf_comparison(
                etf_symbols=etf_list,
                symbol_col=args.symbol_col,
                weight_col=args.weight_col,
                sort_by=args.sort_assets,
            )

            if args.output:
                # Export to CSV
                comparison.to_csv(args.output, index=False)
                print(f"ETF comparison exported to: {args.output}")
            else:
                # Display comparison summary to stdout
                print(f"ETF Comparison: {', '.join(etf_list)}")
                print("=" * 60)
                print()

                # Calculate statistics for each ETF
                stats_data = {}
                for etf in etf_list:
                    etf_holdings = portfolio.df[
                        portfolio.df["etf_symbol"] == etf
                    ]
                    holdings_count = len(etf_holdings)

                    # Count overlapped assets (in any other compared ETF)
                    etf_assets = set(
                        etf_holdings[args.symbol_col].unique()
                    )
                    other_etfs = [e for e in etf_list if e != etf]
                    other_assets = set(
                        portfolio.df[
                            portfolio.df["etf_symbol"].isin(other_etfs)
                        ][args.symbol_col].unique()
                    )
                    overlapped_assets = etf_assets & other_assets
                    overlap_count = len(overlapped_assets)
                    overlap_count_pct = (
                        (overlap_count / holdings_count * 100)
                        if holdings_count > 0
                        else 0
                    )

                    # Calculate weight-based overlap
                    total_weight = 0.0
                    overlapped_weight = 0.0
                    unique_weight = 0.0

                    for _, row in etf_holdings.iterrows():
                        asset = row[args.symbol_col]
                        weight_str = row.get(args.weight_col, "0%")
                        try:
                            weight = float(
                                str(weight_str).replace("%", "").strip()
                            )
                        except (ValueError, AttributeError):
                            weight = 0.0

                        total_weight += weight
                        if asset in overlapped_assets:
                            overlapped_weight += weight
                        else:
                            unique_weight += weight

                    overlap_weight_pct = (
                        (overlapped_weight / total_weight * 100)
                        if total_weight > 0
                        else 0
                    )
                    unique_weight_pct = (
                        (unique_weight / total_weight * 100)
                        if total_weight > 0
                        else 0
                    )

                    stats_data[etf] = {
                        "holdings": holdings_count,
                        "overlapped_count": overlap_count,
                        "overlap_count_pct": overlap_count_pct,
                        "unique_count": holdings_count - overlap_count,
                        "overlapped_weight": overlapped_weight,
                        "overlap_weight_pct": overlap_weight_pct,
                        "unique_weight": unique_weight,
                        "unique_weight_pct": unique_weight_pct,
                    }

                # Print stats table
                print(f"{'':20}", end="")
                for etf in etf_list:
                    print(f"{etf:>12}", end="")
                print()

                print(f"{'Holdings Count':20}", end="")
                for etf in etf_list:
                    print(f"{stats_data[etf]['holdings']:>12}", end="")
                print()
                print()

                # Count-based overlap statistics
                print("By Asset Count:")
                print(f"{'  Overlapped Assets':20}", end="")
                for etf in etf_list:
                    count = stats_data[etf]['overlapped_count']
                    print(f"{count:>12}", end="")
                print()

                print(f"{'  Overlap %':20}", end="")
                for etf in etf_list:
                    pct = stats_data[etf]['overlap_count_pct']
                    print(f"{pct:>11.1f}%", end="")
                print()

                print(f"{'  Unique Assets':20}", end="")
                for etf in etf_list:
                    count = stats_data[etf]['unique_count']
                    print(f"{count:>12}", end="")
                print()

                print(f"{'  Unique %':20}", end="")
                for etf in etf_list:
                    unique_pct = 100 - stats_data[etf]['overlap_count_pct']
                    print(f"{unique_pct:>11.1f}%", end="")
                print()
                print()

                # Weight-based overlap statistics
                print("By Weight:")
                print(f"{'  Overlapped Weight':20}", end="")
                for etf in etf_list:
                    weight = stats_data[etf]['overlapped_weight']
                    print(f"{weight:>11.1f}%", end="")
                print()

                print(f"{'  Overlap %':20}", end="")
                for etf in etf_list:
                    pct = stats_data[etf]['overlap_weight_pct']
                    print(f"{pct:>11.1f}%", end="")
                print()

                print(f"{'  Unique Weight':20}", end="")
                for etf in etf_list:
                    weight = stats_data[etf]['unique_weight']
                    print(f"{weight:>11.1f}%", end="")
                print()

                print(f"{'  Unique %':20}", end="")
                for etf in etf_list:
                    pct = stats_data[etf]['unique_weight_pct']
                    print(f"{pct:>11.1f}%", end="")
                print()
                print()

                # Find common holdings across all ETFs
                all_asset_sets = [
                    set(
                        portfolio.df[portfolio.df["etf_symbol"] == etf][
                            args.symbol_col
                        ].unique()
                    )
                    for etf in etf_list
                ]
                common_all = set.intersection(*all_asset_sets)
                print(
                    f"Common Holdings (in all {len(etf_list)} ETFs): "
                    f"{len(common_all)} assets"
                )
                if len(common_all) > 0:
                    sample = sorted(list(common_all))[:10]
                    print(f"  {', '.join(sample)}", end="")
                    if len(common_all) > 10:
                        print(f", ... ({len(common_all) - 10} more)")
                    else:
                        print()
                print()

                # Count assets by how many ETFs they appear in
                from collections import Counter

                asset_etf_count = Counter()
                for etf in etf_list:
                    etf_assets = portfolio.df[
                        portfolio.df["etf_symbol"] == etf
                    ][args.symbol_col].unique()
                    for asset in etf_assets:
                        asset_etf_count[asset] += 1

                for count in range(len(etf_list) - 1, 0, -1):
                    assets_in_n = [
                        asset
                        for asset, c in asset_etf_count.items()
                        if c == count
                    ]
                    if len(assets_in_n) > 0:
                        print(
                            f"Assets in {count} ETF"
                            f"{'s' if count > 1 else ''}: "
                            f"{len(assets_in_n)} assets"
                        )

                print()

                # Show unique assets for each ETF
                for etf in etf_list:
                    unique_count = stats_data[etf]["unique_count"]
                    if unique_count > 0:
                        print(f"Unique to {etf}: {unique_count} assets")

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
