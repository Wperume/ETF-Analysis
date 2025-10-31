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
        # Extract ETF symbol from filename (e.g., "SPY" from "SPY-etf-holdings.csv")
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

    def load_all_etfs(self, pattern: str = "*-etf-holdings.csv", **kwargs) -> pd.DataFrame:
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
        print(f"\nCombined {len(csv_files)} ETFs with {len(self.df)} total holdings")
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
