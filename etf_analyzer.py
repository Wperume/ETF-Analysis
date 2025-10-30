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
        """
        self.csv_path = Path(csv_path)
        self.df: Optional[pd.DataFrame] = None
        self.etf_name = self.csv_path.stem

    def load_data(self, **kwargs) -> pd.DataFrame:
        """
        Load CSV data into a pandas DataFrame

        Args:
            **kwargs: Additional arguments to pass to pd.read_csv()

        Returns:
            Loaded DataFrame
        """
        try:
            self.df = pd.read_csv(self.csv_path, **kwargs)
            print(f"Loaded {len(self.df)} holdings from {self.csv_path.name}")
            return self.df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            raise

    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the holdings

        Returns:
            Dictionary with summary statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        return {
            'total_holdings': len(self.df),
            'columns': list(self.df.columns),
            'memory_usage': f"{self.df.memory_usage(deep=True).sum() / 1024:.2f} KB"
        }

    def get_top_holdings(self, n: int = 10, weight_column: str = 'weight') -> pd.DataFrame:
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
            raise ValueError(f"Column '{weight_column}' not found in DataFrame")

        return self.df.nlargest(n, weight_column)

    def analyze_by_sector(self, sector_column: str = 'sector') -> pd.DataFrame:
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
            raise ValueError(f"Column '{sector_column}' not found in DataFrame")

        sector_analysis = self.df.groupby(sector_column).agg({
            sector_column: 'count'
        }).rename(columns={sector_column: 'count'})

        return sector_analysis.sort_values('count', ascending=False)

    def plot_top_holdings(self, n: int = 10, weight_column: str = 'weight',
                         name_column: str = 'name'):
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
        plt.yticks(range(len(top_holdings)), top_holdings[name_column])
        plt.xlabel('Weight (%)')
        plt.title(f'Top {n} Holdings - {self.etf_name}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def export_analysis(self, output_path: str, analysis_type: str = 'summary'):
        """
        Export analysis results to a file

        Args:
            output_path: Path to save the output file
            analysis_type: Type of analysis to export ('summary', 'top_holdings', etc.)
        """
        output_path = Path(output_path)

        if analysis_type == 'summary':
            stats = self.get_summary_stats()
            with open(output_path, 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")

        print(f"Analysis exported to {output_path}")


def compare_etfs(etf_paths: List[str], weight_column: str = 'weight') -> Dict[str, Optional[pd.DataFrame]]:
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

    # Create an example of how to use the analyzer
    example_path = "data/sample_etf.csv"

    print(f"\nTo use this analyzer:")
    print(f"1. Place your ETF CSV files in the 'data/' directory")
    print(f"2. Run the following code:\n")
    print(f"analyzer = ETFAnalyzer('{example_path}')")
    print(f"analyzer.load_data()")
    print(f"print(analyzer.get_summary_stats())")
    print(f"print(analyzer.get_top_holdings(10))")
