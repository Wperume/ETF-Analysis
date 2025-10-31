"""
Tests for ETF Analyzer
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from etf_analyzer import ETFAnalyzer, ETFPortfolioAnalyzer


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_etf_data():
    """Sample ETF holdings data"""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            "name": [
                "Apple Inc.",
                "Microsoft Corp",
                "Alphabet Inc",
                "Amazon.com Inc",
                "Tesla Inc",
            ],
            "weight": [5.2, 4.8, 3.9, 3.5, 2.1],
            "sector": [
                "Technology",
                "Technology",
                "Technology",
                "Consumer",
                "Auto",
            ],
            "shares": [1000000, 900000, 500000, 300000, 200000],
        }
    )


@pytest.fixture
def spy_csv_file(temp_data_dir, sample_etf_data):
    """Create a sample SPY ETF CSV file"""
    csv_path = temp_data_dir / "SPY-etf-holdings.csv"
    sample_etf_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def qqq_csv_file(temp_data_dir):
    """Create a sample QQQ ETF CSV file"""
    data = pd.DataFrame(
        {
            "ticker": ["NVDA", "META", "AVGO"],
            "name": ["NVIDIA Corp", "Meta Platforms", "Broadcom Inc"],
            "weight": [8.5, 4.2, 3.8],
            "sector": ["Technology", "Technology", "Technology"],
            "shares": [500000, 300000, 200000],
        }
    )
    csv_path = temp_data_dir / "QQQ-etf-holdings.csv"
    data.to_csv(csv_path, index=False)
    return csv_path


class TestETFAnalyzer:
    """Tests for ETFAnalyzer class"""

    def test_init_extracts_symbol_from_filename(self, spy_csv_file):
        """Test that ETF symbol is correctly extracted from filename"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        assert analyzer.etf_name == "SPY"

    def test_init_uppercase_conversion(self, temp_data_dir):
        """Test that symbol is converted to uppercase"""
        csv_path = temp_data_dir / "spy-etf-holdings.csv"
        csv_path.touch()
        analyzer = ETFAnalyzer(str(csv_path))
        assert analyzer.etf_name == "SPY"

    def test_load_data_returns_dataframe(self, spy_csv_file):
        """Test that load_data returns a DataFrame"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        df = analyzer.load_data()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_load_data_adds_etf_symbol_column(self, spy_csv_file):
        """Test that etf_symbol column is added"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        df = analyzer.load_data()
        assert "etf_symbol" in df.columns
        assert df["etf_symbol"].iloc[0] == "SPY"
        assert all(df["etf_symbol"] == "SPY")

    def test_load_data_symbol_at_first_position(self, spy_csv_file):
        """Test that etf_symbol is the first column"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        df = analyzer.load_data()
        assert df.columns[0] == "etf_symbol"

    def test_get_summary_stats(self, spy_csv_file):
        """Test get_summary_stats returns correct information"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        analyzer.load_data()
        stats = analyzer.get_summary_stats()

        assert "total_holdings" in stats
        assert stats["total_holdings"] == 5
        assert "columns" in stats
        assert "memory_usage" in stats

    def test_get_summary_stats_before_loading_raises_error(
        self, spy_csv_file
    ):
        """Test that get_summary_stats raises error if data not loaded"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        with pytest.raises(ValueError, match="Data not loaded"):
            analyzer.get_summary_stats()

    def test_get_top_holdings(self, spy_csv_file):
        """Test get_top_holdings returns correct top N"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        analyzer.load_data()
        top = analyzer.get_top_holdings(n=3, weight_column="weight")

        assert len(top) == 3
        assert top.iloc[0]["ticker"] == "AAPL"
        assert top.iloc[1]["ticker"] == "MSFT"
        assert top.iloc[2]["ticker"] == "GOOGL"

    def test_get_top_holdings_invalid_column_raises_error(
        self, spy_csv_file
    ):
        """Test that invalid weight column raises error"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        analyzer.load_data()
        with pytest.raises(ValueError, match="not found in DataFrame"):
            analyzer.get_top_holdings(weight_column="invalid_column")

    def test_analyze_by_sector(self, spy_csv_file):
        """Test analyze_by_sector returns correct grouping"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        analyzer.load_data()
        sector_analysis = analyzer.analyze_by_sector(
            sector_column="sector"
        )

        assert "Technology" in sector_analysis.index
        assert sector_analysis.loc["Technology", "count"] == 3

    def test_analyze_by_sector_invalid_column_raises_error(
        self, spy_csv_file
    ):
        """Test that invalid sector column raises error"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        analyzer.load_data()
        with pytest.raises(ValueError, match="not found in DataFrame"):
            analyzer.analyze_by_sector(sector_column="invalid")

    def test_export_analysis_creates_file(
        self, spy_csv_file, temp_data_dir
    ):
        """Test that export_analysis creates output file"""
        analyzer = ETFAnalyzer(str(spy_csv_file))
        analyzer.load_data()
        output_path = temp_data_dir / "analysis.txt"

        analyzer.export_analysis(str(output_path), analysis_type="summary")

        assert output_path.exists()
        content = output_path.read_text()
        assert "total_holdings: 5" in content


class TestETFPortfolioAnalyzer:
    """Tests for ETFPortfolioAnalyzer class"""

    def test_init_sets_data_dir(self, temp_data_dir):
        """Test that data directory is set correctly"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        assert portfolio.data_dir == temp_data_dir

    def test_load_all_etfs_combines_files(
        self, temp_data_dir, spy_csv_file, qqq_csv_file
    ):
        """Test that all ETF files are loaded and combined"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        df = portfolio.load_all_etfs()

        assert len(df) == 8  # 5 from SPY + 3 from QQQ
        assert "etf_symbol" in df.columns

    def test_load_all_etfs_tracks_analyzers(
        self, temp_data_dir, spy_csv_file, qqq_csv_file
    ):
        """Test that individual analyzers are tracked"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        portfolio.load_all_etfs()

        assert "SPY" in portfolio.etf_analyzers
        assert "QQQ" in portfolio.etf_analyzers

    def test_load_all_etfs_no_files_raises_error(self, temp_data_dir):
        """Test that error is raised when no files found"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        with pytest.raises(FileNotFoundError, match="No files matching"):
            portfolio.load_all_etfs()

    def test_get_etf_list(
        self, temp_data_dir, spy_csv_file, qqq_csv_file
    ):
        """Test get_etf_list returns all loaded symbols"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        portfolio.load_all_etfs()
        etf_list = portfolio.get_etf_list()

        assert len(etf_list) == 2
        assert "SPY" in etf_list
        assert "QQQ" in etf_list

    def test_get_etf_summary(
        self, temp_data_dir, spy_csv_file, qqq_csv_file
    ):
        """Test get_etf_summary returns correct counts"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        portfolio.load_all_etfs()
        summary = portfolio.get_etf_summary()

        assert "SPY" in summary.index
        assert "QQQ" in summary.index
        assert summary.loc["SPY", "holdings_count"] == 5
        assert summary.loc["QQQ", "holdings_count"] == 3

    def test_get_etf_summary_before_loading_raises_error(
        self, temp_data_dir
    ):
        """Test that error is raised if data not loaded"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        with pytest.raises(ValueError, match="Data not loaded"):
            portfolio.get_etf_summary()

    def test_filter_by_etf(
        self, temp_data_dir, spy_csv_file, qqq_csv_file
    ):
        """Test filtering by ETF symbol"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        portfolio.load_all_etfs()

        spy_data = portfolio.filter_by_etf("SPY")
        assert len(spy_data) == 5
        assert all(spy_data["etf_symbol"] == "SPY")

        qqq_data = portfolio.filter_by_etf("QQQ")
        assert len(qqq_data) == 3
        assert all(qqq_data["etf_symbol"] == "QQQ")

    def test_filter_by_etf_case_insensitive(
        self, temp_data_dir, spy_csv_file, qqq_csv_file
    ):
        """Test that filtering is case insensitive"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        portfolio.load_all_etfs()

        spy_data = portfolio.filter_by_etf("spy")
        assert len(spy_data) == 5
        assert all(spy_data["etf_symbol"] == "SPY")

    def test_custom_pattern(self, temp_data_dir, spy_csv_file):
        """Test loading with custom file pattern"""
        portfolio = ETFPortfolioAnalyzer(str(temp_data_dir))
        df = portfolio.load_all_etfs(pattern="SPY-*.csv")

        assert len(df) == 5
        assert "SPY" in portfolio.etf_analyzers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
