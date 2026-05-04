"""
Unit tests for the data pipeline components.

Covers:
    - AbstractDataAdapter.validate() — all quality checks
    - AbstractDataAdapter.normalise_columns()
    - ParquetDataStore save/load/exists/merge/date_range
    - UniverseDefinition construction and query methods
    - CSVAdapter loading and date filtering
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from cpat.core.enums import AssetClass, Interval
from cpat.data.base import AbstractDataAdapter, DataAdapterError, DataQualityError
from cpat.data.store import ParquetDataStore
from cpat.data.universe import UniverseDefinition
from cpat.data.adapters.csv_adapter import CSVAdapter


def _make_df(n: int = 20) -> pd.DataFrame:
    import numpy as np
    dates = pd.date_range("2022-01-01", periods=n, freq="B", tz="UTC")
    p = np.array([100.0 + i for i in range(n)], dtype=float)
    df = pd.DataFrame({
        "open": p, "high": p * 1.01, "low": p * 0.99,
        "close": p, "adj_close": p, "volume": np.full(n, 1_000_000.0),
    }, index=dates)
    df.index.name = "timestamp"
    return df


class _ConcreteAdapter(AbstractDataAdapter):
    def fetch(self, symbol, start, end, interval=Interval.ONE_DAY):
        return _make_df()


class TestAbstractDataAdapterValidate:
    def setup_method(self):
        self.adapter = _ConcreteAdapter()

    def test_valid_df_passes(self):
        self.adapter.validate("AAPL", _make_df())

    def test_empty_df_raises(self):
        with pytest.raises(DataQualityError, match="Empty"):
            self.adapter.validate("AAPL", pd.DataFrame())

    def test_missing_column_raises(self):
        with pytest.raises(DataQualityError, match="Missing columns"):
            self.adapter.validate("AAPL", _make_df().drop(columns=["volume"]))

    def test_tz_naive_index_raises(self):
        df = _make_df()
        df.index = df.index.tz_localize(None)
        with pytest.raises(DataQualityError, match="timezone-aware"):
            self.adapter.validate("AAPL", df)

    def test_unsorted_index_raises(self):
        with pytest.raises(DataQualityError, match="sorted"):
            self.adapter.validate("AAPL", _make_df().iloc[::-1])

    def test_high_lt_low_raises(self):
        df = _make_df()
        df.loc[df.index[0], "high"] = df.loc[df.index[0], "low"] - 1.0
        with pytest.raises(DataQualityError, match="high < low"):
            self.adapter.validate("AAPL", df)

    def test_excessive_missing_raises(self):
        df = _make_df()
        df.loc[df.index[:15], "close"] = float("nan")
        with pytest.raises(DataQualityError, match="Missing value"):
            self.adapter.validate("AAPL", df, max_missing_pct=0.05)


class TestNormaliseColumns:
    def test_uppercase_lowercased(self):
        adapter = _ConcreteAdapter()
        df = pd.DataFrame({"Open": [1], "Close": [1], "High": [1],
                           "Low": [1], "Volume": [1], "Adj Close": [1]})
        result = adapter.normalise_columns(df)
        assert "open" in result.columns
        assert "adj_close" in result.columns

    def test_adj_close_fallback(self):
        adapter = _ConcreteAdapter()
        df = pd.DataFrame({"open": [1], "close": [1], "high": [1],
                           "low": [1], "volume": [1]})
        result = adapter.normalise_columns(df)
        assert "adj_close" in result.columns


class TestParquetDataStore:
    def test_save_and_load(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        df = _make_df()
        store.save("AAPL", df, Interval.ONE_DAY)
        loaded = store.load("AAPL", Interval.ONE_DAY)
        assert len(loaded) == len(df)

    def test_exists_true_after_save(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        assert not store.exists("AAPL")
        store.save("AAPL", _make_df())
        assert store.exists("AAPL")

    def test_load_nonexistent_raises(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        with pytest.raises(FileNotFoundError):
            store.load("TSLA")

    def test_merge_on_save_deduplicates(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        store.save("AAPL", _make_df(n=20))
        store.save("AAPL", _make_df(n=30))  # Overlaps first 20
        loaded = store.load("AAPL")
        assert len(loaded) == 30

    def test_date_range_filter(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        store.save("AAPL", _make_df(n=50))
        filtered = store.load("AAPL", start=date(2022, 1, 10), end=date(2022, 1, 20))
        assert len(filtered) < 50

    def test_available_symbols(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        store.save("AAPL", _make_df())
        store.save("MSFT", _make_df())
        assert "AAPL" in store.available_symbols()
        assert "MSFT" in store.available_symbols()

    def test_manifest_tracks_dates(self, temp_data_dir):
        store = ParquetDataStore(temp_data_dir)
        store.save("AAPL", _make_df(n=20))
        first, last = store.date_range("AAPL")
        assert first is not None and last is not None
        assert first <= last


class TestUniverseDefinition:
    def test_symbols_deduplicated(self, minimal_config):
        u = UniverseDefinition.from_config(minimal_config.universe)
        assert len(u.symbols) == len(set(u.symbols))

    def test_contains_operator(self, minimal_config):
        u = UniverseDefinition.from_config(minimal_config.universe)
        assert "AAPL" in u
        assert "NOTREAL" not in u

    def test_by_asset_class_etf(self):
        from cpat.config.loader import UniverseConfig
        cfg = UniverseConfig(etfs=["SPY", "QQQ"], equities=["AAPL"])
        u = UniverseDefinition.from_config(cfg)
        etfs = u.by_asset_class(AssetClass.ETF)
        assert "SPY" in etfs
        assert "AAPL" not in etfs

    def test_size_property(self, minimal_config):
        u = UniverseDefinition.from_config(minimal_config.universe)
        assert u.size == len(u.symbols)


class TestCSVAdapter:
    def test_load_csv_valid(self, tmp_path):
        (tmp_path / "AAPL").mkdir()
        csv_path = tmp_path / "AAPL" / "1d.csv"
        df = _make_df(n=30).reset_index().rename(columns={"timestamp": "Date"})
        df.to_csv(csv_path, index=False)

        adapter = CSVAdapter(base_dir=tmp_path)
        result = adapter.fetch("AAPL", date(2022, 1, 1), date(2022, 12, 31))
        assert not result.empty
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_missing_file_raises(self, tmp_path):
        adapter = CSVAdapter(base_dir=tmp_path)
        with pytest.raises(DataAdapterError):
            adapter.fetch("ZZZZ", date(2022, 1, 1), date(2022, 12, 31))
