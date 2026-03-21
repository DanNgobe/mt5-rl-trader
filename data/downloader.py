"""
MetaTrader 5 data downloader for historical OHLCV data.

This module handles connection to MT5 and downloads historical bar data
for specified forex pairs and timeframes.
"""

import os
from datetime import datetime
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MT5Downloader:
    """Downloader for historical OHLCV data from MetaTrader 5."""

    def __init__(
        self,
        login: str | None = None,
        password: str | None = None,
        server: str | None = None,
        path: str | None = None,
    ):
        """
        Initialize the MT5 downloader.

        Args:
            login: MT5 account login
            password: MT5 account password
            server: MT5 server name
            path: Path to MT5 terminal executable
        """
        self.login = login or os.getenv("MT5_LOGIN")
        self.password = password or os.getenv("MT5_PASSWORD")
        self.server = server or os.getenv("MT5_SERVER")
        self.path = path or os.getenv("MT5_PATH")
        self._connected = False

    def connect(self) -> bool:
        """
        Establish connection to MT5 terminal.

        Returns:
            True if connection successful, False otherwise
        """
        if self._connected and mt5.last_error() == (0, ""):
            return True

        # Initialize MT5
        init_params = {}
        if self.path:
            init_params["path"] = self.path

        if not mt5.initialize(**init_params):
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return False

        # Login if credentials provided
        if self.login and self.password:
            login_params = {
                "login": int(self.login),
                "password": self.password,
            }
            if self.server:
                login_params["server"] = self.server

            if not mt5.login(**login_params):
                print(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False

        self._connected = True
        print("Connected to MetaTrader 5")
        return True

    def disconnect(self) -> None:
        """Close connection to MT5 terminal."""
        if self._connected:
            mt5.shutdown()
            self._connected = False
            print("Disconnected from MetaTrader 5")

    def get_timeframes(self) -> dict[str, int]:
        """
        Get available timeframes mapping.

        Returns:
            Dictionary mapping string codes to MT5 timeframe constants
        """
        return {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
            "W1": mt5.TIMEFRAME_W1,
            "MN1": mt5.TIMEFRAME_MN1,
        }

    def download_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        output_path: str | None = None,
    ) -> pd.DataFrame | None:
        """
        Download historical OHLCV bars for a symbol.

        Args:
            symbol: Forex pair symbol (e.g., "EURUSD")
            timeframe: Timeframe string (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_path: Optional path to save CSV file

        Returns:
            DataFrame with OHLCV data or None if failed
        """
        if not self._connected:
            if not self.connect():
                return None

        timeframes = self.get_timeframes()
        if timeframe not in timeframes:
            print(f"Invalid timeframe: {timeframe}")
            return None

        tf = timeframes[timeframe]
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        print(f"Downloading {symbol} {timeframe} from {start_date} to {end_date}...")

        # Fetch bars
        bars = mt5.copy_rates_range(symbol, tf, start_dt, end_dt)

        if bars is None or len(bars) == 0:
            print(f"No data received for {symbol}: {mt5.last_error()}")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "volume",
            }
        )
        df = df[["open", "high", "low", "close", "volume"]]

        print(f"Downloaded {len(df)} bars for {symbol}")

        # Save to CSV if path provided
        if output_path:
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)
            print(f"Saved to {output_path}")

        return df

    def download_multiple(
        self,
        symbols: list[str],
        timeframe: str,
        start_date: str,
        end_date: str,
        output_dir: str,
    ) -> dict[str, pd.DataFrame]:
        """
        Download historical data for multiple symbols.

        Args:
            symbols: List of forex pair symbols
            timeframe: Timeframe string
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = {}
        for symbol in symbols:
            filename = f"{symbol}_{timeframe}.csv"
            filepath = output_path / filename

            df = self.download_bars(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                output_path=str(filepath),
            )

            if df is not None:
                results[symbol] = df

        return results

    def get_available_symbols(self) -> list[str]:
        """
        Get list of available symbols from MT5.

        Returns:
            List of symbol names
        """
        if not self._connected:
            if not self.connect():
                return []

        symbols = mt5.symbols_get()
        if symbols is None:
            return []

        return [s.name for s in symbols if s.visible]


def main():
    """CLI entry point for downloading data."""
    import argparse

    parser = argparse.ArgumentParser(description="Download MT5 historical data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["EURUSD"],
        help="Forex symbols to download",
    )
    parser.add_argument(
        "--timeframe",
        default="H1",
        choices=["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"],
        help="Timeframe",
    )
    parser.add_argument(
        "--start",
        default="2024-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2024-12-31",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for CSV files",
    )

    args = parser.parse_args()

    downloader = MT5Downloader()

    try:
        if not downloader.connect():
            print("Failed to connect to MT5")
            return

        results = downloader.download_multiple(
            symbols=args.symbols,
            timeframe=args.timeframe,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir,
        )

        print(f"\nSuccessfully downloaded {len(results)} symbols")
        for symbol in results:
            print(f"  - {symbol}: {len(results[symbol])} bars")

    finally:
        downloader.disconnect()


if __name__ == "__main__":
    main()
