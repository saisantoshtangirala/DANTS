"""
Feature engineering pipeline for Astra-Trade QML.
Generates technical, fundamental, and microstructure features for Indian markets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Configuration for feature generation."""
    lookback_periods: int = 60
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    volume_ma_period: int = 20
    vwap_period: int = 14
    adx_period: int = 14


class FeatureEngineer:
    """
    Comprehensive feature engineering for Indian equity markets.
    Generates technical indicators, regime features, and cross-asset signals.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineer.

        Args:
            config: Feature configuration
        """
        self.config = config or FeatureConfig()

    def generate_all_features(
        self,
        df: pd.DataFrame,
        include_vwap: bool = True,
        include_microstructure: bool = True,
    ) -> pd.DataFrame:
        """
        Generate complete feature set from OHLCV data.

        Args:
            df: DataFrame with columns [date, open, high, low, close, volume]
            include_vwap: Include VWAP-based features
            include_microstructure: Include microstructure features

        Returns:
            DataFrame with original data + engineered features
        """
        if df.empty or len(df) < self.config.lookback_periods:
            return df

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Price-based features
        df = self._add_returns_features(df)
        df = self._add_moving_averages(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_atr(df)
        df = self._add_adx(df)

        # Volume features
        df = self._add_volume_features(df)

        if include_vwap:
            df = self._add_vwap_features(df)

        if include_microstructure:
            df = self._add_microstructure_features(df)

        # Momentum and trend features
        df = self._add_momentum_features(df)

        # Replace inf values (from division by zero in indicators) with NaN,
        # then drop all NaN rows from lookback periods and sanitization.
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna().reset_index(drop=True)

        return df

    def _add_returns_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_10d"] = df["close"].pct_change(10)
        df["returns_20d"] = df["close"].pct_change(20)

        # Volatility
        df["volatility_5d"] = df["returns"].rolling(5).std() * np.sqrt(252)
        df["volatility_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)

        return df

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving average features."""
        for period in [5, 10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()
            df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()
            df[f"close_to_sma_{period}"] = df["close"] / df[f"sma_{period}"] - 1

        # Golden/Death cross signals
        df["golden_cross"] = (df["sma_20"] > df["sma_50"]).astype(int)
        df["ma_alignment"] = (
            (df["sma_5"] > df["sma_10"]).astype(int) +
            (df["sma_10"] > df["sma_20"]).astype(int) +
            (df["sma_20"] > df["sma_50"]).astype(int)
        ) / 3.0

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI and RSI-derived features."""
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1.0 / self.config.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / self.config.rsi_period, adjust=False).mean()

        rs = avg_gain / avg_loss
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # RSI momentum
        df["rsi_slope"] = df["rsi_14"].diff(3)
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD features."""
        ema_fast = df["close"].ewm(span=self.config.macd_fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.config.macd_slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=self.config.macd_signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        df["macd_cross"] = (df["macd"] > df["macd_signal"]).astype(int).diff().fillna(0)

        df["macd_norm"] = df["macd"] / df["close"]
        df["macd_signal_norm"] = df["macd_signal"] / df["close"]
        df["macd_hist_norm"] = df["macd_histogram"] / df["close"]

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands features."""
        sma = df["close"].rolling(self.config.bb_period).mean()
        std = df["close"].rolling(self.config.bb_period).std(ddof=0)

        df["bb_upper"] = sma + (self.config.bb_std * std)
        df["bb_lower"] = sma - (self.config.bb_std * std)
        df["bb_middle"] = sma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
        bb_range = df["bb_upper"] - df["bb_lower"]
        df["bb_position"] = (df["close"] - df["bb_lower"]) / bb_range.replace(0, np.nan)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(20).min() * 1.05).astype(int)

        return df

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range features."""
        high_low = df["high"] - df["low"]
        high_close = np.abs(df["high"] - df["close"].shift())
        low_close = np.abs(df["low"] - df["close"].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1.0 / self.config.atr_period, adjust=False).mean()
        df["atr_pct"] = df["atr_14"] / df["close"]

        # ATR-based stop loss levels
        df["atr_stop_long"] = df["close"] - (2 * df["atr_14"])
        df["atr_stop_short"] = df["close"] + (2 * df["atr_14"])

        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ADX (Average Directional Index) features."""
        plus_dm_raw = df["high"].diff()
        minus_dm_raw = -df["low"].diff()

        plus_dm = plus_dm_raw.where((plus_dm_raw > minus_dm_raw) & (plus_dm_raw > 0), 0)
        minus_dm = minus_dm_raw.where((minus_dm_raw > plus_dm_raw) & (minus_dm_raw > 0), 0)

        atr = df["atr_14"]

        plus_di = 100 * plus_dm.rolling(self.config.adx_period).mean() / atr
        minus_di = 100 * minus_dm.rolling(self.config.adx_period).mean() / atr

        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
        df["adx"] = dx.rolling(self.config.adx_period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        df["di_cross"] = (df["plus_di"] > df["minus_di"]).astype(int)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        df["volume_sma_20"] = df["volume"].rolling(self.config.volume_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        df["volume_trend"] = df["volume"].rolling(5).mean() / df["volume"].rolling(20).mean()

        # On-Balance Volume (OBV) — vectorized
        direction = np.sign(df["close"].diff())
        df["obv"] = (direction * df["volume"]).fillna(0).cumsum()
        df["obv_slope"] = df["obv"].diff(5)
        df["obv_roc"] = df["obv"].diff(5) / (df["volume"].rolling(5).sum() + 1)

        # Money Flow Index
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        money_flow = typical_price * df["volume"]

        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        positive_sum = positive_flow.rolling(14).sum()
        negative_sum = negative_flow.rolling(14).sum()

        mfi_ratio = positive_sum / negative_sum
        df["mfi"] = 100 - (100 / (1 + mfi_ratio))

        return df

    def _add_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP (Volume Weighted Average Price) features."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        window = self.config.vwap_period
        vwap = (typical_price * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()

        df["vwap"] = vwap
        df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"]
        df["above_vwap"] = (df["close"] > df["vwap"]).astype(int)

        # Intraday VWAP reset (for 5-min data, reset daily)
        if "date" in df.columns and pd.infer_freq(df["date"]) in ["5min", "15min"]:
            date_only = df["date"].dt.date
            cum_tp_vol = (typical_price * df["volume"]).groupby(date_only).cumsum()
            cum_vol = df["volume"].groupby(date_only).cumsum()
            df["intraday_vwap"] = cum_tp_vol / cum_vol
            df["intraday_vwap_dev"] = (df["close"] - df["intraday_vwap"]) / df["intraday_vwap"]

        return df

    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market microstructure features."""
        # Price impact (how much price moves per unit volume)
        df["price_impact"] = df["returns"].abs() / np.log(df["volume"] + 1)

        # Intraday range
        df["intraday_range"] = (df["high"] - df["low"]) / df["close"]
        df["body_size"] = (df["close"] - df["open"]).abs() / df["close"]
        df["upper_shadow"] = (df["high"] - df[["close", "open"]].max(axis=1)) / df["close"]
        df["lower_shadow"] = (df[["close", "open"]].min(axis=1) - df["low"]) / df["close"]

        # Candlestick patterns
        df["doji"] = (df["body_size"] < 0.001).astype(int)
        df["hammer"] = (
            (df["lower_shadow"] > 2 * df["body_size"]) &
            (df["upper_shadow"] < df["body_size"])
        ).astype(int)
        df["shooting_star"] = (
            (df["upper_shadow"] > 2 * df["body_size"]) &
            (df["lower_shadow"] < df["body_size"])
        ).astype(int)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and trend strength features."""
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}"] = (df["close"] - df["close"].shift(period)) / df["close"].shift(period)

        # Stochastic Oscillator
        lowest_low = df["low"].rolling(14).min()
        highest_high = df["high"].rolling(14).max()
        stoch_range = (highest_high - lowest_low).replace(0, np.nan)
        df["stoch_k"] = 100 * (df["close"] - lowest_low) / stoch_range
        df["stoch_d"] = df["stoch_k"].rolling(3).mean()

        # Williams %R
        df["williams_r"] = -100 * (highest_high - df["close"]) / stoch_range

        # CCI (Commodity Channel Index)
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        tp_sma = typical_price.rolling(20).mean()
        tp_mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci_denom = (0.015 * tp_mad).replace(0, np.nan)
        df["cci"] = (typical_price - tp_sma) / cci_denom

        return df

    def generate_labels(
        self,
        df: pd.DataFrame,
        forward_periods: int = 5,
        noise_threshold: float = 0.003,
    ) -> pd.DataFrame:
        """
        Generate binary classification labels with dead-zone exclusion.

        Samples with |future_return| <= noise_threshold are labeled NaN
        (dead zone) and should be dropped before training.

        Args:
            df: DataFrame with features
            forward_periods: Number of periods to look ahead
            noise_threshold: Returns inside [-threshold, +threshold] are dead zone

        Returns:
            DataFrame with 'label' column (1=UP, 0=DOWN, NaN=dead zone)
        """
        df = df.copy()

        future_return = df["close"].shift(-forward_periods) / df["close"] - 1

        df["future_return"] = future_return

        df = df.dropna(subset=["future_return"]).reset_index(drop=True)

        df["label"] = np.nan
        df.loc[df["future_return"] > noise_threshold, "label"] = 1    # UP
        df.loc[df["future_return"] < -noise_threshold, "label"] = 0   # DOWN

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of engineered feature columns (excluding raw price/volume).

        Args:
            df: DataFrame with all columns

        Returns:
            List of feature column names
        """
        exclude = {
            "date", "open", "high", "low", "close", "volume", "turnover",
            "symbol", "label", "future_return",
            # Absolute-price features that leak symbol identity when pooling
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_5", "ema_10", "ema_20", "ema_50",
            "bb_upper", "bb_lower", "bb_middle",
            "macd", "macd_signal", "macd_histogram",
            "atr_14", "atr_stop_long", "atr_stop_short",
            "volume_sma_20", "obv", "obv_slope",
            "vwap",
        }
        return [col for col in df.columns if col not in exclude]