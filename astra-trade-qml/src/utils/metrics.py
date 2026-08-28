"""
Performance metrics calculation module.
Implements Sharpe ratio, max drawdown, profit factor, and other trading metrics.
"""

import numpy as np
import pandas as pd
from typing import Any, List, Dict, Optional, Tuple


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.06,  # 6% Indian risk-free rate
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default 6% for India)
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if returns.empty or len(returns) < 2 or returns.std() == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    return np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())


def calculate_max_drawdown(equity_curve: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown with peak and trough dates.

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        Tuple of (max_drawdown_pct, peak_date, trough_date)
    """
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max

    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()
    peak_idx = equity_curve.loc[:trough_idx].idxmax()

    return float(max_dd), peak_idx, trough_idx


def calculate_profit_factor(trades_df: pd.DataFrame) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        trades_df: DataFrame with 'pnl' column

    Returns:
        Profit factor (inf if no losses)
    """
    if trades_df.empty:
        return 0.0

    gross_profit = trades_df[trades_df["pnl"] > 0]["pnl"].sum()
    gross_loss = abs(trades_df[trades_df["pnl"] < 0]["pnl"].sum())

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Calculate Kelly criterion fraction for position sizing.
    Uses Quarter-Kelly for safety.

    Args:
        win_rate: Probability of winning trade
        avg_win: Average winning trade return
        avg_loss: Average losing trade return (positive number)
        fraction: Kelly fraction multiplier (default 0.25 for Quarter-Kelly)

    Returns:
        Recommended position size as fraction of capital
    """
    if avg_loss <= 0:
        return 0.0

    win_loss_ratio = avg_win / avg_loss
    kelly = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    return max(0.0, min(kelly * fraction, 0.5))  # Cap at 50%


def calculate_expectancy(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
) -> float:
    """
    Calculate trade expectancy.

    Args:
        win_rate: Win probability
        avg_win: Average win amount
        avg_loss: Average loss amount (positive)

    Returns:
        Expected value per trade
    """
    return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


def calculate_calmar_ratio(
    returns: pd.Series,
    equity_curve: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Daily returns series
        equity_curve: Portfolio value series
        periods_per_year: Trading days per year

    Returns:
        Calmar ratio
    """
    if returns.empty or equity_curve.empty:
        return 0.0

    annualized_return = returns.mean() * periods_per_year
    max_dd, _, _ = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return float("inf") if annualized_return > 0 else 0.0

    return annualized_return / abs(max_dd)


def calculate_confidence_calibration(
    predictions: pd.DataFrame,
    confidence_col: str = "confidence",
    actual_col: str = "actual",
    pred_col: str = "predicted",
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Calculate confidence calibration metrics.

    Args:
        predictions: DataFrame with confidence, predicted, and actual columns
        confidence_col: Name of confidence column
        actual_col: Name of actual outcome column
        pred_col: Name of prediction column
        n_bins: Number of confidence bins

    Returns:
        Dictionary with calibration metrics
    """
    predictions = predictions.copy()
    predictions["correct"] = (predictions[pred_col] == predictions[actual_col]).astype(int)
    predictions["bin"] = pd.cut(predictions[confidence_col], bins=n_bins, labels=False)

    calibration_data = []
    for bin_idx in range(n_bins):
        bin_data = predictions[predictions["bin"] == bin_idx]
        if len(bin_data) > 0:
            avg_confidence = bin_data[confidence_col].mean()
            accuracy = bin_data["correct"].mean()
            calibration_data.append({
                "bin": bin_idx,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "count": len(bin_data),
            })

    cal_df = pd.DataFrame(calibration_data)

    if cal_df.empty:
        return {"expected_calibration_error": 0.0, "max_calibration_error": 0.0}

    cal_df["error"] = abs(cal_df["accuracy"] - cal_df["avg_confidence"])
    cal_df["weighted_error"] = cal_df["error"] * cal_df["count"]

    total_count = cal_df["count"].sum()
    ece = cal_df["weighted_error"].sum() / total_count if total_count > 0 else 0.0
    mce = cal_df["error"].max()

    return {
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
    }


def generate_performance_report(
    trades_df: pd.DataFrame,
    equity_curve: pd.Series,
    period_days: int = 30,
) -> Dict[str, Any]:
    """
    Generate comprehensive performance report.

    Args:
        trades_df: DataFrame of closed trades
        equity_curve: Series of daily portfolio values
        period_days: Analysis period in days

    Returns:
        Dictionary with all key metrics
    """
    if trades_df.empty or equity_curve.empty:
        return {
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "total_trades": 0,
            "total_pnl": 0.0,
            "calmar_ratio": 0.0,
            "avg_confidence": 0.0,
        }

    returns = equity_curve.pct_change().dropna()

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] < 0]

    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0.0
    avg_win = wins["pnl_pct"].mean() if not wins.empty else 0.0
    avg_loss = abs(losses["pnl_pct"].mean()) if not losses.empty else 0.0

    max_dd, peak_date, trough_date = calculate_max_drawdown(equity_curve)

    return {
        "sharpe_ratio": calculate_sharpe_ratio(returns),
        "max_drawdown_pct": float(max_dd),
        "max_dd_peak_date": str(peak_date),
        "max_dd_trough_date": str(trough_date),
        "profit_factor": calculate_profit_factor(trades_df),
        "win_rate": float(win_rate),
        "total_trades": len(trades_df),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "total_pnl": float(trades_df["pnl"].sum()),
        "avg_trade_return_pct": float(trades_df["pnl_pct"].mean()),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "expectancy": calculate_expectancy(win_rate, avg_win, avg_loss),
        "kelly_fraction": calculate_kelly_fraction(win_rate, avg_win, avg_loss),
        "calmar_ratio": calculate_calmar_ratio(returns, equity_curve),
        "avg_confidence": float(trades_df["confidence"].mean()),
    }