"""Performance metrics for the target ticker vs benchmark (SPY).

Computes:
    Beta             - systematic risk vs benchmark (OLS)
    Alpha (CAPM)     - excess return after adjusting for beta, annualized
    Sharpe           - excess return / total volatility
    Sortino          - excess return / downside volatility (more honest)
    Std dev          - annualized volatility
    Max drawdown     - worst peak-to-trough loss in window
    Calmar           - annualized return / |max drawdown|
    Information      - active return / tracking error vs benchmark
    VaR (historical) - 5th percentile daily loss, NOT parametric
                       (equity returns have fat tails; parametric understates)

Decoupled from the signal — performance is a portfolio diagnostic, not part
of the buy/sell decision.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TRADING_DAYS = 252  # annualization factor


# ---------- Data ---------------------------------------------------------

def fetch_returns(ticker: str, benchmark: str, lookback_days: int) -> pd.DataFrame:
    """Daily returns for ticker and benchmark over the lookback window."""
    end = datetime.now()
    # Pull ~1.5x the window in calendar days to ensure enough trading days
    start = end - timedelta(days=int(lookback_days * 1.5))

    data = yf.download([ticker, benchmark], start=start, end=end,
                       progress=False, auto_adjust=True)["Close"]
    if data is None or data.empty:
        raise ValueError(f"No price data for {ticker} or {benchmark}")

    rets = data.pct_change().dropna()
    if len(rets) < 30:
        raise ValueError(f"Insufficient return history: only {len(rets)} days")

    # Trim to most recent `lookback_days` trading days
    rets = rets.tail(lookback_days)
    logger.info("Loaded %d days of returns for %s and %s",
                len(rets), ticker, benchmark)
    return rets


# ---------- Individual metrics -------------------------------------------

def beta_alpha(asset: pd.Series, benchmark: pd.Series,
               risk_free_daily: float) -> tuple[float, float]:
    """OLS regression of excess returns. Returns (beta, alpha_annualized)."""
    asset_excess = asset - risk_free_daily
    bench_excess = benchmark - risk_free_daily

    cov = np.cov(asset_excess, bench_excess, ddof=1)
    beta = cov[0, 1] / cov[1, 1]
    alpha_daily = asset_excess.mean() - beta * bench_excess.mean()
    alpha_annual = alpha_daily * TRADING_DAYS
    return float(beta), float(alpha_annual)


def sharpe(asset: pd.Series, risk_free_daily: float) -> float:
    excess = asset - risk_free_daily
    std = excess.std(ddof=1)
    if std == 0:
        return float("nan")
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(asset: pd.Series, risk_free_daily: float) -> float:
    excess = asset - risk_free_daily
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")  # no down days = undefined
    dd_std = np.sqrt(np.mean(downside ** 2))  # downside deviation
    if dd_std == 0:
        return float("nan")
    return float(excess.mean() / dd_std * np.sqrt(TRADING_DAYS))


def annualized_std(asset: pd.Series) -> float:
    return float(asset.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(asset: pd.Series) -> float:
    """Worst peak-to-trough loss as a negative number (e.g. -0.32 = -32%)."""
    cum = (1 + asset).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    return float(drawdown.min())


def calmar(asset: pd.Series) -> float:
    """Annualized return / |max drawdown|. Higher is better."""
    cum_return = (1 + asset).prod() - 1
    n_years = len(asset) / TRADING_DAYS
    if n_years <= 0:
        return float("nan")
    annual_return = (1 + cum_return) ** (1 / n_years) - 1
    mdd = abs(max_drawdown(asset))
    if mdd == 0:
        return float("inf")
    return float(annual_return / mdd)


def information_ratio(asset: pd.Series, benchmark: pd.Series) -> float:
    """Active return / tracking error, annualized."""
    active = asset - benchmark
    te = active.std(ddof=1)
    if te == 0:
        return float("nan")
    return float(active.mean() / te * np.sqrt(TRADING_DAYS))


def value_at_risk(asset: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR — empirical percentile, not parametric.
    Returns negative number (e.g. -0.034 = expect to lose ≥3.4% on worst 5% of days)."""
    pct = (1 - confidence) * 100
    return float(np.percentile(asset, pct))


# ---------- Top-level ----------------------------------------------------

def compute_performance(ticker: str, cfg: dict) -> dict:
    perf_cfg = cfg["performance"]
    benchmark = perf_cfg["benchmark"]
    rf_annual = perf_cfg["risk_free_rate"]
    rf_daily = rf_annual / TRADING_DAYS

    rets = fetch_returns(ticker, benchmark, perf_cfg["lookback_days"])
    asset = rets[ticker]
    bench = rets[benchmark]

    beta, alpha_a = beta_alpha(asset, bench, rf_daily)

    return {
        "ticker": ticker,
        "benchmark": benchmark,
        "lookback_days": len(rets),
        "period": {
            "start": str(rets.index[0].date()),
            "end":   str(rets.index[-1].date()),
        },
        "metrics": {
            "beta":              beta,
            "alpha_annualized":  alpha_a,
            "sharpe":            sharpe(asset, rf_daily),
            "sortino":           sortino(asset, rf_daily),
            "std_annualized":    annualized_std(asset),
            "max_drawdown":      max_drawdown(asset),
            "calmar":            calmar(asset),
            "information_ratio": information_ratio(asset, bench),
            "var_95_daily":      value_at_risk(asset, perf_cfg["var_confidence"]),
        },
        "benchmark_metrics": {
            "sharpe":         sharpe(bench, rf_daily),
            "std_annualized": annualized_std(bench),
            "max_drawdown":   max_drawdown(bench),
        },
    }


def print_performance_summary(res: dict) -> None:
    ticker = res["ticker"]
    bench  = res["benchmark"]

    print("\n" + "=" * 72)
    print(f"  PERFORMANCE — {ticker} vs {bench}")
    print(f"  {res['period']['start']} → {res['period']['end']}"
          f"  ({res['lookback_days']} trading days)")
    print("=" * 72)

    m = res["metrics"]
    b = res["benchmark_metrics"]

    # (label, ticker_val, bench_val_or_None, note)
    rows = [
        ("Beta",                 f"{m['beta']:.2f}",                  None,                             ""),
        ("Alpha (annualized)",   f"{m['alpha_annualized']:.2%}",       None,                             ""),
        ("Sharpe ratio",         f"{m['sharpe']:.2f}",                 f"{b['sharpe']:.2f}",             ""),
        ("Sortino ratio",        f"{m['sortino']:.2f}",                None,                             ""),
        ("Std dev (ann.)",       f"{m['std_annualized']:.2%}",         f"{b['std_annualized']:.2%}",     ""),
        ("Max drawdown",         f"{m['max_drawdown']:.2%}",           f"{b['max_drawdown']:.2%}",       ""),
        ("Calmar ratio",         f"{m['calmar']:.2f}",                 None,                             ""),
        ("Information ratio",    f"{m['information_ratio']:.2f}",      None,                             ""),
        ("VaR 95% (daily)",      f"{m['var_95_daily']:.2%}",           None,                             "worst 5% of days"),
    ]

    col_w = (26, 10, 10, 18)
    header = (f"  {'Metric':<{col_w[0]}} {ticker:>{col_w[1]}} {bench:>{col_w[2]}}  {'Note':<{col_w[3]}}")
    divider = "  " + "-" * (sum(col_w) + 4)
    print(header)
    print(divider)
    for label, t_val, b_val, note in rows:
        b_str = b_val if b_val is not None else "—"
        print(f"  {label:<{col_w[0]}} {t_val:>{col_w[1]}} {b_str:>{col_w[2]}}  {note:<{col_w[3]}}")
    print(divider)
