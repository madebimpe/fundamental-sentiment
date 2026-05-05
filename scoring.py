"""Composite scoring: combine sentiment + fundamentals into buy/hold/sell.

Simple weighted average of the two legs. If one leg is unavailable
(network failure, no data), fall back to the other rather than crashing —
log it loudly so the user knows the signal is single-legged.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def decide(score: float, buy_threshold: float, sell_threshold: float) -> str:
    if score >= buy_threshold:
        return "BUY"
    if score <= sell_threshold:
        return "SELL"
    return "HOLD"


def composite_score(sentiment_result: dict, fundamentals_result: dict,
                    cfg: dict) -> dict:
    """Combine the two legs into a single 0-100 score and signal."""
    sent = sentiment_result.get("composite_score") if sentiment_result else None
    fund = fundamentals_result.get("composite_score") if fundamentals_result else None

    w_sent = cfg["composite"]["sentiment_weight"]
    w_fund = cfg["composite"]["fundamentals_weight"]

    if sent is None and fund is None:
        return {
            "composite_score": None,
            "signal": "NO_SIGNAL",
            "reason": "both legs unavailable",
            "sentiment_score": None,
            "fundamentals_score": None,
            "weights_used": {"sentiment": 0, "fundamentals": 0},
        }

    if sent is None:
        logger.warning("Sentiment unavailable — falling back to fundamentals only")
        composite = fund
        weights_used = {"sentiment": 0.0, "fundamentals": 1.0}
    elif fund is None:
        logger.warning("Fundamentals unavailable — falling back to sentiment only")
        composite = sent
        weights_used = {"sentiment": 1.0, "fundamentals": 0.0}
    else:
        composite = sent * w_sent + fund * w_fund
        weights_used = {"sentiment": w_sent, "fundamentals": w_fund}

    thresholds = cfg["composite"]["thresholds"]
    signal = decide(composite, thresholds["buy"], thresholds["sell"])

    return {
        "composite_score": composite,
        "signal": signal,
        "sentiment_score": sent,
        "fundamentals_score": fund,
        "weights_used": weights_used,
        "thresholds": thresholds,
    }


def print_composite_summary(res: dict, ticker: str) -> None:
    print("\n" + "=" * 60)
    print(f"COMPOSITE SIGNAL — {ticker}")
    print("=" * 60)
    score = res["composite_score"]
    if score is None:
        print(f"\nNO SIGNAL — {res.get('reason', 'unknown')}")
        return
    print(f"\nSignal: {res['signal']}")
    print(f"Composite score: {score:.1f} / 100")
    print(f"  thresholds: buy ≥ {res['thresholds']['buy']}, "
          f"sell ≤ {res['thresholds']['sell']}")
    sent = res["sentiment_score"]
    fund = res["fundamentals_score"]
    w = res["weights_used"]
    print(f"\nLeg breakdown:")
    print(f"  sentiment:    "
          f"{sent:.1f}" if sent is not None else "  sentiment:    n/a", end="")
    print(f"   (weight {w['sentiment']:.0%})")
    print(f"  fundamentals: "
          f"{fund:.1f}" if fund is not None else "  fundamentals: n/a", end="")
    print(f"   (weight {w['fundamentals']:.0%})")
