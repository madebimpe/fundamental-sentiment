"""Entry point. Run a single leg, performance metrics, or the full pipeline.

Usage:
    python main.py --mode fundamentals    # weekly refresh
    python main.py --mode sentiment       # daily refresh
    python main.py --mode performance     # diagnostic vs SPY
    python main.py --mode full            # all of the above + composite signal
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import yaml

from data.fundamentals import score_fundamentals
from data.sentiment import score_sentiment
from performance import compute_performance, print_performance_summary
from scoring import composite_score, print_composite_summary


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )


def write_output(result: dict, output_dir: Path, mode: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{mode}_{ts}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path


# ---------- Per-mode runners ---------------------------------------------

def print_fundamentals_summary(res: dict) -> None:
    print("\n" + "=" * 60)
    print(f"FUNDAMENTALS REPORT — {res['ticker']}")
    print("=" * 60)
    score = res["composite_score"]
    print(f"\nComposite score: {score:.1f} / 100" if score is not None
          else "\nComposite score: UNAVAILABLE")
    print("\nFeature scores (0=worst vs peers, 100=best):")
    for feat, s in res["feature_scores"].items():
        raw = res["raw_values"].get(feat)
        raw_str = f"{raw:.3f}" if isinstance(raw, (int, float)) else "n/a"
        s_str = f"{s:.1f}" if s is not None else "n/a"
        print(f"  {feat:25s}  score={s_str:>6}   raw={raw_str}")
    if res["errors"]:
        print("\nIssues encountered:")
        for tk, errs in res["errors"].items():
            for e in errs:
                print(f"  [{tk}] {e}")


def print_sentiment_summary(res: dict) -> None:
    print("\n" + "=" * 60)
    print(f"SENTIMENT REPORT — {res['ticker']} ({res['source']})")
    print("=" * 60)
    score = res.get("composite_score")
    if score is None:
        print(f"\nNo data: {res.get('reason', 'unknown')}")
        return
    print(f"\nComposite score: {score:.1f} / 100  (50 = neutral)")
    print(f"Messages analyzed: {res['message_count']}")
    mean, median, std = res.get("mean"), res.get("median"), res.get("std")
    if mean is not None:
        print(f"  mean={mean:.1f}  median={median:.1f}  std={std:.1f}")
    d = res.get("distribution")
    if d:
        print(f"\nDistribution (FinBERT):")
        print(f"  bullish (>60): {d['bullish_pct']:.0f}%")
        print(f"  neutral:       {d['neutral_pct']:.0f}%")
        print(f"  bearish (<40): {d['bearish_pct']:.0f}%")

    legs = res.get("legs", {})
    if legs:
        print(f"\nLeg scores:")
        for leg_name, leg in legs.items():
            leg_score = leg.get("composite_score")
            leg_count = leg.get("message_count", 0)
            score_str = f"{leg_score:.1f}" if leg_score is not None else "n/a"
            print(f"  {leg_name:8s}: {score_str:>5} / 100  ({leg_count} items)")

        reddit_breakdown = legs.get("reddit", {}).get("source_breakdown", {})
        if reddit_breakdown:
            print(f"\nSubreddit breakdown:")
            for name, count in sorted(reddit_breakdown.items(), key=lambda x: -x[1]):
                print(f"  r/{name}: {count} posts")

        news_breakdown = legs.get("news", {}).get("source_breakdown", {})
        if news_breakdown:
            print(f"\nNews sources:")
            for name, count in sorted(news_breakdown.items(), key=lambda x: -x[1]):
                print(f"  {name}: {count} articles")


def run_fundamentals(cfg: dict, output_dir: Path) -> dict:
    res = score_fundamentals(
        target=cfg["target"]["ticker"],
        peers=cfg["peers"],
        weights=cfg["fundamentals"]["weights"],
    )
    print_fundamentals_summary(res)
    path = write_output(res, output_dir, "fundamentals")
    print(f"\nFull output written to: {path}")
    return res


def run_sentiment(cfg: dict, output_dir: Path, device: str = "cpu") -> dict:
    s_cfg = cfg["sentiment"]
    r_cfg = s_cfg["reddit"]
    feeds = s_cfg.get("news", {}).get("feeds", [])
    weights = s_cfg.get("weights", {})
    res = score_sentiment(
        ticker=cfg["target"]["ticker"],
        subreddits=r_cfg["subreddits"],
        feeds=feeds,
        max_items=r_cfg["max_items"],
        lookback_hours=r_cfg["lookback_hours"],
        device=device,
        weights=weights,
    )
    print_sentiment_summary(res)
    path = write_output(res, output_dir, "sentiment")
    print(f"\nFull output written to: {path}")
    return res


def run_performance(cfg: dict, output_dir: Path) -> dict:
    res = compute_performance(cfg["target"]["ticker"], cfg)
    print_performance_summary(res)
    path = write_output(res, output_dir, "performance")
    print(f"\nFull output written to: {path}")
    return res


def run_full(cfg: dict, output_dir: Path, device: str = "cpu") -> dict:
    """Run everything and produce a single composite signal report."""
    print("\n>>> Running fundamentals leg...")
    fund = None
    try:
        fund = run_fundamentals(cfg, output_dir)
    except Exception as e:
        logging.error("Fundamentals leg failed: %s", e)

    print("\n>>> Running sentiment leg...")
    sent = None
    try:
        sent = run_sentiment(cfg, output_dir, device=device)
    except Exception as e:
        logging.error("Sentiment leg failed: %s", e)

    print("\n>>> Computing composite signal...")
    composite = composite_score(sent, fund, cfg)
    print_composite_summary(composite, cfg["target"]["ticker"])

    print("\n>>> Computing performance metrics...")
    perf = None
    try:
        perf = run_performance(cfg, output_dir)
    except Exception as e:
        logging.error("Performance leg failed: %s", e)

    full_report = {
        "ticker": cfg["target"]["ticker"],
        "timestamp": datetime.now().isoformat(),
        "composite": composite,
        "sentiment": sent,
        "fundamentals": fund,
        "performance": perf,
    }
    path = write_output(full_report, output_dir, "full")
    print(f"\nFull report written to: {path}")
    return full_report


# ---------- CLI ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="INTC discretionary signal: sentiment + fundamentals.")
    parser.add_argument("--mode", choices=["fundamentals", "sentiment",
                                            "performance", "full"],
                        default="full")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                        help="Device for FinBERT (sentiment only)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output"])
    setup_logging(output_dir)

    if args.mode == "fundamentals":
        run_fundamentals(cfg, output_dir)
    elif args.mode == "sentiment":
        run_sentiment(cfg, output_dir, device=args.device)
    elif args.mode == "performance":
        run_performance(cfg, output_dir)
    elif args.mode == "full":
        run_full(cfg, output_dir, device=args.device)


if __name__ == "__main__":
    main()
