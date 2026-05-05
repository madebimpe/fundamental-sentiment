# Intel Strategy — Beta

Discretionary buy/hold/sell signal for INTC built from sentiment + fundamentals.
Composite score (0–100) → BUY ≥65, SELL ≤35, HOLD otherwise.

## Quick start in VS Code

1. Open the folder: `File → Open Folder...` → select `intel-strategy/`
2. VS Code will prompt to install recommended extensions — accept (Python, Pylance, YAML).
3. Create the virtual environment in a terminal (`Ctrl+` ` ` `):

   ```bash
   python -m venv .venv
   ```

4. VS Code should auto-detect `.venv` as the interpreter. If not: `Cmd/Ctrl+Shift+P` → "Python: Select Interpreter" → pick `.venv`.
5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Hit **F5** to run with the debugger. Pick any of the four launch configs:
   - Run: full pipeline (default)
   - Run: fundamentals only
   - Run: sentiment only
   - Run: performance only

   Or run from the terminal:

   ```bash
   python main.py --mode full
   python main.py --mode fundamentals
   python main.py --mode sentiment --device cpu     # or cuda if you have a GPU
   python main.py --mode performance
   ```

## Project structure

```
intel-strategy/
  config.yaml              # all tunable knobs — ticker, peers, weights, thresholds
  main.py                  # CLI entry point
  scoring.py               # composite leg combiner + buy/hold/sell decision
  performance.py           # vs-SPY metrics
  data/
    fundamentals.py        # 5-metric percentile rank vs peers
    sentiment.py           # StockTwits + FinBERT
  output/                  # JSON results + run.log (gitignored)
  cache/                   # FinBERT cache (gitignored)
  .vscode/                 # workspace settings and launch configs
```

## How the score is built

**Fundamentals (50% weight, refresh weekly)** — five metrics, each percentile-ranked across peers (AMD, NVDA, QCOM, AVGO, TXN, MU, TSM):

| Metric | Direction | Weight |
|---|---|---|
| EV/EBITDA | lower better | 25% |
| 2yr revenue CAGR | higher better | 20% |
| TTM gross margin trend | higher better | 20% |
| TTM FCF yield | higher better | 20% |
| Net debt / TTM EBITDA | lower better | 15% |

**Sentiment (50% weight, refresh daily)** — StockTwits stream → FinBERT → recency-weighted mean. 0–100 with 50 neutral.

**Performance metrics** (decoupled from the signal — these tell you how the underlying *asset* has behaved, not what to do):
- Beta, Alpha (annualized)
- Sharpe, Sortino, Calmar
- Std dev (annualized)
- Max drawdown, Information ratio
- VaR 95% (historical, not parametric)

## Output

Each run writes timestamped JSON to `output/`. `--mode full` produces a single combined report with composite signal, both legs, and performance metrics.

## Tuning

Edit `config.yaml`:
- **Different ticker**: change `target.ticker` and `peers`
- **Reweight metrics**: edit `fundamentals.weights` (must sum to 1.0)
- **Shift thresholds**: edit `composite.thresholds` (lower buy threshold = more aggressive)
- **Change leg balance**: edit `composite.sentiment_weight` / `fundamentals_weight`

## Status

- [x] Config + skeleton
- [x] Fundamentals (5 metrics, percentile rank vs peers)
- [x] Sentiment (StockTwits + FinBERT, recency-weighted)
- [x] Composite signal + buy/hold/sell
- [x] Performance metrics vs SPY
- [ ] Streamlit dashboard (next)
- [ ] IBKR paper trading integration (later)

## Known caveats

- yfinance is free and convenient but can occasionally rename row labels between versions. The fundamentals module is defensive against minor label drift but a major rename will require a code update.
- StockTwits' `$INTC` stream is heavily retail; expect noisy days. The recency weighting and distribution stats help diagnose this.
- FinBERT first-run downloads ~440MB to your HuggingFace cache. After that, scoring 30 messages takes ~2 seconds on CPU.
