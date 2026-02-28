"""
Bot Configuration - Edit these settings
"""
import os
from dotenv import load_dotenv
load_dotenv()

# ─── DISCORD ────────────────────────────────────────────────────────────────
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Channel IDs - Set these after creating channels in your Discord server
SIGNALS_SCALP_CHANNEL_ID    = int(os.getenv("SIGNALS_SCALP_CHANNEL_ID", 0))
SIGNALS_DAYTRADER_CHANNEL_ID= int(os.getenv("SIGNALS_DAYTRADER_CHANNEL_ID", 0))
SIGNALS_SWING_CHANNEL_ID    = int(os.getenv("SIGNALS_SWING_CHANNEL_ID", 0))
LIQUIDATION_CHANNEL_ID      = int(os.getenv("LIQUIDATION_CHANNEL_ID", 0))
LOG_CHANNEL_ID              = int(os.getenv("LOG_CHANNEL_ID", 0))

# ─── API KEYS ────────────────────────────────────────────────────────────────
CMC_API_KEY       = os.getenv("CMC_API_KEY", "")         # CoinMarketCap
BINANCE_API_KEY   = os.getenv("BINANCE_API_KEY", "")     # Binance (read-only)
BINANCE_SECRET    = os.getenv("BINANCE_SECRET", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Claude AI (for signal analysis + auto-tune)

# ─── SIGNAL ENGINE CONFIG ────────────────────────────────────────────────────
SCAN_INTERVAL_SCALP   = 300    # seconds (5 min)
SCAN_INTERVAL_DAY     = 900    # seconds (15 min)
SCAN_INTERVAL_SWING   = 3600   # seconds (1 hour)
LIQUIDATION_INTERVAL  = 60     # seconds (1 min)

TOP_COINS_LIMIT = 1000         # Top N coins from CMC
CMC_REFRESH_HOURS = 6          # Refresh CMC list every N hours

# ─── SIGNAL THRESHOLDS ───────────────────────────────────────────────────────
MIN_VOLUME_24H_USD  = 5_000_000      # Min 24h volume $5M
MIN_OI_USD          = 2_000_000      # Min Open Interest $2M
VOLUME_SPIKE_MULT   = 1.5            # Volume must be 1.5x 20-period avg (was 2.0 — too strict)
BTC_OUTPERFORM_PCT  = 0.5            # % outperformance vs BTC required (was 1.5 — too strict)

# Grade thresholds (confluence score 0-100)
GRADE_A_PLUS = 80
GRADE_B_PLUS = 60
GRADE_C_PLUS = 40

# ─── RISK CONFIG ─────────────────────────────────────────────────────────────
# All trades enforce minimum 1:2 R:R (first TP must be >= 2× the SL distance)
# Scalp  targets: 1–5%  total move from entry
# Day    targets: 10–12% total move from entry
# Swing  targets: 10–20% total move from entry

# SL is now structure-based (swing high/low + ATR buffer) — not fixed %
# ATR buffer multipliers are set per trade type in signal_scorer.py
# Scalp=0.5×ATR  Day=1.0×ATR  Swing=1.5×ATR beyond the structure level

# TP ratios per GRADE — A+ gets maximum extended targets, B+ standard, C+ conservative
# Minimum 1:2 R:R enforced on all (first ratio always >= 2.0)

TP_RR_RATIOS = {
    "A+": {
        # A+ SCALP: push for full 5% move — 5 TPs, last one is a runner
        "scalp": [2.0, 3.5, 5.0, 7.0, 10.0],
        # A+ DAY: full 12-18% move — 5 TPs, runner at 5× risk
        "day":   [2.0, 3.0, 4.0, 5.5, 7.0],
        # A+ SWING: max 20-30% move — 5 TPs, last TP is a moonshot runner
        "swing": [2.0, 3.0, 4.5, 6.5, 9.0],
    },
    "B+": {
        # B+ SCALP: solid 1-5% — 4 TPs
        "scalp": [2.0, 3.1, 4.5, 6.0],
        # B+ DAY: 10-14% — 4 TPs
        "day":   [2.0, 2.9, 3.8, 5.0],
        # B+ SWING: 10-20% — 4 TPs
        "swing": [2.0, 3.0, 4.2, 5.5],
    },
    "C+": {
        # C+ SCALP: conservative 1-3% — 3 TPs, take profit early
        "scalp": [2.0, 2.8, 3.8],
        # C+ DAY: 7-12% — 3 TPs
        "day":   [2.0, 2.8, 3.5],
        # C+ SWING: 10-16% — 3 TPs
        "swing": [2.0, 2.8, 3.8],
    },
}

# Leverage suggestions
LEVERAGE = {
    "A+": {"scalp": 10, "day": 7, "swing": 5},
    "B+": {"scalp": 7,  "day": 5, "swing": 3},
    "C+": {"scalp": 5,  "day": 3, "swing": 2},
}

# ─── ML CONFIG ───────────────────────────────────────────────────────────────
ML_RETRAIN_HOURS     = 24
ML_MIN_SAMPLES       = 50
ML_MODEL_PATH        = "data/ml_model.pkl"
TRADE_HISTORY_PATH   = "data/trade_history.json"
