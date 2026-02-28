"""
Technical Analysis Engine
Computes all indicators used for signal generation
"""
import numpy as np
import pandas as pd
from typing import Optional


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series: pd.Series, period=20, std_dev=2):
    mid = sma(series, period)
    std = series.rolling(period).std()
    upper = mid + std_dev * std
    lower = mid - std_dev * std
    return upper, mid, lower

def atr(df: pd.DataFrame, period=14) -> pd.Series:
    high = df["high"]
    low  = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period-1, min_periods=period).mean()

def stochastic_rsi(rsi_series: pd.Series, period=14) -> tuple[pd.Series, pd.Series]:
    min_rsi = rsi_series.rolling(period).min()
    max_rsi = rsi_series.rolling(period).max()
    stoch = (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-9)
    k = stoch.rolling(3).mean() * 100
    d = k.rolling(3).mean()
    return k, d

def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol = df["volume"].cumsum()
    cum_tp_vol = (tp * df["volume"]).cumsum()
    return cum_tp_vol / cum_vol

def volume_profile_poc(df: pd.DataFrame, bins=50) -> float:
    """Point of Control - price level with most volume"""
    if df.empty:
        return 0.0
    price_bins = pd.cut(df["close"], bins=bins)
    vol_by_price = df.groupby(price_bins, observed=False)["volume"].sum()
    if vol_by_price.empty:
        return 0.0
    poc_bin = vol_by_price.idxmax()
    if poc_bin is pd.NaT or poc_bin is None:
        return float(df["close"].mean())
    return float(poc_bin.mid)

def supertrend(df: pd.DataFrame, period=10, multiplier=3.0):
    """Supertrend indicator"""
    atr_vals = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals

    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)

    for i in range(1, len(df)):
        if df["close"].iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif df["close"].iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1] if i > 0 else 1

        if direction.iloc[i] == 1:
            supertrend.iloc[i] = lower_band.iloc[i]
        else:
            supertrend.iloc[i] = upper_band.iloc[i]

    return supertrend, direction

def ichimoku(df: pd.DataFrame):
    """Ichimoku Cloud components"""
    high9  = df["high"].rolling(9).max()
    low9   = df["low"].rolling(9).min()
    tenkan = (high9 + low9) / 2

    high26 = df["high"].rolling(26).max()
    low26  = df["low"].rolling(26).min()
    kijun  = (high26 + low26) / 2

    span_a = ((tenkan + kijun) / 2).shift(26)

    high52 = df["high"].rolling(52).max()
    low52  = df["low"].rolling(52).min()
    span_b = ((high52 + low52) / 2).shift(26)

    chikou = df["close"].shift(-26)
    return tenkan, kijun, span_a, span_b, chikou

def pivot_points(df: pd.DataFrame) -> dict:
    """Daily pivot points"""
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    h, l, c = float(prev["high"]), float(prev["low"]), float(prev["close"])
    pivot = (h + l + c) / 3
    return {
        "pivot": round(pivot, 4),
        "r1": round(2 * pivot - l, 4),
        "r2": round(pivot + (h - l), 4),
        "r3": round(h + 2 * (pivot - l), 4),
        "s1": round(2 * pivot - h, 4),
        "s2": round(pivot - (h - l), 4),
        "s3": round(l - 2 * (h - pivot), 4),
    }

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 5) -> str:
    """Detect RSI/MACD divergence. Returns: 'bullish', 'bearish', or 'none'"""
    if len(price) < lookback + 1:
        return "none"
    recent_price = price.iloc[-lookback:]
    recent_ind   = indicator.iloc[-lookback:]

    price_higher = recent_price.iloc[-1] > recent_price.iloc[0]
    ind_higher   = recent_ind.iloc[-1] > recent_ind.iloc[0]

    if price_higher and not ind_higher:
        return "bearish"
    if not price_higher and ind_higher:
        return "bullish"
    return "none"

def detect_patterns(df: pd.DataFrame) -> list[str]:
    """Detect common candlestick patterns"""
    patterns = []
    if len(df) < 3:
        return patterns

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Hammer
    last = df.iloc[-1]
    body = abs(float(last["close"]) - float(last["open"]))
    lower_wick = min(float(last["close"]), float(last["open"])) - float(last["low"])
    upper_wick = float(last["high"]) - max(float(last["close"]), float(last["open"]))
    if lower_wick > 2 * body and upper_wick < body * 0.5:
        patterns.append("Hammer 🔨")

    # Doji
    if body < (float(last["high"]) - float(last["low"])) * 0.1:
        patterns.append("Doji ⚖️")

    # Engulfing
    prev = df.iloc[-2]
    prev_body = abs(float(prev["close"]) - float(prev["open"]))
    if (float(last["close"]) > float(last["open"]) and
        float(prev["close"]) < float(prev["open"]) and
        body > prev_body * 1.1):
        patterns.append("Bullish Engulfing 🟢")

    if (float(last["close"]) < float(last["open"]) and
        float(prev["close"]) > float(prev["open"]) and
        body > prev_body * 1.1):
        patterns.append("Bearish Engulfing 🔴")

    return patterns


def calculate_all_indicators(df: pd.DataFrame) -> dict:
    """Compute all indicators and return as dict of latest values"""
    if df.empty or len(df) < 50:
        return {}

    close = df["close"]
    volume = df["volume"]

    rsi14        = rsi(close, 14)
    rsi7         = rsi(close, 7)
    macd_l, macd_s, macd_h = macd(close)
    bb_u, bb_m, bb_l = bollinger_bands(close)
    atr14        = atr(df, 14)
    stoch_k, stoch_d = stochastic_rsi(rsi14)
    vwap_val     = vwap(df)
    ema9         = ema(close, 9)
    ema21        = ema(close, 21)
    ema50        = ema(close, 50)
    ema200       = ema(close, 200) if len(df) >= 200 else ema(close, len(df))
    vol_sma20    = sma(volume, 20)
    vol_sma5     = sma(volume, 5)

    divergence_rsi  = detect_divergence(close, rsi14)
    divergence_macd = detect_divergence(close, macd_h)
    patterns        = detect_patterns(df)
    pivots          = pivot_points(df)
    poc             = volume_profile_poc(df.tail(50))

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    curr_price = float(close.iloc[-1])

    return {
        "price":          curr_price,
        "rsi14":          float(rsi14.iloc[-1]) if not pd.isna(rsi14.iloc[-1]) else 50,
        "rsi7":           float(rsi7.iloc[-1]) if not pd.isna(rsi7.iloc[-1]) else 50,
        "macd_line":      float(macd_l.iloc[-1]) if not pd.isna(macd_l.iloc[-1]) else 0,
        "macd_signal":    float(macd_s.iloc[-1]) if not pd.isna(macd_s.iloc[-1]) else 0,
        "macd_hist":      float(macd_h.iloc[-1]) if not pd.isna(macd_h.iloc[-1]) else 0,
        "macd_hist_prev": float(macd_h.iloc[-2]) if not pd.isna(macd_h.iloc[-2]) else 0,
        "bb_upper":       float(bb_u.iloc[-1]) if not pd.isna(bb_u.iloc[-1]) else curr_price * 1.02,
        "bb_mid":         float(bb_m.iloc[-1]) if not pd.isna(bb_m.iloc[-1]) else curr_price,
        "bb_lower":       float(bb_l.iloc[-1]) if not pd.isna(bb_l.iloc[-1]) else curr_price * 0.98,
        "atr14":          float(atr14.iloc[-1]) if not pd.isna(atr14.iloc[-1]) else curr_price * 0.01,
        "stoch_k":        float(stoch_k.iloc[-1]) if not pd.isna(stoch_k.iloc[-1]) else 50,
        "stoch_d":        float(stoch_d.iloc[-1]) if not pd.isna(stoch_d.iloc[-1]) else 50,
        "vwap":           float(vwap_val.iloc[-1]) if not pd.isna(vwap_val.iloc[-1]) else curr_price,
        "ema9":           float(ema9.iloc[-1]),
        "ema21":          float(ema21.iloc[-1]),
        "ema50":          float(ema50.iloc[-1]),
        "ema200":         float(ema200.iloc[-1]),
        "vol_current":    float(volume.iloc[-1]),
        "vol_sma20":      float(vol_sma20.iloc[-1]) if not pd.isna(vol_sma20.iloc[-1]) else 1,
        "vol_sma5":       float(vol_sma5.iloc[-1]) if not pd.isna(vol_sma5.iloc[-1]) else 1,
        "obv":            float(obv.iloc[-1]),
        "obv_prev":       float(obv.iloc[-2]),
        "divergence_rsi": divergence_rsi,
        "divergence_macd":divergence_macd,
        "patterns":       patterns,
        "pivots":         pivots,
        "poc":            poc,
    }
