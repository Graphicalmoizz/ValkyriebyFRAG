"""
Signal Scoring Engine
Scores each potential trade from 0-100 based on confluence
Returns signal details including direction, grade, TPs, SL
"""
import logging
from typing import Optional
import config

logger = logging.getLogger("SignalScorer")


class SignalScorer:
    def __init__(self):
        pass

    def score_signal(
        self,
        symbol: str,
        trade_type: str,   # scalp | day | swing
        indicators: dict,
        oi_data: dict,
        funding_rate: float,
        ob_imbalance: float,
        liq_zones: dict,
        taker_ls_ratio: float,
        btc_change: float,
        coin_change: float,
        ml_score: float = 0.0,
        dom_regime: dict = None,   # BTC.D / USDT.D market regime
        btc_corr: float = 0.5,     # Pearson correlation with BTC (-1 to +1)
    ) -> Optional[dict]:
        """
        Score a signal 0-100 and return full signal dict if strong enough
        Returns None if no signal found
        """
        if not indicators:
            return None

        # Use .get() with sane defaults — missing key never crashes the scorer
        price       = indicators.get("price", 0)
        if not price:
            return None

        # ── Time-of-day aware volume thresholds ──────────────────────────────
        # Crypto volume follows a strong intraday cycle. At 01:00-07:00 UTC
        # (Asian dead hours), a single 5m candle is naturally 80-95% below the
        # 20-bar SMA which spans more active periods. Fixed thresholds would
        # kill every signal in off-hours regardless of trend quality.
        #
        # UTC hour zones:
        #   Peak    08-17 UTC  London + NY overlap — full thresholds
        #   Active  06-08, 17-20 UTC  Asian open / NY close
        #   Dead    20-06 UTC  off-hours
        import datetime as _dt
        _utc_hour = _dt.datetime.utcnow().hour

        if 8 <= _utc_hour < 17:
            # Peak hours: highest standards
            _VOL_HARD_REJECT = 0.50
            _VOL_MINIMUM     = 1.20
            _VOL_CONFIRMED   = 1.50
            _VOL_STRONG      = 2.00
        elif 6 <= _utc_hour < 8 or 17 <= _utc_hour < 20:
            # Transition hours: slightly relaxed
            _VOL_HARD_REJECT = 0.30
            _VOL_MINIMUM     = 0.80
            _VOL_CONFIRMED   = 1.20
            _VOL_STRONG      = 1.80
        else:
            # Dead hours 20:00-06:00 UTC: significantly relaxed
            _VOL_HARD_REJECT = 0.20
            _VOL_MINIMUM     = 0.50
            _VOL_CONFIRMED   = 0.80
            _VOL_STRONG      = 1.50

        rsi14          = indicators.get("rsi14", 50.0)
        macd_hist      = indicators.get("macd_hist", 0.0)
        macd_hist_prev = indicators.get("macd_hist_prev", 0.0)
        stoch_k        = indicators.get("stoch_k", 50.0)
        vol_current    = indicators.get("vol_current", 1.0)
        vol_sma20      = indicators.get("vol_sma20", 1.0)
        ema9           = indicators.get("ema9", price)
        ema21          = indicators.get("ema21", price)
        ema50          = indicators.get("ema50", price)
        vwap           = indicators.get("vwap", price)
        bb_lower       = indicators.get("bb_lower", price * 0.97)
        bb_upper       = indicators.get("bb_upper", price * 1.03)
        atr            = indicators.get("atr14", price * 0.01)

        # Pre-calculate vol_ratio needed throughout
        vol_ratio = vol_current / (vol_sma20 + 1e-9)

        # ─── Determine Direction ─────────────────────────────────────────────
        bullish_points = 0
        bearish_points = 0

        # EMA trend
        if ema9 > ema21 > ema50:
            bullish_points += 2
        elif ema9 < ema21 < ema50:
            bearish_points += 2

        # Price vs VWAP
        if price > vwap:
            bullish_points += 1
        else:
            bearish_points += 1

        # RSI
        if rsi14 < 35:
            bullish_points += 2
        elif rsi14 > 65:
            bearish_points += 2
        elif 40 < rsi14 < 60:
            pass  # neutral
        elif rsi14 > 50:
            bullish_points += 1
        else:
            bearish_points += 1

        # MACD histogram turning
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            bullish_points += 2
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            bearish_points += 2

        # Stochastic
        if stoch_k < 20:
            bullish_points += 1
        elif stoch_k > 80:
            bearish_points += 1

        # Order flow
        if ob_imbalance > 0.15:
            bullish_points += 2
        elif ob_imbalance < -0.15:
            bearish_points += 2

        # Taker LS ratio
        if taker_ls_ratio > 1.1:
            bullish_points += 1
        elif taker_ls_ratio < 0.9:
            bearish_points += 1

        # Funding rate (extreme = contrarian)
        if funding_rate > 0.001:    # very positive = longs paying = bear signal
            bearish_points += 1
        elif funding_rate < -0.001: # very negative = shorts paying = bull signal
            bullish_points += 1

        # Coin vs BTC outperformance
        outperform = coin_change - btc_change
        if outperform > config.BTC_OUTPERFORM_PCT:
            bullish_points += 2
        elif outperform < -config.BTC_OUTPERFORM_PCT:
            bearish_points += 1

        # Divergence
        div_rsi  = indicators.get("divergence_rsi", "none")
        div_macd = indicators.get("divergence_macd", "none")
        if div_rsi == "bullish" or div_macd == "bullish":
            bullish_points += 3
        elif div_rsi == "bearish" or div_macd == "bearish":
            bearish_points += 3

        # BB squeeze — reinforce dominant direction, not always bullish
        bb_width = (bb_upper - bb_lower) / price
        if bb_width < 0.03:
            if bullish_points >= bearish_points:
                bullish_points += 1
            else:
                bearish_points += 1

        total = bullish_points + bearish_points
        if total == 0:
            logger.debug(f"{symbol} REJECTED: zero direction points")
            return None

        bull_ratio = bullish_points / total
        if bull_ratio >= 0.54:
            direction = "LONG"
        elif bull_ratio <= 0.46:
            direction = "SHORT"
        else:
            logger.info(f"{symbol} REJECTED: no clear direction bull_ratio={bull_ratio:.2f} (bull={bullish_points} bear={bearish_points})")
            return None

        # ─── Wyckoff Phase Detection ─────────────────────────────────────────
        wyckoff_phase = "none"
        wyckoff_bonus = 0
        obv       = indicators.get("obv", 0)
        obv_prev  = indicators.get("obv_prev", 0)
        obv_rising = obv > obv_prev
        pivots    = indicators.get("pivots", {})

        if (rsi14 < 35 and vol_ratio >= 1.5 and obv_rising and ob_imbalance > -0.1 and direction == "LONG"):
            wyckoff_phase = "accumulation_spring"
            wyckoff_bonus = 12
        elif (ema9 > ema21 > ema50 and vol_ratio >= 1.5 and price > vwap and obv_rising and direction == "LONG"):
            wyckoff_phase = "markup_SOS"
            wyckoff_bonus = 10
        elif (rsi14 > 65 and vol_ratio < 1.2 and not obv_rising and div_rsi == "bearish" and direction == "SHORT"):
            wyckoff_phase = "distribution_UTAD"
            wyckoff_bonus = 12
        elif (ema9 < ema21 < ema50 and vol_ratio >= 1.5 and price < vwap and not obv_rising and direction == "SHORT"):
            wyckoff_phase = "markdown_SOW"
            wyckoff_bonus = 10

        # ─── Price Action Signals ─────────────────────────────────────────────
        pa_bonus   = 0
        pa_signals = []
        atr_pct    = atr / (price + 1e-9)

        # 1. Liquidity sweep / stop hunt
        if direction == "LONG" and price < bb_lower * 1.003 and rsi14 < 45:
            pa_bonus += 8
            pa_signals.append("Liquidity sweep — stop hunt long")
        elif direction == "SHORT" and price > bb_upper * 0.997 and rsi14 > 55:
            pa_bonus += 8
            pa_signals.append("Liquidity sweep — stop hunt short")

        # 2. Fair Value Gap (large ATR expansion + price returning to mean)
        if atr_pct > 0.008:
            if direction == "LONG" and price < vwap:
                pa_bonus += 6
                pa_signals.append("FVG fill — price below VWAP after expansion")
            elif direction == "SHORT" and price > vwap:
                pa_bonus += 6
                pa_signals.append("FVG fill — price above VWAP after expansion")

        # 3. Break of Structure
        if direction == "LONG" and pivots.get("r1", 0) and price > pivots.get("r1", price * 1.1):
            pa_bonus += 5
            pa_signals.append("Break of structure above R1")
        elif direction == "SHORT" and pivots.get("s1", 0) and price < pivots.get("s1", price * 0.9):
            pa_bonus += 5
            pa_signals.append("Break of structure below S1")

        # 4. Order Block imbalance
        if direction == "LONG" and ob_imbalance > 0.20:
            pa_bonus += 6
            pa_signals.append(f"Order block bid imbalance {ob_imbalance:+.2f}")
        elif direction == "SHORT" and ob_imbalance < -0.20:
            pa_bonus += 6
            pa_signals.append(f"Order block ask imbalance {ob_imbalance:+.2f}")

        # 5. Trend continuation
        if direction == "LONG" and ema9 > ema21 > ema50 and macd_hist > 0:
            pa_bonus += 5
            pa_signals.append("Trend continuation — EMAs + MACD aligned bullish")
        elif direction == "SHORT" and ema9 < ema21 < ema50 and macd_hist < 0:
            pa_bonus += 5
            pa_signals.append("Trend continuation — EMAs + MACD aligned bearish")

        # ─── Confluence Score (0-100) ─────────────────────────────────────────
        score = 0

        # Trend alignment (20 pts)
        if direction == "LONG":
            if ema9 > ema21:  score += 7
            if ema21 > ema50: score += 7
            if price > vwap:  score += 6
        else:
            if ema9 < ema21:  score += 7
            if ema21 < ema50: score += 7
            if price < vwap:  score += 6

        # Momentum (20 pts)
        # RSI healthy zone: 40-65 for longs (below 40 = weak, above 65 = chasing)
        #                   35-60 for shorts (above 60 = weak entry, below 35 = chasing)
        if direction == "LONG":
            if macd_hist > 0 and macd_hist > macd_hist_prev: score += 10
            if 35 <= rsi14 <= 65:   score += 8   # healthy momentum — best zone
            elif rsi14 < 35:        score += 4   # oversold bounce — ok but risky
            # RSI > 65 on a LONG = chasing overbought — no points
            if rsi14 > 50:          score += 2   # has upside momentum above midline
        else:
            if macd_hist < 0 and macd_hist < macd_hist_prev: score += 10
            if 35 <= rsi14 <= 65:   score += 8   # healthy downside momentum
            elif rsi14 > 65:        score += 4   # overbought rejection — ok but risky
            # RSI < 35 on a SHORT = chasing oversold — no points
            if rsi14 < 50:          score += 2   # has downside momentum below midline

        # Volume (15 pts)
        # Hard filter: reject dead volume immediately
        if vol_ratio < _VOL_HARD_REJECT:
            logger.info(f"{symbol} REJECTED: vol_ratio={vol_ratio:.2f} < {_VOL_HARD_REJECT} (dead volume, UTC hour={_utc_hour})")
            return None
        if vol_ratio >= _VOL_STRONG * 1.25:
            score += 15   # massive spike — institutional interest
        elif vol_ratio >= _VOL_STRONG:
            score += 12   # strong volume confirmation
        elif vol_ratio >= _VOL_CONFIRMED:
            score += 8    # above average — healthy
        elif vol_ratio >= _VOL_MINIMUM:
            score += 4    # slight uptick — ok but not decisive
        # vol below minimum but above hard-reject = 0 pts (low vol, marginal setup)

        # Order flow (15 pts)
        if direction == "LONG":
            if ob_imbalance > 0.1:  score += 8
            if taker_ls_ratio > 1.0: score += 7
        else:
            if ob_imbalance < -0.1: score += 8
            if taker_ls_ratio < 1.0: score += 7

        # Divergence (15 pts)
        if direction == "LONG" and (div_rsi == "bullish" or div_macd == "bullish"):
            score += 15
        elif direction == "SHORT" and (div_rsi == "bearish" or div_macd == "bearish"):
            score += 15

        # Candlestick patterns (10 pts) — only reward aligned patterns
        patterns = indicators.get("patterns", [])
        bullish_pats = ["Hammer", "Bullish Engulfing"]
        bearish_pats = ["Bearish Engulfing"]
        aligned = 0
        conflicting = 0
        for p in patterns:
            if direction == "LONG" and any(bp in p for bp in bullish_pats):
                aligned += 1
            elif direction == "SHORT" and any(bp in p for bp in bearish_pats):
                aligned += 1
            elif direction == "LONG" and any(bp in p for bp in bearish_pats):
                conflicting += 1
            elif direction == "SHORT" and any(bp in p for bp in bullish_pats):
                conflicting += 1
        score += min(10, aligned * 5)
        score -= conflicting * 5  # penalize conflicting patterns

        # ML bonus (5 pts)
        score += min(5, ml_score * 5)

        # Wyckoff phase bonus (max 12 pts)
        score += wyckoff_bonus

        # Price Action bonus (max 20 pts, capped)
        score += min(20, pa_bonus)

        # ── BTC Correlation score adjustment ─────────────────────────────────
        # Coin correlating strongly with BTC in same direction as BTC = good
        # Coin correlating strongly with BTC against BTC direction = bad
        if btc_corr is not None:
            if direction == "LONG":
                if btc_corr > 0.75 and btc_change > 0:
                    score += 6   # high corr + BTC rising = confirmed tailwind
                elif btc_corr > 0.75 and btc_change < -0.5:
                    score -= 8   # high corr + BTC falling = headwind on long
                elif btc_corr < 0.25 and btc_change < 0:
                    score += 4   # decorrelated coin can run despite BTC weakness
            else:  # SHORT
                if btc_corr > 0.75 and btc_change < -0.5:
                    score += 6   # high corr + BTC falling = confirmed tailwind short
                elif btc_corr > 0.75 and btc_change > 0.5:
                    score -= 8   # high corr + BTC rising = headwind on short
                elif btc_corr < 0.25 and btc_change > 0:
                    score += 4   # decorrelated coin can fall despite BTC strength

        # ── Dominance-adjusted score ─────────────────────────────────────────
        # Boost or penalize based on macro regime alignment
        if dom_regime:
            regime   = dom_regime.get("regime", "neutral")
            btc_d    = dom_regime.get("btc_d", 0)
            usdt_d   = dom_regime.get("usdt_d", 0)
            btc_high = dom_regime.get("btc_d_high", False)
            btc_low  = dom_regime.get("btc_d_low", False)

            if regime == "risk_on_alt" and direction == "LONG":
                score += 8   # altseason — long alts is WITH the flow
            elif regime == "risk_on_btc" and direction == "LONG":
                base = symbol.replace("USDT","").replace("PERP","")
                if base in ["BTC","ETH"]:
                    score += 6   # BTC season — BTC/ETH longs fine
                else:
                    score -= 6   # BTC season — altcoin longs go against flow
            elif regime == "risk_off" and direction == "SHORT":
                score += 8   # fear regime — shorts are WITH the flow
            elif regime == "risk_off" and direction == "LONG":
                score -= 10  # fear regime — longs go against flow (strong penalty)
            elif regime == "neutral":
                pass  # no adjustment

        # Cap at 100
        score = min(100, score)

        # Early kill: score can't possibly reach C+ minimum (58) even with bonuses still to apply
        # At this point: wyckoff, PA, BTC corr, dom bonuses haven't been added yet
        # Max remaining bonus ≈ 38 pts, so if score < 20 it can never reach 58
        if score < 20:
            logger.debug(f"{symbol} REJECTED early: score={score:.1f} — cannot reach C+ minimum")
            return None

        # ─── Grade — Professional Algo Trader Criteria ──────────────────────────
        #
        # Think like a top quant fund: every trade must justify its risk.
        # C+ is NOT a "weak signal" — it's a REAL trade, just with fewer confirmations.
        # The goal: reject garbage, only send setups a real trader would take.
        #
        # ─────────────────────────────────────────────────────────────────────
        # A+ │ "Sniper" — maximum confluence, take it with full size
        #    │ Score ≥ 78  │  All 6 criteria  │  Vol ≥ 2x  │  RSI healthy zone
        #    │ Clean trend + perfect location + structure + no conflicts
        #    │ Expected win rate: 65%+
        #
        # B+ │ "Quality" — solid setup, standard position size
        #    │ Score ≥ 65  │  4+ of 6 criteria  │  Vol ≥ 1.5x  │  Trend required
        #    │ May lack one confirmation — manage actively
        #    │ Expected win rate: 55%+
        #
        # C+ │ "Speculative" — directional bias, reduced size, last resort only
        #    │ Score ≥ 58  │  3+ of 6 criteria  │  Vol ≥ 1.2x  │  Trend OR momentum
        #    │ Only sent when no A+ or B+ found all day
        #    │ Expected win rate: 45%+  (above 50/50 but marginal)
        # ─────────────────────────────────────────────────────────────────────

        # ── 6 Hard Criteria (each is binary — met or not) ────────────────────
        # 1. TREND: EMA stack fully aligned
        trend_clear = (ema9 > ema21 > ema50) or (ema9 < ema21 < ema50)

        # 2. MOMENTUM: RSI in healthy zone (not chasing extremes)
        rsi_healthy = (
            (direction == "LONG"  and 35 <= rsi14 <= 68) or
            (direction == "SHORT" and 32 <= rsi14 <= 65)
        )

        # 3. VOLUME: confirmed by above-average volume (time-of-day adjusted)
        vol_strong    = vol_ratio >= _VOL_STRONG    # A+ requirement
        vol_confirmed = vol_ratio >= _VOL_CONFIRMED # B+ requirement
        vol_minimum   = vol_ratio >= _VOL_MINIMUM   # C+ minimum

        # 4. STRUCTURE: Wyckoff phase / divergence / price action signal
        has_structure = (
            wyckoff_phase != "none" or
            len(pa_signals) > 0 or
            div_rsi  != "none" or
            div_macd != "none"
        )

        # 5. LOCATION: price at a key level (not mid-range noise)
        at_key_level = (
            abs(price - vwap) / (price + 1e-9) < 0.012 or   # near VWAP
            (pivots.get("s1", 0) > 0 and direction == "LONG") or   # near support
            (pivots.get("r1", 0) > 0 and direction == "SHORT") or  # near resistance
            abs(ob_imbalance) > 0.15 or                            # order block
            price <= bb_lower * 1.005 and direction == "LONG" or   # at BB lower
            price >= bb_upper * 0.995 and direction == "SHORT"     # at BB upper
        )

        # 6. ORDER FLOW: market participants confirm direction
        flow_confirmed = (
            (direction == "LONG"  and ob_imbalance > 0.10 and taker_ls_ratio > 1.0) or
            (direction == "SHORT" and ob_imbalance < -0.10 and taker_ls_ratio < 1.0)
        )

        # ── Conflict check: hard opposing signals that invalidate the trade ────
        has_conflict = (
            (direction == "LONG"  and rsi14 > 72) or       # chasing overbought
            (direction == "SHORT" and rsi14 < 28) or       # chasing oversold
            (direction == "LONG"  and ema9 < ema21 < ema50) or  # fully bearish EMA stack on long
            (direction == "SHORT" and ema9 > ema21 > ema50) or  # fully bullish EMA stack on short
            conflicting > 0                                 # opposing candle patterns
        )

        # ── Estimated RR check ────────────────────────────────────────────────
        # Use ATR to estimate RR before full SL calc (fast pre-filter)
        # SL estimate = 1.2x ATR away. TP1 = 1.5x risk minimum.
        est_risk = atr * 1.2
        rr_ok_aplus = atr * 2.0 / (est_risk + 1e-9) >= 1.5  # effectively always true if ATR sane
        # Real RR filter: reject if coin has essentially zero volatility
        if atr / (price + 1e-9) < 0.002:
            logger.debug(f"{symbol} REJECTED: ATR too small ({atr/price:.4%}) — no room for RR")
            return None

        # ── BTC Correlation criterion ─────────────────────────────────────────
        # This is mandatory context — every trade must be aligned with or
        # independent from BTC direction. Trading AGAINST BTC correlation = low edge.
        #
        # btc_corr_aligned = True when:
        #   LONG:  BTC rising (corr > 0.5)  OR coin decorrelated (corr < 0.3)  OR BTC flat
        #   SHORT: BTC falling (corr > 0.5) OR coin decorrelated (corr < 0.3) OR BTC flat
        #   LONG inverse corr coin: BTC falling = coin rising (negative corr + BTC down = ok)
        #
        # btc_corr_aligned = False (HARD BLOCK for A+/B+) when:
        #   LONG  + corr > 0.6 + BTC falling (high-corr coin long into BTC downtrend)
        #   SHORT + corr > 0.6 + BTC rising  (high-corr coin short into BTC uptrend)

        btc_corr_aligned = True  # default: assume aligned unless proven otherwise
        btc_corr_kills   = False  # True = hard kill regardless of grade

        if btc_corr is not None:
            if direction == "LONG":
                if btc_corr > 0.60 and btc_change < -0.8:
                    btc_corr_aligned = False   # high-corr long into BTC dump
                    if btc_corr > 0.75 and btc_change < -1.5:
                        btc_corr_kills = True  # extreme: BTC crashing, don't long corr coin
                elif btc_corr > 0.60 and btc_change < -0.3:
                    btc_corr_aligned = False   # mild headwind — criterion fails but not killer
                elif btc_corr < 0.30:
                    btc_corr_aligned = True    # decorrelated — runs independently
                elif btc_change >= -0.3:
                    btc_corr_aligned = True    # BTC flat or rising — ok for long
            else:  # SHORT
                if btc_corr > 0.60 and btc_change > 0.8:
                    btc_corr_aligned = False   # high-corr short into BTC pump
                    if btc_corr > 0.75 and btc_change > 1.5:
                        btc_corr_kills = True  # extreme: BTC pumping, don't short corr coin
                elif btc_corr > 0.60 and btc_change > 0.3:
                    btc_corr_aligned = False   # mild headwind
                elif btc_corr < 0.30:
                    btc_corr_aligned = True    # decorrelated — can fall independently
                elif btc_change <= 0.3:
                    btc_corr_aligned = True    # BTC flat or falling — ok for short

        # Hard kill: BTC is moving powerfully against this trade
        if btc_corr_kills:
            logger.debug(
                f"{symbol} KILLED by BTC corr: {direction} corr={btc_corr:.2f} "
                f"BTC={btc_change:+.1f}% — trading directly into BTC flow"
            )
            return None

        # ── Count criteria met (7 total now — BTC corr is #7) ─────────────────
        criteria = {
            "trend":     trend_clear,
            "momentum":  rsi_healthy,
            "volume":    vol_confirmed,
            "structure": has_structure,
            "location":  at_key_level,
            "flow":      flow_confirmed,
            "btc_corr":  btc_corr_aligned,   # ← NEW: mandatory macro alignment
        }
        n_criteria = sum(criteria.values())

        # ── Grade assignment ──────────────────────────────────────────────────
        # Use config thresholds so AI auto-tune takes effect live
        a_thresh = config.GRADE_A_PLUS   # default 80
        b_thresh = config.GRADE_B_PLUS   # default 60
        c_thresh = config.GRADE_C_PLUS   # default 40

        if (
            n_criteria >= 7 and
            score >= a_thresh and
            rsi_healthy and
            btc_corr_aligned and
            not has_conflict
        ):
            grade = "A+"

        elif (
            n_criteria >= 6 and
            score >= a_thresh and
            trend_clear and
            rsi_healthy and
            btc_corr_aligned and
            vol_strong and
            not has_conflict
        ):
            grade = "A+"

        elif (
            n_criteria >= 5 and
            score >= b_thresh and
            trend_clear and
            btc_corr_aligned and
            vol_confirmed and
            not has_conflict
        ):
            grade = "B+"

        elif (
            n_criteria >= 6 and
            score >= b_thresh and
            trend_clear and
            btc_corr_aligned and
            not has_conflict
        ):
            grade = "B+"

        elif (
            n_criteria >= 3 and
            score >= c_thresh and
            (trend_clear or rsi_healthy) and
            vol_minimum and
            not has_conflict and
            not (btc_corr is not None and btc_corr > 0.60 and (
                (direction == "LONG"  and btc_change < -1.0) or
                (direction == "SHORT" and btc_change >  1.0)
            ))
        ):
            grade = "C+"

        else:
            reasons = []
            if n_criteria < 3:          reasons.append(f"criteria={n_criteria}/7 (need 3+)")
            if score < c_thresh:        reasons.append(f"score={score:.0f} (need {c_thresh}+)")
            if not (trend_clear or rsi_healthy): reasons.append("no trend/momentum")
            if not vol_minimum:         reasons.append(f"vol={vol_ratio:.2f}x (need {_VOL_MINIMUM}x+)")
            if has_conflict:            reasons.append("conflicting signals")
            if not btc_corr_aligned:    reasons.append(f"btc_corr misaligned corr={btc_corr:.2f}")
            logger.info(f"{symbol} REJECTED [{direction} score={score:.0f} n={n_criteria}]: {', '.join(reasons)}")
            return None

        # ─── Entry / SL — Structure-Based ────────────────────────────────────────
        # SL is placed at logical market structure: swing high/low + ATR buffer
        # NOT a fixed percentage — respects actual price action

        entry   = price
        atr_val = indicators.get("atr14", price * 0.01)
        # pivots already assigned in Wyckoff block above

        # ATR multipliers per trade type (how much buffer beyond structure)
        atr_buffer = {"scalp": 0.5, "day": 1.0, "swing": 1.5}[trade_type]

        if direction == "LONG":
            # SL candidates (pick the most logical one below entry):
            candidates = []

            # 1. Recent swing low (S1 pivot)
            s1 = pivots.get("s1", 0)
            if 0 < s1 < entry:
                candidates.append(s1 - atr_val * 0.3)  # just below S1

            # 2. BB lower band (dynamic support)
            bb_low = indicators.get("bb_lower", 0)
            if 0 < bb_low < entry:
                candidates.append(bb_low - atr_val * 0.2)

            # 3. ATR-based structural SL (always available fallback)
            candidates.append(entry - atr_val * (atr_buffer + 1.0))

            # 4. VWAP as dynamic support (for scalp/day)
            if trade_type in ["scalp", "day"]:
                vwap_val = indicators.get("vwap", 0)
                if 0 < vwap_val < entry * 0.998:
                    candidates.append(vwap_val - atr_val * 0.3)

            # Choose the candidate closest below entry (tightest logical SL)
            valid = [c for c in candidates if 0 < c < entry]
            if valid:
                sl_raw = max(valid)   # highest = closest to entry = tightest logical SL
            else:
                sl_raw = entry - atr_val * (atr_buffer + 1.0)

            # Hard cap: SL can't be more than 15% away (protects from insane swings)
            max_sl_dist = entry * 0.15
            sl = round(max(sl_raw, entry - max_sl_dist), 6)

        else:  # SHORT
            candidates = []

            # 1. Recent swing high (R1 pivot)
            r1 = pivots.get("r1", 0)
            if r1 > entry:
                candidates.append(r1 + atr_val * 0.3)

            # 2. BB upper band (dynamic resistance)
            bb_high = indicators.get("bb_upper", 0)
            if bb_high > entry:
                candidates.append(bb_high + atr_val * 0.2)

            # 3. ATR-based structural SL
            candidates.append(entry + atr_val * (atr_buffer + 1.0))

            # 4. VWAP as resistance
            if trade_type in ["scalp", "day"]:
                vwap_val = indicators.get("vwap", 0)
                if vwap_val > entry * 1.002:
                    candidates.append(vwap_val + atr_val * 0.3)

            valid = [c for c in candidates if c > entry]
            if valid:
                sl_raw = min(valid)   # lowest = closest to entry = tightest logical SL
            else:
                sl_raw = entry + atr_val * (atr_buffer + 1.0)

            max_sl_dist = entry * 0.15
            sl = round(min(sl_raw, entry + max_sl_dist), 6)

        # Grade-based TP ratios — A+ gets maximum extended targets
        rr_ratios = config.TP_RR_RATIOS[grade][trade_type]

        risk = abs(entry - sl)
        tps = []
        for rr in rr_ratios:
            if direction == "LONG":
                tp = round(entry + risk * rr, 6)
            else:
                tp = round(entry - risk * rr, 6)
            tps.append(tp)

        # ─── Confluences List ────────────────────────────────────────────────
        confluences = []
        if direction == "LONG":
            if ema9 > ema21 > ema50: confluences.append("✅ EMA 9>21>50 bullish stack")
            if price > vwap:          confluences.append("✅ Price above VWAP")
            if macd_hist > macd_hist_prev and macd_hist > 0: confluences.append("✅ MACD histogram rising")
            if rsi14 < 35:            confluences.append("✅ RSI oversold bounce")
            if div_rsi == "bullish":  confluences.append("✅ RSI bullish divergence")
            if div_macd == "bullish": confluences.append("✅ MACD bullish divergence")
            if ob_imbalance > 0.15:   confluences.append(f"✅ Bid pressure {ob_imbalance:.1%}")
            if vol_ratio > 2:         confluences.append(f"✅ Volume spike {vol_ratio:.1f}x avg")
            if outperform > 1.5:      confluences.append(f"✅ Outperforming BTC by {outperform:.1f}%")
            if funding_rate < -0.001: confluences.append("✅ Negative funding (shorts squeezable)")
        else:
            if ema9 < ema21 < ema50: confluences.append("✅ EMA 9<21<50 bearish stack")
            if price < vwap:          confluences.append("✅ Price below VWAP")
            if macd_hist < macd_hist_prev and macd_hist < 0: confluences.append("✅ MACD histogram falling")
            if rsi14 > 65:            confluences.append("✅ RSI overbought rejection")
            if div_rsi == "bearish":  confluences.append("✅ RSI bearish divergence")
            if div_macd == "bearish": confluences.append("✅ MACD bearish divergence")
            if ob_imbalance < -0.15:  confluences.append(f"✅ Ask pressure {abs(ob_imbalance):.1%}")
            if vol_ratio > 2:         confluences.append(f"✅ Volume spike {vol_ratio:.1f}x avg")
            if funding_rate > 0.001:  confluences.append("✅ Positive funding (longs squeezable)")

        # Wyckoff phase confluence
        wyckoff_labels = {
            "accumulation_spring": "✅ Wyckoff: Accumulation Spring (demand zone)",
            "markup_SOS":          "✅ Wyckoff: Markup — Sign of Strength",
            "distribution_UTAD":   "✅ Wyckoff: Distribution UTAD (supply zone)",
            "markdown_SOW":        "✅ Wyckoff: Markdown — Sign of Weakness",
        }
        if wyckoff_phase in wyckoff_labels:
            confluences.append(wyckoff_labels[wyckoff_phase])

        # Price action signals
        for pa in pa_signals:
            confluences.append(f"✅ PA: {pa}")

        # Only add patterns that AGREE with direction
        bullish_patterns = ["Hammer", "Bullish Engulfing", "Doji"]
        bearish_patterns = ["Bearish Engulfing", "Doji"]
        for p in patterns:
            if direction == "LONG" and any(bp in p for bp in bullish_patterns):
                confluences.append(f"✅ Pattern: {p}")
            elif direction == "SHORT" and any(bp in p for bp in bearish_patterns):
                confluences.append(f"✅ Pattern: {p}")
            elif direction == "SHORT" and any(bp in p for bp in bullish_patterns):
                confluences.append(f"⚠️ Counter-pattern: {p}")  # warn, don't boost

        # ─── Leverage ──────────────────────────────────────────────────────────
        leverage = config.LEVERAGE[grade][trade_type]

        return {
            "symbol":       symbol,
            "trade_type":   trade_type,
            "direction":    direction,
            "entry":        round(entry, 6),
            "sl":           round(sl, 6),
            "tps":          tps,
            "grade":        grade,
            "score":        round(score, 1),
            "confluences":  confluences,
            "leverage":     leverage,
            "funding_rate": funding_rate,
            "vol_ratio":    round(vol_ratio, 2),
            "rsi14":        round(rsi14, 1),
            "outperform":   round(outperform, 2),
            "patterns":     patterns,
            "liq_zones":    liq_zones,
            "wyckoff_phase":   wyckoff_phase,
            "pa_signals":      pa_signals,
            "btc_corr":        round(btc_corr, 3),
            "grade_criteria": {
                "trend_clear":   trend_clear,
                "vol_confirmed": vol_confirmed,
                "has_structure": has_structure,
                "good_location": at_key_level,
                "strong_conf":   flow_confirmed,
                "criteria_met":  n_criteria,
            },
        }
