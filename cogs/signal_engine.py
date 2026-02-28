"""
Signal Engine Cog
Main trading signal scanner and broadcaster
"""
import asyncio
import logging
import time
from typing import Optional

import discord
from discord.ext import commands, tasks

import config
from utils.data_fetcher import DataFetcher
from utils.indicators import calculate_all_indicators
from utils.signal_scorer import SignalScorer
from cogs.ml_engine import MLEngine

logger = logging.getLogger("SignalEngine")


class SignalEngine(commands.Cog):
    def __init__(self, bot):
        self.bot     = bot
        self.fetcher = DataFetcher(config.CMC_API_KEY, config.BINANCE_API_KEY, config.BINANCE_SECRET)
        self.scorer  = SignalScorer()
        self.ml      = MLEngine(config.ML_MODEL_PATH, config.TRADE_HISTORY_PATH)

        # Dominance cache: (btc_d, usdt_d, timestamp)
        self._dominance_cache = (0.0, 0.0, 0.0)

        # BTC correlation cache: {symbol: (corr, timestamp)} refreshed every 30min per symbol
        self._btc_corr_cache: dict = {}  # {symbol: (float, float)} = (corr, ts)

        # BTC price history for live dominance tracking (last 60 values, 1min each)
        self._btc_price_history: list = []

        # Live dominance trend: tracks direction of BTC.D and USDT.D change
        # (btc_d_prev, usdt_d_prev, btc_d_curr, usdt_d_curr)
        self._dom_trend = {"btc_d_prev": 0.0, "usdt_d_prev": 0.0, "btc_d_curr": 0.0, "usdt_d_curr": 0.0}

        # ── Candle-by-candle dominance state for scalp filtering ─────────────
        # Stores last N candles of USDT.D and BTC.D (1-min resolution from CoinGecko)
        # Shape: [{"t": timestamp, "btc_d": float, "usdt_d": float}, ...]
        self._dom_candles: list = []          # rolling 60-candle history
        self._dom_candle_ts: float = 0.0      # last candle fetch timestamp

        # Derived signals updated every candle:
        self._dom_signal = {
            "usdt_trend":    "flat",   # "rising" | "falling" | "flat"
            "btc_trend":     "flat",   # "rising" | "falling" | "flat"
            "usdt_velocity": 0.0,      # rate of change per candle (last 3)
            "btc_velocity":  0.0,
            "usdt_accel":    0.0,      # acceleration (is it speeding up?)
            "btc_accel":     0.0,
            "scalp_bias":    "neutral",# "long_ok" | "short_ok" | "neutral" | "blocked"
            "scalp_reason":  "",       # human-readable reason
            "candle_count":  0,        # how many candles fetched
        }

        # Active trades: symbol -> signal dict (with message_id, channel_id, tps_hit)
        self.active_trades: dict[str, dict] = {}

        # Last scan timestamps per type
        self._last_scalp = 0
        self._last_day   = 0
        self._last_swing  = 0
        self._last_ml_retrain = 0

        # Cache
        self._valid_symbols: list[str] = []
        self._btc_change_cache = (0.0, 0)  # (value, timestamp)

        # ── Daily quota tracking ─────────────────────────────────────────────
        # Tracks how many signals sent per trade_type per day + per hour for scalps
        #
        # SCALP:  target 24/day (1/hour). A+ = always send (no daily cap).
        #         If no A+ found this hour → allow B+. If no B+ all day → allow C+.
        # DAY:    target 8-10/day. A+ = always send. B+ if no A+. C+ if no B+ all day.
        # SWING:  target 3-4/day.  A+ = always send. B+ if no A+. C+ if no B+ all day.
        #
        # Reset at midnight UTC.
        self._quota = {
            "scalp": {
                "day_start":    0.0,       # timestamp of last daily reset
                "sent_today":   {"A+": 0, "B+": 0, "C+": 0},
                "last_hour_sent": 0.0,     # timestamp of last scalp signal sent
                "hour_has_aplus": False,   # did we find an A+ this hour?
                "day_has_bplus":  False,   # did we find a B+ today?
                "daily_target":  24,
            },
            "day": {
                "day_start":    0.0,
                "sent_today":   {"A+": 0, "B+": 0, "C+": 0},
                "last_hour_sent": 0.0,
                "hour_has_aplus": False,
                "day_has_bplus":  False,
                "daily_target":  9,        # target 8-10, use 9 as midpoint
            },
            "swing": {
                "day_start":    0.0,
                "sent_today":   {"A+": 0, "B+": 0, "C+": 0},
                "last_hour_sent": 0.0,
                "hour_has_aplus": False,
                "day_has_bplus":  False,
                "daily_target":  3,        # target 3-4
            },
        }

    async def cog_load(self):
        logger.info("SignalEngine starting...")
        self.ml.set_bot(self.bot)  # Give ML engine access to Claude via AIEngine
        self.scan_loop.start()
        self.monitor_trades_loop.start()
        self.ml_retrain_loop.start()
        self.dominance_tracker_loop.start()
        self.daily_ml_report_loop.start()
        self.dom_candle_loop.start()

    async def cog_unload(self):
        self.scan_loop.cancel()
        self.monitor_trades_loop.cancel()
        self.ml_retrain_loop.cancel()
        self.dominance_tracker_loop.cancel()
        self.daily_ml_report_loop.cancel()
        self.dom_candle_loop.cancel()
        await self.fetcher.close()

    # ─── Loops ───────────────────────────────────────────────────────────────

    @tasks.loop(seconds=30)
    async def scan_loop(self):
        """Main scanning loop - runs every 30s, triggers type-specific scans"""
        now = time.time()
        try:
            if now - self._last_scalp >= config.SCAN_INTERVAL_SCALP:
                self._last_scalp = now
                await self._run_scan("scalp", "5m", 100)

            if now - self._last_day >= config.SCAN_INTERVAL_DAY:
                self._last_day = now
                await self._run_scan("day", "1h", 100)

            if now - self._last_swing >= config.SCAN_INTERVAL_SWING:
                self._last_swing = now
                await self._run_scan("swing", "4h", 150)
        except Exception as e:
            logger.error(f"Scan loop error: {e}", exc_info=True)

    @scan_loop.before_loop
    async def before_scan_loop(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(5)
        # Prime symbol cache
        top_coins = await self.fetcher.get_top_coins(config.TOP_COINS_LIMIT)
        self._valid_symbols = await self.fetcher.get_valid_futures_symbols(top_coins)
        logger.info(f"Valid futures symbols loaded: {len(self._valid_symbols)}")

    @tasks.loop(seconds=60)
    async def monitor_trades_loop(self):
        """Monitor active trades for TP/SL hits"""
        if not self.active_trades:
            return
        try:
            for symbol, trade in list(self.active_trades.items()):
                ticker = await self.fetcher.get_ticker_24h(symbol)
                if not ticker:
                    continue
                current_price = float(ticker.get("lastPrice", 0))
                if not current_price:
                    continue

                direction = trade["direction"]
                sl = trade["sl"]
                tps = trade["tps"]
                tps_hit = trade.get("tps_hit", 0)

                # Check SL
                sl_hit = (direction == "LONG" and current_price <= sl) or \
                         (direction == "SHORT" and current_price >= sl)

                if sl_hit:
                    await self._handle_sl_hit(symbol, trade, current_price)
                    continue

                # Check TPs in order
                for i in range(tps_hit, len(tps)):
                    tp = tps[i]
                    tp_hit = (direction == "LONG" and current_price >= tp) or \
                             (direction == "SHORT" and current_price <= tp)
                    if tp_hit:
                        await self._handle_tp_hit(symbol, trade, i + 1, tp)
                        trade["tps_hit"] = i + 1
                        if i + 1 == len(tps):
                            # All TPs hit
                            self.ml.record_trade(
                                trade, trade.get("indicators", {}),
                                trade.get("funding_rate", 0),
                                trade.get("ob_imbalance", 0),
                                "win"
                            )
                            del self.active_trades[symbol]
                        break
        except Exception as e:
            logger.error(f"Monitor loop error: {e}", exc_info=True)

    @tasks.loop(minutes=1)
    async def dom_candle_loop(self):
        """
        Fetch dominance every 1 minute — builds a live candle feed for
        USDT.D and BTC.D used by scalp filtering.

        Uses CoinGecko /global (free, no key) as primary source.
        Falls back to CMC if CoinGecko fails.
        """
        try:
            btc_d, usdt_d = await self._fetch_dominance_spot()
            if btc_d == 0 or usdt_d == 0:
                return

            now = time.time()
            candle = {"t": now, "btc_d": btc_d, "usdt_d": usdt_d}

            self._dom_candles.append(candle)
            # Keep last 60 candles (60 minutes)
            if len(self._dom_candles) > 60:
                self._dom_candles = self._dom_candles[-60:]

            self._dom_candle_ts = now
            self._update_dom_signal()

            sig = self._dom_signal
            logger.debug(
                f"DOM candle — BTC.D={btc_d:.3f}%  USDT.D={usdt_d:.3f}%  "
                f"usdt_trend={sig['usdt_trend']}({sig['usdt_velocity']:+.4f})  "
                f"scalp_bias={sig['scalp_bias']}"
            )

            # Also refresh the slow dominance cache used by day/swing
            self._dominance_cache = (btc_d, usdt_d, now)

        except Exception as e:
            logger.error(f"DOM candle loop error: {e}", exc_info=True)

    @dom_candle_loop.before_loop
    async def before_dom_candle_loop(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(10)
        # Pre-fill candle history with 20 rapid samples (1s apart)
        # so scalp filtering works immediately instead of waiting 20 minutes
        logger.info("DOM candle: pre-filling history with 20 startup samples...")
        for i in range(20):
            try:
                btc_d, usdt_d = await self._fetch_dominance_spot()
                if btc_d > 0 and usdt_d > 0:
                    self._dom_candles.append({
                        "t":      time.time() - (19 - i),  # spread slightly in time
                        "btc_d":  btc_d,
                        "usdt_d": usdt_d,
                    })
            except Exception:
                pass
            await asyncio.sleep(1)
        if self._dom_candles:
            self._update_dom_signal()
            logger.info(f"DOM candle: pre-filled {len(self._dom_candles)} candles — "
                        f"bias={self._dom_signal['scalp_bias']}")

    async def _fetch_dominance_spot(self) -> tuple:
        """
        Fetch current BTC.D and USDT.D as a single lightweight call.
        Returns (btc_d, usdt_d) floats. Returns (0, 0) on failure.

        Source 1: CoinGecko /global  — free, no key, exact values
        Source 2: CMC global-metrics — fallback
        """
        import aiohttp

        # ── CoinGecko (primary) ─────────────────────────────────────────────
        # TradingView CRYPTOCAP:USDT.D = all stablecoins / total crypto mcap
        # CoinGecko pct["usdt"] = Tether only (reliable, ~7.9%)
        # CoinGecko pct["usdc"] is UNRELIABLE — returns 1.5-3.5% (data error)
        # Fix: use USDT only + fixed 0.35% offset for USDC/DAI/other small stables
        # Result: 7.944 + 0.35 = 8.294% ≈ TradingView 8.291% ✅
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    "https://api.coingecko.com/api/v3/global",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        pct    = (await resp.json()).get("data", {}).get("market_cap_percentage", {})
                        btc_d  = float(pct.get("btc",  0) or 0)
                        usdt   = float(pct.get("usdt", 0) or 0)
                        # +0.35% accounts for USDC + DAI + other minor stablecoins
                        # This offset is stable and matches TradingView within 0.05%
                        usdt_d = usdt + 0.35
                        logger.info(f"CoinGecko: usdt={usdt:.3f}% + 0.35 offset = USDT.D={usdt_d:.3f}%")
                        if btc_d > 0 and usdt_d > 0:
                            return btc_d, usdt_d
        except Exception as e:
            logger.debug(f"CoinGecko spot fetch failed: {e}")

        # ── CMC fallback ──────────────────────────────────────────────────────
        try:
            import aiohttp as _aiohttp
            url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": config.CMC_API_KEY, "Accept": "application/json"}
            async with _aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=_aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        d = (await resp.json()).get("data", {})
                        btc_d = float(d.get("btc_dominance", 0) or 0)
                        for field in ["stablecoin_market_cap_dominance", "stablecoin_volume_dominance",
                                      "usdt_dominance", "usdt_market_cap_dominance"]:
                            v = d.get(field)
                            if v and float(v) > 0:
                                return btc_d, float(v)
        except Exception as e:
            logger.debug(f"CMC spot fetch failed: {e}")

        return 0.0, 0.0

    def _update_dom_signal(self):
        """
        Analyse the last N candles to produce a scalp bias signal.

        Logic:
          USDT.D rising  = money fleeing to stable = BEARISH for crypto
          USDT.D falling = money entering crypto   = BULLISH
          BTC.D  rising  = BTC taking share from alts = BAD for alt longs
          BTC.D  falling = money moving to alts   = GOOD for alt longs

        Scalp bias combinations:
          USDT falling + BTC.D flat/falling  → long_ok  (best entry condition)
          USDT falling + BTC.D rising        → btc_long_only (alts weak)
          USDT rising  (any speed)           → short_ok (fear entering)
          USDT rising fast (accel > 0.002)   → blocked  (panic, no scalps)
          Flat / mixed                       → neutral
        """
        candles = self._dom_candles
        sig = self._dom_signal
        sig["candle_count"] = len(candles)

        if len(candles) < 3:
            sig["scalp_bias"]   = "neutral"
            sig["scalp_reason"] = "Insufficient candle data"
            return

        # Last 3 candles for velocity
        recent = candles[-3:]
        usdt_vals = [c["usdt_d"] for c in recent]
        btc_vals  = [c["btc_d"]  for c in recent]

        usdt_vel = (usdt_vals[-1] - usdt_vals[0]) / max(len(recent)-1, 1)
        btc_vel  = (btc_vals[-1]  - btc_vals[0])  / max(len(recent)-1, 1)

        # Last 5 candles for acceleration (rate of rate of change)
        if len(candles) >= 5:
            older = candles[-5:-2]
            usdt_old_vel = (older[-1]["usdt_d"] - older[0]["usdt_d"]) / max(len(older)-1, 1)
            btc_old_vel  = (older[-1]["btc_d"]  - older[0]["btc_d"])  / max(len(older)-1, 1)
            usdt_accel = usdt_vel - usdt_old_vel
            btc_accel  = btc_vel  - btc_old_vel
        else:
            usdt_accel = 0.0
            btc_accel  = 0.0

        # Trend labels (threshold: 0.003% per candle = meaningful move on 1min)
        THRESH = 0.003
        usdt_trend = "rising"  if usdt_vel >  THRESH else ("falling" if usdt_vel < -THRESH else "flat")
        btc_trend  = "rising"  if btc_vel  >  THRESH else ("falling" if btc_vel  < -THRESH else "flat")

        sig["usdt_trend"]    = usdt_trend
        sig["btc_trend"]     = btc_trend
        sig["usdt_velocity"] = round(usdt_vel,  5)
        sig["btc_velocity"]  = round(btc_vel,   5)
        sig["usdt_accel"]    = round(usdt_accel, 5)
        sig["btc_accel"]     = round(btc_accel,  5)

        # Current USDT.D level
        current_usdt = candles[-1]["usdt_d"]
        current_btc  = candles[-1]["btc_d"]

        # ── Scalp bias decision tree ──────────────────────────────────────────
        # Panic mode: USDT.D rising fast AND accelerating = no scalps at all
        if usdt_trend == "rising" and usdt_accel > 0.002 and current_usdt > 7.5:
            sig["scalp_bias"]   = "blocked"
            sig["scalp_reason"] = (
                f"⛔ PANIC — USDT.D accelerating +{usdt_vel*100:.3f}%/min "
                f"(accel={usdt_accel:+.4f}) @ {current_usdt:.3f}%"
            )

        # Risk-off: USDT.D rising steadily = shorts only
        elif usdt_trend == "rising" and current_usdt > 7.5:
            sig["scalp_bias"]   = "short_ok"
            sig["scalp_reason"] = (
                f"🔴 USDT.D rising {usdt_vel:+.4f}%/min → fear building "
                f"@ {current_usdt:.3f}% — SHORT bias"
            )

        # Fear easing: USDT.D was high but now falling = shorts still viable, longs cautious
        elif usdt_trend == "falling" and current_usdt > 7.5:
            sig["scalp_bias"]   = "short_ok"
            sig["scalp_reason"] = (
                f"🟠 USDT.D falling from high ({current_usdt:.3f}%) — still elevated, cautious"
            )

        # Best long condition: USDT.D falling + BTC.D flat or falling
        elif usdt_trend == "falling" and btc_trend in ("falling", "flat"):
            sig["scalp_bias"]   = "long_ok"
            sig["scalp_reason"] = (
                f"🟢 USDT.D falling {usdt_vel:+.4f}%/min + BTC.D {btc_trend} "
                f"@ {current_usdt:.3f}% — LONG bias"
            )

        # Alt-weak: USDT.D falling but BTC.D rising = BTC longs ok, alt longs risky
        elif usdt_trend == "falling" and btc_trend == "rising":
            sig["scalp_bias"]   = "btc_long_only"
            sig["scalp_reason"] = (
                f"🟡 USDT.D falling but BTC.D rising {btc_vel:+.4f}%/min "
                f"— BTC/ETH longs ok, alts risky"
            )

        # BTC.D spiking while USDT flat = rotation from alts to BTC
        elif btc_trend == "rising" and btc_accel > 0.002:
            sig["scalp_bias"]   = "short_ok"
            sig["scalp_reason"] = (
                f"🟡 BTC.D surging +{btc_vel:.4f}%/min (accel={btc_accel:+.4f}) "
                f"— alts losing value, SHORT alts"
            )

        else:
            sig["scalp_bias"]   = "neutral"
            sig["scalp_reason"] = (
                f"⚪ No clear candle signal — USDT.D {usdt_trend} "
                f"@ {current_usdt:.3f}%, BTC.D {btc_trend} @ {current_btc:.3f}%"
            )

    @tasks.loop(minutes=5)
    async def dominance_tracker_loop(self):
        """
        Live dominance tracking — runs every 5 min.
        Tracks BTC.D and USDT.D trend direction (rising/falling).
        Broadcasts alert to log channels when regime shifts.
        """
        try:
            now = time.time()
            # Force fresh fetch (bypass 10min cache)
            self._dominance_cache = (0.0, 0.0, 0.0)
            dom = await self._get_dominance()
            if dom["btc_d"] == 0:
                return

            prev = self._dom_trend
            btc_d_curr  = dom["btc_d"]
            usdt_d_curr = dom["usdt_d"]
            btc_d_prev  = prev.get("btc_d_curr", btc_d_curr)
            usdt_d_prev = prev.get("usdt_d_curr", usdt_d_curr)

            btc_d_delta  = btc_d_curr  - btc_d_prev
            usdt_d_delta = usdt_d_curr - usdt_d_prev

            self._dom_trend = {
                "btc_d_prev":  btc_d_prev,
                "usdt_d_prev": usdt_d_prev,
                "btc_d_curr":  btc_d_curr,
                "usdt_d_curr": usdt_d_curr,
                "btc_d_delta":  round(btc_d_delta, 3),
                "usdt_d_delta": round(usdt_d_delta, 3),
            }

            # Detect regime shift (significant move)
            regime_shift = abs(btc_d_delta) > 0.5 or abs(usdt_d_delta) > 0.3

            btc_arrow  = "📈" if btc_d_delta > 0.1 else ("📉" if btc_d_delta < -0.1 else "➡️")
            usdt_arrow = "📈" if usdt_d_delta > 0.1 else ("📉" if usdt_d_delta < -0.1 else "➡️")

            logger.info(
                f"Dominance tracker: BTC.D={btc_d_curr:.2f}% ({btc_d_delta:+.3f}) "
                f"USDT.D={usdt_d_curr:.2f}% ({usdt_d_delta:+.3f}) "
                f"Regime={dom['regime']}"
            )

            # Broadcast shift alert to all log channels
            if regime_shift:
                await self._broadcast_dominance_alert(dom, btc_d_delta, usdt_d_delta, btc_arrow, usdt_arrow)

        except Exception as e:
            logger.error(f"Dominance tracker error: {e}", exc_info=True)

    @dominance_tracker_loop.before_loop
    async def before_dominance_tracker(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(15)

    async def _broadcast_dominance_alert(self, dom, btc_d_delta, usdt_d_delta, btc_arrow, usdt_arrow):
        """Send dominance shift alert to all log channels"""
        from db import get_all_guilds
        regime_colors = {
            "risk_on_alt": 0x00C853,
            "risk_on_btc": 0xFFD600,
            "risk_off":    0xFF1744,
            "neutral":     0x607D8B,
        }
        embed = discord.Embed(
            title="🌍 Market Dominance Shift Detected",
            description=dom["bias"],
            color=regime_colors.get(dom["regime"], 0x607D8B)
        )
        embed.add_field(
            name=f"{btc_arrow} BTC Dominance",
            value=f"`{dom['btc_d']:.2f}%`  ({btc_d_delta:+.3f}%)",
            inline=True
        )
        embed.add_field(
            name=f"{usdt_arrow} USDT Dominance",
            value=f"`{dom['usdt_d']:.2f}%`  ({usdt_d_delta:+.3f}%)",
            inline=True
        )
        regime_advice = {
            "risk_on_alt": "💡 Altseason: Focus on alt LONG setups. Avoid shorts.",
            "risk_on_btc": "💡 BTC season: Prefer BTC/ETH longs. Alt longs risky.",
            "risk_off":    "⚠️ Fear regime: Only SHORT setups. Avoid longs.",
            "neutral":     "💡 Mixed signals: Trade both directions with caution.",
        }
        embed.add_field(
            name="🎯 Trade Bias",
            value=regime_advice.get(dom["regime"], "No specific bias"),
            inline=False
        )
        embed.set_footer(text="Live dominance tracking • Updates every 5 min")

        guilds = get_all_guilds()
        for g in guilds:
            g_dict = dict(g)
            ch_id  = g_dict.get("log_channel_id")
            if not ch_id:
                continue
            ch = self.bot.get_channel(ch_id)
            if not ch:
                continue
            try:
                await ch.send(embed=embed)
            except Exception as e:
                # 403 = bot lacks permissions in that server — skip silently
                if "403" in str(e) or "50013" in str(e) or "50001" in str(e):
                    logger.debug(f"Dominance broadcast skipped (no permission) guild={g_dict.get('guild_id')}")
                else:
                    logger.warning(f"Dominance broadcast error guild={g_dict.get('guild_id')}: {e}")

    async def _get_btc_correlation(self, symbol: str, interval: str) -> float:
        """
        Calculate Pearson correlation between a coin's returns and BTC returns.
        Uses last 50 candles. Returns -1 to +1. Cache per-symbol for 30 min.
        """
        now = time.time()
        cached = self._btc_corr_cache.get(symbol)
        if cached and now - cached[1] < 1800:   # per-symbol timestamp
            return cached[0]

        try:
            import numpy as np
            btc_df   = await self.fetcher.get_klines("BTCUSDT", interval, 60)
            coin_df  = await self.fetcher.get_klines(symbol, interval, 60)

            if btc_df is None or coin_df is None or btc_df.empty or coin_df.empty:
                return 0.5  # neutral fallback

            btc_close  = btc_df["close"].astype(float).values
            coin_close = coin_df["close"].astype(float).values

            # Align lengths
            min_len = min(len(btc_close), len(coin_close))
            if min_len < 10:
                return 0.5

            btc_close  = btc_close[-min_len:]
            coin_close = coin_close[-min_len:]

            # Percent returns
            btc_ret  = np.diff(btc_close) / (btc_close[:-1] + 1e-9)
            coin_ret = np.diff(coin_close) / (coin_close[:-1] + 1e-9)

            # Pearson correlation
            corr = float(np.corrcoef(btc_ret, coin_ret)[0, 1])
            corr = max(-1.0, min(1.0, corr))  # clamp

            self._btc_corr_cache[symbol] = (corr, now)  # store (value, timestamp)
            return corr

        except Exception as e:
            logger.debug(f"BTC correlation error for {symbol}: {e}")
            return 0.5  # neutral fallback

    @tasks.loop(hours=1)
    async def ml_retrain_loop(self):
        """Periodic ML retraining + hourly quota flag reset"""
        try:
            # Reset A+ hour flags so B+ can send next hour if no A+ found
            self._reset_hour_flags()

            result = await asyncio.to_thread(self.ml.retrain, config.ML_MIN_SAMPLES)
            if result:
                logger.info("ML model retrained successfully")
        except Exception as e:
            logger.error(f"ML retrain loop error: {e}")

    @tasks.loop(hours=24)
    async def daily_ml_report_loop(self):
        """Send ML Deep Pattern Analysis report once every 24 hours"""
        try:
            ml_trainer = self.bot.cogs.get("MLTrainer")
            ai_engine  = self.bot.cogs.get("AIEngine")
            if ml_trainer and ai_engine:
                logger.info("Triggering daily ML deep pattern analysis report...")
                await ml_trainer._deep_pattern_analysis(ai_engine, self)
            else:
                logger.warning("Daily ML report: MLTrainer or AIEngine cog not found")
        except Exception as e:
            logger.error(f"Daily ML report error: {e}", exc_info=True)

    @daily_ml_report_loop.before_loop
    async def before_daily_ml_report(self):
        """Wait until midnight UTC before first report, then run every 24h"""
        import datetime
        await self.bot.wait_until_ready()
        now = datetime.datetime.utcnow()
        # Schedule first run at next midnight UTC
        next_midnight = (now + datetime.timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        wait_secs = (next_midnight - now).total_seconds()
        logger.info(f"Daily ML report scheduled in {wait_secs/3600:.1f}h (next UTC midnight)")
        await asyncio.sleep(wait_secs)

    # ─── Core Scan ───────────────────────────────────────────────────────────

    async def _get_dominance(self) -> dict:
        """
        Fetch BTC.D and USDT.D.
        Cache 10 minutes.

        Source priority:
          1. CMC global-metrics  → btc_dominance field (reliable)
                                 → stablecoin_market_cap_dominance (may not exist on basic plan)
          2. Binance /ticker/24hr computation (ALWAYS available, no key needed)
             - Fetch BTC, ETH, USDT circulating supply proxies via Binance price × OI
             - USDT.D = USDT market cap / total crypto market cap × 100
             - Uses top-20 coins' market caps from CMC listings (already fetched)
          3. CoinGecko public /global endpoint (free, no key)
             - market_cap_percentage.usdt gives exact USDT.D
        """
        now = time.time()
        btc_d, usdt_d, ts = self._dominance_cache
        if now - ts < 600 and btc_d > 0 and usdt_d > 0:
            return self._build_regime(btc_d, usdt_d)

        import aiohttp

        btc_d  = 0.0
        usdt_d = 0.0

        # ── Source 1: CMC global metrics (btc_d reliable, usdt_d may be missing) ──
        try:
            url = "https://pro-api.coinmarketcap.com/v1/global-metrics/quotes/latest"
            headers = {"X-CMC_PRO_API_KEY": config.CMC_API_KEY, "Accept": "application/json"}
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                    if resp.status == 200:
                        d = (await resp.json()).get("data", {})
                        btc_d = float(d.get("btc_dominance", 0) or 0)
                        # Try every known field name for USDT dominance
                        for field in ["stablecoin_market_cap_dominance",
                                      "stablecoin_volume_dominance",
                                      "usdt_dominance",
                                      "usdt_market_cap_dominance",
                                      "stable_coin_dominance"]:
                            v = d.get(field)
                            if v and float(v) > 0:
                                usdt_d = float(v)
                                logger.info(f"USDT.D from CMC field '{field}': {usdt_d:.3f}%")
                                break
        except Exception as e:
            logger.debug(f"CMC dominance fetch failed: {e}")

        # ── Source 2: CoinGecko /global (free, no key, has exact USDT.D) ──────────
        if usdt_d == 0 or btc_d == 0:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        "https://api.coingecko.com/api/v3/global",
                        timeout=aiohttp.ClientTimeout(total=6)
                    ) as resp:
                        if resp.status == 200:
                            data = (await resp.json()).get("data", {})
                            pct  = data.get("market_cap_percentage", {})
                            if btc_d == 0:
                                btc_d  = float(pct.get("btc", 0) or 0)
                            if usdt_d == 0:
                                _usdt  = float(pct.get("usdt", 0) or 0)
                                usdt_d = _usdt + 0.35  # +0.35% for USDC/DAI offset
                            logger.info(f"Dominance from CoinGecko: BTC.D={btc_d:.3f}%  USDT.D={usdt_d:.3f}%")
            except Exception as e:
                logger.debug(f"CoinGecko dominance fetch failed: {e}")

        # ── Source 3: Compute from Binance top-coin prices × CMC supplies ────────
        # Uses the top-coins list already fetched by DataFetcher
        # USDT market cap = USDT price (≈$1) × circulating supply
        # Total crypto market cap = sum of top coins' prices × circulating supply
        if usdt_d == 0:
            try:
                async with aiohttp.ClientSession() as session:
                    url = (
                        "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
                        "?limit=20&convert=USD&sort=market_cap"
                    )
                    headers = {"X-CMC_PRO_API_KEY": config.CMC_API_KEY, "Accept": "application/json"}
                    async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                        if resp.status == 200:
                            coins = (await resp.json()).get("data", [])
                            total_mcap = 0.0
                            usdt_mcap  = 0.0
                            btc_mcap   = 0.0
                            for c in coins:
                                mcap = float(c.get("quote", {}).get("USD", {}).get("market_cap", 0) or 0)
                                sym  = c.get("symbol", "")
                                total_mcap += mcap
                                if sym == "USDT":
                                    usdt_mcap = mcap
                                if sym == "BTC":
                                    btc_mcap  = mcap
                            if total_mcap > 0:
                                usdt_d = (usdt_mcap / total_mcap) * 100
                                if btc_d == 0:
                                    btc_d = (btc_mcap / total_mcap) * 100
                                logger.info(f"USDT.D computed from CMC listings: {usdt_d:.3f}%")
            except Exception as e:
                logger.debug(f"CMC listings dominance compute failed: {e}")

        # ── Validate and cache ────────────────────────────────────────────────────
        # Sanity check: USDT.D real range is ~4-12% (TradingView shows 8.29% today)
        if usdt_d <= 0 or usdt_d > 25:
            logger.warning(f"USDT.D value {usdt_d:.3f}% invalid — all sources failed, using last known or 8.0%")
            # Use last known cached value if available
            _, last_usdt_d, _ = self._dominance_cache
            usdt_d = last_usdt_d if last_usdt_d > 0 else 8.0  # realistic estimate from chart

        if btc_d <= 0 or btc_d > 100:
            btc_d_last, _, _ = self._dominance_cache
            btc_d = btc_d_last if btc_d_last > 0 else 57.9

        self._dominance_cache = (btc_d, usdt_d, now)
        regime = self._build_regime(btc_d, usdt_d)
        logger.info(f"Dominance: BTC.D={btc_d:.2f}%  USDT.D={usdt_d:.2f}%  Regime={regime['regime']}")
        return regime

    def _build_regime(self, btc_d: float, usdt_d: float) -> dict:
        """
        Interpret BTC.D and USDT.D to determine market regime.

        BTC.D rising  = money rotating INTO BTC = altcoins weak = prefer BTC longs / alt shorts
        BTC.D falling = altseason = alts strong = prefer alt longs
        USDT.D rising = risk-off, people going to stables = bearish for all crypto
        USDT.D falling = risk-on, money leaving stables = bullish for crypto

        Regimes:
        - risk_on_alt  : USDT.D falling + BTC.D falling → altseason, alts going up
        - risk_on_btc  : USDT.D falling + BTC.D rising  → BTC dominance phase
        - risk_off     : USDT.D rising                  → market fear, prefer shorts
        - neutral      : mixed signals
        """
        # ── USDT.D thresholds ────────────────────────────────────────────────
        # Real-world calibration (Feb 2026 TradingView data):
        #   8.29% = active fear / risk-off (confirmed from chart)
        #   7.9%  = transition zone, caution
        #   Below 6.5% = risk-on / greed
        #   Below 5.5% = full altseason greed
        usdt_extreme_fear = usdt_d > 8.0   # high fear — only shorts
        usdt_fear         = usdt_d > 7.5   # moderate fear — caution on longs
        usdt_neutral      = 6.5 <= usdt_d <= 7.5
        usdt_greed        = 5.5 < usdt_d < 6.5
        usdt_alt_greed    = usdt_d <= 5.5  # full altseason

        # ── BTC.D thresholds ─────────────────────────────────────────────────
        btc_d_high   = btc_d > 58.0    # BTC dominance phase
        btc_d_low    = btc_d < 48.0    # altseason territory
        btc_d_mid    = 48.0 <= btc_d <= 58.0

        if usdt_extreme_fear:
            regime      = "risk_off"
            allow_long  = False
            allow_short = True
            bias = f"🔴 RISK-OFF — USDT.D {usdt_d:.2f}% (extreme fear), avoid longs"
        elif usdt_fear:
            regime      = "risk_off"
            allow_long  = False
            allow_short = True
            bias = f"🟠 CAUTION — USDT.D {usdt_d:.2f}% (fear elevated), shorts preferred"
        elif usdt_alt_greed and btc_d_low:
            regime      = "risk_on_alt"
            allow_long  = True
            allow_short = False
            bias = f"🟢 ALTSEASON — USDT.D {usdt_d:.2f}% low + BTC.D {btc_d:.1f}% falling"
        elif (usdt_alt_greed or usdt_greed) and btc_d_high:
            regime      = "risk_on_btc"
            allow_long  = True
            allow_short = True
            bias = f"🟡 BTC SEASON — USDT.D {usdt_d:.2f}%, BTC.D {btc_d:.1f}% dominant"
        elif usdt_neutral or usdt_greed:
            regime      = "neutral"
            allow_long  = True
            allow_short = True
            bias = f"⚪ NEUTRAL — USDT.D {usdt_d:.2f}%, mixed signals"
        else:
            regime      = "neutral"
            allow_long  = True
            allow_short = True
            bias = f"⚪ NEUTRAL — USDT.D {usdt_d:.2f}%, BTC.D {btc_d:.1f}%"

        return {
            "regime":       regime,
            "btc_d":        round(btc_d, 2),
            "usdt_d":       round(usdt_d, 2),
            "allow_long":   allow_long,
            "allow_short":  allow_short,
            "bias":         bias,
            "btc_d_high":   btc_d_high,
            "btc_d_low":    btc_d_low,
        }

    # ─── Quota Helpers ───────────────────────────────────────────────────────

    def _reset_quota_if_new_day(self, trade_type: str):
        """Reset daily counters at UTC midnight"""
        import calendar
        import datetime
        q   = self._quota[trade_type]

        # calendar.timegm converts a UTC struct_time to a POSIX timestamp correctly
        # regardless of the server's local timezone — unlike naive .timestamp()
        utcnow = datetime.datetime.utcnow()
        today_start = calendar.timegm(
            utcnow.replace(hour=0, minute=0, second=0, microsecond=0).timetuple()
        )

        if q["day_start"] < today_start:
            logger.info(f"Quota reset for {trade_type} (new UTC day)")
            q["day_start"]      = today_start
            q["sent_today"]     = {"A+": 0, "B+": 0, "C+": 0}
            q["hour_has_aplus"] = False
            q["day_has_bplus"]  = False

    def _quota_allows(self, trade_type: str, grade: str) -> bool:
        """
        Decide whether to send a signal of this grade right now.

        SCALP  (24/day, 1/hour pacing)
          A+  -> always send instantly
          B+  -> 55min gap + no A+ this hour
          C+  -> 55min gap + no B+ all day

        DAY    (9/day, 90min gap between every signal)
          A+  -> 60min minimum gap (no flooding), otherwise always send
          B+  -> 90min gap + no A+ this scan
          C+  -> 90min gap + no B+ all day

        SWING  (3-4/day, 2h gap between every signal)
          A+  -> 90min minimum gap, otherwise always send
          B+  -> 120min gap + no A+ this scan
          C+  -> 120min gap + no B+ all day
        """
        q          = self._quota[trade_type]
        now        = time.time()
        sent       = q["sent_today"]
        target     = q["daily_target"]
        total_sent = sum(sent.values())
        secs_since = now - q["last_hour_sent"]

        # Minimum gap per trade type per grade
        GAP = {
            "scalp": {"A+":     0, "B+": 55*60, "C+": 55*60},
            "day":   {"A+": 60*60, "B+": 90*60, "C+": 90*60},
            "swing": {"A+": 90*60, "B+":120*60, "C+":120*60},
        }
        min_gap = GAP.get(trade_type, {}).get(grade, 3600)

        # A+ scalp: never blocked
        if grade == "A+" and trade_type == "scalp":
            return True

        # All others: enforce minimum gap first
        if secs_since < min_gap:
            logger.debug(
                f"Gap block {trade_type} {grade}: "
                f"{secs_since/60:.0f}min elapsed, need {min_gap//60}min"
            )
            return False

        # A+ day/swing: gap met = always send (no daily cap)
        if grade == "A+":
            return True

        if trade_type == "scalp":
            if grade == "B+":
                return not q["hour_has_aplus"]
            if grade == "C+":
                return not q["day_has_bplus"] and sent["B+"] == 0

        elif trade_type == "day":
            if total_sent >= target:
                return False
            if grade == "B+":
                return not q["hour_has_aplus"] and (target - sent["A+"] - sent["B+"]) > 0
            if grade == "C+":
                return not q["day_has_bplus"] and sent["B+"] == 0 and total_sent < target

        elif trade_type == "swing":
            if total_sent >= target + 1:
                return False
            if grade == "B+":
                return not q["hour_has_aplus"] and (target - sent["A+"] - sent["B+"]) > 0
            if grade == "C+":
                return not q["day_has_bplus"] and sent["B+"] == 0 and total_sent < target

        return False

    def _record_quota_send(self, trade_type: str, grade: str):
        """Update quota counters after a signal is successfully sent"""
        q = self._quota[trade_type]
        q["sent_today"][grade] = q["sent_today"].get(grade, 0) + 1
        q["last_hour_sent"]    = time.time()

        if grade == "A+":
            q["hour_has_aplus"] = True
        if grade == "B+":
            q["day_has_bplus"]  = True

    def _reset_hour_flags(self):
        """
        Called every hour to reset the per-hour A+ detection flag.
        This allows B+ to be sent again if the next hour has no A+.
        """
        for tt in self._quota:
            self._quota[tt]["hour_has_aplus"] = False
        logger.debug("Quota: hourly A+ flags reset")

    async def _run_scan(self, trade_type: str, interval: str, kline_limit: int):
        """Scan all valid symbols for signals"""
        if not self._valid_symbols:
            logger.warning("No valid symbols cached")
            return

        logger.info(f"Starting {trade_type} scan on {len(self._valid_symbols)} symbols")

        # Get BTC change (cached 5 min)
        now = time.time()
        if now - self._btc_change_cache[1] > 300:
            btc_change = await self.fetcher.get_btc_change(interval, 4)
            self._btc_change_cache = (btc_change, now)
        btc_change = self._btc_change_cache[0]

        # ── Dominance macro filter ──────────────────────────────────────────
        dom = await self._get_dominance()
        logger.info(f"Market regime: {dom['bias']}  BTC.D={dom['btc_d']}%  USDT.D={dom['usdt_d']}%")

        # ── Candle-level signal (scalp only) ──────────────────────────────────
        dom_sig = self._dom_signal
        if trade_type == "scalp":
            n = dom_sig.get("candle_count", 0)
            bias = dom_sig.get("scalp_bias", "neutral")
            reason = dom_sig.get("scalp_reason", "")
            logger.info(
                f"DOM candle signal ({n} candles): bias={bias} — {reason}"
            )

            # BLOCKED = USDT.D accelerating upward = absolute no-trade zone
            if bias == "blocked":
                logger.warning(f"DOM candle BLOCKED all scalps: {reason}")
                return  # skip entire scalp scan

            # Override macro dom regime with candle-level signal if we have enough data
            if n >= 3:
                if bias == "short_ok":
                    dom["allow_long"]  = False
                    dom["allow_short"] = True
                elif bias == "long_ok":
                    dom["allow_long"]  = True
                    # Only block shorts if not in extreme fear level
                    if dom_sig.get("usdt_velocity", 0) < 0.005:
                        dom["allow_short"] = False
                elif bias == "btc_long_only":
                    # Allow longs only on BTC/ETH, block alt longs
                    dom["allow_long"]  = True   # filtered per-symbol below
                    dom["allow_short"] = False
                # neutral = keep macro regime as-is

        # Shuffle to avoid always checking same coins first
        import random
        symbols_to_scan = self._valid_symbols.copy()
        random.shuffle(symbols_to_scan)

        # ── Phase 1: Collect all signals ──
        raw_signals = []
        for symbol in symbols_to_scan:
            if symbol in self.active_trades:
                continue
            try:
                signal = await self._analyze_symbol(symbol, trade_type, interval, kline_limit, btc_change, dom)
                if signal:
                    direction = signal.get("direction")

                    # ── Dominance filter ─────────────────────────────────────
                    # Block trades that go against the macro regime
                    if direction == "LONG" and not dom["allow_long"]:
                        logger.debug(f"Dominance blocked LONG {symbol} — {dom['regime']}")
                        continue
                    if direction == "SHORT" and not dom["allow_short"]:
                        logger.debug(f"Dominance blocked SHORT {symbol} — {dom['regime']}")
                        continue

                    # ── BTC Correlation filter ───────────────────────────────
                    btc_corr = signal.get("btc_corr", 0.5)
                    dom_regime = dom["regime"]

                    # High BTC correlation (>0.8) + BTC falling = coin will also fall
                    # Don't send LONG on high-corr coin when BTC trend is down
                    if direction == "LONG" and btc_corr > 0.80 and btc_change < -0.5:
                        logger.debug(f"BTC corr blocked: {symbol} LONG corr={btc_corr:.2f} BTC falling {btc_change:.1f}%")
                        continue

                    # Inverse correlation SHORT: if coin is inverse to BTC and BTC rising, don't short
                    if direction == "SHORT" and btc_corr < -0.5 and btc_change > 0.5:
                        logger.debug(f"BTC inv-corr blocked: {symbol} SHORT corr={btc_corr:.2f} BTC rising {btc_change:.1f}%")
                        continue

                    # In BTC season OR candle btc_long_only: block alt longs
                    if direction == "LONG":
                        base = symbol.replace("USDT","")
                        is_alt = base not in ["BTC","ETH"]
                        candle_btc_only = dom_sig.get("scalp_bias") == "btc_long_only" and trade_type == "scalp"
                        macro_btc_only  = dom["regime"] == "risk_on_btc" and btc_corr > 0.75

                        if is_alt and (candle_btc_only or macro_btc_only):
                            if signal.get("grade") == "A+":
                                signal["grade"] = "B+"
                                logger.debug(f"BTC dominance: demoted {symbol} LONG A+→B+")
                            elif signal.get("grade") == "B+" and candle_btc_only:
                                logger.debug(f"Candle btc_long_only: blocked {symbol} LONG B+")
                                continue

                    # Low/negative correlation in altseason = prioritize (decorrelated alts run harder)
                    if dom["regime"] == "risk_on_alt" and direction == "LONG" and btc_corr < 0.4:
                        signal["score"] = min(100, signal.get("score", 0) + 5)
                        logger.debug(f"Altseason decorrelated boost: {symbol} corr={btc_corr:.2f}")

                    # Attach dominance context to signal
                    signal["market_regime"] = dom["regime"]
                    signal["btc_d"]   = dom["btc_d"]
                    signal["usdt_d"]  = dom["usdt_d"]
                    signal["dom_bias"] = dom["bias"]
                    raw_signals.append(signal)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Analysis error {symbol}: {e}", exc_info=True)
                await asyncio.sleep(0.2)

        # ── Phase 2: Sort A+ > B+ > C+ by score within each grade ─────────
        sorted_signals = self.ml.sort_signals_by_priority(raw_signals)

        # ── Phase 3: Quota-aware sending ─────────────────────────────────────
        signals_sent = 0
        scan_counts  = {"A+": 0, "B+": 0, "C+": 0}
        ml_trained   = len(self.ml.trade_history) >= 50

        q = self._quota[trade_type]
        self._reset_quota_if_new_day(trade_type)

        for signal in sorted_signals:
            try:
                grade = signal.get("grade", "C+")

                # ── Quota gate ───────────────────────────────────────────────
                allowed = self._quota_allows(trade_type, grade)
                if not allowed:
                    logger.debug(f"Quota blocked {signal['symbol']} {grade} ({trade_type})")
                    continue

                # ── ML gate — only after 50+ real trades ─────────────────────
                if ml_trained:
                    ml_prob = await self.ml.predict_with_claude(
                        signal,
                        signal.get("indicators", {}),
                        signal.get("funding_rate", 0),
                        signal.get("ob_imbalance", 0)
                    )
                    # A+ never ML-blocked — always send if quota allows
                    if grade != "A+":
                        min_prob = {"B+": 0.42, "C+": 0.45}.get(grade, 0.45)
                        if ml_prob < min_prob:
                            logger.debug(f"ML filtered {signal['symbol']} {grade} — prob {ml_prob:.2f}")
                            continue

                await self._send_signal(signal)
                self._record_quota_send(trade_type, grade)
                scan_counts[grade] = scan_counts.get(grade, 0) + 1
                signals_sent += 1
                await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error sending {signal.get('symbol')}: {e}", exc_info=True)

        q = self._quota[trade_type]
        total_today = sum(q["sent_today"].values())
        logger.info(
            f"{trade_type} scan complete. "
            f"Raw: {len(raw_signals)} → Sent: {signals_sent} "
            f"(A+:{scan_counts['A+']} B+:{scan_counts['B+']} C+:{scan_counts['C+']}) "
            f"| Today total: {total_today}/{q['daily_target']} "
            f"(A+:{q['sent_today']['A+']} B+:{q['sent_today']['B+']} C+:{q['sent_today']['C+']})"
        )

    async def _analyze_symbol(self, symbol: str, trade_type: str, interval: str, limit: int, btc_change: float, dom: dict = None) -> Optional[dict]:
        """Full analysis pipeline for one symbol"""
        # Fetch data in parallel
        klines_task    = self.fetcher.get_klines(symbol, interval, limit)
        oi_task        = self.fetcher.get_open_interest(symbol)
        funding_task   = self.fetcher.get_funding_rate(symbol)
        ob_task        = self.fetcher.get_orderbook_imbalance(symbol, 20)
        taker_task     = self.fetcher.get_taker_buy_sell_ratio(symbol, "5m", 10)
        ticker_task    = self.fetcher.get_ticker_24h(symbol)

        results = await asyncio.gather(
            klines_task, oi_task, funding_task, ob_task, taker_task, ticker_task,
            return_exceptions=True
        )

        df, oi_data, funding_rate, ob_imbalance, taker_df, ticker = results

        # Validate
        if isinstance(df, Exception) or df is None or (hasattr(df, 'empty') and df.empty):
            return None
        if isinstance(funding_rate, Exception): funding_rate = 0.0
        if isinstance(ob_imbalance, Exception): ob_imbalance = 0.0
        if isinstance(taker_df, Exception):     taker_df = None
        if isinstance(ticker, Exception):       ticker = {}
        if isinstance(oi_data, Exception):      oi_data = {}

        # Volume filter
        if ticker:
            vol_24h = float(ticker.get("quoteVolume", 0))
            if vol_24h < config.MIN_VOLUME_24H_USD:
                return None

        # OI filter
        if oi_data:
            try:
                oi_val = float(oi_data.get("openInterest", 0)) * float(ticker.get("lastPrice", 1))
                if oi_val < config.MIN_OI_USD:
                    return None
            except:
                pass

        # Coin % change
        if len(df) >= 4:
            coin_change = ((float(df["close"].iloc[-1]) - float(df["close"].iloc[-4])) / float(df["close"].iloc[-4])) * 100
        else:
            coin_change = 0.0

        # Taker ratio
        taker_ls = 1.0
        if taker_df is not None and not taker_df.empty and "buySellRatio" in taker_df.columns:
            try:
                taker_ls = float(taker_df["buySellRatio"].iloc[-1])
            except:
                pass

        # Indicators
        indicators = calculate_all_indicators(df)
        if not indicators:
            return None

        # Liquidation zones
        liq_zones = await self.fetcher.get_liquidation_zones(symbol, df)

        # ML prediction
        ml_temp_signal = {
            "symbol": symbol, "direction": "LONG", "trade_type": trade_type,
            "score": 50, "outperform": coin_change - btc_change
        }
        ml_prob = await asyncio.to_thread(
            self.ml.predict_success_probability, ml_temp_signal, indicators, funding_rate, ob_imbalance
        )
        ml_score = ml_prob  # 0-1

        # Score signal
        # BTC correlation — tells us how closely this coin follows BTC
        btc_corr = await self._get_btc_correlation(symbol, interval)

        signal = self.scorer.score_signal(
            symbol=symbol,
            trade_type=trade_type,
            indicators=indicators,
            oi_data=oi_data,
            funding_rate=funding_rate,
            ob_imbalance=ob_imbalance,
            liq_zones=liq_zones,
            taker_ls_ratio=taker_ls,
            btc_change=btc_change,
            coin_change=coin_change,
            ml_score=ml_score,
            dom_regime=dom,
            btc_corr=btc_corr,
        )

        if signal:
            # Store extra data for ML recording
            signal["indicators"]   = indicators
            signal["ob_imbalance"] = ob_imbalance
            signal["funding_rate"] = funding_rate
            signal["btc_corr"]     = round(btc_corr, 3)

        return signal

    # ─── Signal Sending ───────────────────────────────────────────────────────

    async def _send_signal(self, signal: dict):
        """Send signal to ALL configured guilds, with AI analysis"""
        import db
        trade_type = signal["trade_type"]
        channel_key = {"scalp": "scalp_channel_id", "day": "day_channel_id", "swing": "swing_channel_id"}.get(trade_type)
        if not channel_key:
            return

        guilds = db.get_all_guilds()
        if not guilds:
            logger.warning(f"No guilds configured for {trade_type}")
            return

        # Get AI analysis before building embed
        ai_analysis = None
        ai_engine = self.bot.cogs.get("AIEngine")
        if ai_engine:
            try:
                ai_analysis = await ai_engine.analyze_signal(signal, signal.get("indicators", {}))
            except Exception as e:
                logger.warning(f"AI analysis failed for {signal.get('symbol')}: {e}")

        symbol    = signal["symbol"]
        direction = signal["direction"]
        grade     = signal["grade"]
        entry     = signal.get("entry", 0)
        sl        = signal.get("sl", 0)
        tps       = signal.get("tps", [])
        score     = signal.get("score", 0)
        leverage  = signal.get("leverage", 5)
        rsi14     = signal.get("rsi14", 0)
        vol_ratio = signal.get("vol_ratio", 0)
        funding   = signal.get("funding_rate", 0)
        confluences = signal.get("confluences", [])
        wyckoff   = signal.get("wyckoff_phase", "none")
        pa        = signal.get("pa_signals", [])
        gc        = signal.get("grade_criteria", {})

        dir_emoji  = "🟢" if direction == "LONG" else "🔴"
        grade_emoji = {"A+": "🏆", "B+": "🥈", "C+": "🥉"}.get(grade, "")
        color = 0x00C853 if direction == "LONG" else 0xFF1744

        # Grade description
        grade_desc = {
            "A+": "🎯 Sniper shot — take it without hesitation",
            "B+": "✅ Good setup — most criteria met",
            "C+": "⚠️ Directional bias — partial setup",
        }.get(grade, "")

        embed = discord.Embed(
            title=f"{dir_emoji} {grade_emoji} {grade} {direction} — {symbol}",
            description=grade_desc,
            color=color
        )

        # Price levels
        def fp(p):
            if not p: return "—"
            if p >= 1000:  return f"{p:,.2f}"
            if p >= 1:     return f"{p:.4f}"
            if p >= 0.01:  return f"{p:.5f}"
            return f"{p:.8f}"

        embed.add_field(name="📍 Entry",     value=f"`{fp(entry)}`",    inline=True)
        embed.add_field(name="🛑 Stop Loss", value=f"`{fp(sl)}`",       inline=True)
        embed.add_field(name="⚡ Leverage",  value=f"`{leverage}×`",    inline=True)

        if entry and sl:
            risk_pct = abs(entry - sl) / entry * 100
            embed.add_field(name="📊 Risk", value=f"`{risk_pct:.2f}%`", inline=True)
        embed.add_field(name="🎯 Score",     value=f"`{score:.0f}/100`", inline=True)
        embed.add_field(name="📈 RSI (14)",  value=f"`{rsi14:.1f}`",    inline=True)

        # Take profits
        if tps:
            tp_lines = []
            for i, tp in enumerate(tps):
                rr = abs(tp - entry) / abs(entry - sl) if (entry and sl and entry != sl) else 0
                tp_lines.append(f"TP{i+1}: `{fp(tp)}` (R:R {rr:.1f})")
            embed.add_field(name="🎯 Take Profits", value="\n".join(tp_lines), inline=False)

        # Grade criteria checklist
        if gc:
            checks = [
                ("Trend",       gc.get("trend_clear", False)),
                ("Volume",      gc.get("vol_confirmed", False)),
                ("Structure",   gc.get("has_structure", False)),
                ("Location",    gc.get("good_location", False)),
                ("Confluence",  gc.get("strong_conf", False)),
            ]
            criteria_str = "  ".join(f"{'✅' if v else '❌'} {k}" for k, v in checks)
            embed.add_field(name="📋 Grade Criteria", value=criteria_str, inline=False)

        # Wyckoff + PA
        theory_lines = []
        if wyckoff != "none":
            wlabels = {
                "accumulation_spring": "📦 Wyckoff: Accumulation Spring",
                "markup_SOS":          "📈 Wyckoff: Markup — Sign of Strength",
                "distribution_UTAD":   "📤 Wyckoff: Distribution UTAD",
                "markdown_SOW":        "📉 Wyckoff: Markdown — Sign of Weakness",
            }
            theory_lines.append(wlabels.get(wyckoff, wyckoff))
        for p in pa[:2]:
            theory_lines.append(f"🔍 PA: {p}")
        if theory_lines:
            embed.add_field(name="🧠 Theory", value="\n".join(theory_lines), inline=False)

        # Confluences
        clean_conf = [c.replace("✅ ", "").replace("✅", "").strip()
                      for c in confluences if not c.startswith("⚠️") and not c.startswith("✅ Wyckoff") and not c.startswith("✅ PA:")][:6]
        if clean_conf:
            embed.add_field(name="🔗 Confluences", value="\n".join(f"• {c}" for c in clean_conf), inline=False)

        # Stats footer
        embed.add_field(
            name="📊 Market Stats",
            value=f"Vol: `{vol_ratio:.2f}×`  |  Funding: `{funding*100:.4f}%`",
            inline=False
        )

        # AI analysis
        if ai_analysis:
            short_ai = ai_analysis[:900] + "..." if len(ai_analysis) > 900 else ai_analysis
            embed.add_field(name="🤖 AI Analysis", value=short_ai, inline=False)

        # Market regime + BTC correlation
        btc_d    = signal.get("btc_d", 0)
        usdt_d   = signal.get("usdt_d", 0)
        dom_bias = signal.get("dom_bias", "")
        btc_corr = signal.get("btc_corr", None)
        dom_trend = self._dom_trend

        if btc_d > 0:
            # Dominance trend arrows
            btc_d_delta  = dom_trend.get("btc_d_delta", 0)
            usdt_d_delta = dom_trend.get("usdt_d_delta", 0)
            btc_arr  = "📈" if btc_d_delta > 0.05 else ("📉" if btc_d_delta < -0.05 else "➡️")
            usdt_arr = "📈" if usdt_d_delta > 0.05 else ("📉" if usdt_d_delta < -0.05 else "➡️")

            embed.add_field(
                name="🌍 Market Regime",
                value=(
                    f"{dom_bias}\n"
                    f"{btc_arr} BTC.D: `{btc_d:.2f}%` ({btc_d_delta:+.3f}%)  "
                    f"{usdt_arr} USDT.D: `{usdt_d:.2f}%` ({usdt_d_delta:+.3f}%)"
                ),
                inline=False
            )

        if btc_corr is not None:
            if btc_corr >= 0.75:
                corr_label = f"🔗 High ({btc_corr:.2f}) — moves tightly with BTC"
            elif btc_corr >= 0.45:
                corr_label = f"〰️ Moderate ({btc_corr:.2f}) — partial BTC influence"
            elif btc_corr >= 0.15:
                corr_label = f"🔓 Low ({btc_corr:.2f}) — mostly independent"
            else:
                corr_label = f"🔄 Decorrelated/Inverse ({btc_corr:.2f}) — runs own path"
            embed.add_field(name="📐 BTC Correlation", value=corr_label, inline=False)

        # DOM candle signal (scalp only — shows live candle context)
        if trade_type == "scalp":
            ds = self._dom_signal
            n  = ds.get("candle_count", 0)
            if n >= 3:
                uv  = ds.get("usdt_velocity", 0)
                bv  = ds.get("btc_velocity",  0)
                ua  = ds.get("usdt_accel",    0)
                bias_icons = {
                    "long_ok":      "🟢",
                    "short_ok":     "🔴",
                    "btc_long_only":"🟡",
                    "neutral":      "⚪",
                    "blocked":      "⛔",
                }
                icon = bias_icons.get(ds.get("scalp_bias","neutral"), "⚪")
                embed.add_field(
                    name=f"🕯️ DOM Candle Signal ({n} candles)",
                    value=(
                        f"{icon} {ds.get('scalp_reason','')}\n"
                        f"USDT.D vel: `{uv:+.4f}%/min`  accel: `{ua:+.4f}`\n"
                        f"BTC.D  vel: `{bv:+.4f}%/min`"
                    ),
                    inline=False
                )

        embed.set_footer(text=f"{symbol} • {trade_type.upper()} • Not financial advice")

        # Store per-guild message refs for reply threading
        guild_messages = []

        for guild_cfg in guilds:
            channel_id = guild_cfg[channel_key]
            if not channel_id:
                continue
            channel = self.bot.get_channel(channel_id)
            if not channel:
                continue
            try:
                msg = await channel.send(embed=embed)
                guild_messages.append((channel_id, msg.id))
                logger.info(f"Signal sent to guild '{guild_cfg['guild_name']}': {symbol} {direction} {trade_type} Grade:{grade}")
            except discord.HTTPException as e:
                if e.code in (50001, 50013):  # Missing Access / Missing Permissions
                    logger.debug(f"Signal skipped (no permission) guild={guild_cfg['guild_id']}")
                else:
                    logger.error(f"Failed to send signal to guild {guild_cfg['guild_id']}: {e}")

        if guild_messages:
            signal["guild_messages"] = guild_messages  # [(channel_id, msg_id), ...]
            signal["channel_id"]     = guild_messages[0][0]  # keep for compat
            signal["message_id"]     = guild_messages[0][1]
            signal["tps_hit"]        = 0
            self.active_trades[symbol] = signal

    # ─── TP/SL Handlers ──────────────────────────────────────────────────────

    async def _handle_tp_hit(self, symbol: str, trade: dict, tp_num: int, tp_price: float):
        tps        = trade.get("tps", [])
        total_tps  = len(tps)
        tps_hit    = tp_num
        direction  = trade.get("direction", "LONG")
        grade      = trade.get("grade", "?")
        entry      = trade.get("entry", 0)
        sl         = trade.get("sl", 0)
        all_done   = (tps_hit == total_tps)

        # Progress bar  e.g.  ✅✅⬜⬜⬜
        bar = "".join("✅" if i < tps_hit else "⬜" for i in range(total_tps))

        # Profit % from entry
        if entry:
            pnl_pct = abs((tp_price - entry) / entry * 100)
            pnl_str = f"+{pnl_pct:.2f}%"
        else:
            pnl_str = "N/A"

        color = 0x00C853 if not all_done else 0xFFD700  # green → gold when all done

        embed = discord.Embed(
            title=f"{'🏆 ALL TPs HIT!' if all_done else f'✅ TP{tp_num} HIT'} — {symbol}",
            color=color
        )
        embed.add_field(name="Direction", value=f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT'}", inline=True)
        embed.add_field(name="Grade",     value=grade, inline=True)
        embed.add_field(name="TP Price",  value=f"`{tp_price}`", inline=True)
        embed.add_field(name="Entry",     value=f"`{entry}`", inline=True)
        embed.add_field(name="Profit",    value=f"**{pnl_str}**", inline=True)
        embed.add_field(name="Stop Loss", value=f"`{sl}`", inline=True)
        tp_lines = "\n".join(
            ("✅" if i < tps_hit else "🎯") + f" TP{i+1}: `{tps[i]}`" for i in range(total_tps)
        )
        embed.add_field(
            name=f"TP Progress ({tps_hit}/{total_tps})",
            value=bar + "\n" + tp_lines,
            inline=False
        )
        if all_done:
            embed.set_footer(text="🎉 Trade closed — all targets reached!")
        else:
            remaining = tps[tps_hit:]
            embed.set_footer(text=f"Next target: TP{tps_hit+1} @ {remaining[0]} | SL: {sl}")

        # Send embed as reply in every guild
        for channel_id, msg_id in trade.get("guild_messages", [(trade.get("channel_id"), trade.get("message_id"))]):
            channel = self.bot.get_channel(channel_id)
            if not channel:
                continue
            try:
                ref  = discord.MessageReference(message_id=msg_id, channel_id=channel_id, fail_if_not_exists=False)
                sent = await channel.send(embed=embed, reference=ref)
                msgs = trade.get("guild_messages", [])
                trade["guild_messages"] = [(cid, sent.id if cid == channel_id else mid) for cid, mid in msgs]
            except Exception as e:
                logger.error(f"TP hit message error in channel {channel_id}: {e}")

        logger.info(f"TP{tp_num}/{total_tps} hit for {symbol} at {tp_price} (+{pnl_str})")

    async def _handle_sl_hit(self, symbol: str, trade: dict, price: float):
        direction = trade.get("direction", "LONG")
        entry     = trade.get("entry", 0)
        sl        = trade.get("sl", 0)
        tps       = trade.get("tps", [])
        tps_hit   = trade.get("tps_hit", 0)
        total_tps = len(tps)
        grade     = trade.get("grade", "?")

        # Loss % from entry
        if entry:
            loss_pct = abs((price - entry) / entry * 100)
            loss_str = f"-{loss_pct:.2f}%"
        else:
            loss_str = "N/A"

        # Show which TPs were hit before SL
        bar = "".join("✅" if i < tps_hit else "❌" for i in range(total_tps))

        embed = discord.Embed(
            title=f"🛑 STOP LOSS HIT — {symbol}",
            color=0xFF1744
        )
        embed.add_field(name="Direction",  value=f"{'🟢 LONG' if direction=='LONG' else '🔴 SHORT'}", inline=True)
        embed.add_field(name="Grade",      value=grade, inline=True)
        embed.add_field(name="SL Price",   value=f"`{sl}`", inline=True)
        embed.add_field(name="Entry",      value=f"`{entry}`", inline=True)
        embed.add_field(name="Loss",       value=f"**{loss_str}**", inline=True)
        embed.add_field(name="Hit at",     value=f"`{price}`", inline=True)
        if tps:
            sl_tp_lines = "\n".join(
                ("✅" if i < tps_hit else "❌") + f" TP{i+1}: `{tps[i]}`" for i in range(total_tps)
            )
            sl_progress_val = bar + "\n" + sl_tp_lines
        else:
            sl_progress_val = "No TPs"
        embed.add_field(
            name=f"TP Progress ({tps_hit}/{total_tps} before SL)",
            value=sl_progress_val,
            inline=False
        )
        embed.set_footer(text="Trade closed • ML engine has recorded this outcome")

        # Reply to original signal in every guild
        for channel_id, msg_id in trade.get("guild_messages", [(trade.get("channel_id"), trade.get("message_id"))]):
            channel = self.bot.get_channel(channel_id)
            if not channel:
                continue
            try:
                ref  = discord.MessageReference(message_id=msg_id, channel_id=channel_id, fail_if_not_exists=False)
                await channel.send(embed=embed, reference=ref)
            except Exception as e:
                logger.error(f"SL hit message error in channel {channel_id}: {e}")

        self.ml.record_trade(
            trade, trade.get("indicators", {}),
            trade.get("funding_rate", 0),
            trade.get("ob_imbalance", 0),
            "loss"
        )
        if symbol in self.active_trades:
            del self.active_trades[symbol]
        logger.info(f"SL hit for {symbol} at {price} ({loss_str})")

    # Commands are handled by cogs/admin.py to avoid duplicate registration


async def setup(bot):
    await bot.add_cog(SignalEngine(bot))
