"""
ML Trainer Cog - 24/7 Claude-assisted ML training
Runs continuously to improve the ML model using:
1. Real closed trade outcomes (TP/SL hits)
2. Claude-generated synthetic training examples from market patterns
3. Claude reviewing and labeling ambiguous historical signals
4. Periodic deep analysis of what indicator combos actually work
"""
import asyncio
import re
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta
from typing import Optional

import discord
from discord.ext import commands, tasks

import config
import db

logger = logging.getLogger("MLTrainer")

TRAINING_LOG_PATH  = "data/training_log.json"
SYNTHETIC_DATA_PATH = "data/synthetic_trades.json"

# How many synthetic samples Claude generates per cycle
SYNTHETIC_BATCH_SIZE = 20
# Training cycle interval (minutes)
TRAIN_CYCLE_MINUTES = 30


def load_json(path: str, default):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return default


def save_json(path: str, data):
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class MLTrainer(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.training_log = load_json(TRAINING_LOG_PATH, [])
        self.synthetic_data = load_json(SYNTHETIC_DATA_PATH, [])
        self._cycle_count = 0
        self._last_deep_analysis = 0

    async def cog_load(self):
        logger.info("MLTrainer starting 24/7 Claude-assisted training...")
        self.training_cycle.start()

    async def cog_unload(self):
        self.training_cycle.cancel()

    # ─── Main Training Loop ──────────────────────────────────────────────────

    @tasks.loop(minutes=TRAIN_CYCLE_MINUTES)
    async def training_cycle(self):
        try:
            await self._run_training_cycle()
        except Exception as e:
            logger.error(f"Training cycle error: {e}", exc_info=True)

    @training_cycle.before_loop
    async def before_training(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(120)  # Wait 2 min after startup

    async def _run_training_cycle(self):
        ai_engine = self.bot.cogs.get("AIEngine")
        engine    = self.bot.cogs.get("SignalEngine")
        if not ai_engine or not ai_engine.api_key:
            logger.info("MLTrainer: no API key, skipping cycle")
            return

        self._cycle_count += 1
        logger.info(f"MLTrainer: starting cycle #{self._cycle_count}")

        # Step 1: Generate synthetic training data via Claude
        new_samples = await self._generate_synthetic_data(ai_engine, engine)

        # Step 2: Deep pattern analysis is now triggered by signal_engine
        # daily at UTC midnight. MLTrainer keeps a 24h fallback here in case
        # signal_engine is not running.
        now = time.time()
        if now - self._last_deep_analysis > 86400:  # 24 hours
            await self._deep_pattern_analysis(ai_engine, engine)
            self._last_deep_analysis = now

        # Step 3: Retrain ML with combined real + synthetic data
        if new_samples > 0 or self._cycle_count % 4 == 0:
            await self._retrain_with_all_data(engine)

        # Step 4: Log cycle results to Discord
        await self._log_to_discord(new_samples)

    # ─── Synthetic Data Generation ───────────────────────────────────────────

    async def _generate_synthetic_data(self, ai_engine, engine) -> int:
        """Ask Claude to generate realistic training examples based on market patterns"""

        # Get recent market context from active trades if available
        context = ""
        if engine and engine.active_trades:
            sample = list(engine.active_trades.values())[:3]
            context = f"Recent active signals for context: " + ", ".join(
                f"{t.get('symbol')} {t.get('direction')} {t.get('grade')}" for t in sample
            )

        system = (
            "You are an expert quantitative crypto trader with deep knowledge of technical analysis. "
            "Generate realistic crypto futures trading scenarios with their outcomes for ML training. "
            "Each scenario must be based on REAL market behavior patterns you know from experience. "
            "CRITICAL: Respond with ONLY a raw JSON array starting with [ and ending with ]. "
            "Absolutely no markdown, no code fences, no explanation text before or after. "
            "The very first character of your response must be [ and the last must be ]."
        )

        user = f"""Generate {SYNTHETIC_BATCH_SIZE} realistic crypto futures trading scenarios for ML training.
{context}

Each scenario represents a moment when a signal was generated and what happened.
Base these on REAL patterns — how RSI, MACD, EMA, volume, funding rate combinations actually play out.

Rules:
- Mix of LONG/SHORT, scalp/day/swing, A+/B+/C+ grades
- outcome: 1 = hit TP1 (win), 0 = hit SL (loss)
- Make outcomes realistic (good setups win ~65%, bad ones ~35%)
- Vary all indicators realistically
- Include edge cases (RSI divergence, high funding, low volume failures)

Required JSON format — array of objects with these exact fields:
{{
  "rsi14": float,
  "rsi7": float,
  "macd_hist": float,
  "stoch_k": float,
  "vol_ratio": float,
  "ema_bullish": int,
  "ema21_50_bullish": int,
  "vwap_distance": float,
  "ob_imbalance": float,
  "funding_rate_scaled": float,
  "outperform": float,
  "score": float,
  "is_long": int,
  "trade_type_encoded": int,
  "has_divergence": int,
  "pattern_count": int,
  "grade": "A+" | "B+" | "C+",
  "direction": "LONG" | "SHORT",
  "trade_type": "scalp" | "day" | "swing",
  "outcome": 0 | 1,
  "reasoning": "brief explanation of why this outcome"
}}

Generate exactly {SYNTHETIC_BATCH_SIZE} scenarios:"""

        response = await ai_engine._call_claude(system, user, max_tokens=3000)
        if not response:
            return 0

        try:
            # Robust JSON extraction — handles markdown, text before/after, etc.
            raw = response.strip()

            # Strip all markdown code fences
            raw = re.sub(r"```(?:json)?", "", raw).strip()

            # Find the outermost JSON array
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start == -1 or end <= 1:
                # Maybe Claude returned a single object — try wrapping it
                obj_start = raw.find("{")
                obj_end   = raw.rfind("}") + 1
                if obj_start != -1 and obj_end > 1:
                    raw = "[" + raw[obj_start:obj_end] + "]"
                    start, end = 0, len(raw)
                else:
                    logger.warning(f"MLTrainer: no JSON found in Claude response: {raw[:200]}")
                    return 0

            json_str = raw[start:end]

            # Fix common Claude JSON issues: trailing commas
            json_str = re.sub(r",\s*([}\]])", r"\1", json_str)  # noqa

            try:
                scenarios = json.loads(json_str)
            except json.JSONDecodeError as je:
                logger.warning(f"MLTrainer: JSON decode failed: {je} — snippet: {json_str[:300]}")
                return 0
            added = 0
            for s in scenarios:
                if "outcome" not in s or "score" not in s:
                    continue
                # Convert to ML feature format
                features = [
                    s.get("rsi14", 50),
                    s.get("rsi7", 50),
                    s.get("macd_hist", 0),
                    s.get("stoch_k", 50),
                    s.get("vol_ratio", 1),
                    s.get("ema_bullish", 0),
                    s.get("ema21_50_bullish", 0),
                    s.get("vwap_distance", 0),
                    s.get("ob_imbalance", 0),
                    s.get("funding_rate_scaled", 0),
                    s.get("outperform", 0),
                    s.get("score", 50),
                    s.get("is_long", 1),
                    s.get("trade_type_encoded", 0),
                    s.get("has_divergence", 0),
                    s.get("pattern_count", 0),
                ]
                record = {
                    "timestamp":   datetime.utcnow().isoformat(),
                    "symbol":      f"SYNTHETIC_{s.get('trade_type','scalp').upper()}",
                    "grade":       s.get("grade", "B+"),
                    "score":       s.get("score", 60),
                    "direction":   s.get("direction", "LONG"),
                    "trade_type":  s.get("trade_type", "scalp"),
                    "outcome":     int(s.get("outcome", 0)),
                    "features":    features,
                    "source":      "claude_synthetic",
                    "reasoning":   s.get("reasoning", ""),
                }
                self.synthetic_data.append(record)
                added += 1

            save_json(SYNTHETIC_DATA_PATH, self.synthetic_data)
            logger.info(f"MLTrainer: generated {added} synthetic training samples")
            return added

        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"MLTrainer: synthetic data parse error: {e}")
            return 0

    # ─── Deep Pattern Analysis ───────────────────────────────────────────────

    async def _deep_pattern_analysis(self, ai_engine, engine):
        """Every 2 hours, Claude deeply analyzes what's working and what's not"""
        ml = engine.ml if engine else None
        if not ml:
            return

        real_trades = ml.trade_history
        synthetic   = self.synthetic_data[-100:]  # last 100 synthetic

        if len(real_trades) < 5 and len(synthetic) < 20:
            return

        # Build pattern breakdown
        def breakdown(trades):
            result = {}
            for t in trades:
                key = f"{t.get('grade','?')}_{t.get('direction','?')}_{t.get('trade_type','?')}"
                if key not in result:
                    result[key] = {"wins": 0, "total": 0}
                result[key]["total"] += 1
                if t.get("outcome") == 1:
                    result[key]["wins"] += 1
            return {k: {"win_rate": round(v["wins"]/v["total"]*100,1), "n": v["total"]}
                    for k, v in result.items() if v["total"] >= 2}

        real_breakdown = breakdown(real_trades) if real_trades else {}
        synth_breakdown = breakdown(synthetic)

        system = (
            "You are a quantitative trading system optimizer. "
            "Analyze trading performance patterns and identify what indicator combinations "
            "are most predictive of success. Be specific and actionable. "
            "Respond in plain text, max 300 words."
        )

        user = f"""Analyze this crypto futures signal bot performance data:

Real Closed Trades ({len(real_trades)} total):
{json.dumps(real_breakdown, indent=2) if real_breakdown else "Not enough real data yet"}

Synthetic Training Patterns ({len(synthetic)} samples):
{json.dumps(synth_breakdown, indent=2)}

Questions to answer:
1. Which grade+direction+type combos have the best win rates?
2. Which combinations are consistently losing and should be filtered harder?
3. What indicator thresholds should the ML focus on most?
4. Any concerning patterns in the data?
5. Recommended priority adjustments for A+/B+/C+ signal filtering?

Be direct and specific with numbers."""

        analysis = await ai_engine._call_claude(system, user, max_tokens=400)
        if not analysis:
            return

        logger.info(f"MLTrainer deep analysis:\n{analysis}")

        # Save analysis log
        self.training_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": "deep_analysis",
            "analysis": analysis,
            "real_trades": len(real_trades),
            "synthetic_trades": len(synthetic),
        })
        save_json(TRAINING_LOG_PATH, self.training_log[-100:])  # keep last 100

        # Broadcast to log channels
        await self._broadcast_analysis(analysis, len(real_trades), len(synthetic))

    async def _broadcast_analysis(self, analysis: str, real_count: int, synth_count: int):
        """Send daily ML Deep Pattern Analysis report to all guild log channels"""
        from datetime import datetime as dt
        guilds = db.get_all_guilds()

        # Split analysis into chunks (Discord field limit 1024 chars)
        chunks = []
        remaining = analysis
        while remaining:
            if len(remaining) <= 1000:
                chunks.append(remaining)
                break
            split_at = remaining[:1000].rfind("\n")
            if split_at == -1:
                split_at = 1000
            chunks.append(remaining[:split_at])
            remaining = remaining[split_at:].lstrip()

        embed = discord.Embed(
            title="🧠 ML Daily Deep Pattern Analysis",
            description=f"**24-Hour Performance Report** — {dt.utcnow().strftime('%Y-%m-%d UTC')}",
            color=0x6A1B9A,
            timestamp=dt.utcnow()
        )
        for i, chunk in enumerate(chunks[:4]):  # max 4 fields
            embed.add_field(
                name=f"📊 Analysis {'(cont.)' if i > 0 else ''}",
                value=chunk,
                inline=False
            )
        embed.add_field(
            name="📈 Data Used",
            value=f"Real trades: `{real_count}` | Synthetic samples: `{synth_count}`",
            inline=False
        )
        embed.set_footer(text="Daily report • Next analysis in 24h at UTC midnight")

        for g in guilds:
            g_dict = dict(g)
            log_ch_id = g_dict.get("log_channel_id")
            if not log_ch_id:
                continue
            ch = self.bot.get_channel(log_ch_id)
            if ch:
                try:
                    await ch.send(embed=embed)
                    logger.info(f"Daily ML report sent to channel {log_ch_id}")
                except Exception as e:
                    logger.error(f"MLTrainer broadcast error: {e}")

    # ─── Retrain with All Data ───────────────────────────────────────────────

    async def _retrain_with_all_data(self, engine):
        """Combine real + synthetic data and retrain ML model"""
        if not engine:
            return

        ml = engine.ml
        real_trades = ml.trade_history

        # Merge real + synthetic into one training set
        all_data = real_trades + self.synthetic_data

        if len(all_data) < 30:
            logger.info(f"MLTrainer: not enough data to retrain ({len(all_data)}/30)")
            return

        # Temporarily inject synthetic into ml.trade_history for retraining
        original_history = ml.trade_history
        ml.trade_history = all_data

        result = await asyncio.to_thread(ml.retrain, 30)

        # Restore only real trades in ml.trade_history
        ml.trade_history = original_history

        if result:
            logger.info(
                f"MLTrainer: retrained with {len(all_data)} samples "
                f"({len(real_trades)} real + {len(self.synthetic_data)} synthetic)"
            )
            # Log to training log
            self.training_log.append({
                "timestamp":       datetime.utcnow().isoformat(),
                "type":            "retrain",
                "real_samples":    len(real_trades),
                "synthetic_samples": len(self.synthetic_data),
                "total_samples":   len(all_data),
            })
            save_json(TRAINING_LOG_PATH, self.training_log[-100:])

    # ─── Cycle Log ───────────────────────────────────────────────────────────

    async def _log_to_discord(self, new_samples: int):
        """Send brief training update to log channels every 4 cycles"""
        if self._cycle_count % 4 != 0:
            return

        engine = self.bot.cogs.get("SignalEngine")
        real_count  = len(engine.ml.trade_history) if engine else 0
        synth_count = len(self.synthetic_data)

        guilds = db.get_all_guilds()
        embed = discord.Embed(
            title="🤖 ML Training Update",
            color=0x4A148C,
            timestamp=datetime.utcnow()
        )
        embed.add_field(name="Cycle",             value=f"#{self._cycle_count}", inline=True)
        embed.add_field(name="Real Trades",       value=str(real_count),         inline=True)
        embed.add_field(name="Synthetic Samples", value=str(synth_count),        inline=True)
        embed.add_field(name="New This Cycle",    value=f"+{new_samples}",       inline=True)
        embed.add_field(name="Next Cycle",        value=f"In {TRAIN_CYCLE_MINUTES} min", inline=True)
        embed.set_footer(text="Claude is continuously training your ML model 24/7")

        for g in guilds:
            log_ch_id = g["log_channel_id"]
            if not log_ch_id:
                continue
            ch = self.bot.get_channel(log_ch_id)
            if ch:
                try:
                    await ch.send(embed=embed)
                except Exception:
                    pass

    # ─── Admin Commands ──────────────────────────────────────────────────────

    @commands.command(name="trainstatus")
    async def train_status(self, ctx):
        """Show ML training status"""
        engine = self.bot.cogs.get("SignalEngine")
        real_count  = len(engine.ml.trade_history) if engine else 0
        synth_count = len(self.synthetic_data)
        last_log    = self.training_log[-1] if self.training_log else None

        embed = discord.Embed(title="🧠 ML Training Status", color=0x4A148C)
        embed.add_field(name="Training Cycles Run",   value=str(self._cycle_count),  inline=True)
        embed.add_field(name="Real Trade Samples",    value=str(real_count),          inline=True)
        embed.add_field(name="Synthetic Samples",     value=str(synth_count),         inline=True)
        embed.add_field(name="Total Training Data",   value=str(real_count + synth_count), inline=True)
        embed.add_field(name="Cycle Interval",        value=f"Every {TRAIN_CYCLE_MINUTES} min", inline=True)
        embed.add_field(name="Deep Analysis",         value="Every 2 hours",          inline=True)

        if last_log:
            embed.add_field(
                name="Last Activity",
                value=f"{last_log.get('type','?')} at {last_log.get('timestamp','?')[:19]}",
                inline=False
            )
        embed.set_footer(text="Claude generates synthetic training data every 30 min 24/7")
        await ctx.send(embed=embed)

    @commands.command(name="forcetrain")
    @commands.has_permissions(administrator=True)
    async def force_train(self, ctx):
        """Force an immediate training cycle"""
        ai_engine = self.bot.cogs.get("AIEngine")
        if not ai_engine or not ai_engine.api_key:
            await ctx.send("❌ `ANTHROPIC_API_KEY` not set — AI training disabled.")
            return
        msg = await ctx.send("🧠 Running immediate ML training cycle with Claude...")
        await self._run_training_cycle()
        engine = self.bot.cogs.get("SignalEngine")
        real  = len(engine.ml.trade_history) if engine else 0
        synth = len(self.synthetic_data)
        await msg.edit(content=f"✅ Training cycle complete! Model trained on **{real + synth}** samples ({real} real + {synth} synthetic).")


async def setup(bot):
    await bot.add_cog(MLTrainer(bot))
