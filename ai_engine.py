"""
AI Engine - Claude-powered market analysis and auto-tuning
- Generates per-signal market analysis using Claude API
- Auto-tunes config thresholds based on win/loss history
- Runs as a background cog
"""
import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

import aiohttp
import discord
from discord.ext import commands, tasks

import config
import db

logger = logging.getLogger("AIEngine")

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"  # Fast + cheap for 24/7 use

# Auto-tune settings saved here
TUNE_PATH = "data/auto_tune.json"


def load_tune() -> dict:
    if os.path.exists(TUNE_PATH):
        try:
            with open(TUNE_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_tune(data: dict):
    os.makedirs("data", exist_ok=True)
    with open(TUNE_PATH, "w") as f:
        json.dump(data, f, indent=2)


class AIEngine(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._tune_data = load_tune()
        self._last_tune = self._tune_data.get("last_tuned", 0)
        # Apply any saved tuned thresholds on startup
        self._apply_tuned_thresholds()

    async def cog_load(self):
        self._session = aiohttp.ClientSession()
        if not self.api_key:
            logger.warning("ANTHROPIC_API_KEY not set — AI analysis disabled")
        else:
            logger.info("AIEngine ready with Claude API")
        self.auto_tune_loop.start()

    async def cog_unload(self):
        self.auto_tune_loop.cancel()
        if self._session:
            await self._session.close()

    # ─── Claude API Call ─────────────────────────────────────────────────────

    async def _call_claude(self, system: str, user: str, max_tokens: int = 400) -> Optional[str]:
        if not self.api_key or not self._session:
            return None
        try:
            payload = {
                "model": CLAUDE_MODEL,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}]
            }
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            async with self._session.post(
                ANTHROPIC_API_URL, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data["content"][0]["text"].strip()
                else:
                    err = await resp.text()
                    logger.error(f"Claude API error {resp.status}: {err[:200]}")
                    return None
        except asyncio.TimeoutError:
            logger.warning("Claude API timeout")
            return None
        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            return None

    # ─── Signal Analysis ─────────────────────────────────────────────────────

    async def analyze_signal(self, signal: dict, indicators: dict) -> Optional[str]:
        """Generate a short AI market analysis for a signal. Called by signal_engine."""
        if not self.api_key:
            return None

        system = (
            "You are a professional crypto futures trader. "
            "Given a trading signal with technical indicators, write a concise 2-3 sentence market analysis. "
            "Be direct and specific. Mention the key confluence factors driving this signal. "
            "Do not use disclaimers. Format: plain text, no markdown."
        )

        user = f"""Signal Details:
Symbol: {signal.get('symbol')}
Direction: {signal.get('direction')}
Type: {signal.get('trade_type')} trade
Grade: {signal.get('grade')}
Score: {signal.get('score')}/100
Entry: {signal.get('entry')}
Stop Loss: {signal.get('sl')}
Take Profits: {signal.get('tps')}
Confluences: {', '.join(signal.get('confluences', []))}

Key Indicators:
RSI(14): {indicators.get('rsi14', 'N/A')}
MACD Hist: {indicators.get('macd_hist', 'N/A')}
EMA Stack: {'Bullish' if indicators.get('ema9', 0) > indicators.get('ema21', 0) > indicators.get('ema50', 0) else 'Bearish' if indicators.get('ema9', 0) < indicators.get('ema21', 0) < indicators.get('ema50', 0) else 'Mixed'}
Volume Spike: {indicators.get('vol_ratio', 'N/A')}x avg
Funding Rate: {signal.get('funding_rate', 'N/A')}
OB Imbalance: {signal.get('ob_imbalance', 'N/A')}
Patterns: {', '.join(indicators.get('patterns', [])) or 'None'}

Write the market analysis:"""

        analysis = await self._call_claude(system, user, max_tokens=200)
        return analysis

    # ─── Auto-Tune Loop ──────────────────────────────────────────────────────

    @tasks.loop(hours=6)
    async def auto_tune_loop(self):
        """Every 6 hours, analyze win/loss history and auto-tune thresholds"""
        try:
            await self._run_auto_tune()
        except Exception as e:
            logger.error(f"Auto-tune error: {e}", exc_info=True)

    @auto_tune_loop.before_loop
    async def before_tune_loop(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(60)  # Wait 1 min after startup before first tune

    async def _run_auto_tune(self):
        if not self.api_key:
            return

        # Load trade history for ML engine
        engine = self.bot.cogs.get("SignalEngine")
        if not engine:
            return

        history = engine.ml.trade_history
        if len(history) < 20:
            logger.info(f"Auto-tune: not enough data yet ({len(history)}/20 trades)")
            return

        logger.info(f"Auto-tune: analyzing {len(history)} trades with Claude...")

        # Build stats breakdown
        by_type  = {}
        by_grade = {}
        by_dir   = {}
        for t in history:
            tt = t.get("trade_type", "scalp")
            g  = t.get("grade", "?")
            d  = t.get("direction", "LONG")
            o  = t.get("outcome", 0)

            by_type.setdefault(tt,   {"wins": 0, "total": 0})
            by_grade.setdefault(g,   {"wins": 0, "total": 0})
            by_dir.setdefault(d,     {"wins": 0, "total": 0})

            by_type[tt]["total"]   += 1
            by_grade[g]["total"]   += 1
            by_dir[d]["total"]     += 1
            if o == 1:
                by_type[tt]["wins"]  += 1
                by_grade[g]["wins"]  += 1
                by_dir[d]["wins"]    += 1

        def pct(d): return round(d["wins"]/d["total"]*100, 1) if d["total"] else 0

        stats_summary = {
            "total_trades": len(history),
            "overall_win_rate": pct({"wins": sum(1 for t in history if t.get("outcome")==1), "total": len(history)}),
            "by_trade_type": {k: {"win_rate": pct(v), "total": v["total"]} for k, v in by_type.items()},
            "by_grade":      {k: {"win_rate": pct(v), "total": v["total"]} for k, v in by_grade.items()},
            "by_direction":  {k: {"win_rate": pct(v), "total": v["total"]} for k, v in by_dir.items()},
        }

        current_config = {
            "GRADE_A_PLUS":      config.GRADE_A_PLUS,
            "GRADE_B_PLUS":      config.GRADE_B_PLUS,
            "GRADE_C_PLUS":      config.GRADE_C_PLUS,
            "VOLUME_SPIKE_MULT": config.VOLUME_SPIKE_MULT,
            "BTC_OUTPERFORM_PCT": config.BTC_OUTPERFORM_PCT,
            "MIN_VOLUME_24H_USD": config.MIN_VOLUME_24H_USD,
        }

        system = (
            "You are an expert quant trader and algorithmic trading system optimizer. "
            "You will be given performance stats for a crypto signal bot and its current config thresholds. "
            "Your job is to suggest optimized threshold values to improve win rate and TP hit rate. "
            "Respond ONLY with a valid JSON object. No explanation, no markdown, just raw JSON."
        )

        user = f"""Performance Stats:
{json.dumps(stats_summary, indent=2)}

Current Config Thresholds:
{json.dumps(current_config, indent=2)}

Rules for optimization:
- If a grade's win rate is below 45%, raise its score threshold by 3-8 points
- If scalp win rate < 50%, increase VOLUME_SPIKE_MULT by 0.2-0.5
- If swing win rate > 65%, you can slightly lower GRADE_C_PLUS threshold to get more signals
- SL is now structure-based, do not touch it
- Never set GRADE_A_PLUS above 92, GRADE_B_PLUS above 75, GRADE_C_PLUS above 55
- Never set VOLUME_SPIKE_MULT below 1.2 or above 4.0
- BTC_OUTPERFORM_PCT range: 0.2 to 2.0
- Keep changes conservative (max 10% change per tune cycle)

Respond with ONLY a JSON object with the optimized values, example format:
{{"GRADE_A_PLUS": 82, "GRADE_B_PLUS": 63, "GRADE_C_PLUS": 42, "VOLUME_SPIKE_MULT": 2.2, "BTC_OUTPERFORM_PCT": 0.5, "MIN_VOLUME_24H_USD": 5000000}}"""

        response = await self._call_claude(system, user, max_tokens=300)
        if not response:
            return

        try:
            # Strip markdown code fences properly using regex (str.strip("```json")
            # strips individual characters, NOT the substring — use re.sub instead)
            import re as _re
            clean = _re.sub(r"```(?:json)?", "", response.strip()).strip()
            new_vals = json.loads(clean)
            self._apply_new_thresholds(new_vals, stats_summary)
        except json.JSONDecodeError as e:
            logger.error(f"Auto-tune: failed to parse Claude response: {e}\nResponse: {response[:300]}")

    def _apply_new_thresholds(self, new_vals: dict, stats: dict):
        """Apply Claude-suggested thresholds to live config"""
        changes = []

        def safe_set(attr, key, min_val, max_val):
            if key in new_vals:
                old = getattr(config, attr)
                new = max(min_val, min(max_val, new_vals[key]))
                if abs(new - old) > 0.01:
                    setattr(config, attr, new)
                    changes.append(f"{attr}: {old} → {new}")

        safe_set("GRADE_A_PLUS",       "GRADE_A_PLUS",       70, 92)
        safe_set("GRADE_B_PLUS",       "GRADE_B_PLUS",       50, 75)
        safe_set("GRADE_C_PLUS",       "GRADE_C_PLUS",       30, 55)
        safe_set("VOLUME_SPIKE_MULT",  "VOLUME_SPIKE_MULT",  1.2, 4.0)
        safe_set("BTC_OUTPERFORM_PCT", "BTC_OUTPERFORM_PCT", 0.2, 2.0)
        safe_set("MIN_VOLUME_24H_USD", "MIN_VOLUME_24H_USD", 1_000_000, 50_000_000)

        if changes:
            logger.info(f"Auto-tune applied {len(changes)} changes: {', '.join(changes)}")
            # Save for persistence across restarts
            self._tune_data = {
                "last_tuned": time.time(),
                "last_tuned_str": datetime.utcnow().isoformat(),
                "changes": changes,
                "stats_at_tune": stats,
                "current_thresholds": {
                    "GRADE_A_PLUS":       config.GRADE_A_PLUS,
                    "GRADE_B_PLUS":       config.GRADE_B_PLUS,
                    "GRADE_C_PLUS":       config.GRADE_C_PLUS,
                    "VOLUME_SPIKE_MULT":  config.VOLUME_SPIKE_MULT,
                    "BTC_OUTPERFORM_PCT": config.BTC_OUTPERFORM_PCT,
                    "MIN_VOLUME_24H_USD": config.MIN_VOLUME_24H_USD,
                }
            }
            save_tune(self._tune_data)
            # Notify all log channels
            asyncio.create_task(self._broadcast_tune_update(changes, stats))
        else:
            logger.info("Auto-tune: thresholds already optimal, no changes needed")

    def _apply_tuned_thresholds(self):
        """Re-apply saved tuned thresholds on bot restart"""
        saved = self._tune_data.get("current_thresholds", {})
        if not saved:
            return
        for attr, val in saved.items():
            if hasattr(config, attr):
                setattr(config, attr, val)
        if saved:
            logger.info(f"Restored {len(saved)} auto-tuned thresholds from last session")

    async def _broadcast_tune_update(self, changes: list, stats: dict):
        """Send auto-tune report to all guild log channels"""
        await asyncio.sleep(2)
        guilds = db.get_all_guilds()
        embed = discord.Embed(
            title="🤖 AI Auto-Tune Complete",
            description=f"Claude analyzed **{stats.get('total_trades', 0)} trades** and optimized signal thresholds.",
            color=0x7B2FBE,
            timestamp=datetime.utcnow()
        )
        embed.add_field(
            name="📊 Performance at Tune",
            value=f"Overall Win Rate: **{stats.get('overall_win_rate', '?')}%**",
            inline=False
        )
        embed.add_field(
            name="⚙️ Threshold Changes",
            value="\n".join(f"• {c}" for c in changes) or "No changes needed",
            inline=False
        )
        embed.set_footer(text="Thresholds auto-saved • Next tune in 6 hours")

        for g in guilds:
            log_ch_id = g["log_channel_id"]
            if not log_ch_id:
                continue
            ch = self.bot.get_channel(log_ch_id)
            if ch:
                try:
                    await ch.send(embed=embed)
                except Exception as e:
                    logger.error(f"Failed to send tune report to guild {g['guild_id']}: {e}")

    # ─── Admin Commands ──────────────────────────────────────────────────────

    @commands.command(name="tune")
    @commands.has_permissions(administrator=True)
    async def force_tune(self, ctx):
        """Force an immediate AI auto-tune cycle"""
        if not self.api_key:
            await ctx.send("❌ `ANTHROPIC_API_KEY` not set in `.env` — AI features disabled.")
            return
        msg = await ctx.send("🤖 Running AI auto-tune analysis with Claude...")
        await self._run_auto_tune()
        if self._tune_data.get("changes"):
            changes = self._tune_data["changes"]
            await msg.edit(content=f"✅ Auto-tune complete! Applied **{len(changes)}** threshold changes. Check your log channel.")
        else:
            await msg.edit(content="✅ Auto-tune complete. Thresholds were already optimal — no changes needed.")

    @commands.command(name="thresholds")
    async def show_thresholds(self, ctx):
        """Show current (possibly auto-tuned) signal thresholds"""
        embed = discord.Embed(title="⚙️ Current Signal Thresholds", color=0x7B2FBE)
        embed.add_field(name="Grade A+", value=f"Score ≥ {config.GRADE_A_PLUS}", inline=True)
        embed.add_field(name="Grade B+", value=f"Score ≥ {config.GRADE_B_PLUS}", inline=True)
        embed.add_field(name="Grade C+", value=f"Score ≥ {config.GRADE_C_PLUS}", inline=True)
        embed.add_field(name="Volume Spike", value=f"{config.VOLUME_SPIKE_MULT}x avg", inline=True)
        embed.add_field(name="SL Mode",   value="Structure-based (ATR)", inline=True)
        embed.add_field(name="Vol Mult",  value=f"{config.VOLUME_SPIKE_MULT}x", inline=True)
        embed.add_field(name="BTC Δ Min", value=f"{config.BTC_OUTPERFORM_PCT}%", inline=True)
        embed.add_field(name="Min Volume", value=f"${config.MIN_VOLUME_24H_USD:,.0f}", inline=True)

        last = self._tune_data.get("last_tuned_str")
        if last:
            embed.set_footer(text=f"Last auto-tuned by Claude: {last} UTC")
        else:
            embed.set_footer(text="Not yet auto-tuned (need 20+ closed trades)")
        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(AIEngine(bot))
