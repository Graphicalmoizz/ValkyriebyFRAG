"""
BTC Liquidation Monitor — Live WebSocket
Connects directly to Binance forceOrder stream for real-time liquidation alerts
No polling delay — alerts fire within milliseconds of liquidation
"""
import asyncio
import json
import logging
import time
import aiohttp

import discord
from discord.ext import commands, tasks

import config
import db
from utils.card_builder import build_liquidation_alert, build_massive_liq_alert

logger = logging.getLogger("LiquidationMonitor")

# Thresholds
MIN_LIQ_USD      = 50_000      # $50K min for individual alert
WHALE_USD        = 500_000     # $500K = whale label
MEGA_USD         = 5_000_000   # $5M = massive alert
AGG_WINDOW_SEC   = 300         # 5 min aggregate window
AGG_THRESHOLD    = 2_000_000   # $2M in window = aggregate alert

BINANCE_WSS = "wss://fstream.binance.com/ws/btcusdt@forceOrder"


class LiquidationMonitor(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self._ws_task = None
        self._liq_longs_usd  = 0.0
        self._liq_shorts_usd = 0.0
        self._window_start   = time.time()
        self._last_agg_alert = 0
        self._seen_ids       = set()

    async def cog_load(self):
        logger.info("LiquidationMonitor: starting live WebSocket stream...")
        self._ws_task = asyncio.create_task(self._ws_loop())
        self.agg_loop.start()

    async def cog_unload(self):
        if self._ws_task:
            self._ws_task.cancel()
        self.agg_loop.cancel()

    # ── Live WebSocket ────────────────────────────────────────────────────────

    async def _ws_loop(self):
        """Persistent WebSocket connection to Binance liquidation stream"""
        await self.bot.wait_until_ready()
        await asyncio.sleep(5)

        backoff = 5
        while True:
            try:
                logger.info("LiquidationMonitor: connecting to Binance WS...")
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(BINANCE_WSS, heartbeat=30) as ws:
                        logger.info("LiquidationMonitor: WebSocket connected ✅")
                        backoff = 5  # reset on success
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._handle_liq(json.loads(msg.data))
                            elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                              aiohttp.WSMsgType.ERROR):
                                break
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.error(f"WS error: {e} — reconnecting in {backoff}s")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    async def _handle_liq(self, data: dict):
        """Process a single liquidation event from WebSocket"""
        order = data.get("o", {})
        uid   = order.get("T", 0)  # trade time as unique ID

        if uid in self._seen_ids:
            return
        self._seen_ids.add(uid)
        if len(self._seen_ids) > 2000:
            self._seen_ids = set(list(self._seen_ids)[-500:])

        side  = order.get("S", "BUY")   # BUY=short liq, SELL=long liq
        qty   = float(order.get("q", 0))
        price = float(order.get("ap", 0) or order.get("p", 0))
        usd   = qty * price

        if usd < MIN_LIQ_USD:
            return

        # Aggregate tracking
        if side == "SELL":
            self._liq_longs_usd += usd
        else:
            self._liq_shorts_usd += usd

        # Build and send alert image card
        await self._send_liq_alert(side, qty, price, usd)

    async def _send_liq_alert(self, side: str, qty: float, price: float, usd: float):
        guilds = [dict(g) for g in db.get_all_guilds()]
        channels = [
            self.bot.get_channel(g["liquidation_channel_id"])
            for g in guilds
            if g.get("liquidation_channel_id")
        ]
        channels = [c for c in channels if c]
        if not channels:
            return

        is_long_liq = side == "SELL"
        color  = 0xFF4560 if is_long_liq else 0x00E396
        label  = "🔴 LONG LIQUIDATED" if is_long_liq else "🟢 SHORT LIQUIDATED"
        impact = ""
        if usd >= MEGA_USD:
            impact = f"🚨 MEGA LIQUIDATION — ${usd/1_000_000:.2f}M"
        elif usd >= WHALE_USD:
            impact = f"🐋 WHALE LIQUIDATION — ${usd/1_000:.0f}K"

        embed = discord.Embed(
            title=f"💥 {label}",
            color=color,
            timestamp=discord.utils.utcnow()
        )
        embed.add_field(
            name="BTC/USDT Perpetual",
            value=(
                f"```\n"
                f"Type  : {'Long' if is_long_liq else 'Short'} Liquidated\n"
                f"Price : ${price:,.2f}\n"
                f"Size  : {qty:.4f} BTC\n"
                f"Value : ${usd:,.0f}\n"
                f"```"
            ),
            inline=False
        )
        if impact:
            embed.add_field(name="⚠️ Impact", value=impact, inline=False)

        direction_hint = (
            "⚡ Long squeeze — watch for short opportunity"
            if is_long_liq else
            "⚡ Short squeeze — watch for long opportunity"
        )
        embed.set_footer(text=direction_hint)

        for ch in channels:
            try:
                await ch.send(embed=embed)
            except Exception as e:
                logger.error(f"Liq send error: {e}")

    # ── Aggregate Window (every 5 min) ────────────────────────────────────────

    @tasks.loop(seconds=AGG_WINDOW_SEC)
    async def agg_loop(self):
        now = time.time()
        total = self._liq_longs_usd + self._liq_shorts_usd

        if total >= AGG_THRESHOLD and now - self._last_agg_alert > AGG_WINDOW_SEC:
            guilds = [dict(g) for g in db.get_all_guilds()]
            channels = [
                self.bot.get_channel(g["liquidation_channel_id"])
                for g in guilds if g.get("liquidation_channel_id")
            ]
            channels = [c for c in channels if c]

            embed = discord.Embed(
                title="🚨 MASS LIQUIDATION EVENT — BTC/USDT",
                color=0xFF6600,
                timestamp=discord.utils.utcnow()
            )
            embed.add_field(
                name="Last 5 Minutes",
                value=(
                    f"```\n"
                    f"Longs Wiped : ${self._liq_longs_usd/1000:,.0f}K\n"
                    f"Shorts Wiped: ${self._liq_shorts_usd/1000:,.0f}K\n"
                    f"Total       : ${total/1000:,.0f}K\n"
                    f"Pressure    : {'SHORT ↓' if self._liq_longs_usd > self._liq_shorts_usd else 'LONG ↑'}\n"
                    f"```"
                ), inline=False
            )
            net = self._liq_longs_usd - self._liq_shorts_usd
            embed.set_footer(
                text=f"Net: {'${:.0f}K more longs wiped'.format(abs(net)/1000) if net > 0 else '${:.0f}K more shorts wiped'.format(abs(net)/1000)}"
            )

            for ch in channels:
                try:
                    await ch.send(embed=embed)
                except Exception as e:
                    logger.error(f"Agg liq error: {e}")

            self._last_agg_alert = now

        # Reset window
        self._liq_longs_usd  = 0.0
        self._liq_shorts_usd = 0.0

    @agg_loop.before_loop
    async def before_agg(self):
        await self.bot.wait_until_ready()


async def setup(bot):
    await bot.add_cog(LiquidationMonitor(bot))
