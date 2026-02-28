"""
Signal Card Builder
Creates beautiful Discord embeds for signals
Blue color scheme with different shades for different states
"""
import discord
from datetime import datetime, timezone

# Color palette (blues)
COLORS = {
    "A+_LONG":   0x1565C0,   # Deep Blue
    "B+_LONG":   0x1E88E5,   # Medium Blue
    "C+_LONG":   0x64B5F6,   # Light Blue
    "A+_SHORT":  0x880E4F,   # Deep Rose
    "B+_SHORT":  0xC2185B,   # Medium Rose
    "C+_SHORT":  0xF06292,   # Light Rose
    "TP_HIT":    0x0D47A1,   # Dark Blue - TP Hit
    "SL_HIT":    0x37474F,   # Dark Grey - SL Hit
    "SCALP":     0x0277BD,   # Cyan Blue
    "DAY":       0x283593,   # Indigo
    "SWING":     0x1A237E,   # Navy
    "LIQ":       0xB71C1C,   # Dark Red for liquidations
    "ALERT":     0xFF6F00,   # Amber for alerts
}

GRADE_EMOJI = {"A+": "🏆", "B+": "⭐", "C+": "✨"}
TYPE_EMOJI  = {"scalp": "⚡", "day": "📈", "swing": "🌊"}
DIR_EMOJI   = {"LONG": "🟢 LONG", "SHORT": "🔴 SHORT"}


def format_price(price: float) -> str:
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.4f}"
    elif price >= 0.001:
        return f"{price:.6f}"
    else:
        return f"{price:.8f}"


def build_signal_card(signal: dict) -> discord.Embed:
    sym   = signal["symbol"]
    base  = sym.replace("USDT", "")
    grade = signal["grade"]
    direction = signal["direction"]
    trade_type = signal["trade_type"]

    color_key = f"{grade}_{direction}"
    color = COLORS.get(color_key, COLORS["B+_LONG"])

    # Title
    g_emoji = GRADE_EMOJI.get(grade, "")
    t_emoji = TYPE_EMOJI.get(trade_type, "")
    d_text  = DIR_EMOJI.get(direction, direction)
    title   = f"{g_emoji} {t_emoji} {base}/USDT — {d_text}"

    type_label = trade_type.upper()
    embed = discord.Embed(
        title=title,
        color=color,
        timestamp=datetime.now(timezone.utc)
    )

    # Header info
    embed.add_field(
        name="📊 Signal Info",
        value=(
            f"```\n"
            f"Grade     : {grade} (Score: {signal['score']}/100)\n"
            f"Type      : {type_label}\n"
            f"Direction : {direction}\n"
            f"Leverage  : {signal['leverage']}x (Suggested)\n"
            f"```"
        ),
        inline=False
    )

    # Entry / SL
    embed.add_field(
        name="🎯 Entry & Stop Loss",
        value=(
            f"```\n"
            f"Entry : {format_price(signal['entry'])}\n"
            f"SL    : {format_price(signal['sl'])}\n"
            f"```"
        ),
        inline=True
    )

    # TPs
    tp_lines = []
    for i, tp in enumerate(signal["tps"], 1):
        tp_lines.append(f"TP{i}   : {format_price(tp)}")
    embed.add_field(
        name="💰 Take Profits",
        value="```\n" + "\n".join(tp_lines) + "\n```",
        inline=True
    )

    # Market data
    funding_str = f"{signal['funding_rate'] * 100:.4f}%"
    embed.add_field(
        name="📉 Market Data",
        value=(
            f"```\n"
            f"RSI(14)     : {signal['rsi14']}\n"
            f"Vol Spike   : {signal['vol_ratio']}x\n"
            f"Funding Rate: {funding_str}\n"
            f"BTC Delta   : {signal['outperform']:+.2f}%\n"
            f"```"
        ),
        inline=False
    )

    # Liquidation zones
    lz = signal.get("liq_zones", {})
    if lz:
        embed.add_field(
            name="💥 Liquidity Zones",
            value=(
                f"```\n"
                f"Liq Above : {format_price(lz.get('liq_zone_above', 0))}\n"
                f"Resistance: {format_price(lz.get('resistance', 0))}\n"
                f"Pivot     : {format_price(lz.get('pivot', 0))}\n"
                f"Support   : {format_price(lz.get('support', 0))}\n"
                f"Liq Below : {format_price(lz.get('liq_zone_below', 0))}\n"
                f"```"
            ),
            inline=True
        )

    # Confluences
    confluences = signal.get("confluences", [])
    if confluences:
        conf_text = "\n".join(confluences[:8])  # max 8 lines
        embed.add_field(
            name="🔬 Confluences",
            value=conf_text,
            inline=False
        )

    # Patterns
    patterns = signal.get("patterns", [])
    if patterns:
        embed.add_field(
            name="🕯️ Candle Patterns",
            value="  ".join(patterns),
            inline=False
        )

    # Footer
    embed.set_footer(
        text=f"CryptoQuant Bot • {base}/USDT Futures • Not financial advice",
        icon_url="https://cdn.discordapp.com/emojis/0.png"
    )

    return embed


def build_tp_hit_card(original_embed: discord.Embed, tp_number: int, tp_price: float, symbol: str) -> discord.Embed:
    """Build updated card when TP is hit"""
    base = symbol.replace("USDT", "")
    embed = discord.Embed(
        title=f"✅ TP{tp_number} HIT — {base}/USDT",
        description=f"**Take Profit {tp_number}** reached at `{format_price(tp_price)}`\n🎉 Lock in profits & move SL to entry!",
        color=COLORS["TP_HIT"],
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text=f"CryptoQuant Bot • {base}/USDT Futures")
    return embed


def build_sl_hit_card(symbol: str, sl_price: float) -> discord.Embed:
    """Build card when SL is hit"""
    base = symbol.replace("USDT", "")
    embed = discord.Embed(
        title=f"❌ SL HIT — {base}/USDT",
        description=f"Stop loss triggered at `{format_price(sl_price)}`\nRisk managed. Analyzing next setup...",
        color=COLORS["SL_HIT"],
        timestamp=datetime.now(timezone.utc)
    )
    embed.set_footer(text=f"CryptoQuant Bot • {base}/USDT Futures")
    return embed


def build_liquidation_alert(data: dict) -> discord.Embed:
    """Build BTC liquidation alert embed"""
    side = data.get("side", "BUY")
    qty  = float(data.get("origQty", 0))
    price = float(data.get("price", 0))
    usd_val = qty * price

    liq_type = "🐋 LONG LIQUIDATED" if side == "SELL" else "🐳 SHORT LIQUIDATED"
    color = COLORS["LIQ"]

    embed = discord.Embed(
        title=f"💥 {liq_type}",
        color=color,
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(
        name="BTC/USDT Liquidation",
        value=(
            f"```\n"
            f"Type  : {'Long' if side == 'SELL' else 'Short'} Liquidation\n"
            f"Price : {format_price(price)}\n"
            f"Size  : {qty:.4f} BTC\n"
            f"Value : ${usd_val:,.0f}\n"
            f"```"
        ),
        inline=False
    )
    if usd_val > 1_000_000:
        embed.add_field(name="⚠️ Whale Alert", value=f"${usd_val/1_000_000:.2f}M liquidation detected!", inline=False)
    embed.set_footer(text="CryptoQuant Bot • BTC Liquidation Monitor")
    return embed


def build_massive_liq_alert(total_longs_usd: float, total_shorts_usd: float) -> discord.Embed:
    """Aggregate liquidation alert"""
    embed = discord.Embed(
        title="🚨 MASSIVE LIQUIDATION EVENT — BTC/USDT",
        color=COLORS["ALERT"],
        timestamp=datetime.now(timezone.utc)
    )
    embed.add_field(
        name="📊 Last 5 Minutes",
        value=(
            f"```\n"
            f"Longs Wiped : ${total_longs_usd/1_000:,.0f}K\n"
            f"Shorts Wiped: ${total_shorts_usd/1_000:,.0f}K\n"
            f"Net Flow    : {'SHORT PRESSURE 🔴' if total_longs_usd > total_shorts_usd else 'LONG PRESSURE 🟢'}\n"
            f"```"
        ),
        inline=False
    )
    embed.set_footer(text="CryptoQuant Bot • Liquidation Tracker")
    return embed
