"""
Admin Cog - Bot management and channel setup (Multi-Server)
"""
import discord
from discord.ext import commands
import logging
import config
import db

logger = logging.getLogger("Admin")


class Admin(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="setup")
    @commands.has_permissions(administrator=True)
    async def setup_channels(self, ctx):
        """Auto-create all required channels for this server"""
        await ctx.send("⏳ Setting up channels for this server, please wait...")
        guild = ctx.guild

        category = discord.utils.get(guild.categories, name="📊 CRYPTO SIGNALS")
        if not category:
            category = await guild.create_category("📊 CRYPTO SIGNALS")

        channels_config = [
            ("⚡-scalp-signals",    "scalp"),
            ("📈-day-trade-signals", "day"),
            ("🌊-swing-signals",     "swing"),
            ("💥-btc-liquidations",  "liquidation"),
            ("📋-bot-logs",          "log"),
        ]

        channel_ids = {}
        results = []
        for ch_name, key in channels_config:
            existing = discord.utils.get(guild.text_channels, name=ch_name)
            if existing:
                channel_ids[key] = existing.id
                results.append((key, existing.id, ch_name, "already existed"))
            else:
                ch = await guild.create_text_channel(ch_name, category=category)
                channel_ids[key] = ch.id
                results.append((key, ch.id, ch_name, "created ✅"))

        # Save to database — no .env needed
        db.save_guild_channels(
            guild_id   = guild.id,
            guild_name = guild.name,
            scalp_id   = channel_ids["scalp"],
            day_id     = channel_ids["day"],
            swing_id   = channel_ids["swing"],
            liq_id     = channel_ids["liquidation"],
            log_id     = channel_ids["log"],
        )

        embed = discord.Embed(
            title="✅ Server Setup Complete!",
            description=f"**{guild.name}** is now configured! Signals will start appearing automatically.",
            color=0x1565C0
        )
        for key, ch_id, ch_name, status in results:
            embed.add_field(
                name=f"#{ch_name} ({status})",
                value=f"<#{ch_id}>",
                inline=False
            )
        embed.set_footer(text="No need to edit any files — this server is ready to go!")
        await ctx.send(embed=embed)
        logger.info(f"Setup complete for guild: {guild.name} ({guild.id})")

    @setup_channels.error
    async def setup_error(self, ctx, error):
        if isinstance(error, commands.MissingPermissions):
            await ctx.send("❌ You need **Administrator** permission to run `!setup`.")
        else:
            await ctx.send(f"❌ Setup error: {error}")

    @commands.command(name="botinfo")
    async def info(self, ctx):
        cfg = db.get_guild_config(ctx.guild.id)
        embed = discord.Embed(title="🤖 CryptoQuant Bot Info", color=0x1565C0)
        embed.add_field(name="Top Coins",   value=f"{config.TOP_COINS_LIMIT}", inline=True)
        embed.add_field(name="Scalp Scan",  value=f"Every {config.SCAN_INTERVAL_SCALP}s", inline=True)
        embed.add_field(name="Day Scan",    value=f"Every {config.SCAN_INTERVAL_DAY}s", inline=True)
        embed.add_field(name="Swing Scan",  value=f"Every {config.SCAN_INTERVAL_SWING}s", inline=True)
        embed.add_field(name="Min Volume",  value=f"${config.MIN_VOLUME_24H_USD:,.0f}", inline=True)
        embed.add_field(name="Servers",     value=f"{len(ctx.bot.guilds)}", inline=True)

        if cfg:
            ch_map = {
                "Scalp":       cfg["scalp_channel_id"],
                "Day Trade":   cfg["day_channel_id"],
                "Swing":       cfg["swing_channel_id"],
                "Liquidation": cfg["liquidation_channel_id"],
            }
            ch_status = []
            for name, cid in ch_map.items():
                if cid and self.bot.get_channel(cid):
                    ch_status.append(f"✅ {name}")
                else:
                    ch_status.append(f"❌ {name} (not configured)")
            embed.add_field(name="Channel Status", value="\n".join(ch_status), inline=False)
        else:
            embed.add_field(name="⚠️ Not Set Up", value="Run `!setup` to configure this server.", inline=False)

        await ctx.send(embed=embed)

    @commands.command(name="active")
    async def active_trades(self, ctx):
        engine = self.bot.cogs.get("SignalEngine")
        guild_id = ctx.guild.id
        if not engine:
            await ctx.send("❌ Signal engine not loaded.")
            return
        cfg = db.get_guild_config(guild_id)
        if not cfg:
            await ctx.send("⚠️ This server isn't set up yet. Run `!setup` first.")
            return
        guild_channel_ids = {
            cfg["scalp_channel_id"],
            cfg["day_channel_id"],
            cfg["swing_channel_id"],
        }
        guild_trades = {
            sym: t for sym, t in engine.active_trades.items()
            if t.get("channel_id") in guild_channel_ids
        }
        if not guild_trades:
            await ctx.send("📭 No active trades for this server.")
            return
        lines = []
        for sym, t in guild_trades.items():
            tps_done  = t.get("tps_hit", 0)
            total_tps = len(t.get("tps", []))
            lines.append(f"**{sym}** | {t['direction']} | {t['grade']} | TPs: {tps_done}/{total_tps}")
        embed = discord.Embed(title="📊 Active Trades", description="\n".join(lines), color=0x1565C0)
        await ctx.send(embed=embed)

    @commands.command(name="stats")
    async def stats_cmd(self, ctx):
        engine = self.bot.cogs.get("SignalEngine")
        if not engine:
            await ctx.send("❌ Signal engine not loaded.")
            return
        stats = engine.ml.get_stats()
        if not stats:
            await ctx.send("📭 No trade history yet. Stats appear after trades close.")
            return
        embed = discord.Embed(title="📊 Signal Performance", color=0x1565C0)
        embed.add_field(name="Total Trades", value=str(stats["total"]), inline=True)
        embed.add_field(name="Win Rate",     value=f"{stats['win_rate']}%", inline=True)
        embed.add_field(name="Last Retrain", value=stats.get("last_trained") or "Not yet", inline=True)
        for grade, gs in stats.get("by_grade", {}).items():
            embed.add_field(name=f"Grade {grade}", value=f"{gs['win_rate']}% ({gs['total']} trades)", inline=True)
        await ctx.send(embed=embed)

    @commands.command(name="scan")
    @commands.has_permissions(administrator=True)
    async def force_scan(self, ctx, trade_type: str = "scalp"):
        if trade_type not in ["scalp", "day", "swing"]:
            await ctx.send("❌ Use: `!scan scalp`, `!scan day`, or `!scan swing`")
            return
        engine = self.bot.cogs.get("SignalEngine")
        if not engine:
            await ctx.send("❌ Signal engine not loaded.")
            return
        msg = await ctx.send(f"🔍 Forcing **{trade_type}** scan on {len(engine._valid_symbols)} symbols...")
        intervals = {"scalp": "5m", "day": "1h", "swing": "4h"}
        limits    = {"scalp": 100,  "day": 100,  "swing": 150}
        await engine._run_scan(trade_type, intervals[trade_type], limits[trade_type])
        await msg.edit(content=f"✅ **{trade_type}** scan done! Check your signal channels.")

    @commands.command(name="symbols")
    async def symbols_count(self, ctx):
        engine = self.bot.cogs.get("SignalEngine")
        count = len(engine._valid_symbols) if engine else 0
        await ctx.send(f"📊 Tracking **{count}** USDT perpetual futures from top {config.TOP_COINS_LIMIT} CMC coins.")

    @commands.command(name="removeserver")
    @commands.has_permissions(administrator=True)
    async def remove_server(self, ctx):
        """Remove this server's config from the database"""
        db.delete_guild(ctx.guild.id)
        await ctx.send("🗑️ This server's configuration has been removed. Run `!setup` to reconfigure.")

    @commands.command(name="help_bot", aliases=["commands"])
    async def help_cmd(self, ctx):
        embed = discord.Embed(title="📚 CryptoQuant Bot Commands", color=0x1565C0)
        for cmd, desc in [
            ("!setup",         "Create all signal channels for THIS server (Admin only)"),
            ("!botinfo",       "Show config and channel status for this server"),
            ("!scan [type]",   "Force scan: scalp / day / swing (Admin)"),
            ("!active",        "Show open trades for this server"),
            ("!stats",         "Win rate and ML stats"),
            ("!symbols",       "How many coins are tracked"),
            ("!thresholds",    "Show current AI-tuned signal thresholds"),
            ("!tune",          "Force AI auto-tune cycle now (Admin)"),
            ("!trainstatus",   "Show ML 24/7 training status"),
            ("!forcetrain",    "Force immediate ML training cycle (Admin)"),
            ("!removeserver",  "Remove this server's config (Admin)"),
        ]:
            embed.add_field(name=f"`{cmd}`", value=desc, inline=False)
        await ctx.send(embed=embed)


async def setup(bot):
    await bot.add_cog(Admin(bot))
