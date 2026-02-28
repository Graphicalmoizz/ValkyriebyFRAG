"""
CryptoQuant Signal Bot - Main Entry Point
"""

import asyncio
import logging
import os
import sys
import io

from dotenv import load_dotenv
import discord
from discord.ext import commands

# load_dotenv() works locally (reads .env file)
# On Railway it does nothing — Railway injects variables directly into os.getenv()
load_dotenv()

# Initialize database for multi-server support
import db
db.init_db()

# ── UTF-8 safe logging ──
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('CryptoSignalBot')

intents = discord.Intents.default()
intents.message_content = True
intents.guilds = True

COGS = ['cogs.signal_engine', 'cogs.liquidation_monitor', 'cogs.admin', 'cogs.ai_engine', 'cogs.ml_trainer']

bot = commands.Bot(command_prefix='!', intents=intents)

async def load_cogs():
    """Load all cogs, with helpful error messages for common failures."""
    import traceback
    from discord.ext.commands.errors import CommandRegistrationError, ExtensionFailed

    for cog in COGS:
        try:
            await bot.load_extension(cog)
            logger.info(f'Loaded cog: {cog}')
        except ExtensionFailed as e:
            cause = e.original if hasattr(e, 'original') else e
            if isinstance(cause, CommandRegistrationError):
                logger.error(
                    f'Failed to load {cog}: duplicate command name "{cause.name}". '
                    f'Rename one of them in the cog file to fix this.'
                )
            else:
                logger.error(f'Failed to load cog {cog}:\n{traceback.format_exc()}')
        except Exception:
            logger.error(f'Failed to load cog {cog}:\n{traceback.format_exc()}')

@bot.event
async def setup_hook():
    await load_cogs()

@bot.event
async def on_ready():
    logger.info(f'Bot connected as {bot.user} (ID: {bot.user.id})')
    logger.info(f'Connected to {len(bot.guilds)} guilds')
    for guild in bot.guilds:
        db.upsert_guild(guild.id, guild.name)
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching,
        name="Scanning 1000 Markets..."
    ))

@bot.event
async def on_guild_join(guild):
    logger.info(f'Joined guild: {guild.name} (ID: {guild.id})')
    db.upsert_guild(guild.id, guild.name)

@bot.event
async def on_guild_remove(guild):
    logger.info(f'Removed from guild: {guild.name} (ID: {guild.id})')
    db.delete_guild(guild.id)

if __name__ == '__main__':
    token = os.getenv('DISCORD_TOKEN')
    if not token:
        logger.error("DISCORD_TOKEN not found! Make sure it is set in Railway Variables!")
        exit(1)
    logger.info("DISCORD_TOKEN found! Starting bot...")
    bot.run(token, log_handler=None)
