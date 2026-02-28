"""
CryptoQuant Signal Bot - Main Entry Point
"""

# ════════════════════════════════════════════════════════════════════════════
# WINDOWS DNS FIX — must be FIRST, before any other imports
# aiodns (used by aiohttp by default) cannot contact DNS on many Windows
# machines. We forcibly remove aiodns from aiohttp's awareness so it falls
# back to the threaded (system) resolver, which works perfectly on Windows.
# ════════════════════════════════════════════════════════════════════════════
import sys

# Remove aiodns from sys.modules so aiohttp thinks it's not installed
for mod in list(sys.modules.keys()):
    if 'aiodns' in mod:
        del sys.modules[mod]

# Block aiodns from being imported at all
import importlib
_real_import = importlib.import_module
def _blocked_import(name, *args, **kwargs):
    if 'aiodns' in str(name):
        raise ImportError(f"aiodns blocked for Windows DNS fix")
    return _real_import(name, *args, **kwargs)

# Patch aiohttp resolver to only know about ThreadedResolver
import aiohttp.resolver as _resolver
_resolver.aiodns = None  # type: ignore
_resolver.DefaultResolver = _resolver.ThreadedResolver

# Also patch the connector default
import aiohttp
original_tcp_connector_init = aiohttp.TCPConnector.__init__

def _patched_connector_init(self, *args, **kwargs):
    if 'resolver' not in kwargs:
        kwargs['resolver'] = _resolver.ThreadedResolver()
    original_tcp_connector_init(self, *args, **kwargs)

aiohttp.TCPConnector.__init__ = _patched_connector_init
# ════════════════════════════════════════════════════════════════════════════

import asyncio
import logging
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands

load_dotenv()

# Initialize database for multi-server support
import db
db.init_db()

# ── UTF-8 safe logging (fixes emoji UnicodeEncodeError on Windows cp1252) ──
import io
_utf8_stream = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler(_utf8_stream)
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
            logger.info(f'✅ Loaded cog: {cog}')
        except ExtensionFailed as e:
            # Unwrap the inner cause for a cleaner diagnosis
            cause = e.original if hasattr(e, 'original') else e
            if isinstance(cause, CommandRegistrationError):
                logger.error(
                    f'❌ Failed to load {cog}: duplicate command name "{cause.name}". '
                    f'Another cog already registered a command called "{cause.name}". '
                    f'Rename one of them in the cog file to fix this.'
                )
            else:
                logger.error(f'❌ Failed to load cog {cog}:\n{traceback.format_exc()}')
        except Exception:
            logger.error(f'❌ Failed to load cog {cog}:\n{traceback.format_exc()}')

@bot.event
async def setup_hook():
    """
    setup_hook runs BEFORE on_ready and is the correct place to load extensions.
    Loading here guarantees all commands (including !setup) are registered
    before the bot connects to the gateway.
    """
    await load_cogs()

@bot.event
async def on_ready():
    logger.info(f'Bot connected as {bot.user} (ID: {bot.user.id})')
    logger.info(f'Connected to {len(bot.guilds)} guilds')
    # Register all current guilds in DB (in case bot was offline when added)
    for guild in bot.guilds:
        db.upsert_guild(guild.id, guild.name)
    await bot.change_presence(activity=discord.Activity(
        type=discord.ActivityType.watching,
        name="📊 Scanning 1000 Markets..."
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
        logger.error("DISCORD_TOKEN not found in .env file!")
        exit(1)
    bot.run(token, log_handler=None)
