"""
Database Manager - SQLite-based multi-server config storage
Replaces .env channel IDs with per-guild database records
"""
import sqlite3
import logging
import os
from typing import Optional

logger = logging.getLogger("DB")

DB_PATH = "data/guilds.db"


def get_conn() -> sqlite3.Connection:
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist"""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS guild_config (
                guild_id              INTEGER PRIMARY KEY,
                guild_name            TEXT,
                scalp_channel_id      INTEGER DEFAULT 0,
                day_channel_id        INTEGER DEFAULT 0,
                swing_channel_id      INTEGER DEFAULT 0,
                liquidation_channel_id INTEGER DEFAULT 0,
                log_channel_id        INTEGER DEFAULT 0,
                setup_at              TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    logger.info("Database initialized")


def upsert_guild(guild_id: int, guild_name: str):
    """Register a guild if not already in DB"""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO guild_config (guild_id, guild_name)
            VALUES (?, ?)
            ON CONFLICT(guild_id) DO UPDATE SET guild_name=excluded.guild_name
        """, (guild_id, guild_name))
        conn.commit()


def save_guild_channels(guild_id: int, guild_name: str,
                        scalp_id: int, day_id: int, swing_id: int,
                        liq_id: int, log_id: int):
    """Save or update channel IDs for a guild"""
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO guild_config
                (guild_id, guild_name, scalp_channel_id, day_channel_id,
                 swing_channel_id, liquidation_channel_id, log_channel_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(guild_id) DO UPDATE SET
                guild_name=excluded.guild_name,
                scalp_channel_id=excluded.scalp_channel_id,
                day_channel_id=excluded.day_channel_id,
                swing_channel_id=excluded.swing_channel_id,
                liquidation_channel_id=excluded.liquidation_channel_id,
                log_channel_id=excluded.log_channel_id,
                updated_at=CURRENT_TIMESTAMP
        """, (guild_id, guild_name, scalp_id, day_id, swing_id, liq_id, log_id))
        conn.commit()


def get_guild_config(guild_id: int) -> Optional[sqlite3.Row]:
    """Get channel config for a guild. Returns None if not set up."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM guild_config WHERE guild_id = ?", (guild_id,)
        ).fetchone()
    return row


def get_all_guilds() -> list:
    """Get all configured guilds"""
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM guild_config").fetchall()
    return rows


def delete_guild(guild_id: int):
    """Remove guild config (called when bot is kicked)"""
    with get_conn() as conn:
        conn.execute("DELETE FROM guild_config WHERE guild_id = ?", (guild_id,))
        conn.commit()
    logger.info(f"Removed guild {guild_id} from database")
