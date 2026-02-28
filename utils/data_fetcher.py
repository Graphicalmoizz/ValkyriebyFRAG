"""
Market Data Fetcher
Pulls OHLCV, OI, funding rate, liquidation data from Binance Futures
and top coins list from CoinMarketCap
"""
import asyncio
import aiohttp
import logging
import time
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger("DataFetcher")

CMC_BASE   = "https://pro-api.coinmarketcap.com/v1"
BINANCE_BASE = "https://fapi.binance.com"
BINANCE_SPOT = "https://api.binance.com"


class DataFetcher:
    def __init__(self, cmc_api_key: str, binance_key: str = "", binance_secret: str = ""):
        self.cmc_api_key = cmc_api_key
        self.binance_key = binance_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._top_coins_cache = []
        self._top_coins_ts = 0
        self._futures_symbols_cache = set()
        self._futures_symbols_ts = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            # Use ThreadedResolver to avoid aiodns DNS failures on Windows
            connector = aiohttp.TCPConnector(resolver=aiohttp.resolver.ThreadedResolver())
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    # ─── CMC Top Coins ──────────────────────────────────────────────────────
    async def get_top_coins(self, limit: int = 1000, refresh_hours: int = 6) -> list[str]:
        """Returns list of top N symbols by market cap"""
        now = time.time()
        if self._top_coins_cache and (now - self._top_coins_ts) < refresh_hours * 3600:
            return self._top_coins_cache

        if not self.cmc_api_key:
            logger.warning("No CMC API key - using default top coins")
            return self._top_coins_cache or []

        session = await self._get_session()
        url = f"{CMC_BASE}/cryptocurrency/listings/latest"
        params = {"limit": limit, "sort": "market_cap", "cryptocurrency_type": "coins"}
        headers = {"X-CMC_PRO_API_KEY": self.cmc_api_key}

        try:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    symbols = [coin["symbol"] for coin in data.get("data", [])]
                    self._top_coins_cache = symbols
                    self._top_coins_ts = now
                    logger.info(f"Refreshed CMC top {len(symbols)} coins")
                    return symbols
                else:
                    logger.error(f"CMC API error: {resp.status}")
        except Exception as e:
            logger.error(f"CMC fetch error: {e}")
        return self._top_coins_cache

    # ─── Binance Futures Symbols ─────────────────────────────────────────────
    async def get_futures_symbols(self) -> set[str]:
        """Get all USDT perpetual futures symbols"""
        now = time.time()
        if self._futures_symbols_cache and (now - self._futures_symbols_ts) < 3600:
            return self._futures_symbols_cache

        session = await self._get_session()
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/exchangeInfo") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    syms = {
                        s["symbol"]
                        for s in data.get("symbols", [])
                        if s.get("quoteAsset") == "USDT"
                        and s.get("contractType") == "PERPETUAL"
                        and s.get("status") == "TRADING"
                    }
                    self._futures_symbols_cache = syms
                    self._futures_symbols_ts = now
                    return syms
        except Exception as e:
            logger.error(f"Error fetching futures symbols: {e}")
        return self._futures_symbols_cache

    async def get_valid_futures_symbols(self, top_coins: list[str]) -> list[str]:
        """Return futures symbols that are in top coins list"""
        futures = await self.get_futures_symbols()
        valid = []
        for sym in top_coins:
            perp = f"{sym}USDT"
            if perp in futures:
                valid.append(perp)
        return valid

    # ─── OHLCV ───────────────────────────────────────────────────────────────
    async def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV klines from Binance Futures"""
        session = await self._get_session()
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/klines", params=params) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    df = pd.DataFrame(raw, columns=[
                        "open_time","open","high","low","close","volume",
                        "close_time","quote_vol","trades","taker_buy_base",
                        "taker_buy_quote","ignore"
                    ])
                    for col in ["open","high","low","close","volume","quote_vol","taker_buy_base","taker_buy_quote"]:
                        df[col] = pd.to_numeric(df[col])
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                    df.set_index("open_time", inplace=True)
                    return df
        except Exception as e:
            logger.error(f"Klines error {symbol} {interval}: {e}")
        return pd.DataFrame()

    # ─── Open Interest ────────────────────────────────────────────────────────
    async def get_open_interest(self, symbol: str) -> dict:
        """Fetch current OI"""
        session = await self._get_session()
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/openInterest", params={"symbol": symbol}) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"OI error {symbol}: {e}")
        return {}

    async def get_oi_history(self, symbol: str, period: str = "5m", limit: int = 50) -> pd.DataFrame:
        """OI historical data"""
        session = await self._get_session()
        params = {"symbol": symbol, "period": period, "limit": limit}
        try:
            async with session.get(f"{BINANCE_BASE}/futures/data/openInterestHist", params=params) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    if raw:
                        df = pd.DataFrame(raw)
                        df["sumOpenInterest"] = pd.to_numeric(df["sumOpenInterest"])
                        df["sumOpenInterestValue"] = pd.to_numeric(df["sumOpenInterestValue"])
                        return df
        except Exception as e:
            logger.error(f"OI history error {symbol}: {e}")
        return pd.DataFrame()

    # ─── Funding Rate ─────────────────────────────────────────────────────────
    async def get_funding_rate(self, symbol: str) -> float:
        """Get current funding rate"""
        session = await self._get_session()
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/premiumIndex", params={"symbol": symbol}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("lastFundingRate", 0))
        except Exception as e:
            logger.error(f"Funding rate error {symbol}: {e}")
        return 0.0

    # ─── Liquidation Data ────────────────────────────────────────────────────
    async def get_recent_liquidations(self, symbol: str = "BTCUSDT", limit: int = 20) -> list:
        """Get recent liquidation orders"""
        session = await self._get_session()
        params = {"symbol": symbol, "limit": limit}
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/allForceOrders", params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Liquidations error {symbol}: {e}")
        return []

    async def get_liquidation_zones(self, symbol: str, df: pd.DataFrame) -> dict:
        """Estimate major liquidation zones based on price levels"""
        if df.empty:
            return {}
        current_price = float(df["close"].iloc[-1])
        # Approximate liq zones using recent swing highs/lows
        recent = df.tail(50)
        resistance = float(recent["high"].max())
        support    = float(recent["low"].min())
        pivot      = (resistance + support + current_price) / 3

        # Typical liquidation clusters
        liq_above = current_price * 1.05  # 5% above
        liq_below = current_price * 0.95  # 5% below

        return {
            "support": round(support, 4),
            "resistance": round(resistance, 4),
            "pivot": round(pivot, 4),
            "liq_zone_above": round(liq_above, 4),
            "liq_zone_below": round(liq_below, 4),
        }

    # ─── Volume Profile ───────────────────────────────────────────────────────
    async def get_taker_buy_sell_ratio(self, symbol: str, period: str = "5m", limit: int = 20) -> pd.DataFrame:
        """Long/Short taker ratio"""
        session = await self._get_session()
        params = {"symbol": symbol, "period": period, "limit": limit}
        try:
            async with session.get(f"{BINANCE_BASE}/futures/data/takerlongshortRatio", params=params) as resp:
                if resp.status == 200:
                    raw = await resp.json()
                    if raw:
                        return pd.DataFrame(raw)
        except Exception as e:
            logger.error(f"Taker ratio error {symbol}: {e}")
        return pd.DataFrame()

    # ─── BTC Data ─────────────────────────────────────────────────────────────
    async def get_btc_change(self, interval: str = "15m", lookback: int = 4) -> float:
        """Get BTC % change over last N candles"""
        df = await self.get_klines("BTCUSDT", interval, lookback + 1)
        if df.empty:
            return 0.0
        start = float(df["close"].iloc[0])
        end   = float(df["close"].iloc[-1])
        return ((end - start) / start) * 100

    async def get_ticker_24h(self, symbol: str) -> dict:
        """24h ticker statistics"""
        session = await self._get_session()
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/ticker/24hr", params={"symbol": symbol}) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.error(f"Ticker 24h error {symbol}: {e}")
        return {}

    async def get_orderbook_imbalance(self, symbol: str, limit: int = 20) -> float:
        """Calculate bid/ask volume imbalance (-1 to 1, positive = bullish)"""
        session = await self._get_session()
        try:
            async with session.get(f"{BINANCE_BASE}/fapi/v1/depth", params={"symbol": symbol, "limit": limit}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bid_vol = sum(float(b[1]) for b in data.get("bids", []))
                    ask_vol = sum(float(a[1]) for a in data.get("asks", []))
                    total = bid_vol + ask_vol
                    if total == 0:
                        return 0.0
                    return (bid_vol - ask_vol) / total
        except Exception as e:
            logger.error(f"Orderbook error {symbol}: {e}")
        return 0.0
