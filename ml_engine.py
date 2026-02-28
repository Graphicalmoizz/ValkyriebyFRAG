"""
ML Engine - Enhanced with Claude AI co-prediction
- Local sklearn ensemble (Gradient Boosting + Random Forest)
- Claude API consulted for borderline signals to boost accuracy
- Auto-retrains every 24h
- A+/B+ signals get priority queue over C+
"""
import json
import logging
import os
import pickle
import time
import asyncio
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger("MLEngine")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn not installed. ML scoring disabled.")

# Claude is consulted when local ML confidence is in this range (uncertain zone)
CLAUDE_CONSULT_MIN = 0.40
CLAUDE_CONSULT_MAX = 0.65

# Grade priority weights — A+ and B+ signals jump the queue
GRADE_PRIORITY = {"A+": 3, "B+": 2, "C+": 1}


class MLEngine:
    def __init__(self, model_path: str = "data/ml_model.pkl",
                 history_path: str = "data/trade_history.json"):
        self.model_path   = model_path
        self.history_path = history_path
        self.model        = None
        self.scaler       = None
        self.last_trained = 0
        self.trade_history = []
        self._bot = None  # injected after bot is ready
        os.makedirs("data", exist_ok=True)
        self._load_history()
        self._load_model()

    def set_bot(self, bot):
        """Inject bot reference so MLEngine can reach AIEngine cog"""
        self._bot = bot

    # ─── Persistence ─────────────────────────────────────────────────────────

    def _load_history(self):
        if os.path.exists(self.history_path):
            try:
                with open(self.history_path, "r") as f:
                    self.trade_history = json.load(f)
                logger.info(f"Loaded {len(self.trade_history)} historical trades")
            except Exception as e:
                logger.error(f"Error loading trade history: {e}")
                self.trade_history = []

    def _save_history(self):
        try:
            with open(self.history_path, "w") as f:
                json.dump(self.trade_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trade history: {e}")

    def _load_model(self):
        if not ML_AVAILABLE or not os.path.exists(self.model_path):
            return
        try:
            with open(self.model_path, "rb") as f:
                saved = pickle.load(f)
                self.model        = saved.get("model")
                self.scaler       = saved.get("scaler")
                self.last_trained = saved.get("timestamp", 0)
            logger.info("ML model loaded from disk")
        except Exception as e:
            # numpy/sklearn version mismatch (e.g. BitGenerator error after upgrade)
            # Delete the incompatible model so it rebuilds cleanly on next retrain
            logger.warning(f"ML model incompatible ({e}) — deleting and will retrain fresh")
            try:
                os.remove(self.model_path)
                logger.info("Deleted incompatible ML model — will rebuild after enough trades")
            except Exception:
                pass
            self.model  = None
            self.scaler = None

    def _save_model(self):
        if not ML_AVAILABLE or not self.model:
            return
        try:
            with open(self.model_path, "wb") as f:
                pickle.dump({
                    "model": self.model,
                    "scaler": self.scaler,
                    "timestamp": time.time()
                }, f)
            logger.info("ML model saved")
        except Exception as e:
            logger.error(f"Error saving ML model: {e}")

    # ─── Feature Extraction ──────────────────────────────────────────────────

    def extract_features(self, signal: dict, indicators: dict,
                         funding_rate: float, ob_imbalance: float) -> list:
        return [
            indicators.get("rsi14", 50),
            indicators.get("rsi7", 50),
            indicators.get("macd_hist", 0),
            indicators.get("stoch_k", 50),
            (indicators.get("vol_current", 1) / (indicators.get("vol_sma20", 1) + 1e-9)),
            1 if indicators.get("ema9", 0) > indicators.get("ema21", 1) else 0,
            1 if indicators.get("ema21", 0) > indicators.get("ema50", 1) else 0,
            abs(indicators.get("price", 1) - indicators.get("vwap", 1)) / (indicators.get("price", 1) + 1e-9),
            ob_imbalance,
            funding_rate * 1000,
            signal.get("outperform", 0),
            signal.get("score", 50),
            1 if signal.get("direction") == "LONG" else 0,
            {"scalp": 0, "day": 1, "swing": 2}.get(signal.get("trade_type", "scalp"), 0),
            1 if indicators.get("divergence_rsi") in ["bullish", "bearish"] else 0,
            len(indicators.get("patterns", [])),
        ]

    # ─── Local ML Prediction ─────────────────────────────────────────────────

    def _local_predict(self, signal: dict, indicators: dict,
                       funding_rate: float, ob_imbalance: float) -> float:
        """Returns 0-1 probability from local sklearn model"""
        if not ML_AVAILABLE or not self.model:
            return 0.5
        try:
            features = self.extract_features(signal, indicators, funding_rate, ob_imbalance)
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            return float(self.model.predict_proba(X_scaled)[0][1])
        except Exception as e:
            logger.error(f"Local ML prediction error: {e}")
            return 0.5

    # ─── Claude Co-Prediction ────────────────────────────────────────────────

    async def _claude_predict(self, signal: dict, indicators: dict,
                               funding_rate: float, ob_imbalance: float,
                               local_prob: float) -> float:
        """
        Ask Claude for a probability estimate when local ML is uncertain.
        Returns blended probability.
        """
        if not self._bot:
            return local_prob

        ai_engine = self._bot.cogs.get("AIEngine")
        if not ai_engine or not ai_engine.api_key:
            return local_prob

        system = (
            "You are a professional quantitative crypto trader. "
            "Given a futures signal's technical data, estimate the probability (0.0 to 1.0) "
            "that this trade will hit at least TP1. "
            "Respond with ONLY a single float number between 0.0 and 1.0. Nothing else."
        )

        user = f"""Signal:
Symbol: {signal.get('symbol')}
Direction: {signal.get('direction')}
Type: {signal.get('trade_type')}
Score: {signal.get('score')}/100
Grade: {signal.get('grade', 'unknown')}
Confluences: {', '.join(signal.get('confluences', []))}

Indicators:
RSI14: {indicators.get('rsi14', 50):.1f}
RSI7: {indicators.get('rsi7', 50):.1f}
MACD Hist: {indicators.get('macd_hist', 0):.4f}
Stoch K: {indicators.get('stoch_k', 50):.1f}
EMA9>EMA21: {indicators.get('ema9', 0) > indicators.get('ema21', 0)}
EMA21>EMA50: {indicators.get('ema21', 0) > indicators.get('ema50', 0)}
Volume vs avg: {indicators.get('vol_current', 1) / (indicators.get('vol_sma20', 1) + 1e-9):.2f}x
Funding Rate: {funding_rate:.5f}
OB Imbalance: {ob_imbalance:.3f}
Patterns: {', '.join(indicators.get('patterns', [])) or 'None'}
Divergence: {indicators.get('divergence_rsi', 'none')}
Local ML confidence: {local_prob:.2f}

Probability TP1 will be hit (0.0-1.0):"""

        try:
            response = await ai_engine._call_claude(system, user, max_tokens=10)
            if response:
                claude_prob = float(response.strip())
                claude_prob = max(0.0, min(1.0, claude_prob))
                # Blend: 60% Claude, 40% local ML when in uncertain zone
                blended = (claude_prob * 0.6) + (local_prob * 0.4)
                logger.debug(f"Claude co-predict {signal.get('symbol')}: local={local_prob:.2f} claude={claude_prob:.2f} blended={blended:.2f}")
                return blended
        except Exception as e:
            logger.warning(f"Claude co-predict failed: {e}")

        return local_prob

    # ─── Main Predict (sync + async versions) ────────────────────────────────

    def predict_success_probability(self, signal: dict, indicators: dict,
                                    funding_rate: float, ob_imbalance: float) -> float:
        """Sync version — used during scan (Claude consult happens async in send pipeline)"""
        return self._local_predict(signal, indicators, funding_rate, ob_imbalance)

    async def predict_with_claude(self, signal: dict, indicators: dict,
                                   funding_rate: float, ob_imbalance: float) -> float:
        """
        Full async prediction — local ML first, Claude consulted if uncertain.
        Called before deciding whether to send a signal.
        """
        local_prob = self._local_predict(signal, indicators, funding_rate, ob_imbalance)

        # Only call Claude for uncertain signals (saves API cost)
        if CLAUDE_CONSULT_MIN <= local_prob <= CLAUDE_CONSULT_MAX:
            return await self._claude_predict(signal, indicators, funding_rate, ob_imbalance, local_prob)

        return local_prob

    # ─── Grade Priority Queue ────────────────────────────────────────────────

    @staticmethod
    def sort_signals_by_priority(signals: list) -> list:
        """
        Sort signals so A+ comes first, then B+, then C+.
        Within same grade, higher score wins.
        """
        return sorted(
            signals,
            key=lambda s: (GRADE_PRIORITY.get(s.get("grade", "C+"), 1), s.get("score", 0)),
            reverse=True
        )

    # ─── Record & Retrain ────────────────────────────────────────────────────

    def record_trade(self, signal: dict, indicators: dict,
                     funding_rate: float, ob_imbalance: float, outcome: str):
        features = self.extract_features(signal, indicators, funding_rate, ob_imbalance)
        record = {
            "timestamp":  datetime.utcnow().isoformat(),
            "symbol":     signal.get("symbol"),
            "grade":      signal.get("grade"),
            "score":      signal.get("score"),
            "direction":  signal.get("direction"),
            "trade_type": signal.get("trade_type"),
            "outcome":    1 if outcome == "win" else 0,
            "features":   features,
        }
        self.trade_history.append(record)
        self._save_history()
        logger.info(f"Recorded trade: {signal.get('symbol')} -> {outcome}")

    def retrain(self, min_samples: int = 50) -> bool:
        if not ML_AVAILABLE:
            return False
        if len(self.trade_history) < min_samples:
            logger.info(f"Not enough data to retrain ({len(self.trade_history)}/{min_samples})")
            return False
        try:
            X = np.array([t["features"] for t in self.trade_history])
            y = np.array([t["outcome"] for t in self.trade_history])

            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
            rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
            ensemble = VotingClassifier(estimators=[("gb", gb), ("rf", rf)], voting="soft")
            ensemble.fit(X_scaled, y)

            cv_scores = cross_val_score(ensemble, X_scaled, y, cv=min(5, len(y) // 10 + 1))
            logger.info(f"ML retrain complete. CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            self.model = ensemble
            self.last_trained = time.time()
            self._save_model()
            return True
        except Exception as e:
            logger.error(f"ML retrain error: {e}")
            return False

    def get_stats(self) -> dict:
        if not self.trade_history:
            return {}
        total = len(self.trade_history)
        wins  = sum(1 for t in self.trade_history if t["outcome"] == 1)
        by_grade = {}
        for t in self.trade_history:
            g = t.get("grade", "?")
            if g not in by_grade:
                by_grade[g] = {"wins": 0, "total": 0}
            by_grade[g]["total"] += 1
            if t["outcome"] == 1:
                by_grade[g]["wins"] += 1
        return {
            "total":      total,
            "wins":       wins,
            "win_rate":   round(wins / total * 100, 1),
            "by_grade":   {g: {"win_rate": round(v["wins"] / v["total"] * 100, 1), "total": v["total"]} for g, v in by_grade.items()},
            "last_trained": datetime.fromtimestamp(self.last_trained).isoformat() if self.last_trained else None,
        }
