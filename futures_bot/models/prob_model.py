from typing import Dict
from math import exp

LABELS = ["LONG", "SHORT", "NEUTRAL", "NO_TRADE"]


class ProbModel:
    """
    Simple linear + softmax model over DOM features.
    Produces probabilities for four classes.
    """

    def __init__(self, weights: Dict[str, float] | None = None, spread_penalty: float = 1.0):
        self.w = weights or {
            "imbalance": 1.2,
            "depth_pressure": 1.0,
            "depth_ratio": 0.4,
            "spread": -0.6,
            "bid_top": 0.05,
            "ask_top": -0.05,
        }
        self.spread_penalty = spread_penalty

    def _score(self, f: Dict[str, float]) -> Dict[str, float]:
        long_signal = (
            self.w["imbalance"] * f["imbalance"] +
            self.w["depth_pressure"] * f["depth_pressure"] +
            self.w["depth_ratio"] * min(3.0, f["depth_ratio"]) +
            self.w["spread"] * f["spread"] +
            self.w["bid_top"] * f["bid_top"]
        )

        short_signal = (
            -self.w["imbalance"] * f["imbalance"] +
            -self.w["depth_pressure"] * f["depth_pressure"] +
            -self.w["depth_ratio"] * min(3.0, f["depth_ratio"]) +
            self.w["spread"] * f["spread"] -
            self.w["ask_top"] * f["ask_top"]
        )

        neutral_signal = -0.5 * (abs(f["imbalance"]) + abs(f["depth_pressure"])) + (-0.2 * f["spread"])

        no_trade_signal = -0.3 + (f["spread"] * self.spread_penalty) + (0.2 * (1.0 - abs(f["imbalance"])))

        return {
            "LONG": long_signal,
            "SHORT": short_signal,
            "NEUTRAL": neutral_signal,
            "NO_TRADE": no_trade_signal,
        }

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = self._score(features)
        exps = {k: exp(max(-20.0, min(20.0, v))) for k, v in scores.items()}
        s = sum(exps.values())
        if s == 0:
            return {k: 0.25 for k in LABELS}
        return {k: exps[k] / s for k in LABELS}
