from dataclasses import dataclass
from typing import Dict

from config import CONFIG


@dataclass
class ExecutionConfig:
    # Simple placeholder execution config
    max_size: float = 1.0


class ExecutionEngine:
    """
    Handles acting on signals.

    Behavior:
    - If CONFIG.safety.live_trading or CONFIG.safety.auto_trading is False:
        -> only logs what it WOULD do (no real trading).
    - To wire real trading, replace the TODO section with broker API calls.
    """

    def __init__(self, max_size: float | None = None):
        self.cfg = ExecutionConfig(max_size=max_size or ExecutionConfig.max_size)

    def handle_signal(
        self,
        symbol: str,
        action: str,
        conf: float,
        probs: Dict[str, float],
        feats: Dict[str, float],
    ) -> None:
        # Only act on LONG/SHORT
        if action not in ("LONG", "SHORT"):
            return

        if not CONFIG.safety.live_trading or not CONFIG.safety.auto_trading:
            print(
                f"[AUTO-TRADING DISABLED] Would place {action} order on {symbol} "
                f"(size={self.cfg.max_size}, conf={conf:0.2f})"
            )
            return

        # TODO: Implement actual order placement using your broker API here.
        # This section intentionally left as a placeholder to keep the bot broker-agnostic.
        print(f"[TODO ORDER] {action} {symbol} size={self.cfg.max_size} (conf={conf:0.2f})")
