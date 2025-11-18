# ui_console.py
#
# Console rendering with percentages and traffic-light indicators.

from typing import Dict

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
GREY = "\033[90m"

COLOR_MAP = {
    "LONG": GREEN,
    "SHORT": RED,
    "NEUTRAL": YELLOW,
    "NO_TRADE": GREY,
}


def _light(color: str, on: bool) -> str:
    """
    Returns a colored light symbol.
    on=True -> solid circle, on=False -> hollow circle.
    """
    symbol = "●" if on else "○"
    return f"{color}{symbol}{RESET}"


def _percent(v: float) -> str:
    return f"{v * 100:5.1f}%"


def render_tick(
    i: int,
    symbol: str,
    probs: Dict[str, float],
    action: str,
    conf: float,
    feats: Dict[str, float],
    explanation: str | None = None,
    light_threshold: float = 0.40,
) -> None:
    """
    Pretty console block:
    - Percentages
    - Red/green/yellow/grey lights based on probability
    - Core DOM metrics
    - Optional GPT explanation text
    """
    segments = []
    for label in ["LONG", "SHORT", "NEUTRAL", "NO_TRADE"]:
        p = probs.get(label, 0.0)
        color = COLOR_MAP[label]
        on = p >= light_threshold
        light = _light(color, on)
        pct = _percent(p)
        segments.append(f"{light} {label:<8} {pct}")

    line1 = f"[{i}] {symbol}  action={action:<8} conf={conf:0.2f}"
    line2 = "  ".join(segments)
    line3 = (
        f"spread={feats['spread']:.2f}  "
        f"imb={feats['imbalance']:.2f}  "
        f"dp={feats['depth_pressure']:.2f}  "
        f"depth_ratio={feats['depth_ratio']:.2f}"
    )

    print(line1)
    print(line2)
    print(line3)

    if explanation:
        print(f"GPT: {explanation}")

    print("-" * 80)
