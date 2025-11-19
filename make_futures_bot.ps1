# make_futures_bot.ps1
# Creates the full futures_bot project with:
# - DOM analysis and probabilities
# - Console UI with percentages + red/green/yellow/grey lights
# - Optional GPT explanation text
# - Auto-trading toggle (dry-run by default, broker-agnostic)

$root = "futures_bot"
New-Item -ItemType Directory -Path $root -Force | Out-Null
$dirs = @(
    "$root\adapters",
    "$root\features",
    "$root\models",
    "$root\decision",
    "$root\utils",
    "$root\trading"
)
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Path $d -Force | Out-Null
}

# --- __init__ files ---
"" | Set-Content -Path "$root\adapters\__init__.py" -Encoding UTF8
"" | Set-Content -Path "$root\features\__init__.py" -Encoding UTF8
"" | Set-Content -Path "$root\models\__init__.py" -Encoding UTF8
"" | Set-Content -Path "$root\decision\__init__.py" -Encoding UTF8
"" | Set-Content -Path "$root\utils\__init__.py" -Encoding UTF8
"" | Set-Content -Path "$root\trading\__init__.py" -Encoding UTF8

# --- config.py ---
@"
from dataclasses import dataclass

@dataclass
class Symbols:
    symbol: str = "ESZ5"          # set your futures symbol
    venue: str = "CME"

@dataclass
class Depth:
    levels: int = 10              # DOM levels
    poll_ms: int = 200            # polling interval in ms
    price_tick: float = 0.25      # tick size of the contract

@dataclass
class Safety:
    # Master switch for any live order sending
    live_trading: bool = False
    # Whether to act automatically on signals (when live_trading is True)
    auto_trading: bool = False
    max_notional: float = 0.0

@dataclass
class LLM:
    enable_explanations: bool = False
    provider: str = "openai"
    api_key: str = "TODO_OPENAI_API_KEY"  # TODO: replace with your ChatGPT API key

@dataclass
class DeskAPI:
    api_key: str = "TODO_TRADING_DESK_API_KEY"  # TODO: replace with your desk API key
    base_url: str = "https://api.your-trading-desk.example"  # TODO: replace with real URL

@dataclass
class Config:
    symbols: Symbols = Symbols()
    depth: Depth = Depth()
    safety: Safety = Safety()
    llm: LLM = LLM()
    desk: DeskAPI = DeskAPI()

CONFIG = Config()
"@ | Set-Content -Path "$root\config.py" -Encoding UTF8

# --- adapters/base.py ---
@"
from typing import List, Dict, Protocol

OrderBookSide = List[Dict[str, float]]
OrderBook = Dict[str, OrderBookSide]  # {"bids": [...], "asks": [...]}

class MarketDataAdapter(Protocol):
    def get_order_book(self, symbol: str, levels: int) -> OrderBook:
        ...
"@ | Set-Content -Path "$root\adapters\base.py" -Encoding UTF8

# --- adapters/simulated.py ---
@"
import random
from typing import Dict, List
from adapters.base import OrderBook

class SimulatedAdapter:
    """
    Synthetic DOM generator for testing without any broker connection.
    """

    def __init__(self, mid: float = 5000.0, tick: float = 0.25, base_size: float = 20.0):
        self.mid = mid
        self.tick = tick
        self.base_size = base_size
        random.seed(42)

    def _gen_side(self, start_price: float, is_bid: bool, levels: int):
        side = []
        for i in range(levels):
            price = start_price - i * self.tick if is_bid else start_price + i * self.tick
            size = max(1.0, random.gauss(self.base_size, self.base_size * 0.5))
            side.append({"price": round(price, 2), "size": round(size, 2)})
        return side

    def get_order_book(self, symbol: str, levels: int) -> OrderBook:
        # simple random walk for mid price
        self.mid += random.gauss(0, self.tick * 0.2)
        best_bid = self.mid - self.tick / 2
        best_ask = self.mid + self.tick / 2
        bids = self._gen_side(best_bid, True, levels)
        asks = self._gen_side(best_ask, False, levels)
        return {"bids": bids, "asks": asks}
"@ | Set-Content -Path "$root\adapters\simulated.py" -Encoding UTF8

# --- adapters/tradingdesk.py ---
@"
from typing import Dict
import requests

from config import CONFIG
from adapters.base import OrderBook

class TradingDeskAdapter:
    """
    Template adapter that calls your trading desk's order-book endpoint.
    You must adjust 'path' and JSON field names to match your API.
    """

    def __init__(self):
        self.base = CONFIG.desk.base_url
        self.key = CONFIG.desk.api_key
        self.session = requests.Session()
        # Adjust auth scheme if your desk uses something else
        self.session.headers.update({
            "Authorization": f"Bearer {self.key}"
        })

    def _request(self, path: str, params: Dict) -> Dict:
        url = f"{self.base}{path}"
        resp = self.session.get(url, params=params, timeout=2.5)
        resp.raise_for_status()
        return resp.json()

    def get_order_book(self, symbol: str, levels: int) -> OrderBook:
        # TODO: Update 'path' and 'params' according to your desk's API documentation
        data = self._request(
            path="/v1/market/orderbook",  # TODO: replace with real path
            params={"symbol": symbol, "depth": levels}
        )

        bids_raw = data.get("bids", [])
        asks_raw = data.get("asks", [])

        bids = []
        for x in bids_raw:
            if isinstance(x, list) and len(x) >= 2:
                bids.append({"price": float(x[0]), "size": float(x[1])})
            else:
                bids.append({"price": float(x["price"]), "size": float(x["size"])})

        asks = []
        for x in asks_raw:
            if isinstance(x, list) and len(x) >= 2:
                asks.append({"price": float(x[0]), "size": float(x[1])})
            else:
                asks.append({"price": float(x["price"]), "size": float(x["size"])})

        bids.sort(key=lambda r: r["price"], reverse=True)
        asks.sort(key=lambda r: r["price"])

        return {"bids": bids[:levels], "asks": asks[:levels]}
"@ | Set-Content -Path "$root\adapters\tradingdesk.py" -Encoding UTF8

# --- features/dom.py ---
@"
from typing import Dict, List
from adapters.base import OrderBook

def _sum_sizes(side: List[Dict[str, float]]) -> float:
    return sum(lvl["size"] for lvl in side)

def _weighted_depth(side: List[Dict[str, float]]) -> float:
    # Weight nearer levels more strongly: 1 / (1 + level_index)
    total = 0.0
    for i, lvl in enumerate(side):
        weight = 1.0 / (1.0 + i)
        total += lvl["size"] * weight
    return total

def extract_dom_features(book: OrderBook) -> Dict[str, float]:
    bids = book["bids"]
    asks = book["asks"]

    if not bids or not asks:
        return {
            "imbalance": 0.0,
            "depth_pressure": 0.0,
            "spread": float("inf"),
            "bid_top": 0.0,
            "ask_top": 0.0,
            "depth_ratio": 1.0
        }

    bid_vol = _sum_sizes(bids)
    ask_vol = _sum_sizes(asks)

    total_vol = bid_vol + ask_vol
    if total_vol == 0:
        imbalance = 0.0
    else:
        imbalance = (bid_vol - ask_vol) / total_vol  # in [-1, 1]

    bid_top = bids[0]["size"]
    ask_top = asks[0]["size"]

    spread = max(0.0, asks[0]["price"] - bids[0]["price"])

    w_bid = _weighted_depth(bids)
    w_ask = _weighted_depth(asks)
    if (w_bid + w_ask) == 0:
        depth_pressure = 0.0
    else:
        depth_pressure = (w_bid - w_ask) / (w_bid + w_ask)

    if w_ask == 0:
        depth_ratio = 1.0
    else:
        depth_ratio = w_bid / w_ask

    return {
        "imbalance": imbalance,
        "depth_pressure": depth_pressure,
        "spread": spread,
        "bid_top": bid_top,
        "ask_top": ask_top,
        "depth_ratio": depth_ratio
    }
"@ | Set-Content -Path "$root\features\dom.py" -Encoding UTF8

# --- models/prob_model.py ---
@"
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
            "ask_top": -0.05
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
            "NO_TRADE": no_trade_signal
        }

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        scores = self._score(features)
        exps = {k: exp(max(-20.0, min(20.0, v))) for k, v in scores.items()}
        s = sum(exps.values())
        if s == 0:
            return {k: 0.25 for k in LABELS}
        return {k: exps[k] / s for k in LABELS}
"@ | Set-Content -Path "$root\models\prob_model.py" -Encoding UTF8

# --- decision/selector.py ---
@"
from typing import Dict, Tuple

def select_action(probs: Dict[str, float], min_conf: float = 0.35) -> Tuple[str, float]:
    """
    Select the label with highest probability.
    If its probability < min_conf, force NEUTRAL.
    """
    best_label, best_p = max(probs.items(), key=lambda kv: kv[1])
    if best_p < min_conf:
        return "NEUTRAL", best_p
    return best_label, best_p

def pretty_percentages(probs: Dict[str, float]) -> Dict[str, str]:
    return {k: f"{v*100:.1f}%" for k, v in probs.items()}
"@ | Set-Content -Path "$root\decision\selector.py" -Encoding UTF8

# --- utils/timeutils.py ---
@"
import time

def now_ms() -> int:
    return int(time.time() * 1000)
"@ | Set-Content -Path "$root\utils\timeutils.py" -Encoding UTF8

# --- trading/execution.py ---
@"
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
            print(f"[AUTO-TRADING DISABLED] Would place {action} order on {symbol} (size={self.cfg.max_size}, conf={conf:0.2f})")
            return

        # TODO: Implement actual order placement using your broker API here.
        # This section intentionally left as a placeholder to keep the bot broker-agnostic.
        print(f"[TODO ORDER] {action} {symbol} size={self.cfg.max_size} (conf={conf:0.2f})")
"@ | Set-Content -Path "$root\trading\execution.py" -Encoding UTF8

# --- ui_console.py ---
@"
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
"@ | Set-Content -Path "$root\ui_console.py" -Encoding UTF8

# --- gpt_explain.py ---
@"
# gpt_explain.py
#
# Optional short explanation per tick using OpenAI / ChatGPT.
# Controlled by CONFIG.llm.enable_explanations.

from typing import Dict

from config import CONFIG

try:
    import openai  # type: ignore
except ImportError:
    openai = None

def explain_tick(
    features: Dict[str, float],
    probs: Dict[str, float],
    action: str,
) -> str:
    \"""
    Returns a one-line explanation using GPT if enabled and openai is available.
    Otherwise returns empty string.
    \"""
    if not CONFIG.llm.enable_explanations:
        return ""

    if openai is None:
        return "[explain disabled: openai package not installed]"

    try:
        client = openai.OpenAI(api_key=CONFIG.llm.api_key)

        prompt = (
            "You are a trading microstructure analyst.\\n"
            "Given DOM features and class probabilities, explain in one short line "
            "why the chosen action makes sense. Avoid hype, be factual.\\n\\n"
            f"features = {features}\\n"
            f"probs = {probs}\\n"
            f"action = {action}\\n"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception as e:
        return f"[explain error: {e}]"
"@ | Set-Content -Path "$root\gpt_explain.py" -Encoding UTF8

# --- README.md ---
@"
# Futures DOM Prob Bot (Toggle Auto-Trading)

Python project that reads futures Depth of Market (DOM), computes features, and
outputs probabilities over four classes: LONG, SHORT, NEUTRAL, NO_TRADE.

Console UI:
- Percentages per class
- Traffic-light style indicators (green, red, yellow, grey)
- DOM metrics
- Optional GPT-generated explanation per tick

## Setup

- Python 3.10+
- pip install requests
- (optional) pip install openai

## Config

Edit `config.py`:

- Set your symbol and tick size.
- Wire `DeskAPI` to your trading desk.
- Auto-trading controls:

  - `Safety.live_trading`:
    - False -> never send real orders (only logs).
  - `Safety.auto_trading`:
    - False -> do not act automatically on signals (logs what would happen).
    - True  -> call the execution engine; real trading still requires
               live_trading=True and your broker API logic in `trading/execution.py`.

By default both are False, so the bot is non-executing.

## Run (simulated DOM)

```bash
python main.py --source sim --symbol ESZ5 --iters 20
