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
