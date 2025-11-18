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
