from typing import List, Dict, Protocol

OrderBookSide = List[Dict[str, float]]
OrderBook = Dict[str, OrderBookSide]  # {"bids": [...], "asks": [...]}


class MarketDataAdapter(Protocol):
    def get_order_book(self, symbol: str, levels: int) -> OrderBook:
        ...
