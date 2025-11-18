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
            path="/v1/market/orderbook",  # replace with real path
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
