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
    # Replace with your actual OpenAI / ChatGPT API key locally
    api_key: str = "TODO_OPENAI_API_KEY"


@dataclass
class DeskAPI:
    api_key: str = "TODO_TRADING_DESK_API_KEY"  # replace with your desk API key
    base_url: str = "https://api.your-trading-desk.example"  # replace with real URL


@dataclass
class Config:
    symbols: Symbols = Symbols()
    depth: Depth = Depth()
    safety: Safety = Safety()
    llm: LLM = LLM()
    desk: DeskAPI = DeskAPI()


CONFIG = Config()
