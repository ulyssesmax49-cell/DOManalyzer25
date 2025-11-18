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
- `pip install requests`
- (optional) `pip install openai`

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
               `live_trading=True` and your broker API logic in `trading/execution.py`.

By default both are False, so the bot is non-executing.

## Run (simulated DOM)

```bash
python main.py --source sim --symbol ESZ5 --iters 20
