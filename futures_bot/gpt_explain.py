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
    """
    Returns a one-line explanation using GPT if enabled and openai is available.
    Otherwise returns empty string.
    """
    if not CONFIG.llm.enable_explanations:
        return ""

    if openai is None:
        return "[explain disabled: openai package not installed]"

    try:
        client = openai.OpenAI(api_key=CONFIG.llm.api_key)

        prompt = (
            "You are a trading microstructure analyst.\n"
            "Given DOM features and class probabilities, explain in one short line "
            "why the chosen action makes sense. Avoid hype, be factual.\n\n"
            f"features = {features}\n"
            f"probs = {probs}\n"
            f"action = {action}\n"
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
