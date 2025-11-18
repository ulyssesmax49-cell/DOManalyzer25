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
    return {k: f"{v * 100:.1f}%" for k, v in probs.items()}
