from ..state import RubricState


def f1_score(predicted: set[str], gold: set[str]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0

    true_positives = len(predicted & gold)
    if true_positives == 0:
        return 0.0

    precision = true_positives / len(predicted)
    recall = true_positives / len(gold)
    return 2 * precision * recall / (precision + recall)


def series_name_f1(state: RubricState, info) -> float:
    parsed_answer = state["parsed_answer"] if "parsed_answer" in state else None
    if parsed_answer is None:
        return 0.0

    predicted_names = {item.name for item in parsed_answer.series if item.name}
    gold_names = {name for name in info.get("legend_names", []) if name}
    return f1_score(predicted_names, gold_names)
