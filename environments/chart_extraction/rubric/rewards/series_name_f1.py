def trimmed_name_set(names) -> set[str]:
    return {str(name).strip() for name in names if str(name).strip()}


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


async def series_name_f1_reward(state, info) -> float:
    parsed_answer = state.get("parsed_answer")
    if parsed_answer is None:
        return 0.0

    predicted_names = trimmed_name_set(item.name for item in parsed_answer.series)
    gold_names = trimmed_name_set(info.get("legend_names", []))
    return f1_score(predicted_names, gold_names)
