from typing import List, AnyStr


def calculate_accuracy(labels: List[int], answers: List[AnyStr], models_name: str) -> float:
    correct = 0
    total = len(labels)
    not_processed = 0
    for label, answer in zip(labels, answers):
        try:
            if ": answer:" in answer:
                pred = int(answer.split(": answer:")[1].strip())
            elif ": " in answer:
                pred = int(answer.split(": ")[1].strip())
            elif "internvl3_5-8b" in models_name:
                pred = int(answer[0])
            else:
                pred = int(answer.strip())
        except ValueError:
            # need to implement it better to handle cases where the model's answer is not a valid integer
            try:
                pred = int(answer.strip().split()[0])
            except Exception:
                not_processed += 1
                continue
        if pred == label:
            correct += 1
    total = total - not_processed
    accuracy = correct / total if total > 0 else 0
    return accuracy, not_processed
