import os
import json
from collections import Counter, defaultdict


# ============================================================
# PATH HELPERS
# ============================================================

def project_root():
    # debug/ -> ARCs4/
    return os.path.dirname(os.path.dirname(__file__))


def load_wrong_cases():
    path = os.path.join(project_root(), "wrong_cases.json")

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find wrong_cases.json at:\n{path}\n\n"
            "Run main.py first so it creates wrong_cases.json."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def safe_value(case, key, default="None"):
    value = case.get(key)
    if value is None:
        return default
    return str(value)


def print_counter(title, counter, limit=25):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    if not counter:
        print("None")
        return

    for name, count in counter.most_common(limit):
        print(f"{name}: {count}")


def group_wrong_cases(wrong_cases):
    by_task = defaultdict(list)
    by_strategy = defaultdict(list)

    for case in wrong_cases:
        task_id = safe_value(case, "task_id")
        strategy = safe_value(case, "strategy")

        by_task[task_id].append(case)
        by_strategy[strategy].append(case)

    return by_task, by_strategy


# ============================================================
# MAIN REPORT
# ============================================================

def main():
    wrong_cases = load_wrong_cases()

    print("\n" + "=" * 60)
    print("WRONG CASE ANALYSIS")
    print("=" * 60)
    print(f"Total wrong train pairs: {len(wrong_cases)}")

    if not wrong_cases:
        print("No wrong cases found.")
        return

    by_task, by_strategy = group_wrong_cases(wrong_cases)

    # ------------------------------------------------------------
    # Strategy counts
    # ------------------------------------------------------------
    strategy_counter = Counter(
        safe_value(case, "strategy") for case in wrong_cases
    )
    print_counter("WRONG CASES BY STRATEGY", strategy_counter)

    # ------------------------------------------------------------
    # Task counts
    # ------------------------------------------------------------
    task_counter = Counter(
        safe_value(case, "task_id") for case in wrong_cases
    )
    print_counter("TASKS WITH MOST WRONG PAIRS", task_counter)

    # ------------------------------------------------------------
    # Selector counts
    # ------------------------------------------------------------
    selector_counter = Counter(
        safe_value(case, "selector") for case in wrong_cases
    )
    print_counter("WRONG CASES BY SELECTOR", selector_counter)

    # ------------------------------------------------------------
    # Transform counts
    # ------------------------------------------------------------
    transform_counter = Counter(
        safe_value(case, "transform") for case in wrong_cases
    )
    print_counter("WRONG CASES BY TRANSFORM", transform_counter)

    # ------------------------------------------------------------
    # Score buckets
    # ------------------------------------------------------------
    score_buckets = Counter()

    for case in wrong_cases:
        score = case.get("score")

        if score is None:
            score_buckets["None"] += 1
        elif score < 0:
            score_buckets["negative"] += 1
        elif score == 0:
            score_buckets["zero"] += 1
        elif score < 100:
            score_buckets["1-99"] += 1
        elif score < 500:
            score_buckets["100-499"] += 1
        elif score < 1000:
            score_buckets["500-999"] += 1
        elif score < 10000:
            score_buckets["1000-9999"] += 1
        else:
            score_buckets["10000+"] += 1

    print_counter("WRONG CASES BY SCORE BUCKET", score_buckets)

    # ------------------------------------------------------------
    # Detailed task list
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DETAILED WRONG TASK LIST")
    print("=" * 60)

    for task_id, cases in sorted(by_task.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"\nTASK: {task_id}")
        print(f"Wrong pairs: {len(cases)}")

        strategy_counts = Counter(safe_value(c, "strategy") for c in cases)
        print("Strategies:")
        for strategy, count in strategy_counts.most_common():
            print(f"  {strategy}: {count}")

        for case in cases:
            print(
                f"  Pair {safe_value(case, 'pair_index')} | "
                f"strategy={safe_value(case, 'strategy')} | "
                f"score={safe_value(case, 'score')} | "
                f"selector={safe_value(case, 'selector')} | "
                f"transform={safe_value(case, 'transform')}"
            )

    # ------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("NEXT DEBUG TARGET")
    print("=" * 60)

    most_common_strategy, count = strategy_counter.most_common(1)[0]

    print(f"Most common failing strategy: {most_common_strategy}")
    print(f"Wrong cases from this strategy: {count}")

    print("\nRecommended next move:")
    print("1. Pick the top failing strategy.")
    print("2. Open 3-5 tasks from that group.")
    print("3. Compare INPUT / EXPECTED / PREDICTED.")
    print("4. Fix that one family only.")
    print("5. Rerun main.py and compare total wrong count.")


if __name__ == "__main__":
    main()