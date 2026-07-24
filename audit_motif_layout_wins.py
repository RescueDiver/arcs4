# audit_motif_layout_wins.py
import contextlib
import io
import json
import os
import traceback

from reasoning.motif_layout_rule import solve_pair_motif_layout_rule


# ============================================================
# SETTINGS
# ============================================================

COPY_FULLY_SOLVED_TO_FAILURES = True
COPY_PARTIAL_SOLVED_TO_FAILURES = False

DATA_PATH = os.path.join("data", "data.json")
FAILURE_EXTRACT_DIR = os.path.join("data_failures", "extracted_tasks")

REPORT_JSON = os.path.join("data_failures", "motif_layout_audit_report.json")
FULLY_SOLVED_IDS_TXT = os.path.join("data_failures", "motif_layout_fully_solved_ids.txt")
PARTIAL_SOLVED_IDS_TXT = os.path.join("data_failures", "motif_layout_partial_solved_ids.txt")


# ============================================================
# FILE HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_lines(path, lines):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        for line in lines:
            file.write(str(line) + "\n")


def load_tasks():
    data = load_json(DATA_PATH)

    if isinstance(data, dict) and "train" in data:
        return {"single_task": data}

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        tasks = {}
        for idx, task in enumerate(data):
            task_id = task.get("id", f"task_{idx:04d}")
            tasks[task_id] = task
        return tasks

    raise ValueError("Unknown data/data.json format")


def save_extracted_task(task_id, task):
    os.makedirs(FAILURE_EXTRACT_DIR, exist_ok=True)

    out_path = os.path.join(
        FAILURE_EXTRACT_DIR,
        f"{task_id}.json",
    )

    save_json(out_path, task)
    return out_path


# ============================================================
# MOTIF AUDIT
# ============================================================

def run_motif_rule_silent(input_grid, output_grid):
    """
    motif_layout_rule.py prints a lot.
    Suppress prints so this audit stays readable.
    """
    buffer = io.StringIO()

    with contextlib.redirect_stdout(buffer):
        result = solve_pair_motif_layout_rule(
            input_grid,
            output_grid,
        )

    return result


def audit_task(task_id, task):
    train_pairs = task.get("train", [])

    pair_results = []
    exact_count = 0
    total_score = 0
    had_prediction_count = 0

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        try:
            result = run_motif_rule_silent(
                input_grid,
                output_grid,
            )

            if result is None:
                pair_results.append({
                    "pair_index": pair_index,
                    "predicted": False,
                    "exact": False,
                    "score": None,
                    "error": None,
                })
                continue

            score = result.get("score")
            exact = bool(result.get("exact"))

            had_prediction_count += 1

            if exact:
                exact_count += 1

            if isinstance(score, int):
                total_score += score

            pair_results.append({
                "pair_index": pair_index,
                "predicted": True,
                "exact": exact,
                "score": score,
                "error": None,
            })

        except Exception as exc:
            pair_results.append({
                "pair_index": pair_index,
                "predicted": False,
                "exact": False,
                "score": None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })

    train_count = len(train_pairs)

    fully_solved = train_count > 0 and exact_count == train_count
    partially_solved = exact_count > 0 and exact_count < train_count

    return {
        "task_id": task_id,
        "train_count": train_count,
        "predicted_count": had_prediction_count,
        "exact_count": exact_count,
        "total_score": total_score,
        "fully_solved": fully_solved,
        "partially_solved": partially_solved,
        "pair_results": pair_results,
    }


def main():
    print()
    print("=" * 80)
    print("AUDIT MOTIF_LAYOUT_RULE WINS")
    print("=" * 80)

    tasks = load_tasks()

    fully_solved = []
    partially_solved = []
    failed = []
    copied = []

    report = {
        "source": DATA_PATH,
        "rule": "motif_layout_rule",
        "fully_solved": [],
        "partially_solved": [],
        "failed": [],
        "copied_to_failure_extract_dir": [],
    }

    for idx, (task_id, task) in enumerate(tasks.items(), start=1):
        result = audit_task(task_id, task)

        if result["fully_solved"]:
            fully_solved.append(task_id)
            report["fully_solved"].append(result)

            print(
                f"[FULL] {task_id} "
                f"exact={result['exact_count']}/{result['train_count']} "
                f"score={result['total_score']}"
            )

            if COPY_FULLY_SOLVED_TO_FAILURES:
                out_path = save_extracted_task(task_id, task)
                copied.append(task_id)
                report["copied_to_failure_extract_dir"].append({
                    "task_id": task_id,
                    "path": out_path,
                    "reason": "motif_layout_rule_fully_solved_needs_reaudit",
                })

        elif result["partially_solved"]:
            partially_solved.append(task_id)
            report["partially_solved"].append(result)

            print(
                f"[PART] {task_id} "
                f"exact={result['exact_count']}/{result['train_count']} "
                f"score={result['total_score']}"
            )

            if COPY_PARTIAL_SOLVED_TO_FAILURES:
                out_path = save_extracted_task(task_id, task)
                copied.append(task_id)
                report["copied_to_failure_extract_dir"].append({
                    "task_id": task_id,
                    "path": out_path,
                    "reason": "motif_layout_rule_partially_solved_needs_reaudit",
                })

        else:
            failed.append(task_id)
            report["failed"].append(result)

    save_json(REPORT_JSON, report)
    save_lines(FULLY_SOLVED_IDS_TXT, fully_solved)
    save_lines(PARTIAL_SOLVED_IDS_TXT, partially_solved)

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tasks checked       : {len(tasks)}")
    print(f"Fully solved by motif rule: {len(fully_solved)}")
    print(f"Partial motif wins        : {len(partially_solved)}")
    print(f"Not solved by motif rule  : {len(failed)}")
    print(f"Copied to failures        : {len(copied)}")
    print()
    print(f"Report written: {REPORT_JSON}")
    print(f"Full IDs      : {FULLY_SOLVED_IDS_TXT}")
    print(f"Partial IDs   : {PARTIAL_SOLVED_IDS_TXT}")
    print(f"Extract folder: {FAILURE_EXTRACT_DIR}")


if __name__ == "__main__":
    main()