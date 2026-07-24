from __future__ import annotations

import argparse
import json
import os
import tkinter as tk
from collections import Counter, defaultdict
from dataclasses import dataclass
from tkinter import messagebox
from typing import Any


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.join(PROJECT_DIR, "data", "data.json")
DEFAULT_REPORT_PATH = os.path.join(PROJECT_DIR, "honest_benchmark_results.json")

ARC_COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}

TEXT_COLORS = {
    0: "#FFFFFF",
    1: "#FFFFFF",
    2: "#FFFFFF",
    3: "#000000",
    4: "#000000",
    5: "#000000",
    6: "#FFFFFF",
    7: "#000000",
    8: "#000000",
    9: "#FFFFFF",
}


@dataclass(frozen=True)
class ArcObject:
    color: int
    cells: frozenset[tuple[int, int]]
    top: int
    left: int
    height: int
    width: int

    @property
    def size(self) -> int:
        return len(self.cells)

    @property
    def normalized_cells(self) -> frozenset[tuple[int, int]]:
        return frozenset((row - self.top, col - self.left) for row, col in self.cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visually review failed honest leave-one-out predictions."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="ARC task JSON file.",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT_PATH,
        help="Detailed JSON report created by run_honest_benchmark.py.",
    )
    parser.add_argument(
        "--family",
        default="region_object_family",
        help=(
            "Only show this chosen family. Use --family all to show every family. "
            "Default: region_object_family"
        ),
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Only review this task id.",
    )
    parser.add_argument(
        "--numbers",
        action="store_true",
        help="Show color numbers inside cells.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_tasks(path: str) -> dict[str, dict[str, Any]]:
    raw = load_json(path)

    if not isinstance(raw, dict):
        raise ValueError("Task data must be a JSON object.")

    if "train" in raw or "test" in raw:
        task_id = os.path.splitext(os.path.basename(path))[0]
        return {task_id: raw}

    return {
        str(task_id): task
        for task_id, task in raw.items()
        if isinstance(task, dict)
    }


def grid_shape(grid: Any) -> tuple[int, int]:
    if not isinstance(grid, list) or not grid:
        return 0, 0

    width = len(grid[0]) if isinstance(grid[0], list) else 0
    return len(grid), width


def valid_grid(grid: Any) -> bool:
    height, width = grid_shape(grid)
    return (
        height > 0
        and width > 0
        and all(isinstance(row, list) and len(row) == width for row in grid)
    )


def background_color(grid: list[list[int]]) -> int:
    counts = Counter(cell for row in grid for cell in row)
    if not counts:
        return 0
    return counts.most_common(1)[0][0]


def connected_components(
    grid: list[list[int]],
    ignore_background: bool = True,
) -> list[ArcObject]:
    if not valid_grid(grid):
        return []

    height, width = grid_shape(grid)
    background = background_color(grid)
    visited: set[tuple[int, int]] = set()
    objects: list[ArcObject] = []

    for row in range(height):
        for col in range(width):
            if (row, col) in visited:
                continue

            color = grid[row][col]
            if ignore_background and color == background:
                visited.add((row, col))
                continue

            stack = [(row, col)]
            visited.add((row, col))
            cells: set[tuple[int, int]] = set()

            while stack:
                current_row, current_col = stack.pop()
                cells.add((current_row, current_col))

                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in visited:
                        continue
                    if grid[next_row][next_col] != color:
                        continue

                    visited.add((next_row, next_col))
                    stack.append((next_row, next_col))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            top = min(rows)
            left = min(cols)
            bottom = max(rows)
            right = max(cols)

            objects.append(
                ArcObject(
                    color=color,
                    cells=frozenset(cells),
                    top=top,
                    left=left,
                    height=bottom - top + 1,
                    width=right - left + 1,
                )
            )

    return sorted(
        objects,
        key=lambda obj: (obj.top, obj.left, obj.color, obj.height, obj.width),
    )


def compare_cells(
    expected: list[list[int]],
    predicted: list[list[int]] | None,
) -> dict[str, int]:
    if not valid_grid(expected):
        return {
            "compared": 0,
            "exact_cells": 0,
            "wrong_color": 0,
            "missing_area": 0,
            "extra_area": 0,
        }

    expected_height, expected_width = grid_shape(expected)

    if not valid_grid(predicted):
        return {
            "compared": 0,
            "exact_cells": 0,
            "wrong_color": 0,
            "missing_area": expected_height * expected_width,
            "extra_area": 0,
        }

    predicted_height, predicted_width = grid_shape(predicted)
    overlap_height = min(expected_height, predicted_height)
    overlap_width = min(expected_width, predicted_width)

    exact_cells = 0
    wrong_color = 0

    for row in range(overlap_height):
        for col in range(overlap_width):
            if expected[row][col] == predicted[row][col]:
                exact_cells += 1
            else:
                wrong_color += 1

    expected_area = expected_height * expected_width
    predicted_area = predicted_height * predicted_width
    overlap_area = overlap_height * overlap_width

    return {
        "compared": overlap_area,
        "exact_cells": exact_cells,
        "wrong_color": wrong_color,
        "missing_area": expected_area - overlap_area,
        "extra_area": predicted_area - overlap_area,
    }


def match_objects(
    expected_objects: list[ArcObject],
    predicted_objects: list[ArcObject],
) -> list[str]:
    notes: list[str] = []
    unused_predicted = set(range(len(predicted_objects)))

    for expected_index, expected_object in enumerate(expected_objects, start=1):
        exact_match = next(
            (
                index
                for index in unused_predicted
                if predicted_objects[index] == expected_object
            ),
            None,
        )

        if exact_match is not None:
            unused_predicted.remove(exact_match)
            continue

        same_shape_color = [
            index
            for index in unused_predicted
            if predicted_objects[index].color == expected_object.color
            and predicted_objects[index].normalized_cells
            == expected_object.normalized_cells
        ]

        if same_shape_color:
            index = min(
                same_shape_color,
                key=lambda candidate: (
                    abs(predicted_objects[candidate].top - expected_object.top)
                    + abs(predicted_objects[candidate].left - expected_object.left)
                ),
            )
            predicted_object = predicted_objects[index]
            unused_predicted.remove(index)
            row_shift = predicted_object.top - expected_object.top
            col_shift = predicted_object.left - expected_object.left
            notes.append(
                f"Object {expected_index}: correct shape/color but shifted "
                f"(row {row_shift:+d}, col {col_shift:+d})."
            )
            continue

        same_shape_position = [
            index
            for index in unused_predicted
            if predicted_objects[index].top == expected_object.top
            and predicted_objects[index].left == expected_object.left
            and predicted_objects[index].normalized_cells
            == expected_object.normalized_cells
        ]

        if same_shape_position:
            index = same_shape_position[0]
            predicted_object = predicted_objects[index]
            unused_predicted.remove(index)
            notes.append(
                f"Object {expected_index}: correct shape/position but color "
                f"{predicted_object.color} should be {expected_object.color}."
            )
            continue

        same_color = [
            index
            for index in unused_predicted
            if predicted_objects[index].color == expected_object.color
        ]

        if same_color:
            index = min(
                same_color,
                key=lambda candidate: (
                    abs(predicted_objects[candidate].top - expected_object.top)
                    + abs(predicted_objects[candidate].left - expected_object.left)
                    + abs(predicted_objects[candidate].size - expected_object.size)
                ),
            )
            predicted_object = predicted_objects[index]
            unused_predicted.remove(index)

            differences = []
            if (
                predicted_object.height != expected_object.height
                or predicted_object.width != expected_object.width
                or predicted_object.normalized_cells
                != expected_object.normalized_cells
            ):
                differences.append(
                    f"shape {predicted_object.height}x{predicted_object.width} "
                    f"should be {expected_object.height}x{expected_object.width}"
                )

            if (
                predicted_object.top != expected_object.top
                or predicted_object.left != expected_object.left
            ):
                differences.append(
                    f"position ({predicted_object.top},{predicted_object.left}) "
                    f"should be ({expected_object.top},{expected_object.left})"
                )

            if predicted_object.size != expected_object.size:
                differences.append(
                    f"{predicted_object.size} cells should be {expected_object.size}"
                )

            detail = "; ".join(differences) if differences else "wrong geometry"
            notes.append(f"Object {expected_index}: {detail}.")
            continue

        notes.append(
            f"Object {expected_index}: missing color {expected_object.color} object "
            f"at ({expected_object.top},{expected_object.left}), "
            f"size {expected_object.height}x{expected_object.width}."
        )

    for index in sorted(unused_predicted):
        predicted_object = predicted_objects[index]
        notes.append(
            f"Extra object: color {predicted_object.color} at "
            f"({predicted_object.top},{predicted_object.left}), "
            f"size {predicted_object.height}x{predicted_object.width}."
        )

    return notes


def describe_failure(
    expected: list[list[int]],
    predicted: list[list[int]] | None,
) -> list[str]:
    notes: list[str] = []

    if predicted is None:
        return ["No prediction was returned."]

    expected_shape = grid_shape(expected)
    predicted_shape = grid_shape(predicted)

    if predicted_shape != expected_shape:
        notes.append(
            f"Output shape is {predicted_shape[0]}x{predicted_shape[1]}; "
            f"expected {expected_shape[0]}x{expected_shape[1]}."
        )
    else:
        notes.append(
            f"Output shape is correct: {expected_shape[0]}x{expected_shape[1]}."
        )

    cell_stats = compare_cells(expected, predicted)
    notes.append(
        f"Overlapping area: {cell_stats['exact_cells']} correct cells, "
        f"{cell_stats['wrong_color']} wrong cells."
    )

    if cell_stats["missing_area"]:
        notes.append(
            f"Expected canvas has {cell_stats['missing_area']} cells outside "
            "the predicted canvas."
        )

    if cell_stats["extra_area"]:
        notes.append(
            f"Predicted canvas has {cell_stats['extra_area']} extra cells outside "
            "the expected canvas."
        )

    expected_objects = connected_components(expected)
    predicted_objects = connected_components(predicted)

    notes.append(
        f"Foreground objects: predicted {len(predicted_objects)}, "
        f"expected {len(expected_objects)}."
    )

    object_notes = match_objects(expected_objects, predicted_objects)

    if object_notes:
        notes.append("")
        notes.extend(object_notes)
    else:
        notes.append("Foreground objects match exactly.")

    return notes


def collect_failures(
    tasks: dict[str, dict[str, Any]],
    report: dict[str, Any],
    family_filter: str,
    task_filter: str | None,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []

    for task_result in report.get("tasks") or []:
        task_id = str(task_result.get("task_id"))

        if task_filter is not None and task_id != task_filter:
            continue

        task = tasks.get(task_id)
        if task is None:
            continue

        train_pairs = task.get("train") or []

        for fold in task_result.get("folds") or []:
            if fold.get("exact"):
                continue

            family = fold.get("strategy") or "no_strategy"
            if family_filter.lower() != "all" and family != family_filter:
                continue

            hidden_pair_number = fold.get("hidden_pair")
            if not isinstance(hidden_pair_number, int):
                continue

            hidden_index = hidden_pair_number - 1
            if not 0 <= hidden_index < len(train_pairs):
                continue

            pair = train_pairs[hidden_index]
            hidden_input = pair.get("input")
            expected = pair.get("output")
            predicted = fold.get("predicted")

            failures.append(
                {
                    "task_id": task_id,
                    "hidden_pair": hidden_pair_number,
                    "family": family,
                    "rule_type": fold.get("rule_type"),
                    "chosen_inner_strategy": fold.get("chosen_inner_strategy"),
                    "error": fold.get("error"),
                    "input": hidden_input,
                    "expected": expected,
                    "predicted": predicted,
                    "notes": describe_failure(expected, predicted),
                }
            )

    return failures


class GridPanel(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        title: str,
        show_numbers: bool,
    ) -> None:
        super().__init__(master, background="#202020")
        self.show_numbers = show_numbers
        self.grid_data: Any = None

        self.title_label = tk.Label(
            self,
            text=title,
            font=("Segoe UI", 13, "bold"),
            background="#202020",
            foreground="#FFFFFF",
        )
        self.title_label.pack(pady=(0, 6))

        self.canvas = tk.Canvas(
            self,
            background="#303030",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def set_grid(self, grid: Any) -> None:
        self.grid_data = grid
        self.draw()

    def draw(self) -> None:
        self.canvas.delete("all")

        if not valid_grid(self.grid_data):
            self.canvas.create_text(
                max(10, self.canvas.winfo_width() // 2),
                max(10, self.canvas.winfo_height() // 2),
                text="NO GRID",
                fill="#FFFFFF",
                font=("Segoe UI", 18, "bold"),
            )
            return

        rows, cols = grid_shape(self.grid_data)
        available_width = max(1, self.canvas.winfo_width() - 20)
        available_height = max(1, self.canvas.winfo_height() - 20)

        cell_size = max(
            2,
            min(
                available_width / cols,
                available_height / rows,
                36,
            ),
        )

        grid_width = cell_size * cols
        grid_height = cell_size * rows
        start_x = (self.canvas.winfo_width() - grid_width) / 2
        start_y = (self.canvas.winfo_height() - grid_height) / 2

        for row in range(rows):
            for col in range(cols):
                value = self.grid_data[row][col]
                fill = ARC_COLORS.get(value, "#FFFFFF")

                x1 = start_x + col * cell_size
                y1 = start_y + row * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=fill,
                    outline="#555555",
                    width=1,
                )

                if self.show_numbers and cell_size >= 16:
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(value),
                        fill=TEXT_COLORS.get(value, "#000000"),
                        font=(
                            "Consolas",
                            max(7, int(cell_size * 0.35)),
                            "bold",
                        ),
                    )


class FailureReviewApp:
    def __init__(
        self,
        failures: list[dict[str, Any]],
        show_numbers: bool,
    ) -> None:
        self.failures = failures
        self.index = 0

        self.root = tk.Tk()
        self.root.title("ARCs4 Honest Failure Review")
        self.root.geometry("1500x900")
        self.root.minsize(1000, 650)
        self.root.configure(background="#181818")

        header = tk.Frame(self.root, background="#181818")
        header.pack(fill="x", padx=12, pady=(10, 6))

        self.header_label = tk.Label(
            header,
            text="",
            anchor="w",
            font=("Segoe UI", 14, "bold"),
            background="#181818",
            foreground="#FFFFFF",
        )
        self.header_label.pack(side="left", fill="x", expand=True)

        previous_button = tk.Button(
            header,
            text="◀ Previous",
            command=self.previous,
            width=12,
        )
        previous_button.pack(side="right", padx=(6, 0))

        next_button = tk.Button(
            header,
            text="Next ▶",
            command=self.next,
            width=12,
        )
        next_button.pack(side="right", padx=(6, 0))

        grid_frame = tk.Frame(self.root, background="#202020")
        grid_frame.pack(fill="both", expand=True, padx=12, pady=6)

        grid_frame.grid_rowconfigure(0, weight=1)
        for column in range(3):
            grid_frame.grid_columnconfigure(column, weight=1)

        self.input_panel = GridPanel(grid_frame, "INPUT", show_numbers)
        self.expected_panel = GridPanel(grid_frame, "EXPECTED", show_numbers)
        self.predicted_panel = GridPanel(grid_frame, "PREDICTED", show_numbers)

        self.input_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.expected_panel.grid(row=0, column=1, sticky="nsew", padx=4)
        self.predicted_panel.grid(row=0, column=2, sticky="nsew", padx=(4, 0))

        analysis_frame = tk.Frame(self.root, background="#181818")
        analysis_frame.pack(fill="x", padx=12, pady=(6, 10))

        tk.Label(
            analysis_frame,
            text="DIFFERENCE REPORT",
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            background="#181818",
            foreground="#FFFFFF",
        ).pack(fill="x")

        self.analysis_text = tk.Text(
            analysis_frame,
            height=12,
            wrap="word",
            font=("Consolas", 11),
            background="#101010",
            foreground="#FFFFFF",
            insertbackground="#FFFFFF",
            relief="flat",
            padx=10,
            pady=10,
        )
        self.analysis_text.pack(fill="x", pady=(4, 0))
        self.analysis_text.configure(state="disabled")

        self.root.bind("<Left>", lambda _event: self.previous())
        self.root.bind("<Right>", lambda _event: self.next())
        self.root.bind("<Prior>", lambda _event: self.previous())
        self.root.bind("<Next>", lambda _event: self.next())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())

        self.show_current()

    def show_current(self) -> None:
        if not self.failures:
            return

        failure = self.failures[self.index]

        details = [
            f"{self.index + 1}/{len(self.failures)}",
            f"Task {failure['task_id']}",
            f"Hidden pair {failure['hidden_pair']}",
            f"Family: {failure['family']}",
        ]

        if failure.get("rule_type"):
            details.append(f"Rule: {failure['rule_type']}")

        if failure.get("chosen_inner_strategy"):
            details.append(f"Inner: {failure['chosen_inner_strategy']}")

        self.header_label.configure(text="    |    ".join(details))
        self.input_panel.set_grid(failure["input"])
        self.expected_panel.set_grid(failure["expected"])
        self.predicted_panel.set_grid(failure["predicted"])

        report_lines = []

        if failure.get("error"):
            report_lines.append(f"ERROR: {failure['error']}")
            report_lines.append("")

        for note in failure["notes"]:
            if note:
                report_lines.append(f"• {note}")
            else:
                report_lines.append("")

        self.analysis_text.configure(state="normal")
        self.analysis_text.delete("1.0", "end")
        self.analysis_text.insert("1.0", "\n".join(report_lines))
        self.analysis_text.configure(state="disabled")

    def next(self) -> None:
        if not self.failures:
            return
        self.index = (self.index + 1) % len(self.failures)
        self.show_current()

    def previous(self) -> None:
        if not self.failures:
            return
        self.index = (self.index - 1) % len(self.failures)
        self.show_current()

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    args = parse_args()

    try:
        tasks = load_tasks(args.data)
        report = load_json(args.report)
        failures = collect_failures(
            tasks=tasks,
            report=report,
            family_filter=args.family,
            task_filter=args.task,
        )
    except Exception as exc:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "ARCs4 Honest Failure Review",
            f"{type(exc).__name__}: {exc}",
        )
        root.destroy()
        return 1

    if not failures:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "ARCs4 Honest Failure Review",
            "No matching failed predictions were found.",
        )
        root.destroy()
        return 0

    print(f"Loaded failed folds: {len(failures)}")
    print(f"Family filter: {args.family}")
    print("Use Left/Right arrow keys to move between failures.")
    print("Press Escape to close.")

    app = FailureReviewApp(
        failures=failures,
        show_numbers=args.numbers,
    )
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())