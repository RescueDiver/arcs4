from __future__ import annotations
import argparse
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


Grid = list[list[int]]


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def shape(grid: Grid) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def area(grid: Grid) -> int:
    h, w = shape(grid)
    return h * w


def flatten(grid: Grid) -> list[int]:
    return [cell for row in grid for cell in row]


def colors(grid: Grid) -> set[int]:
    return set(flatten(grid))


def color_counts(grid: Grid) -> Counter[int]:
    return Counter(flatten(grid))


def most_common_color(grid: Grid) -> int | None:
    counts = color_counts(grid)
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def changed_cells(a: Grid, b: Grid) -> int | None:
    if shape(a) != shape(b):
        return None
    return sum(
        1
        for r in range(len(a))
        for c in range(len(a[0]))
        if a[r][c] != b[r][c]
    )


def border_colors(grid: Grid) -> Counter[int]:
    h, w = shape(grid)
    result: Counter[int] = Counter()

    if h == 0 or w == 0:
        return result

    for c in range(w):
        result[grid[0][c]] += 1
        if h > 1:
            result[grid[h - 1][c]] += 1

    for r in range(1, h - 1):
        result[grid[r][0]] += 1
        if w > 1:
            result[grid[r][w - 1]] += 1

    return result


def infer_background(grid: Grid) -> int | None:
    """
    Prefer a dominant border color. Fall back to the most common color.
    This is only for broad visual statistics, not task solving.
    """
    border = border_colors(grid)
    if border:
        border_color, border_count = border.most_common(1)[0]
        total_border = sum(border.values())
        if total_border and border_count / total_border >= 0.50:
            return border_color

    return most_common_color(grid)


def bounding_box(
    grid: Grid,
    ignored_colors: set[int] | None = None,
) -> tuple[int, int, int, int] | None:
    ignored = ignored_colors or set()
    cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value not in ignored
    ]

    if not cells:
        return None

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def bbox_shape(box: tuple[int, int, int, int] | None) -> tuple[int, int] | None:
    if box is None:
        return None
    top, left, bottom, right = box
    return bottom - top + 1, right - left + 1


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Component:
    color: int
    size: int
    top: int
    left: int
    bottom: int
    right: int

    @property
    def height(self) -> int:
        return self.bottom - self.top + 1

    @property
    def width(self) -> int:
        return self.right - self.left + 1


def connected_components(
    grid: Grid,
    ignored_colors: set[int] | None = None,
) -> list[Component]:
    """
    Four-neighbor, same-color connected components.
    Background is normally ignored.
    """
    ignored = ignored_colors or set()
    h, w = shape(grid)
    seen: set[tuple[int, int]] = set()
    result: list[Component] = []

    for start_r in range(h):
        for start_c in range(w):
            if (start_r, start_c) in seen:
                continue

            value = grid[start_r][start_c]
            if value in ignored:
                seen.add((start_r, start_c))
                continue

            queue = deque([(start_r, start_c)])
            seen.add((start_r, start_c))
            cells: list[tuple[int, int]] = []

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = r + dr
                    nc = c + dc

                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if (nr, nc) in seen:
                        continue
                    if grid[nr][nc] != value:
                        continue

                    seen.add((nr, nc))
                    queue.append((nr, nc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]

            result.append(
                Component(
                    color=value,
                    size=len(cells),
                    top=min(rows),
                    left=min(cols),
                    bottom=max(rows),
                    right=max(cols),
                )
            )

    return result


# ---------------------------------------------------------------------------
# Repetition and symmetry signals
# ---------------------------------------------------------------------------

def repeated_rows(grid: Grid) -> int:
    """
    Number of row copies beyond the first occurrence.
    Example: [A, B, A, A] contributes 2 repeated rows.
    """
    rows = Counter(tuple(row) for row in grid)
    return sum(count - 1 for count in rows.values() if count > 1)


def repeated_columns(grid: Grid) -> int:
    h, w = shape(grid)
    if h == 0 or w == 0:
        return 0

    columns = Counter(
        tuple(grid[r][c] for r in range(h))
        for c in range(w)
    )
    return sum(count - 1 for count in columns.values() if count > 1)


def horizontal_symmetry_score(grid: Grid) -> float:
    h, w = shape(grid)
    if h == 0 or w == 0:
        return 0.0

    matches = 0
    comparisons = 0

    for r in range(h):
        mirror_r = h - 1 - r
        if r > mirror_r:
            break

        for c in range(w):
            matches += int(grid[r][c] == grid[mirror_r][c])
            comparisons += 1

    return matches / comparisons if comparisons else 0.0


def vertical_symmetry_score(grid: Grid) -> float:
    h, w = shape(grid)
    if h == 0 or w == 0:
        return 0.0

    matches = 0
    comparisons = 0

    for c in range(w):
        mirror_c = w - 1 - c
        if c > mirror_c:
            break

        for r in range(h):
            matches += int(grid[r][c] == grid[r][mirror_c])
            comparisons += 1

    return matches / comparisons if comparisons else 0.0


def best_symmetry_score(grid: Grid) -> float:
    return max(
        horizontal_symmetry_score(grid),
        vertical_symmetry_score(grid),
    )


# ---------------------------------------------------------------------------
# Pair analysis
# ---------------------------------------------------------------------------

@dataclass
class PairSummary:
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    size_relation: str
    input_colors: set[int]
    output_colors: set[int]
    colors_added: set[int]
    colors_removed: set[int]
    changed_count: int | None
    changed_fraction: float | None
    input_background: int | None
    output_background: int | None
    input_objects: int
    output_objects: int
    object_relation: str
    input_bbox_shape: tuple[int, int] | None
    output_bbox_shape: tuple[int, int] | None
    input_repetition: int
    output_repetition: int
    input_symmetry: float
    output_symmetry: float
    action_candidates: list[str]


def size_relation(input_grid: Grid, output_grid: Grid) -> str:
    ih, iw = shape(input_grid)
    oh, ow = shape(output_grid)

    if (ih, iw) == (oh, ow):
        return "same"

    input_area = ih * iw
    output_area = oh * ow

    if output_area < input_area:
        return "smaller"

    if output_area > input_area:
        return "larger"

    return "reshaped"


def object_relation(input_count: int, output_count: int) -> str:
    if output_count == input_count:
        return "same"
    if output_count > input_count:
        return "more"
    return "fewer"


def analyze_pair(input_grid: Grid, output_grid: Grid) -> PairSummary:
    input_bg = infer_background(input_grid)
    output_bg = infer_background(output_grid)

    input_ignored = {input_bg} if input_bg is not None else set()
    output_ignored = {output_bg} if output_bg is not None else set()

    input_components = connected_components(input_grid, input_ignored)
    output_components = connected_components(output_grid, output_ignored)

    input_color_set = colors(input_grid)
    output_color_set = colors(output_grid)

    change_count = changed_cells(input_grid, output_grid)
    change_fraction = None

    if change_count is not None and area(input_grid):
        change_fraction = change_count / area(input_grid)

    in_repeat = repeated_rows(input_grid) + repeated_columns(input_grid)
    out_repeat = repeated_rows(output_grid) + repeated_columns(output_grid)

    in_symmetry = best_symmetry_score(input_grid)
    out_symmetry = best_symmetry_score(output_grid)

    actions: list[str] = []
    relation = size_relation(input_grid, output_grid)

    if relation == "same":
        if change_fraction == 0:
            actions.append("copies the input unchanged")
        elif change_fraction is not None and change_fraction <= 0.10:
            actions.append("makes a sparse local repair")
        elif change_fraction is not None and change_fraction <= 0.35:
            actions.append("edits selected regions while preserving the canvas")
        else:
            actions.append("reconstructs much of the same-sized canvas")

    elif relation == "smaller":
        actions.append("extracts or compresses information into a smaller output")

    elif relation == "larger":
        actions.append("expands, copies, or arranges information on a larger canvas")

    else:
        actions.append("reshapes the canvas without changing total area")

    added = output_color_set - input_color_set
    removed = input_color_set - output_color_set

    if added and removed:
        actions.append("changes the color vocabulary")
    elif added:
        actions.append("introduces new colors")
    elif removed:
        actions.append("removes some colors")

    if len(output_components) > len(input_components):
        actions.append("creates or separates objects")
    elif len(output_components) < len(input_components):
        actions.append("removes, merges, or selects objects")

    if out_repeat >= in_repeat + 2:
        actions.append("increases visible repetition")

    if out_symmetry >= 0.95 and out_symmetry >= in_symmetry + 0.10:
        actions.append("completes or creates symmetry")

    input_box = bbox_shape(bounding_box(input_grid, input_ignored))
    output_box = bbox_shape(bounding_box(output_grid, output_ignored))

    return PairSummary(
        input_shape=shape(input_grid),
        output_shape=shape(output_grid),
        size_relation=relation,
        input_colors=input_color_set,
        output_colors=output_color_set,
        colors_added=added,
        colors_removed=removed,
        changed_count=change_count,
        changed_fraction=change_fraction,
        input_background=input_bg,
        output_background=output_bg,
        input_objects=len(input_components),
        output_objects=len(output_components),
        object_relation=object_relation(
            len(input_components),
            len(output_components),
        ),
        input_bbox_shape=input_box,
        output_bbox_shape=output_box,
        input_repetition=in_repeat,
        output_repetition=out_repeat,
        input_symmetry=in_symmetry,
        output_symmetry=out_symmetry,
        action_candidates=actions,
    )


# ---------------------------------------------------------------------------
# Task-level headline generation
# ---------------------------------------------------------------------------

@dataclass
class TaskHeadline:
    task_id: str
    training_pairs: int
    test_pairs: int
    headline: str
    size_story: str
    color_story: str
    object_story: str
    consistency: str
    pair_summaries: list[PairSummary]


def all_same(values: Iterable[Any]) -> bool:
    values = list(values)
    return bool(values) and len(set(values)) == 1


def dominant_value(values: Sequence[str]) -> tuple[str, int]:
    counts = Counter(values)
    return counts.most_common(1)[0]


def describe_size_story(pairs: list[PairSummary]) -> str:
    relations = [pair.size_relation for pair in pairs]

    if all_same(relations):
        relation = relations[0]

        if relation == "same":
            return "Output keeps the input canvas size."
        if relation == "smaller":
            return "Output is consistently smaller than the input."
        if relation == "larger":
            return "Output is consistently larger than the input."
        return "Output reshapes the canvas while keeping equal area."

    counts = Counter(relations)
    parts = [f"{name}={count}" for name, count in counts.most_common()]
    return "Output-size behavior is mixed: " + ", ".join(parts) + "."


def describe_color_story(pairs: list[PairSummary]) -> str:
    any_added = any(pair.colors_added for pair in pairs)
    any_removed = any(pair.colors_removed for pair in pairs)

    if not any_added and not any_removed:
        return "Color vocabulary is preserved."

    if any_added and any_removed:
        return "Some colors are introduced while others disappear."

    if any_added:
        added = sorted(set().union(*(pair.colors_added for pair in pairs)))
        return f"New output colors appear: {added}."

    removed = sorted(set().union(*(pair.colors_removed for pair in pairs)))
    return f"Some input colors disappear: {removed}."


def describe_object_story(pairs: list[PairSummary]) -> str:
    relations = [pair.object_relation for pair in pairs]

    if all_same(relations):
        relation = relations[0]

        if relation == "same":
            return "Foreground object count is broadly preserved."
        if relation == "more":
            return "Output generally creates or separates more objects."
        return "Output generally removes, merges, or selects objects."

    return "Object-count behavior varies between training pairs."


def choose_headline(pairs: list[PairSummary]) -> str:
    relations = [pair.size_relation for pair in pairs]
    same_size = all_same(relations) and relations[0] == "same"
    smaller = all_same(relations) and relations[0] == "smaller"
    larger = all_same(relations) and relations[0] == "larger"

    fractions = [
        pair.changed_fraction
        for pair in pairs
        if pair.changed_fraction is not None
    ]

    object_relations = [pair.object_relation for pair in pairs]

    any_added = any(pair.colors_added for pair in pairs)
    any_removed = any(pair.colors_removed for pair in pairs)

    repetition_gains = [
        pair.output_repetition - pair.input_repetition
        for pair in pairs
    ]

    symmetry_gains = [
        pair.output_symmetry - pair.input_symmetry
        for pair in pairs
    ]

    # Most specific broad stories first.
    if same_size and fractions and max(fractions) <= 0.10:
        if any(gain >= 2 for gain in repetition_gains):
            return "Small damaged areas are repaired to restore a repeating pattern."
        if any(gain >= 0.10 for gain in symmetry_gains):
            return "A few cells are repaired to complete a symmetric design."
        return "A small number of cells are corrected while the scene stays intact."

    if smaller:
        if all_same(object_relations) and object_relations[0] == "fewer":
            return "Selected visual information is extracted into a smaller summary."
        return "The large scene is compressed into a smaller symbolic representation."

    if larger:
        if all_same(object_relations) and object_relations[0] == "more":
            return "Objects are copied or expanded across a larger canvas."
        return "The input is expanded or arranged into a larger composition."

    if same_size and any_added and any_removed:
        return "The scene is preserved structurally while colors are transformed."

    if same_size and all_same(object_relations):
        relation = object_relations[0]

        if relation == "fewer":
            return "Objects are removed, merged, or selected on the original canvas."
        if relation == "more":
            return "New objects or separated parts are created on the original canvas."

    if same_size and fractions and max(fractions) <= 0.35:
        return "Selected regions are changed while most of the canvas is preserved."

    if same_size:
        return "The original canvas is substantially reconstructed without resizing."

    return "Training pairs show a mixed structural transformation."


def consistency_label(pairs: list[PairSummary]) -> str:
    checks = [
        all_same(pair.size_relation for pair in pairs),
        all_same(pair.object_relation for pair in pairs),
        all_same(bool(pair.colors_added) for pair in pairs),
        all_same(bool(pair.colors_removed) for pair in pairs),
    ]

    score = sum(checks)

    if score == 4:
        return "High: the same broad story appears across all training pairs."
    if score >= 2:
        return "Medium: the main structure agrees, but some details vary."
    return "Low: training pairs show noticeably different broad behaviors."


def analyze_task(task_id: str, task: dict[str, Any]) -> TaskHeadline:
    train = task.get("train", [])
    test = task.get("test", [])

    pair_summaries = [
        analyze_pair(pair["input"], pair["output"])
        for pair in train
        if "input" in pair and "output" in pair
    ]

    if not pair_summaries:
        headline = "No complete training input/output pairs were found."
        return TaskHeadline(
            task_id=task_id,
            training_pairs=len(train),
            test_pairs=len(test),
            headline=headline,
            size_story="Unknown.",
            color_story="Unknown.",
            object_story="Unknown.",
            consistency="Unknown.",
            pair_summaries=[],
        )

    return TaskHeadline(
        task_id=task_id,
        training_pairs=len(train),
        test_pairs=len(test),
        headline=choose_headline(pair_summaries),
        size_story=describe_size_story(pair_summaries),
        color_story=describe_color_story(pair_summaries),
        object_story=describe_object_story(pair_summaries),
        consistency=consistency_label(pair_summaries),
        pair_summaries=pair_summaries,
    )


# ---------------------------------------------------------------------------
# Loading ARC JSON formats
# ---------------------------------------------------------------------------

def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Format A:
    # {
    #   "135a2760": {"train": [...], "test": [...]},
    #   ...
    # }
    if isinstance(data, dict):
        if "train" in data and "test" in data:
            return {path.stem: data}

        tasks = {
            str(task_id): task
            for task_id, task in data.items()
            if isinstance(task, dict)
            and "train" in task
            and "test" in task
        }

        if tasks:
            return tasks

    # Format B:
    # [
    #   {"id": "135a2760", "train": [...], "test": [...]},
    #   ...
    # ]
    if isinstance(data, list):
        tasks: dict[str, dict[str, Any]] = {}

        for index, task in enumerate(data):
            if not isinstance(task, dict):
                continue
            if "train" not in task or "test" not in task:
                continue

            task_id = str(
                task.get("id")
                or task.get("task_id")
                or task.get("name")
                or f"task_{index:04d}"
            )
            tasks[task_id] = task

        if tasks:
            return tasks

    raise ValueError(
        "Could not recognize the ARC JSON structure. Expected either a "
        "dictionary keyed by task ID, one single task, or a list of tasks."
    )


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def percentage(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def format_pair_detail(index: int, pair: PairSummary) -> list[str]:
    lines = [
        f"    Pair {index}",
        (
            f"      Shape   : {pair.input_shape[0]}x{pair.input_shape[1]}"
            f" -> {pair.output_shape[0]}x{pair.output_shape[1]}"
            f" ({pair.size_relation})"
        ),
        (
            f"      Colors  : {sorted(pair.input_colors)}"
            f" -> {sorted(pair.output_colors)}"
        ),
        (
            f"      Objects : {pair.input_objects}"
            f" -> {pair.output_objects}"
            f" ({pair.object_relation})"
        ),
    ]

    if pair.changed_count is not None:
        lines.append(
            f"      Changes : {pair.changed_count} cells "
            f"({percentage(pair.changed_fraction)})"
        )
    else:
        lines.append("      Changes : different canvas sizes")

    if pair.colors_added:
        lines.append(f"      Added   : {sorted(pair.colors_added)}")

    if pair.colors_removed:
        lines.append(f"      Removed : {sorted(pair.colors_removed)}")

    if pair.action_candidates:
        lines.append(
            "      Signals : " + "; ".join(pair.action_candidates)
        )

    return lines


def format_task_report(
    task: TaskHeadline,
    details: bool,
) -> list[str]:
    lines = [
        task.task_id,
        "-" * 72,
        f"HEADLINE   : {task.headline}",
        (
            f"PAIRS      : {task.training_pairs} train, "
            f"{task.test_pairs} test"
        ),
        f"SIZE       : {task.size_story}",
        f"COLORS     : {task.color_story}",
        f"OBJECTS    : {task.object_story}",
        f"CONSISTENCY: {task.consistency}",
    ]

    if details:
        lines.append("PAIR DETAILS:")

        for index, pair in enumerate(task.pair_summaries, start=1):
            lines.extend(format_pair_detail(index, pair))

    lines.append("")
    return lines


def build_report(
    headlines: list[TaskHeadline],
    source_path: Path,
    details: bool,
) -> str:
    lines = [
        "=" * 72,
        "ARCs4 TASK HEADLINES",
        "=" * 72,
        f"Source file : {source_path}",
        f"Task count  : {len(headlines)}",
        (
            "Purpose     : Broad visual summaries only. "
            "No test prediction and no solving."
        ),
        "=" * 72,
        "",
    ]

    for task in headlines:
        lines.extend(format_task_report(task, details))

    return "\n".join(lines)


def headline_to_dict(task: TaskHeadline) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "training_pairs": task.training_pairs,
        "test_pairs": task.test_pairs,
        "headline": task.headline,
        "size_story": task.size_story,
        "color_story": task.color_story,
        "object_story": task.object_story,
        "consistency": task.consistency,
        "pairs": [
            {
                "input_shape": list(pair.input_shape),
                "output_shape": list(pair.output_shape),
                "size_relation": pair.size_relation,
                "input_colors": sorted(pair.input_colors),
                "output_colors": sorted(pair.output_colors),
                "colors_added": sorted(pair.colors_added),
                "colors_removed": sorted(pair.colors_removed),
                "changed_count": pair.changed_count,
                "changed_fraction": pair.changed_fraction,
                "input_background": pair.input_background,
                "output_background": pair.output_background,
                "input_objects": pair.input_objects,
                "output_objects": pair.output_objects,
                "object_relation": pair.object_relation,
                "input_bbox_shape": pair.input_bbox_shape,
                "output_bbox_shape": pair.output_bbox_shape,
                "input_repetition": pair.input_repetition,
                "output_repetition": pair.output_repetition,
                "input_symmetry": pair.input_symmetry,
                "output_symmetry": pair.output_symmetry,
                "action_candidates": pair.action_candidates,
            }
            for pair in task.pair_summaries
        ],
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print newspaper-style headlines describing the broad "
            "input-to-output story of ARC training tasks."
        )
    )

    parser.add_argument(
        "json_path",
        nargs="?",
        default="data/data.json",
        help="ARC JSON file. Default: data/data.json",
    )

    parser.add_argument(
        "--task",
        help="Show only one task ID.",
    )

    parser.add_argument(
        "--details",
        action="store_true",
        help="Include pair-level statistics under each headline.",
    )

    parser.add_argument(
        "--save",
        help="Also save the text report to this file.",
    )

    parser.add_argument(
        "--save-json",
        help="Save machine-readable headline results to this JSON file.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_path = Path(args.json_path)

    if not source_path.exists():
        print(f"ERROR: JSON file not found: {source_path.resolve()}")
        return 1

    try:
        tasks = load_tasks(source_path)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"ERROR: Could not load tasks: {error}")
        return 1

    if args.task:
        if args.task not in tasks:
            print(f"ERROR: Task ID not found: {args.task}")
            print(f"Available task count: {len(tasks)}")
            return 1

        tasks = {args.task: tasks[args.task]}

    headlines = [
        analyze_task(task_id, task)
        for task_id, task in tasks.items()
    ]

    report = build_report(
        headlines=headlines,
        source_path=source_path,
        details=args.details,
    )

    print(report)

    if args.save:
        save_path = Path(args.save)
        save_path.write_text(report, encoding="utf-8")
        print(f"Saved text report: {save_path.resolve()}")

    if args.save_json:
        json_path = Path(args.save_json)
        payload = {
            "source_file": str(source_path),
            "task_count": len(headlines),
            "tasks": [
                headline_to_dict(task)
                for task in headlines
            ],
        }
        json_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(f"Saved JSON report: {json_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

