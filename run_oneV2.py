import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from reasoning.seed_placement_expansion_rule import (
    discover_seed_placement_expansion_rule_for_task,
    apply_seed_placement_expansion_rule,
    solve_pair_seed_placement_expansion,
)
from reasoning.pattern_expansion_task_discovery import (
    discover_pattern_expansion_rule_for_task,
    apply_pattern_expansion_task_rule,
)

from reasoning.task_router import (
    get_all_strategy_results,
    choose_task_level_strategy,
    solve_pair_with_forced_strategy,
)
from reasoning.multi_seed_composition_rule import (
    discover_multi_seed_composition_rule_for_task,
    solve_pair_multi_seed_composition,
    apply_multi_seed_composition_rule,
)

# ============================================================
# TASK TO DEBUG
# ============================================================
# Change this ID when you want to debug a different ARC task.
# This file is meant for one-task-at-a-time investigation.
TARGET_TASK_ID = "269e22fb"

# Folder where this run_oneV2.py file lives.
# Used to build reliable paths to data/, data_failures/, etc.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# BASIC GRID HELPERS
# ============================================================
def grid_shape(grid):
    """
    Return grid height and width.

    ARC grids are lists of rows.
    If grid is None, return 0x0 so debug printing does not crash.
    """
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def print_grid(title, grid):
    """
    Print a numeric ARC grid in the terminal.

    This is the most useful view when debugging exact cell values.
    """
    if grid is None:
        print(f"{title}: None")
        return

    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")

    for row in grid:
        print(" ".join(str(v) for v in row))


def print_color_grid(title, grid):
    """
    Print a rough emoji color view in the terminal.

    This helps visually inspect shapes without opening the popup.
    """
    if grid is None:
        print(f"{title} COLOR VIEW: None")
        return

    color_map = {
        0: "⬛",
        1: "🟦",
        2: "🟥",
        3: "🟩",
        4: "🟨",
        5: "⬜",
        6: "🟪",
        7: "🟧",
        8: "🟫",
        9: "🟦",
    }

    h, w = grid_shape(grid)
    print(f"{title} COLOR VIEW (h={h}, w={w})")

    for row in grid:
        print(" ".join(color_map.get(v, "❓") for v in row))


def recolor_seed_match(expected_grid, seed_grid, start_r, start_c, highlight_color=9):
    """
    Return a copy of expected_grid where the matching seed area is recolored.

    This does not change the real expected grid.
    It only makes the seed location visually stand out.
    """
    marked = [row[:] for row in expected_grid]

    seed_h, seed_w = grid_shape(seed_grid)

    for r in range(seed_h):
        for c in range(seed_w):
            marked[start_r + r][start_c + c] = highlight_color

    return marked


def find_seed_inside_expected(seed_grid, expected_grid):
    """
    Search for the exact seed grid inside the expected output grid.

    Returns a list of top-left positions where the full seed appears.
    """
    seed_h, seed_w = grid_shape(seed_grid)
    exp_h, exp_w = grid_shape(expected_grid)

    matches = []

    for start_r in range(exp_h - seed_h + 1):
        for start_c in range(exp_w - seed_w + 1):
            ok = True

            for r in range(seed_h):
                for c in range(seed_w):
                    if expected_grid[start_r + r][start_c + c] != seed_grid[r][c]:
                        ok = False
                        break

                if not ok:
                    break

            if ok:
                matches.append((start_r, start_c))

    return matches


# ============================================================
# CROSS-SEED DETECTOR
# ============================================================
MIN_CROSS_SEED_MATCH_RATIO = 0.75


def rotate_grid_90(grid):
    return [list(row) for row in zip(*grid[::-1])]


def rotate_grid_180(grid):
    return rotate_grid_90(rotate_grid_90(grid))


def rotate_grid_270(grid):
    return rotate_grid_90(rotate_grid_180(grid))


def flip_grid_horizontal(grid):
    return [row[::-1] for row in grid]


def flip_grid_vertical(grid):
    return grid[::-1]


def get_grid_transforms(grid):
    return [
        ("identity", grid),
        ("rotate_90", rotate_grid_90(grid)),
        ("rotate_180", rotate_grid_180(grid)),
        ("rotate_270", rotate_grid_270(grid)),
        ("flip_horizontal", flip_grid_horizontal(grid)),
        ("flip_vertical", flip_grid_vertical(grid)),
    ]


def score_partial_shape_match(seed_grid, expected_grid, top, left):
    """
    Match a transformed seed inside an expected output.

    Allows:
    - rotation / flip
    - recoloring
    - color values changing

    Requires:
    - consistent color mapping
    - background staying background
    """
    seed_h, seed_w = grid_shape(seed_grid)
    exp_h, exp_w = grid_shape(expected_grid)

    color_map = {}
    reverse_map = {}

    checked = 0
    matched = 0

    for r in range(seed_h):
        for c in range(seed_w):
            sr = seed_grid[r][c]

            rr = top + r
            cc = left + c

            if not (0 <= rr < exp_h and 0 <= cc < exp_w):
                continue

            ev = expected_grid[rr][cc]

            if sr == 0:
                checked += 1
                if ev == 0:
                    matched += 1
                continue

            if ev == 0:
                checked += 1
                continue

            checked += 1

            if sr in color_map:
                if color_map[sr] == ev:
                    matched += 1
            else:
                if ev in reverse_map:
                    continue

                color_map[sr] = ev
                reverse_map[ev] = sr
                matched += 1

    if checked == 0:
        return 0.0, 0, 0

    return matched / checked, matched, checked





def debug_cross_seed_matches(train_pairs):
    """
    Search every expected output for every OTHER train input seed.
    """
    print("\n" + "=" * 60)
    print("CROSS-SEED MATCH DETECTOR")
    print("=" * 60)

    all_seeds = [pair["input"] for pair in train_pairs]

    for expected_index, pair in enumerate(train_pairs, start=1):
        expected_grid = pair["output"]

        print("\n" + "=" * 60)
        print(f"SEARCHING EXPECTED OUTPUT {expected_index}")
        print("=" * 60)

        for seed_index, seed_grid in enumerate(all_seeds, start=1):
            if seed_index == expected_index:
                continue

    print("\nSEED PLACEMENT TASK DISCOVERY")
    print("-" * 60)

    seed_rule = discover_seed_placement_expansion_rule_for_task(train_pairs)

    if seed_rule is None:
        print("No seed placement rule found.")
    else:
        print(f"Family      : {seed_rule['family']}")
        print(f"Rule type   : {seed_rule['rule_type']}")
        print(f"Anchor      : {seed_rule['anchor']}")
        print(f"Output shape: {seed_rule['output_shape']}")
        print(f"Fill color  : {seed_rule['fill_color']}")
        print(f"Total score : {seed_rule['total_score']}")
        print(f"Exact pairs : {seed_rule['exact_count']} / {seed_rule['pair_count']}")

        print("\nPer-pair seed placement:")
        for ex in seed_rule["examples"]:
            print(
                f"Pair {ex['pair_index'] + 1} | "
                f"found={ex['found']} | "
                f"anchor={ex['anchor']} | "
                f"score={ex['score']} | "
                f"exact={ex['exact']}"
            )
    print("\nMULTI-SEED COMPOSITION TASK DISCOVERY")
    print("-" * 60)

    multi_seed_rule = discover_multi_seed_composition_rule_for_task(train_pairs)

    if multi_seed_rule is None:
        print("No multi-seed composition rule found.")
    else:
        print(f"Family      : {multi_seed_rule['family']}")
        print(f"Rule type   : {multi_seed_rule['rule_type']}")
        print(f"Output shape: {multi_seed_rule['output_shape']}")
        print(f"Total score : {multi_seed_rule['total_score']}")
        print(f"Exact pairs : {multi_seed_rule['exact_count']} / {multi_seed_rule['pair_count']}")

        print("\nPer-pair multi-seed composition:")
        for ex in multi_seed_rule["examples"]:
            print(
                f"Pair {ex['pair_index'] + 1} | "
                f"score={ex['score']} | "
                f"exact={ex['exact']} | "
                f"placements={ex.get('placement_count', 0)}"
            )

            for placement in ex.get("placements", [])[:8]:
                percent = placement["ratio"] * 100
                print(
                    f"  seed={placement['seed_index']} "
                    f"transform={placement['transform']} "
                    f"top={placement['top']} "
                    f"left={placement['left']} "
                    f"size={placement['height']}x{placement['width']} "
                    f"match={percent:.1f}%"
                )
# ============================================================
# POPUP VISUALIZER
# ============================================================
def show_grids_popup(input_grid, expected_grid, predicted_grid, title_prefix="PAIR"):
    """
    Show three grids side-by-side:

        INPUT | EXPECTED | PREDICTED

    This is only for human debugging.
    It does not affect solver behavior.
    """
    arc_colors = [
        "#000000",  # 0 black
        "#0074D9",  # 1 blue
        "#FF4136",  # 2 red
        "#2ECC40",  # 3 green
        "#FFDC00",  # 4 yellow
        "#AAAAAA",  # 5 gray
        "#F012BE",  # 6 magenta
        "#FF851B",  # 7 orange
        "#7FDBFF",  # 8 light blue
        "#870C25",  # 9 dark red / brown
    ]

    cmap = ListedColormap(arc_colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    grids = [
        ("Input", input_grid),
        ("Expected", expected_grid),
        ("Predicted", predicted_grid),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title_prefix, fontsize=16)

    for ax, (name, grid) in zip(axes, grids):
        if grid is None:
            ax.set_title(f"{name}\n(None)")
            ax.axis("off")
            continue

        arr = np.array(grid)
        h, w = arr.shape

        ax.imshow(arr, cmap=cmap, norm=norm)

        # Draw cell borders so ARC shapes are easier to inspect.
        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


# ============================================================
# TASK LOADER
# ============================================================
def load_task(task_id):
    """
    Load a single ARC task by ID.

    This searches several common locations so run_oneV2.py works whether
    the task is in data/, data_failures/, or extracted failure tasks.

    Supported formats:
    1. {task_id: {"train": [...], "test": [...]}}
    2. {"train": [...], "test": [...]}
    3. {"some_only_key": {"train": [...], "test": [...]}}
    """
    exact_filename = f"{task_id}.json"

    short_id = task_id
    if len(task_id) > 3 and task_id[:3].isdigit():
        short_id = task_id[:3]

    candidate_paths = [
        os.path.join(BASE_DIR, "data_failures", "extracted_tasks", exact_filename),
        os.path.join(BASE_DIR, "data_failures", exact_filename),
        os.path.join(BASE_DIR, "data", exact_filename),
        os.path.join(BASE_DIR, "data", f"{short_id}.json"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            if isinstance(raw, dict) and task_id in raw:
                return raw[task_id]

            if isinstance(raw, dict) and ("train" in raw or "test" in raw):
                return raw

            if isinstance(raw, dict) and len(raw) == 1:
                only_value = next(iter(raw.values()))
                if isinstance(only_value, dict) and (
                    "train" in only_value or "test" in only_value
                ):
                    return only_value

            raise ValueError(f"Unexpected task format in {path}")

    raise FileNotFoundError(
        "Task file not found in any expected location:\n"
        + "\n".join(candidate_paths)
    )


# ============================================================
# CANDIDATE DEBUG SUMMARY
# ============================================================
def summarize_candidates(input_grid, output_grid, min_adjusted=-999999):
    """
    Print all useful strategy candidates for one train pair.

    Important:
    This is still pair-level diagnostic output.
    It is useful for debugging, but it is NOT the final ARC goal.

    The real goal is task-level rule discovery:
        use all train pairs together -> infer one rule -> apply to test.
    """
    candidates = get_all_strategy_results(input_grid, output_grid)

    useful = []

    for c in candidates:
        pred = c.get("predicted")

        # Hide candidates that failed to produce a prediction.
        if pred is None:
            continue

        adjusted = c.get("adjusted_score", c.get("score", 0))

        if adjusted < min_adjusted:
            continue

        useful.append(c)

    useful.sort(
        key=lambda x: x.get("adjusted_score", x.get("score", 0)),
        reverse=True,
    )

    print("\nUSEFUL CANDIDATES")
    print("-" * 60)

    if not useful:
        print("No useful candidates.")
        return []

    for c in useful:
        pred = c.get("predicted")
        ph, pw = grid_shape(pred)

        print(
            f"{c.get('strategy'):<28} "
            f"raw={c.get('raw_score', c.get('score')):<5} "
            f"adj={c.get('adjusted_score', c.get('score')):<5} "
            f"shape={ph}x{pw} "
            f"exact={c.get('exact', False)}"
        )

    return useful


# ============================================================
# DIRECT TEST SOLVER FALLBACK
# ============================================================
def solve_test_pair_direct(input_grid, forced_strategy):
    """
    Solve a test pair using the chosen strategy.

    Why this exists:
    Some strategy families expect an output_grid during training.
    Test pairs may not have an output_grid.
    So this avoids running the full router on test inputs.

    For pattern_expansion_rule, the preferred path is now:
        learn rule from all train pairs
        apply learned rule to test

    This function is only the fallback path.
    """
    if forced_strategy == "pattern_expansion_rule":
        from reasoning.pattern_expansion_rule import solve_pair_pattern_expansion

        result = solve_pair_pattern_expansion(input_grid, None)

        if result is not None:
            result["strategy"] = "pattern_expansion_rule"

        return result

    if forced_strategy == "object_grid_rule":
        from reasoning.object_grid_rule import solve_pair_object_grid_rule

        result = solve_pair_object_grid_rule(input_grid, None)

        if result is not None:
            result["strategy"] = "object_grid_rule"

        return result

    if forced_strategy == "pattern_rule":
        from reasoning.pattern_rule_engine import solve_pair_pattern_rule

        result = solve_pair_pattern_rule(input_grid, None)

        if result is not None:
            result["strategy"] = "pattern_rule"

        return result

    if forced_strategy == "motif_layout_rule":
        from reasoning.motif_layout_rule import solve_pair_motif_layout_rule

        result = solve_pair_motif_layout_rule(input_grid, None)

        if result is not None:
            result["strategy"] = "motif_layout_rule"

        return result

    if forced_strategy == "local_constraint_rule":
        print("local_constraint_rule needs train-pair learning for test.")
        return None

    print(f"Strategy {forced_strategy} not supported for direct test in run_oneV2 yet.")
    return None


# ============================================================
# MAIN DEBUG FLOW
# ============================================================
def main():
    """
    Main one-task debug workflow.

    Current intended ARC flow:

    1. Load one task.
    2. Use all train pairs to choose the best strategy family.
    3. If the chosen family is pattern_expansion_rule:
       learn ONE pattern expansion rule from all train pairs.
    4. Print train-pair diagnostics.
    5. Apply the learned task-level rule to test inputs.

    This keeps us aligned with ARC:
        not "solve each pair separately"
        but "infer the hidden task rule"
    """
    task = load_task(TARGET_TASK_ID)

    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print("=" * 60)
    print(f"RUN_ONE_V2 — TASK {TARGET_TASK_ID}")
    print("=" * 60)
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")
    debug_cross_seed_matches(train_pairs)
    # --------------------------------------------------------
    # Choose the best strategy family for the whole task.
    # This should use all train pairs, not just one pair.
    # --------------------------------------------------------
    task_choice = choose_task_level_strategy(train_pairs)
    forced_strategy = task_choice.get("best_strategy")

    print("\nTASK-LEVEL STRATEGY")
    print("-" * 60)
    print(f"Chosen strategy: {forced_strategy}")

    # --------------------------------------------------------
    # Special task-level learning for expansion tasks.
    #
    # This is the key upgrade:
    # Instead of letting each train pair choose its own expansion mode,
    # we try to learn ONE expansion rule that explains the full task.
    # --------------------------------------------------------
    learned_pattern_expansion_rule = None

    if forced_strategy == "pattern_expansion_rule":
        print("\nTASK-LEVEL PATTERN EXPANSION DISCOVERY")
        print("-" * 60)

        learned_pattern_expansion_rule = discover_pattern_expansion_rule_for_task(
            train_pairs
        )

        if learned_pattern_expansion_rule is None:
            print("No task-level pattern expansion rule found.")
        else:
            print(f"Learned family : {learned_pattern_expansion_rule['family']}")
            print(f"Default mode   : {learned_pattern_expansion_rule.get('default_mode')}")
            print(f"Rule type      : {learned_pattern_expansion_rule.get('rule_type')}")
            print(f"Total score    : {learned_pattern_expansion_rule['total_score']}")
            print(
                f"Exact pairs    : "
                f"{learned_pattern_expansion_rule['exact_count']} / {len(train_pairs)}"
            )

            print("\nPer-pair learned mapping:")
            for r in learned_pattern_expansion_rule["results"]:
                print(
                    f"Pair {r['pair_index'] + 1} | "
                    f"Mode: {r.get('mode')} | "
                    f"Score: {r['score']} | "
                    f"Exact: {r['exact']} | "
                    f"Features: "
                    f"h={r['features'].get('height')} "
                    f"w={r['features'].get('width')} "
                    f"bbox={r['features'].get('bbox_h')}x{r['features'].get('bbox_w')} "
                    f"colors={r['features'].get('color_count')}"
                )

    # --------------------------------------------------------
    # Print strategy-family summary.
    # This explains why the router chose the selected family.
    # --------------------------------------------------------
    print("\nSTRATEGY SUMMARY")
    print("-" * 60)

    for strategy, stats in sorted(
        task_choice.get("strategy_stats", {}).items(),
        key=lambda item: item[1].get("total_adjusted_score", 0),
        reverse=True,
    ):
        if stats.get("pair_count", 0) == 0:
            continue

        print(
            f"{strategy:<28} "
            f"exact={stats.get('exact_count', 0)} "
            f"pairs={stats.get('pair_count', 0)} "
            f"total_adj={stats.get('total_adjusted_score', 0)}"
        )

    total_right = 0
    total_wrong = 0

    # ========================================================
    # TRAIN PAIR DEBUGGING
    # ========================================================
    # This section shows what the chosen family does on each train pair.
    # It is diagnostic only.
    #
    # For ARC, we do not want the final test answer to come from
    # unrelated pair-by-pair choices. We want the learned task rule.
    # ========================================================
    for idx, pair in enumerate(train_pairs, start=1):
        input_grid = pair["input"]
        expected_grid = pair["output"]

        print("\n" + "=" * 60)
        print(f"TRAIN PAIR {idx}")
        print("=" * 60)

        ih, iw = grid_shape(input_grid)
        eh, ew = grid_shape(expected_grid)

        print(f"Input shape   : {ih}x{iw}")
        print(f"Expected shape: {eh}x{ew}")

        # ----------------------------------------------------
        # SEED-IN-EXPECTED CHECK
        # ----------------------------------------------------
        seed_matches = find_seed_inside_expected(input_grid, expected_grid)

        print("\nSEED-IN-EXPECTED CHECK")
        print("-" * 60)

        if not seed_matches:
            print("Seed grid was NOT found exactly inside expected output.")
        else:
            print(f"Seed grid found {len(seed_matches)} time(s): {seed_matches}")

            for match_index, (sr, sc) in enumerate(seed_matches, start=1):
                marked_expected = recolor_seed_match(
                    expected_grid,
                    input_grid,
                    sr,
                    sc,
                    highlight_color=9,
                )

                print()
                print_grid(
                    f"EXPECTED WITH SEED HIGHLIGHTED #{match_index} at r={sr}, c={sc}",
                    marked_expected,
                )

                print()
                print_color_grid(
                    f"EXPECTED WITH SEED HIGHLIGHTED #{match_index}",
                    marked_expected,
                )

                try:
                    show_grids_popup(
                        input_grid,
                        expected_grid,
                        marked_expected,
                        title_prefix=(
                            f"{TARGET_TASK_ID} - TRAIN PAIR {idx} "
                            f"SEED MATCH #{match_index} at r={sr}, c={sc}"
                        ),
                    )
                except Exception as e:
                    print("[POPUP ERROR]")
                    print(e)
        # Show all useful candidate families for this pair.
        summarize_candidates(input_grid, expected_grid)

        # Force the task-level chosen strategy so we can inspect it.
        result = solve_pair_with_forced_strategy(
            input_grid,
            expected_grid,
            forced_strategy,
        )

        predicted = result.get("predicted") if result else None
        score = result.get("score") if result else None
        adjusted = result.get("adjusted_score") if result else None
        exact = result.get("exact", False) if result else False

        print("\nCHOSEN RESULT")
        print("-" * 60)
        print(f"Strategy: {forced_strategy}")
        print(f"Score   : {score}")
        print(f"Adjusted: {adjusted}")
        print(f"Exact   : {exact}")

        if result and "mode" in result:
            print(f"Mode    : {result.get('mode')}")

        if exact:
            total_right += 1
        else:
            total_wrong += 1

        print("\nINPUT")
        print_grid("INPUT", input_grid)

        print("\nEXPECTED")
        print_grid("EXPECTED", expected_grid)

        print("\nPREDICTED")
        print_grid("PREDICTED", predicted)

        print("\nCOLOR VIEW")
        print_color_grid("INPUT", input_grid)
        print_color_grid("EXPECTED", expected_grid)
        print_color_grid("PREDICTED", predicted)

        try:
            show_grids_popup(
                input_grid,
                expected_grid,
                predicted,
                title_prefix=f"{TARGET_TASK_ID} - TRAIN PAIR {idx}",
            )
        except Exception as e:
            print("[POPUP ERROR]")
            print(e)

    # ========================================================
    # TRAIN SUMMARY
    # ========================================================
    print("\n" + "=" * 60)
    print("TASK SUMMARY")
    print("=" * 60)
    print(f"Right: {total_right}")
    print(f"Wrong: {total_wrong}")

    total = total_right + total_wrong

    if total:
        print(f"Percent right: {(total_right / total) * 100:.2f}%")

    # ========================================================
    # TEST PAIR SOLVING
    # ========================================================
    # Preferred path:
    #   If we learned a task-level pattern expansion rule,
    #   apply that learned rule to the test input.
    #
    # Fallback path:
    #   If no learned expansion rule exists,
    #   use the older direct forced-strategy test solver.
    # ========================================================
    if test_pairs:
        print("\n" + "=" * 60)
        print("TEST PAIRS")
        print("=" * 60)

        for idx, pair in enumerate(test_pairs, start=1):
            input_grid = pair["input"]
            expected_grid = pair.get("output")

            print("\n" + "-" * 60)
            print(f"TEST PAIR {idx}")
            print("-" * 60)

            if learned_pattern_expansion_rule is not None:
                predicted = apply_pattern_expansion_task_rule(
                    learned_pattern_expansion_rule,
                    input_grid,
                )
            else:
                result = solve_test_pair_direct(input_grid, forced_strategy)
                predicted = result.get("predicted") if result else None

            print_grid("TEST INPUT", input_grid)
            print()
            print_grid("EXPECTED", expected_grid)
            print()
            print_grid("PREDICTED", predicted)

            print()
            print_color_grid("INPUT", input_grid)
            print_color_grid("EXPECTED", expected_grid)
            print_color_grid("PREDICTED", predicted)

            try:
                show_grids_popup(
                    input_grid,
                    expected_grid,
                    predicted,
                    title_prefix=f"{TARGET_TASK_ID} - TEST PAIR {idx}",
                )
            except Exception as e:
                print("[POPUP ERROR]")
                print(e)


if __name__ == "__main__":
    main()