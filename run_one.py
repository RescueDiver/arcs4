import json
import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm

from reasoning.task_router import (
    solve_pair_with_multiple_strategies,
    choose_task_level_strategy,
    solve_pair_with_forced_strategy,
)
from vision.global_pattern_reader import (
    read_global_pattern,
    print_global_pattern_summary,
)
from reasoning.pattern_break_detector import (
    detect_pattern_break,
    print_pattern_break_report,
)
from memory.rule_memory import save_successful_task_memory

# ============================================================
# CHANGE THIS TO THE TASK YOU WANT TO DEBUG
# ============================================================
TARGET_TASK_ID = "136b0064"


# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "data_failures", "extracted_tasks")


# ============================================================
# VISUAL BLOCKS
# ============================================================
def print_grid_color_blocks(grid, title="GRID"):
    """
    Print ARC grid as colored square emoji blocks.
    Good for quick visual reference in the console.
    """
    if grid is None:
        print(f"{title} COLOR VIEW: None")
        print()
        return

    color_map = {
        0: "⬛",  # black
        1: "🟦",  # blue
        2: "🟥",  # red
        3: "🟩",  # green
        4: "🟨",  # yellow
        5: "⬜",  # white
        6: "🟪",  # magenta
        7: "🟧",  # orange
        8: "🟫",  # brown
        9: "🟦",  # fallback/alt blue
    }

    h = len(grid)
    w = len(grid[0]) if h else 0
    print(f"{title} COLOR VIEW (h={h}, w={w})")
    for row in grid:
        print(" ".join(color_map.get(cell, "❓") for cell in row))
    print()


def show_grids_popup(input_grid, predicted_grid, expected_grid, title_prefix="PAIR"):
    """
    Show Input / Predicted / Expected side by side in a popup window.
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
        "#870C25",  # 9 dark red / brown-ish
    ]

    cmap = ListedColormap(arc_colors)
    norm = BoundaryNorm(np.arange(-0.5, 10.5, 1), cmap.N)

    grids = [
        ("Input", input_grid),
        ("Predicted", predicted_grid),
        ("Expected", expected_grid),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(title_prefix, fontsize=16)

    for ax, (name, grid) in zip(axes, grids):
        if grid is None:
            ax.set_title(f"{name}\n(None)")
            ax.axis("off")
            continue

        arr = np.array(grid)
        h, w = arr.shape

        ax.imshow(arr, cmap=cmap, norm=norm)

        ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(name)

    plt.tight_layout()
    plt.show()


# ============================================================
# HELPERS
# ============================================================
def load_task(task_id):
    filename = f"{task_id}.json"
    path = os.path.join(TASKS_DIR, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Task file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and task_id in raw:
        return raw[task_id]

    if isinstance(raw, dict) and ("train" in raw or "test" in raw):
        return raw

    raise ValueError(f"Unexpected task format in {path}")


def grid_shape(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def print_grid(title, grid):
    if grid is None:
        print(f"{title}: None")
        return

    h, w = grid_shape(grid)
    print(f"{title} (h={h}, w={w})")
    for row in grid:
        print(" ".join(str(v) for v in row))


def print_block_list(title, blocks):
    print(f"\n{title}")
    print("=" * len(title))

    if not blocks:
        print("  None")
        return

    for i, block in enumerate(blocks, start=1):
        color = block.get("color")
        local_x = block.get("local_x")
        min_r = block.get("min_r")
        patch = block.get("patch")

        print(f"\nBlock {i}")
        print(f"  color   : {color}")
        print(f"  local_x : {local_x}")
        print(f"  min_r   : {min_r}")

        if patch is None:
            print("  patch   : None")
        else:
            print_grid("  PATCH", patch)


def print_result_details(result):
    if result is None:
        print("Strategy: None")
        print("Score: None")
        print("Exact: False")
        return

    print(f"Strategy: {result.get('strategy')}")
    print(f"Score: {result.get('score')}")
    print(f"Exact: {result.get('exact', False)}")

    if "transform" in result:
        print(f"Transform: {result.get('transform')}")

    if "mode" in result:
        print(f"Mode: {result.get('mode')}")

    if "placement_mode" in result:
        print(f"Placement Mode: {result.get('placement_mode')}")

    if "v_align" in result:
        print(f"V Align: {result.get('v_align')}")

    if "h_align" in result:
        print(f"H Align: {result.get('h_align')}")

    if "group_size" in result:
        print(f"Group Size: {result.get('group_size')}")


# ============================================================
# MAIN
# ============================================================
def main():
    task = load_task(TARGET_TASK_ID)
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    task_level_choice = choose_task_level_strategy(train_pairs)
    forced_strategy = task_level_choice["best_strategy"]

    print("\n" + "=" * 60)
    print("TASK-LEVEL ROUTER CHOICE")
    print("=" * 60)
    print(f"Chosen strategy for this task: {forced_strategy}")

    for strategy_name, stats in sorted(task_level_choice["strategy_stats"].items()):
        print(
            f"  {strategy_name}: "
            f"exact={stats['exact_count']} "
            f"pairs_seen={stats['pair_count']} "
            f"total_adjusted={stats['total_adjusted_score']}"
        )

    print("=" * 60)
    print(f"RUNNING SINGLE TASK: {TARGET_TASK_ID}")
    print("=" * 60)
    print(f"Train pairs: {len(train_pairs)}")
    print(f"Test pairs : {len(test_pairs)}")

    total_right = 0
    total_wrong = 0
    strategy_wins = {}

    for idx, pair in enumerate(train_pairs, start=1):
        input_grid = pair["input"]
        output_grid = pair["output"]

        print("\n" + "-" * 60)
        print(f"TRAIN PAIR {idx}")
        print("-" * 60)

        # Pattern debug
        try:
            global_pattern = read_global_pattern(input_grid)
            if global_pattern is not None:
                print_global_pattern_summary(global_pattern)
        except Exception:
            pass

        try:
            break_report = detect_pattern_break(input_grid)
            if break_report is not None:
                print_pattern_break_report(break_report)
        except Exception:
            pass

        # Existing solver
        if forced_strategy is not None:
            result = solve_pair_with_forced_strategy(input_grid, output_grid, forced_strategy)
        else:
            result = solve_pair_with_multiple_strategies(input_grid, output_grid)

        predicted = result.get("predicted") if result else None

        # ======================================================
        # MOTIF ENGINE BLOCK VISUALS
        # ======================================================
        try:
            from reasoning.motif_projection_rule_engine import MotifEngine

            print("\n############################################")
            print("### ENTERING MOTIF ENGINE BLOCK VISUALS ###")
            print("############################################")

            engine = MotifEngine()

            left_grid, divider_col, right_grid = engine.split_sides(input_grid, divider_color=4)
            left_blocks = engine.get_blocks(left_grid)
            right_blocks = engine.get_blocks(right_grid)

            print(f"\nDivider col: {divider_col}")
            print_grid("LEFT SIDE GRID", left_grid)
            print()
            print_grid("RIGHT SIDE GRID", right_grid)

            print_block_list("LEFT BLOCKS", left_blocks)
            print_block_list("RIGHT BLOCKS", right_blocks)

            motif_predicted = engine.solve(input_grid, debug=True)

            print("\n=== MOTIF ENGINE OUTPUT ===")
            print_grid("MOTIF PREDICTED", motif_predicted)

        except Exception as e:
            print("\n[MOTIF ENGINE ERROR]")
            print(e)

        # Print chosen router result
        print_result_details(result)
        print()

        # INPUT
        print_grid("INPUT", input_grid)
        print_grid_color_blocks(input_grid, "INPUT")

        # EXPECTED
        print_grid("EXPECTED", output_grid)
        print_grid_color_blocks(output_grid, "EXPECTED")

        # PREDICTED
        print_grid("PREDICTED", predicted)
        print_grid_color_blocks(predicted, "PREDICTED")

        # Popup window
        try:
            show_grids_popup(
                input_grid,
                predicted,
                output_grid,
                title_prefix=f"{TARGET_TASK_ID} - TRAIN PAIR {idx}"
            )
        except Exception as e:
            print("\n[POPUP VISUAL ERROR]")
            print(e)

        if result is not None:
            strategy = result.get("strategy", "unknown")
            strategy_wins[strategy] = strategy_wins.get(strategy, 0) + 1

            if result.get("exact", False):
                total_right += 1
            else:
                total_wrong += 1
        else:
            total_wrong += 1

    print("\n" + "=" * 60)
    print(f"TASK SUMMARY: {TARGET_TASK_ID}")
    print("=" * 60)
    print(f"Right: {total_right}")
    print(f"Wrong: {total_wrong}")

    total = total_right + total_wrong
    if total > 0:
        percent_right = (total_right / total) * 100
        percent_wrong = (total_wrong / total) * 100
    else:
        percent_right = 0.0
        percent_wrong = 0.0

    print(f"Percent Right: {percent_right:.2f}%")
    print(f"Percent Wrong: {percent_wrong:.2f}%")

    print("Strategy Wins:")
    for strategy, count in sorted(strategy_wins.items()):
        print(f"  {strategy}: {count}")

    # Save successful task result into memory
    if forced_strategy is not None:
        save_successful_task_memory(
            task_id=TARGET_TASK_ID,
            strategy_name=forced_strategy,
            train_pairs=train_pairs,
            exact_count=total_right,
            total_pairs=len(train_pairs),
        )
        print("\nSaved task result to memory/rule_memory.json")

    if test_pairs:
        print("\n" + "=" * 60)
        print("TEST SOLUTION")
        print("=" * 60)

        for idx, pair in enumerate(test_pairs, start=1):
            input_grid = pair["input"]

            print(f"\nTEST PAIR {idx}")

            # 🚨 CALL STRATEGY DIRECTLY (NO ROUTER)
            if forced_strategy == "motif_layout_rule":
                from reasoning.motif_layout_rule import solve_pair_motif_layout_rule

                result = solve_pair_motif_layout_rule(input_grid, None)

            else:
                print(f"⚠️ Strategy {forced_strategy} not supported for direct test yet")
                result = None

            predicted = result.get("predicted") if result else None

            # PRINT
            print_grid("TEST INPUT", input_grid)
            print_grid_color_blocks(input_grid, "TEST INPUT")

            print_grid("PREDICTED", predicted)
            print_grid_color_blocks(predicted, "PREDICTED")

            # POPUP
            try:
                show_grids_popup(
                    input_grid,
                    predicted,
                    None,
                    title_prefix=f"{TARGET_TASK_ID} - TEST PAIR {idx}"
                )
            except Exception as e:
                print("\n[POPUP ERROR]")
                print(e)

if __name__ == "__main__":
    main()