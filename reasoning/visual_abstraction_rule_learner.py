# reasoning/visual_abstraction_rule_learner.py

from reasoning.visual_abstraction_discovery import discover_visual_abstractions


# ============================================================
# VISUAL ABSTRACTION RULE LEARNER
# ============================================================
#
# This file learns WHICH visual abstraction explains a task.
#
# It does not force "rings and blobs."
# It asks:
#
#   Which view type best matches the train outputs?
#
# Current view candidates come from visual_abstraction_discovery.py:
#
#   largest_component_view
#   all_components_view
#   ring_blob_view
#   foreground_box_view
#
# First goal:
#   Make ring_blob_view win on 2d0172a1 because it explains:
#       - multiple rings
#       - blobs inside rings
#       - blobs outside all rings
#
# ============================================================


def grid_shape(grid):
    if grid is None:
        return 0, 0

    h = len(grid)
    w = len(grid[0]) if h else 0
    return h, w


def find_view(summary, view_type):
    for view in summary.get("views", []):
        if view.get("view_type") == view_type:
            return view

    return None


def output_has_nested_frame_style(output_grid):
    """
    Soft check:
    Does the output look like a framed / nested-frame object?

    This is not meant to be perfect.
    It just gives the learner evidence that ring/blob abstraction
    may explain the output.
    """
    if output_grid is None:
        return False

    h, w = grid_shape(output_grid)

    if h < 3 or w < 3:
        return False

    border_color = output_grid[0][0]

    # Outer border should mostly match the corner/border color.
    border_cells = []

    for c in range(w):
        border_cells.append(output_grid[0][c])
        border_cells.append(output_grid[h - 1][c])

    for r in range(1, h - 1):
        border_cells.append(output_grid[r][0])
        border_cells.append(output_grid[r][w - 1])

    same_border = sum(1 for v in border_cells if v == border_color)
    border_ratio = same_border / max(1, len(border_cells))

    if border_ratio < 0.85:
        return False

    # Look for at least one interior cell with the border color.
    # That suggests a nested marker/frame, not just a plain crop.
    interior_border_cells = 0

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if output_grid[r][c] == border_color:
                interior_border_cells += 1

    return interior_border_cells > 0


def count_output_marker_cells(output_grid):
    """
    Count interior cells that match the border color.

    In these ring/blob tasks, blobs often become interior markers.
    """
    if output_grid is None:
        return 0

    h, w = grid_shape(output_grid)

    if h < 3 or w < 3:
        return 0

    border_color = output_grid[0][0]

    count = 0

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if output_grid[r][c] == border_color:
                count += 1

    return count


def score_ring_blob_view(view, output_grid):
    """
    Score whether ring_blob_view explains the output.

    The goal is not exact prediction yet.
    The goal is choosing the right abstraction.
    """
    if view is None:
        return -10_000

    score = 0

    ring_count = view.get("ring_count", 0)
    blob_count = view.get("blob_count", 0)
    assignments = view.get("blob_assignments", [])

    if ring_count >= 1:
        score += 100

    if blob_count >= 1:
        score += 50

    # Multiple rings are valuable evidence.
    if ring_count >= 2:
        score += 100

    # If blobs are assigned to rings/outside, that is strong structure.
    if assignments:
        score += len(assignments) * 20

    if output_has_nested_frame_style(output_grid):
        score += 200

    marker_cells = count_output_marker_cells(output_grid)

    # More marker evidence in output supports blob logic.
    score += min(marker_cells, 20)

    return score


def score_largest_component_view(view, output_grid):
    """
    Largest-component view is a simpler fallback.
    """
    if view is None:
        return -10_000

    score = 0

    if view.get("main_component_id") is not None:
        score += 80

    extras = view.get("extra_component_ids", [])

    if extras:
        score += len(extras) * 10

    if output_has_nested_frame_style(output_grid):
        score += 80

    return score


def score_all_components_view(view, output_grid):
    """
    All-components view is useful when every object matters equally.
    """
    if view is None:
        return -10_000

    count = view.get("component_count", 0)

    return count * 20


def score_foreground_box_view(view, output_grid):
    """
    Foreground-box view is useful for simple crop/region tasks.
    """
    if view is None:
        return -10_000

    fg_box = view.get("foreground_box")

    if fg_box is None:
        return -10_000

    out_h, out_w = grid_shape(output_grid)

    score = 50

    # If output shape is close to foreground box shape, foreground box
    # may explain the task.
    score -= abs(fg_box["height"] - out_h)
    score -= abs(fg_box["width"] - out_w)

    return score


def score_view_type_on_pair(summary, output_grid, view_type):
    view = find_view(summary, view_type)

    if view_type == "ring_blob_view":
        return score_ring_blob_view(view, output_grid)

    if view_type == "largest_component_view":
        return score_largest_component_view(view, output_grid)

    if view_type == "all_components_view":
        return score_all_components_view(view, output_grid)

    if view_type == "foreground_box_view":
        return score_foreground_box_view(view, output_grid)

    return -10_000


def discover_visual_abstraction_rule_for_task(train_pairs):
    """
    Learn which visual abstraction type best explains all train pairs.
    """
    if not train_pairs:
        return None

    view_types = [
        "largest_component_view",
        "all_components_view",
        "ring_blob_view",
        "foreground_box_view",
    ]

    totals = {
        view_type: 0
        for view_type in view_types
    }

    pair_details = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        summary = discover_visual_abstractions(input_grid)

        pair_scores = {}

        for view_type in view_types:
            score = score_view_type_on_pair(
                summary=summary,
                output_grid=output_grid,
                view_type=view_type,
            )

            pair_scores[view_type] = score
            totals[view_type] += score

        pair_details.append({
            "pair_index": pair_index,
            "scores": pair_scores,
        })

    best_view_type = max(
        totals,
        key=lambda view_type: totals[view_type],
    )

    return {
        "family": "visual_abstraction_rule",
        "best_view_type": best_view_type,
        "scores": totals,
        "pair_details": pair_details,
    }


def print_visual_abstraction_rule(rule):
    if rule is None:
        print("No visual abstraction rule learned.")
        return

    print("\nVISUAL ABSTRACTION RULE")
    print("-" * 60)
    print(f"Best view type: {rule.get('best_view_type')}")

    print("\nTotal scores")
    print("-" * 60)

    for view_type, score in rule.get("scores", {}).items():
        print(f"{view_type:25s}: {score}")

    print("\nPair details")
    print("-" * 60)

    for detail in rule.get("pair_details", []):
        print(f"pair {detail['pair_index'] + 1}: {detail['scores']}")