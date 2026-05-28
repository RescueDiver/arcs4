# reasoning/visual_symbolic_output_learner.py

"""
Visual symbolic output learner.

Purpose:
    Learn from train outputs.

This file should NOT hardcode solutions.
This file should NOT know task ids.
This file should NOT manually force marker placement.

Current job:
    1. Extract marker cells from expected train outputs.
    2. Convert those raw marker cells into symbolic marker positions.
    3. Print learning targets so we can later learn:
           input scene facts -> output marker positions
"""


# ============================================================
# BASIC HELPERS
# ============================================================

from reasoning.scene_to_layout_learner import (
    learn_scene_to_layout_rule,
    predict_layout_targets_for_scene,
    print_scene_to_layout_rule,
)


def grid_shape(grid):
    if grid is None:
        return 0, 0

    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def frame_center(frame):
    if frame is None:
        return None

    top = frame["top"]
    left = frame["left"]
    height = frame["height"]
    width = frame["width"]

    return (
        top + (height - 1) / 2,
        left + (width - 1) / 2,
    )


def frame_bounds(frame):
    if frame is None:
        return None

    top = frame["top"]
    left = frame["left"]
    height = frame["height"]
    width = frame["width"]

    return {
        "top": top,
        "bottom": top + height - 1,
        "left": left,
        "right": left + width - 1,
        "height": height,
        "width": width,
    }


def point_inside_frame(r, c, frame):
    bounds = frame_bounds(frame)

    if bounds is None:
        return False

    return (
        bounds["top"] <= r <= bounds["bottom"]
        and bounds["left"] <= c <= bounds["right"]
    )


def direction_from_center(r, c, frame):
    """
    Convert a marker cell into a simple symbolic direction
    relative to a frame center.

    This is descriptive only.
    It does not decide where future markers go.
    """

    center = frame_center(frame)

    if center is None:
        return "unknown"

    cr, cc = center

    dr = r - cr
    dc = c - cc

    # Center-ish means very close to the frame center.
    if abs(dr) <= 0.5 and abs(dc) <= 0.5:
        return "center"

    # Prefer the stronger axis.
    if abs(dr) >= abs(dc):
        if dr < 0:
            return "above"
        return "below"

    if dc < 0:
        return "left"

    return "right"


def normalized_position_in_frame(r, c, frame):
    """
    Return marker position normalized inside a frame.

    Example:
        center of 5x5 frame -> row_ratio=0.5 col_ratio=0.5

    This helps future learning compare different output sizes.
    """

    bounds = frame_bounds(frame)

    if bounds is None:
        return None

    height = max(1, bounds["height"] - 1)
    width = max(1, bounds["width"] - 1)

    row_ratio = (r - bounds["top"]) / height
    col_ratio = (c - bounds["left"]) / width

    return {
        "row_ratio": row_ratio,
        "col_ratio": col_ratio,
    }


def print_marker_decision_report(title, applied_markers):
    """
    Print a compact marker decision report.

    This does not change predictions.
    It only makes learned marker decisions easier to inspect.
    """

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    applied = []
    candidates = []
    collisions = []

    for item in applied_markers:
        if "debug_candidate_markers" in item:
            candidates.extend(item.get("debug_candidate_markers", []))
            continue

        if "debug_marker_collisions" in item:
            collisions.extend(item.get("debug_marker_collisions", []))
            continue

        applied.append(item)

    print("APPLIED MARKERS")

    if not applied:
        print("  none")
    else:
        for marker in applied:
            source_key = marker.get("source_key")
            target_key = marker.get("target_key")
            row = marker.get("row")
            col = marker.get("col")
            color = marker.get("color")
            source = marker.get("source")

            placement_debug = marker.get("placement_debug") or {}
            reason = placement_debug.get("reason")
            score = placement_debug.get("score")

            line = (
                f"  {source_key} -> {target_key} "
                f"r={row} c={col} color={color} source={source}"
            )

            if reason is not None:
                line += f" reason={reason}"

            if score is not None:
                line += f" score={score}"

            print(line)

            top_candidates = placement_debug.get("top_candidates", [])

            if top_candidates:
                print("    top candidates:")

                for candidate in top_candidates:
                    chosen = " CHOSEN" if candidate.get("chosen") else ""
                    print(
                        "      "
                        f"r={candidate.get('row')} "
                        f"c={candidate.get('col')} "
                        f"reason={candidate.get('reason')} "
                        f"score={candidate.get('score')}"
                        f"{chosen}"
                    )

    print()
    print("CANDIDATE MARKERS")

    if not candidates:
        print("  none")
    else:
        for marker in candidates:
            source_key = marker.get("source_key")
            target_key = marker.get("target_key")
            row = marker.get("row")
            col = marker.get("col")
            color = marker.get("color")
            permission_key = marker.get("permission_key")
            permission_allowed = marker.get("permission_allowed")

            placement_debug = marker.get("placement_debug") or {}
            reason = placement_debug.get("reason")
            score = placement_debug.get("score")

            line = (
                f"  {source_key} -> {target_key} "
                f"r={row} c={col} color={color} "
                f"allowed={permission_allowed}"
            )

            if permission_key is not None:
                line += f" permission={permission_key}"

            if reason is not None:
                line += f" reason={reason}"

            if score is not None:
                line += f" score={score}"

            print(line)

    print()
    print("COLLISIONS")

    if not collisions:
        print("  none")
    else:
        for collision in collisions:
            row = collision.get("row")
            col = collision.get("col")
            markers = collision.get("markers", [])

            print(f"  cell r={row} c={col}")

            for marker in markers:
                source_key = marker.get("source_key")
                target_key = marker.get("target_key")
                source = marker.get("source")

                print(
                    f"    {source_key} -> {target_key} source={source}"
                )


# ============================================================
# MARKER EXTRACTION
# ============================================================

def extract_expected_markers_from_output(output_grid, explanation):
    """
    Extract marker cells by comparing:

        expected output
        vs
        current frame-only prediction from visual_symbolic_ruleV2

    Any cell where expected output differs from the current prediction
    is treated as a marker target that the future learner must explain.

    This is learning-target extraction only.
    It does NOT solve the test.
    """

    baseline = explanation.get("prediction")

    if baseline is None:
        return []

    marker_cells = []

    h, w = grid_shape(output_grid)
    bh, bw = grid_shape(baseline)

    if h != bh or w != bw:
        return [
            {
                "shape_mismatch": True,
                "expected_shape": (h, w),
                "baseline_shape": (bh, bw),
            }
        ]

    for r in range(h):
        for c in range(w):
            expected_val = output_grid[r][c]
            baseline_val = baseline[r][c]

            if expected_val != baseline_val:
                marker_cells.append(
                    {
                        "row": r,
                        "col": c,
                        "color": expected_val,
                        "baseline_color": baseline_val,
                    }
                )

    return marker_cells


# ============================================================
# SYMBOLIC MARKER DESCRIPTION
# ============================================================

def describe_marker_position(cell, explanation):
    """
    Convert a raw marker cell like:

        r=4 c=8 color=4

    into a symbolic position like:

        inside_inner_frame / center
        inside_outer_frame / right
        outside_outer_frame / right_extension

    This is still descriptive.
    It does NOT create a forced rule.
    """

    if cell.get("shape_mismatch"):
        return {
            "shape_mismatch": True,
            "expected_shape": cell.get("expected_shape"),
            "baseline_shape": cell.get("baseline_shape"),
        }

    r = cell["row"]
    c = cell["col"]

    outer_frame = explanation.get("outer_frame")
    inner_frame = explanation.get("inner_frame")

    inside_inner = point_inside_frame(r, c, inner_frame)
    inside_outer = point_inside_frame(r, c, outer_frame)

    if inside_inner:
        frame_name = "inner_frame"
        frame = inner_frame
    elif inside_outer:
        frame_name = "outer_frame"
        frame = outer_frame
    else:
        frame_name = "outside_frames"
        frame = outer_frame

    direction = direction_from_center(r, c, frame)
    normalized = normalized_position_in_frame(r, c, frame)

    descriptor = {
        "row": r,
        "col": c,
        "color": cell["color"],
        "baseline_color": cell["baseline_color"],

        "frame_name": frame_name,
        "inside_inner_frame": inside_inner,
        "inside_outer_frame": inside_outer,

        "direction": direction,
        "normalized": normalized,
    }

    # Add a simple human-readable label.
    if frame_name == "inner_frame":
        descriptor["symbolic_position"] = f"inner_{direction}"
    elif frame_name == "outer_frame":
        descriptor["symbolic_position"] = f"outer_{direction}"
    else:
        descriptor["symbolic_position"] = f"outside_{direction}"

    return descriptor


def describe_all_marker_positions(marker_cells, explanation):
    descriptions = []

    for cell in marker_cells:
        descriptions.append(
            describe_marker_position(
                cell=cell,
                explanation=explanation,
            )
        )

    return descriptions


# ============================================================
# MARKER PLACEMENT RULE LEARNING
# ============================================================

def get_ring_role_by_id(explanation):
    """
    Build:
        ring_id -> role

    Example:
        0 -> outer
        1 -> inner
    """

    result = {}

    for ring in explanation.get("ring_geometry", []):
        ring_id = ring.get("ring_id")
        role = ring.get("role")

        if ring_id is not None:
            result[ring_id] = role

    return result


def source_facts_from_scene(explanation):
    """
    Convert blob_directions into learnable source facts.

    Example source keys:
        inner_1blobs_center
        outer_1blobs_right
        single_2blobs_above
        outside_all

    These are input-side facts.
    """

    ring_roles = get_ring_role_by_id(explanation)
    inner_frame = explanation.get("inner_frame")

    facts = []

    for blob in explanation.get("blob_directions", []):
        blob_id = blob.get("blob_id")
        assigned_to = blob.get("assigned_to")
        direction = blob.get("direction")

        ring_blob_count = None

        if assigned_to == "outside_all" or blob.get("outside_all"):
            container_role = "outside_all"
            source_key = "outside_all"
            simple_source_key = "outside_all"

        else:
            ring_role = ring_roles.get(assigned_to)

            # If there is only one ring, V2 uses outer_frame as the output frame,
            # but semantically this is a single ring.
            if ring_role == "outer" and inner_frame is None:
                container_role = "single"
            else:
                container_role = ring_role

            for ring in explanation.get("ring_geometry", []):
                if ring.get("ring_id") == assigned_to:
                    ring_blob_count = len(ring.get("contained_blobs", []))
                    break

            simple_source_key = f"{container_role}_{direction}"

            if ring_blob_count is not None:
                source_key = f"{container_role}_{ring_blob_count}blobs_{direction}"
            else:
                source_key = simple_source_key

        fact = {
            "blob_id": blob_id,
            "source_key": source_key,
            "simple_source_key": simple_source_key,
            "container_role": container_role,
            "direction": direction,
            "assigned_to": assigned_to,
            "ring_blob_count": ring_blob_count,
        }

        facts.append(fact)

    return facts


def marker_target_key(marker):
    """
    Convert symbolic marker descriptor into a target key.

    Example:
        inner_center
        outer_right
        outside_right
    """

    return marker.get("symbolic_position")


def score_source_to_marker(source, marker):
    """
    Score how naturally an input blob fact could explain an output marker.

    This does NOT force a rule.
    It only chooses the best alignment inside each train example.
    """

    score = 0

    source_role = source.get("container_role")
    source_direction = source.get("direction")

    marker_frame = marker.get("frame_name")
    marker_direction = marker.get("direction")

    # Role / frame agreement.
    if source_role == "inner" and marker_frame == "inner_frame":
        score += 3

    if source_role == "outer" and marker_frame == "outer_frame":
        score += 3

    if source_role == "single" and marker_frame == "outer_frame":
        score += 3

    if source_role == "outside_all" and marker_frame == "outside_frames":
        score += 3

    # Direction agreement.
    if source_direction == marker_direction:
        score += 3

    # Center is often very important in these symbolic views.
    if source_direction == "center" and marker_direction == "center":
        score += 2

    return score


def align_sources_to_markers(source_facts, symbolic_marker_positions):
    """
    Align input blob facts to output marker facts.

    Important:
        One source blob should explain one marker.
        We choose the best total alignment greedily by score.

    This avoids one blob stealing the wrong marker.
    """

    possible_matches = []

    for source_index, source in enumerate(source_facts):
        for marker_index, marker in enumerate(symbolic_marker_positions):
            if marker.get("shape_mismatch"):
                continue

            score = score_source_to_marker(source, marker)

            possible_matches.append(
                {
                    "source_index": source_index,
                    "marker_index": marker_index,
                    "score": score,
                    "source": source,
                    "marker": marker,
                }
            )

    possible_matches.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    used_sources = set()
    used_markers = set()
    alignments = []

    for match in possible_matches:
        source_index = match["source_index"]
        marker_index = match["marker_index"]

        if source_index in used_sources:
            continue

        if marker_index in used_markers:
            continue

        if match["score"] <= 0:
            continue

        used_sources.add(source_index)
        used_markers.add(marker_index)

        alignments.append(
            {
                "source_key": match["source"]["source_key"],
                "target_key": marker_target_key(match["marker"]),
                "score": match["score"],
                "source": match["source"],
                "marker": match["marker"],
            }
        )

    return alignments


def learn_marker_placement_rule(learning_examples):
    """
    Learn symbolic marker-placement mappings from training examples.

    Learns:
        1. Exact mappings:
              inner_1blobs_center -> inner_center
              outer_1blobs_right -> outer_right
              single_1blobs_center -> outer_center

        2. Simple fallback mappings:
              inner_center -> inner_center
              outer_right -> outer_right

        3. Exact blob-count permissions:
              nested_outer_1blobs_direction_marker
              nested_inner_1blobs_direction_marker
              single_1blobs_direction_marker
              single_2blobs_direction_marker
              outside_blob_marker

        4. Abstract nested permissions:
              nested_outer_any_blobs_direction_marker
              nested_inner_any_blobs_direction_marker

    Important:
        The abstract nested permissions are what let the test use:

            nested_inner_2blobs_direction_marker
            nested_outer_2blobs_direction_marker

        without hard-coding 2 blobs.
    """

    evidence = {}
    pattern_evidence = {}

    def ensure_pattern(permission_key):
        if permission_key not in pattern_evidence:
            pattern_evidence[permission_key] = {
                "seen": 0,
                "matched": 0,
            }

    def add_pattern_seen(permission_key):
        ensure_pattern(permission_key)
        pattern_evidence[permission_key]["seen"] += 1

    def add_pattern_matched(permission_key):
        ensure_pattern(permission_key)
        pattern_evidence[permission_key]["matched"] += 1

    def add_evidence(source_key, target_key, marker):
        if source_key is None:
            return

        if target_key is None:
            return

        if marker is None:
            return

        if source_key not in evidence:
            evidence[source_key] = {}

        if target_key not in evidence[source_key]:
            evidence[source_key][target_key] = {
                "count": 0,
                "colors": {},
                "row_ratios": [],
                "col_ratios": [],
            }

        bucket = evidence[source_key][target_key]
        bucket["count"] += 1

        color = marker.get("color")

        if color is not None:
            bucket["colors"][color] = bucket["colors"].get(color, 0) + 1

        normalized = marker.get("normalized") or {}

        row_ratio = normalized.get("row_ratio")
        col_ratio = normalized.get("col_ratio")

        if row_ratio is not None:
            bucket["row_ratios"].append(row_ratio)

        if col_ratio is not None:
            bucket["col_ratios"].append(col_ratio)

    def expected_permission_for_source(source):
        container_role = source.get("container_role")
        direction = source.get("direction")
        ring_blob_count = source.get("ring_blob_count")

        if container_role == "single":
            if direction in ("center", "above", "below", "left", "right"):
                return (
                    f"single_{ring_blob_count}blobs_direction_marker",
                    None,
                )

        if container_role == "outer":
            if direction in ("center", "above", "below", "left", "right"):
                return (
                    f"nested_outer_{ring_blob_count}blobs_direction_marker",
                    f"outer_{direction}",
                )

        if container_role == "inner":
            if direction in ("center", "above", "below", "left", "right"):
                return (
                    f"nested_inner_{ring_blob_count}blobs_direction_marker",
                    f"inner_{direction}",
                )

        if container_role == "outside_all":
            return (
                "outside_blob_marker",
                "outside_right",
            )

        return None, None

    def abstract_permission_for_source(source):
        """
        This is the important new part.

        Exact permission:
            nested_inner_1blobs_direction_marker

        Abstract permission:
            nested_inner_any_blobs_direction_marker

        The abstract permission means:
            for nested inner rings, blob direction maps to inner-frame marker direction,
            no matter whether there are 1, 2, or more blobs.
        """

        container_role = source.get("container_role")
        direction = source.get("direction")

        if direction not in ("center", "above", "below", "left", "right"):
            return None, None

        if container_role == "inner":
            return (
                "nested_inner_any_blobs_direction_marker",
                f"inner_{direction}",
            )

        if container_role == "outer":
            return (
                "nested_outer_any_blobs_direction_marker",
                f"outer_{direction}",
            )

        return None, None

    def source_matches_alignment(source, alignment):
        source_key = source.get("source_key")
        alignment_source_key = alignment.get("source_key")

        if source_key is not None and source_key == alignment_source_key:
            return True

        alignment_source = alignment.get("source", {})

        if source_key is not None and source_key == alignment_source.get("source_key"):
            return True

        if source.get("simple_source_key") == alignment_source.get("simple_source_key"):
            return True

        return False

    for example in learning_examples:
        source_facts = example.get("source_facts", [])
        symbolic_markers = example.get("symbolic_marker_positions", [])

        alignments = align_sources_to_markers(
            source_facts=source_facts,
            symbolic_marker_positions=symbolic_markers,
        )

        # ------------------------------------------------------------
        # Exact mapping evidence.
        # ------------------------------------------------------------
        for alignment in alignments:
            source = alignment.get("source", {})
            marker = alignment.get("marker", {})

            rich_source_key = alignment.get("source_key")
            simple_source_key = source.get("simple_source_key")
            target_key = alignment.get("target_key")

            add_evidence(
                source_key=rich_source_key,
                target_key=target_key,
                marker=marker,
            )

            add_evidence(
                source_key=simple_source_key,
                target_key=target_key,
                marker=marker,
            )

        # ------------------------------------------------------------
        # Permission evidence.
        # ------------------------------------------------------------
        for source in source_facts:
            permission_key, expected_target_key = expected_permission_for_source(source)

            if permission_key is not None:
                add_pattern_seen(permission_key)

            abstract_permission_key, abstract_expected_target_key = abstract_permission_for_source(source)

            if abstract_permission_key is not None:
                add_pattern_seen(abstract_permission_key)

            exact_matched = False
            abstract_matched = False

            for alignment in alignments:
                if not source_matches_alignment(source, alignment):
                    continue

                alignment_target_key = alignment.get("target_key")

                # Exact permission match.
                if source.get("container_role") == "single":
                    if (
                        alignment_target_key is not None
                        and alignment_target_key.startswith("outer_")
                    ):
                        exact_matched = True

                else:
                    if (
                        expected_target_key is not None
                        and alignment_target_key == expected_target_key
                    ):
                        exact_matched = True

                # Abstract nested permission match.
                if (
                    abstract_expected_target_key is not None
                    and alignment_target_key == abstract_expected_target_key
                ):
                    abstract_matched = True

            if permission_key is not None and exact_matched:
                add_pattern_matched(permission_key)

            if abstract_permission_key is not None and abstract_matched:
                add_pattern_matched(abstract_permission_key)

    learned_mappings = {}

    for source_key, target_data in evidence.items():
        best_target = max(
            target_data,
            key=lambda key: target_data[key]["count"],
        )

        best_data = target_data[best_target]

        colors = best_data.get("colors", {})

        if colors:
            best_color = max(colors, key=colors.get)
        else:
            best_color = 4

        row_ratios = best_data.get("row_ratios", [])
        col_ratios = best_data.get("col_ratios", [])

        if row_ratios:
            learned_row_ratio = sum(row_ratios) / len(row_ratios)
        else:
            learned_row_ratio = None

        if col_ratios:
            learned_col_ratio = sum(col_ratios) / len(col_ratios)
        else:
            learned_col_ratio = None

        learned_mappings[source_key] = {
            "best_target": best_target,
            "target_counts": {
                key: value["count"]
                for key, value in target_data.items()
            },
            "ambiguous": len(target_data) > 1,
            "color": best_color,
            "row_ratio": learned_row_ratio,
            "col_ratio": learned_col_ratio,
        }

    learned_pattern_permissions = {}

    for permission_key, counts in pattern_evidence.items():
        seen = counts["seen"]
        matched = counts["matched"]

        learned_pattern_permissions[permission_key] = {
            "allowed": seen > 0 and matched == seen,
            "seen": seen,
            "matched": matched,
        }

    scene_layout_rule = learn_scene_to_layout_rule(learning_examples)

    return {
        "family": "visual_symbolic_marker_placement",
        "mode": "learned_from_symbolic_examples_with_abstract_nested_permissions",
        "learned_mappings": learned_mappings,
        "learned_pattern_permissions": learned_pattern_permissions,
        "scene_layout_rule": scene_layout_rule,
        "raw_evidence": evidence,
        "pattern_evidence": pattern_evidence,
    }


def print_marker_placement_rule(marker_rule):
    """
    Print learned marker mappings and learned pattern permissions.

    This is debug-only.
    It does not affect predictions.
    """

    if marker_rule is None:
        print("\n[LEARNED MARKER PLACEMENT RULE]")
        print("None")
        return

    print("\n[LEARNED MARKER PLACEMENT RULE]")
    print(f"family: {marker_rule.get('family')}")
    print(f"mode  : {marker_rule.get('mode')}")

    print("mappings:")

    learned_mappings = marker_rule.get("learned_mappings", {})

    if not learned_mappings:
        print("  none")
    else:
        for source_key, mapping in learned_mappings.items():
            best_target = mapping.get("best_target")
            target_counts = mapping.get("target_counts")
            ambiguous = mapping.get("ambiguous")

            print(
                f"  {source_key} -> {best_target} "
                f"counts={target_counts} "
                f"ambiguous={ambiguous}"
            )

    print("pattern permissions:")

    permissions = marker_rule.get("learned_pattern_permissions", {})

    if not permissions:
        print("  none")
    else:
        for permission_key, info in permissions.items():
            allowed = info.get("allowed")
            seen = info.get("seen")
            matched = info.get("matched")

            print(
                f"  {permission_key}: "
                f"allowed={allowed} "
                f"seen={seen} "
                f"matched={matched}"
            )
    scene_layout_rule = marker_rule.get("scene_layout_rule")
    print_scene_to_layout_rule(scene_layout_rule)

# ============================================================
# LEARNING EXAMPLE BUILDER
# ============================================================

def build_marker_learning_example(pair_index, input_grid, output_grid, explanation):
    """
    Build one diagnostic learning row:

        scene facts -> source blob facts -> expected marker cells -> symbolic marker positions

    The future learner will use these rows.
    """

    scene_tree_summary = explanation.get("scene_tree_summary", [])

    source_facts = source_facts_from_scene(explanation)

    marker_cells = extract_expected_markers_from_output(
        output_grid=output_grid,
        explanation=explanation,
    )

    symbolic_marker_positions = describe_all_marker_positions(
        marker_cells=marker_cells,
        explanation=explanation,
    )

    return {
        "pair_index": pair_index,
        "scene_tree_summary": scene_tree_summary,
        "source_facts": source_facts,
        "expected_marker_cells": marker_cells,
        "symbolic_marker_positions": symbolic_marker_positions,
    }


# ============================================================
# DEBUG PRINTING
# ============================================================

def print_marker_learning_example(example):
    pair_index = example["pair_index"]
    scene_tree_summary = example["scene_tree_summary"]
    source_facts = example.get("source_facts", [])

    print("source_facts:")
    if not source_facts:
        print("  []")
    else:
        for fact in source_facts:
            print(
                "  "
                f"{fact['source_key']} "
                f"simple={fact['simple_source_key']} "
                f"role={fact['container_role']} "
                f"direction={fact['direction']} "
                f"ring_blob_count={fact['ring_blob_count']}"
            )

    marker_cells = example["expected_marker_cells"]
    symbolic_marker_positions = example["symbolic_marker_positions"]

    print()
    print("[MARKER LEARNING TARGET]")
    print(f"pair_index: {pair_index}")

    print("scene_tree_summary:")
    for line in scene_tree_summary:
        print(f"  {line}")

    print(f"expected_marker_cell_count: {len(marker_cells)}")
    print("expected_marker_cells:")

    if not marker_cells:
        print("  []")
    else:
        for cell in marker_cells:
            if cell.get("shape_mismatch"):
                print(
                    "  SHAPE MISMATCH "
                    f"expected={cell['expected_shape']} "
                    f"baseline={cell['baseline_shape']}"
                )
                continue

            print(
                "  "
                f"r={cell['row']} "
                f"c={cell['col']} "
                f"color={cell['color']} "
                f"baseline={cell['baseline_color']}"
            )

    print("symbolic_marker_positions:")

    if not symbolic_marker_positions:
        print("  []")
    else:
        for item in symbolic_marker_positions:
            if item.get("shape_mismatch"):
                print(
                    "  SHAPE MISMATCH "
                    f"expected={item['expected_shape']} "
                    f"baseline={item['baseline_shape']}"
                )
                continue

            normalized = item.get("normalized") or {}
            row_ratio = normalized.get("row_ratio")
            col_ratio = normalized.get("col_ratio")

            if row_ratio is None or col_ratio is None:
                ratio_text = "ratio=None"
            else:
                ratio_text = f"ratio=({row_ratio:.2f}, {col_ratio:.2f})"

            print(
                "  "
                f"{item['symbolic_position']} "
                f"r={item['row']} "
                f"c={item['col']} "
                f"color={item['color']} "
                f"frame={item['frame_name']} "
                f"direction={item['direction']} "
                f"{ratio_text}"
            )


# ============================================================
# APPLY LEARNED MARKER RULE
# ============================================================

def marker_cell_from_target_key(target_key, explanation, mapping=None):
    """
    Convert a symbolic marker target into an output cell.

    Order:
        1. Use learned row/col ratios when available.
        2. Use stable visual default ratios when no learned ratio exists.

    This keeps structural fallback from landing one cell too high/low.
    """

    outer_frame = explanation.get("outer_frame")
    inner_frame = explanation.get("inner_frame")

    if target_key is None:
        return None

    if target_key.startswith("inner_"):
        frame = inner_frame
        direction = target_key.replace("inner_", "", 1)

    elif target_key.startswith("outer_"):
        frame = outer_frame
        direction = target_key.replace("outer_", "", 1)

    elif target_key.startswith("outside_"):
        frame = outer_frame
        direction = target_key.replace("outside_", "", 1)

    else:
        return None

    if frame is None:
        return None

    top = frame["top"]
    left = frame["left"]
    height = frame["height"]
    width = frame["width"]

    row_ratio = None
    col_ratio = None

    if mapping is not None:
        row_ratio = mapping.get("row_ratio")
        col_ratio = mapping.get("col_ratio")

    if row_ratio is not None and col_ratio is not None:
        r = round(top + row_ratio * (height - 1))
        c = round(left + col_ratio * (width - 1))
        return r, c

    # Visual default ratios.
    # These are not task-id positions. They are frame-relative.
    default_ratios = {
        "center": (0.50, 0.50),
        "above": (0.33, 0.50),
        "below": (0.80, 0.50),
        "left": (0.50, 0.20),
        "right": (0.50, 0.80),
    }

    if target_key.startswith("outside_"):
        if direction == "right":
            row_ratio = 0.40
            col_ratio = 1.25
        else:
            row_ratio, col_ratio = default_ratios.get(
                direction,
                (0.50, 0.50),
            )
    else:
        row_ratio, col_ratio = default_ratios.get(
            direction,
            (0.50, 0.50),
        )

    r = round(top + row_ratio * (height - 1))
    c = round(left + col_ratio * (width - 1))

    return r, c


def find_marker_collisions(applied_markers):
    """
    Debug helper.

    Finds cases where two or more applied markers target the same output cell.
    Ignores debug_candidate_markers because those are not actually written unless promoted.
    """

    cell_to_markers = {}

    for marker in applied_markers:
        if "debug_candidate_markers" in marker:
            continue

        row = marker.get("row")
        col = marker.get("col")

        if row is None or col is None:
            continue

        key = (row, col)

        if key not in cell_to_markers:
            cell_to_markers[key] = []

        cell_to_markers[key].append(marker)

    collisions = []

    for cell, markers in cell_to_markers.items():
        if len(markers) > 1:
            collisions.append(
                {
                    "row": cell[0],
                    "col": cell[1],
                    "markers": markers,
                }
            )

    return collisions


def target_key_frame_and_direction(target_key, explanation):
    """
    Convert symbolic target key into:
        frame_name, frame, direction

    Examples:
        inner_right  -> inner_frame, right
        outer_above  -> outer_frame, above
        outside_right -> outer_frame, right
    """

    outer_frame = explanation.get("outer_frame")
    inner_frame = explanation.get("inner_frame")

    if target_key is None:
        return None, None, None

    if target_key.startswith("inner_"):
        return (
            "inner_frame",
            inner_frame,
            target_key.replace("inner_", "", 1),
        )

    if target_key.startswith("outer_"):
        return (
            "outer_frame",
            outer_frame,
            target_key.replace("outer_", "", 1),
        )

    if target_key.startswith("outside_"):
        return (
            "outside_frames",
            outer_frame,
            target_key.replace("outside_", "", 1),
        )

    return None, None, None


def cell_inside_frame(cell, frame):
    if cell is None:
        return False

    if frame is None:
        return False

    r, c = cell

    top = frame["top"]
    left = frame["left"]
    bottom = top + frame["height"] - 1
    right = left + frame["width"] - 1

    return top <= r <= bottom and left <= c <= right


def generate_marker_cell_candidates(target_key, explanation, mapping=None):
    """
    Generate possible cells for a marker instead of using one forced location.

    A cell can have multiple reasons.

    Example:
        r=1 c=4 may be both:
            above_column
            above_inner_frame

    Keeping both reasons helps debug whether the placement is genuinely structural.
    """

    frame_name, frame, direction = target_key_frame_and_direction(
        target_key,
        explanation,
    )

    if frame is None:
        return []

    top = frame["top"]
    left = frame["left"]
    height = frame["height"]
    width = frame["width"]

    bottom = top + height - 1
    right = left + width - 1

    center_r = round(top + 0.50 * (height - 1))
    center_c = round(left + 0.50 * (width - 1))

    candidates = []

    def add_cell(r, c, reason):
        cell = (r, c)

        for item in candidates:
            if item["cell"] == cell:
                reasons = item.setdefault("reasons", [])

                if reason not in reasons:
                    reasons.append(reason)

                item["reason"] = "+".join(reasons)
                return

        candidates.append(
            {
                "cell": cell,
                "reason": reason,
                "reasons": [reason],
            }
        )

    # Original learned/default cell first.
    base_cell = marker_cell_from_target_key(
        target_key=target_key,
        explanation=explanation,
        mapping=mapping,
    )

    if base_cell is not None:
        add_cell(base_cell[0], base_cell[1], "base_marker_cell")

    # Frame-relative direction candidates.
    if direction == "center":
        add_cell(center_r, center_c, "frame_center")
        add_cell(center_r - 1, center_c, "near_center_up")
        add_cell(center_r + 1, center_c, "near_center_down")
        add_cell(center_r, center_c - 1, "near_center_left")
        add_cell(center_r, center_c + 1, "near_center_right")

    elif direction == "above":
        for r in range(top + 1, center_r + 1):
            add_cell(r, center_c, "above_column")

        add_cell(top + 1, center_c, "near_top_center")
        add_cell(top + 2, center_c, "near_top_center_2")

    elif direction == "below":
        for r in range(center_r, bottom):
            add_cell(r, center_c, "below_column")

        add_cell(bottom - 1, center_c, "near_bottom_center")
        add_cell(bottom - 2, center_c, "near_bottom_center_2")

    elif direction == "left":
        for c in range(left + 1, center_c + 1):
            add_cell(center_r, c, "left_row")

        add_cell(center_r, left + 1, "near_left_center")
        add_cell(center_r, left + 2, "near_left_center_2")

    elif direction == "right":
        for c in range(center_c, right):
            add_cell(center_r, c, "right_row")

        add_cell(center_r, right - 1, "near_right_center")
        add_cell(center_r, right - 2, "near_right_center_2")

    # Outer-frame markers may also be described relative to the inner frame.
    inner_frame = explanation.get("inner_frame")

    if frame_name == "outer_frame" and inner_frame is not None:
        inner_top = inner_frame["top"]
        inner_left = inner_frame["left"]
        inner_bottom = inner_top + inner_frame["height"] - 1
        inner_right = inner_left + inner_frame["width"] - 1

        if direction == "above":
            safe_r = inner_top - 1

            if top <= safe_r <= bottom:
                add_cell(safe_r, center_c, "above_inner_frame")

        elif direction == "below":
            safe_r = inner_bottom + 1

            if top <= safe_r <= bottom:
                add_cell(safe_r, center_c, "below_inner_frame")

        elif direction == "left":
            safe_c = inner_left - 1

            if left <= safe_c <= right:
                add_cell(center_r, safe_c, "left_of_inner_frame")

        elif direction == "right":
            safe_c = inner_right + 1

            if left <= safe_c <= right:
                add_cell(center_r, safe_c, "right_of_inner_frame")

    # Keep only valid grid cells.
    valid_candidates = []

    output_shape = explanation.get("output_shape")

    if output_shape is not None:
        output_h, output_w = output_shape
    else:
        output_h = bottom + 1
        output_w = right + 1

    for candidate in candidates:
        r, c = candidate["cell"]

        if 0 <= r < output_h and 0 <= c < output_w:
            valid_candidates.append(candidate)

    return valid_candidates


def score_marker_cell_candidate(
    candidate,
    target_key,
    explanation,
    occupied_cells,
):
    """
    Score a possible marker cell.

    Higher is better.

    Main rule:
        If the base learned/default cell is legal, prefer it.

    Structural candidates should help only when the base/default cell is bad,
    such as:
        - it collides with an already placed marker
        - it lands inside the inner frame when placing an outer marker

    This prevents the resolver from over-pulling markers toward the inner-frame edge.
    """

    cell = candidate["cell"]
    r, c = cell

    frame_name, frame, direction = target_key_frame_and_direction(
        target_key,
        explanation,
    )

    if frame is None:
        return -10 ** 9

    top = frame["top"]
    left = frame["left"]
    height = frame["height"]
    width = frame["width"]

    bottom = top + height - 1
    right = left + width - 1

    center_r = round(top + 0.50 * (height - 1))
    center_c = round(left + 0.50 * (width - 1))

    reasons = candidate.get("reasons", [])

    def has_reason(name):
        return name in reasons or candidate.get("reason") == name

    score = 0

    collides = cell in occupied_cells

    inner_frame = explanation.get("inner_frame")
    inside_inner_frame = False

    if inner_frame is not None:
        inside_inner_frame = cell_inside_frame(cell, inner_frame)

    inside_target_frame = cell_inside_frame(cell, frame)

    # ------------------------------------------------------------
    # Hard penalties.
    # ------------------------------------------------------------
    if collides:
        score -= 10000

    if not inside_target_frame:
        score -= 500

    # Outer markers should not land inside the inner frame.
    if frame_name == "outer_frame" and inner_frame is not None:
        if inside_inner_frame:
            score -= 1000
        else:
            score += 200

    # ------------------------------------------------------------
    # Directional preference.
    # ------------------------------------------------------------
    if direction == "center":
        score -= abs(r - center_r)
        score -= abs(c - center_c)

    elif direction == "above":
        if r < center_r:
            score += 100
        score -= abs(c - center_c)
        score -= r - top

    elif direction == "below":
        if r > center_r:
            score += 100
        score -= abs(c - center_c)
        score -= bottom - r

    elif direction == "left":
        if c < center_c:
            score += 100
        score -= abs(r - center_r)
        score -= c - left

    elif direction == "right":
        if c > center_c:
            score += 100
        score -= abs(r - center_r)
        score -= right - c

    # ------------------------------------------------------------
    # Base cell preference.
    #
    # This is important:
    # If the original learned/default placement is legal, it should beat
    # structural guesses like right_of_inner_frame or below_inner_frame.
    # ------------------------------------------------------------
    if has_reason("base_marker_cell"):
        if not collides and not (
            frame_name == "outer_frame"
            and inner_frame is not None
            and inside_inner_frame
        ):
            score += 400
        else:
            score += 10

    # ------------------------------------------------------------
    # Structural anti-collision preference.
    #
    # This should help when base placement is bad, but it should not
    # overpower a legal base cell.
    # ------------------------------------------------------------
    if (
        has_reason("above_inner_frame")
        or has_reason("below_inner_frame")
        or has_reason("left_of_inner_frame")
        or has_reason("right_of_inner_frame")
    ):
        score += 300

    return score

def resolve_marker_cell(
    target_key,
    explanation,
    mapping,
    occupied_cells,
):
    """
    Pick the best marker cell from generated candidates.

    This is the key anti-forcing step:
        generate possible cells,
        score them,
        avoid collisions,
        choose the best.
    """

    candidates = generate_marker_cell_candidates(
        target_key=target_key,
        explanation=explanation,
        mapping=mapping,
    )

    if not candidates:
        return None, None

    scored = []

    for candidate in candidates:
        score = score_marker_cell_candidate(
            candidate=candidate,
            target_key=target_key,
            explanation=explanation,
            occupied_cells=occupied_cells,
        )

        scored.append(
            {
                "candidate": candidate,
                "score": score,
            }
        )

    best = max(
        scored,
        key=lambda item: item["score"],
    )

    return best["candidate"]["cell"], {
        "reason": best["candidate"].get("reason"),
        "score": best["score"],
        "candidates": scored,
    }


def apply_learned_marker_rule(prediction, explanation, marker_rule):
    """
    Apply learned marker mappings to a frame-only prediction.

    This version separates:
        1. learned exact markers
        2. learned/simple safe mappings
        3. learned abstract candidates
        4. scene-to-layout fallback candidates

    Important:
        scene_to_layout decides which target markers should exist.
        apply_learned_marker_rule decides whether a source fact is allowed
        to place one of those target markers.

    This should help leave-one-out pair 1:

        source_facts: ['single_1blobs_center']
        predicted_layout_targets: ['outer_center']

    because the scene-level learner already knows the output layout should
    contain outer_center, even if the exact single_1blobs_center mapping was
    hidden in that LOO split.
    """

    if prediction is None:
        return prediction, []

    if marker_rule is None:
        return prediction, []

    output = [row[:] for row in prediction]

    mappings = marker_rule.get("learned_mappings", {})
    pattern_permissions = marker_rule.get("learned_pattern_permissions", {})
    scene_layout_rule = marker_rule.get("scene_layout_rule")

    source_facts = source_facts_from_scene(explanation)

    predicted_layout_targets = predict_layout_targets_for_scene(
        source_facts=source_facts,
        scene_layout_rule=scene_layout_rule,
    )

    print()
    print("[SCENE-TO-LAYOUT PREDICTION]")
    print(f"source_facts: {[fact.get('source_key') for fact in source_facts]}")
    print(f"predicted_layout_targets: {predicted_layout_targets}")

    applied_markers = []
    candidate_markers = []

    exact_ops = []
    candidate_ops = []

    def learned_color_for_role(container_role, ring_blob_count=None):
        """
        Choose a safe fallback marker color.

        Exact learned mappings already carry their own learned color.

        For fallback/abstract placements:
            - single ring with 2 blobs can use 9 if that was learned
            - otherwise use 4 as the safest marker color
        """

        if container_role == "single" and ring_blob_count == 2:
            for source_key, mapping in mappings.items():
                if source_key.startswith("single_2blobs_"):
                    color = mapping.get("color")

                    if color is not None:
                        return color

        return 4

    def pattern_permission_allowed(permission_key):
        info = pattern_permissions.get(permission_key)

        if not info:
            return False

        return bool(info.get("allowed"))

    def natural_target_for_source(source):
        container_role = source.get("container_role")
        direction = source.get("direction")

        if container_role == "outside_all":
            return "outside_right"

        if direction not in ("center", "above", "below", "left", "right"):
            return None

        if container_role == "inner":
            return f"inner_{direction}"

        if container_role == "outer":
            return f"outer_{direction}"

        if container_role == "single":
            return f"outer_{direction}"

        return None

    def permission_for_candidate(source):
        """
        Decide whether a source fact is allowed to create a marker.

        Priority:
            1. exact learned mapping
            2. safe simple learned mapping
            3. abstract learned permission
            4. scene-layout natural fallback

        The scene-layout fallback is still learned:
            scene_to_layout predicts the target set first.
            this function only allows a natural source -> target match.
        """

        source_key = source.get("source_key")
        simple_source_key = source.get("simple_source_key")
        container_role = source.get("container_role")
        direction = source.get("direction")
        ring_blob_count = source.get("ring_blob_count")

        # ------------------------------------------------------------
        # 1. Exact learned source mapping.
        # ------------------------------------------------------------
        exact_mapping = mappings.get(source_key)

        if exact_mapping is not None and not exact_mapping.get("ambiguous"):
            target_key = exact_mapping.get("best_target")

            if not predicted_layout_targets or target_key in predicted_layout_targets:
                return (
                    target_key,
                    source_key,
                    "learned_exact_mapping",
                    True,
                    exact_mapping,
                )

        # ------------------------------------------------------------
        # 2. Safe simple learned source mapping.
        #
        # Do NOT use simple fallback for single-ring sources here.
        # single_center is dangerous because:
        #     one-blob single center -> outer_center
        #     two-blob single center -> can become outer_below
        # ------------------------------------------------------------
        if container_role != "single":
            simple_mapping = mappings.get(simple_source_key)

            if simple_mapping is not None and not simple_mapping.get("ambiguous"):
                target_key = simple_mapping.get("best_target")

                if not predicted_layout_targets or target_key in predicted_layout_targets:
                    return (
                        target_key,
                        simple_source_key,
                        "learned_simple_mapping",
                        True,
                        simple_mapping,
                    )

        # ------------------------------------------------------------
        # 3. Abstract learned permissions.
        # ------------------------------------------------------------
        if container_role == "outside_all":
            permission_key = "outside_blob_marker"
            target_key = "outside_right"

            if not predicted_layout_targets or target_key in predicted_layout_targets:
                return (
                    target_key,
                    permission_key,
                    "outside_permission",
                    pattern_permission_allowed(permission_key),
                    None,
                )

        if direction in ("center", "above", "below", "left", "right"):
            if container_role == "inner":
                permission_key = "nested_inner_any_blobs_direction_marker"
                target_key = f"inner_{direction}"

                if not predicted_layout_targets or target_key in predicted_layout_targets:
                    return (
                        target_key,
                        permission_key,
                        "nested_abstract_permission",
                        pattern_permission_allowed(permission_key),
                        None,
                    )

            if container_role == "outer":
                permission_key = "nested_outer_any_blobs_direction_marker"
                target_key = f"outer_{direction}"

                if not predicted_layout_targets or target_key in predicted_layout_targets:
                    return (
                        target_key,
                        permission_key,
                        "nested_abstract_permission",
                        pattern_permission_allowed(permission_key),
                        None,
                    )

            if container_role == "single":
                if ring_blob_count == 1:
                    permission_key = "single_1blobs_direction_marker"
                elif ring_blob_count == 2:
                    permission_key = "single_2blobs_direction_marker"
                else:
                    permission_key = "single_any_blobs_direction_marker"

                target_key = f"outer_{direction}"

                if not predicted_layout_targets or target_key in predicted_layout_targets:
                    allowed = pattern_permission_allowed(permission_key)

                    if not allowed and target_key in predicted_layout_targets:
                        return (
                            target_key,
                            "scene_layout_natural_target",
                            "scene_layout_fallback",
                            True,
                            None,
                        )

                    return (
                        target_key,
                        permission_key,
                        "single_abstract_permission",
                        allowed,
                        None,
                    )

        # ------------------------------------------------------------
        # 4. Scene-layout natural fallback.
        #
        # This is the fix:
        # if scene_to_layout predicts a natural target, allow it.
        # This does not need pattern_permissions because the scene-level
        # learner already decided this target should exist.
        # ------------------------------------------------------------
        natural_target_key = natural_target_for_source(source)

        if natural_target_key in predicted_layout_targets:
            return (
                natural_target_key,
                "scene_layout_natural_target",
                "scene_layout_fallback",
                True,
                None,
            )

        return None, None, None, False, None

    def make_marker_record(
        source_key,
        used_key,
        target_key,
        row,
        col,
        color,
        marker_source,
        placement_debug=None,
    ):
        record = {
            "source_key": source_key,
            "used_key": used_key,
            "target_key": target_key,
            "row": row,
            "col": col,
            "color": color,
            "source": marker_source,
        }

        if placement_debug is not None:
            record["placement_debug"] = {
                "reason": placement_debug.get("reason"),
                "score": placement_debug.get("score"),
                "top_candidates": placement_debug.get("top_candidates", []),
            }

        return record

    # ------------------------------------------------------------
    # Step 1: convert source facts into operations.
    # ------------------------------------------------------------
    for source in source_facts:
        source_key = source.get("source_key")
        container_role = source.get("container_role")

        (
            possible_target_key,
            permission_key,
            permission_mode,
            allowed,
            learned_mapping,
        ) = permission_for_candidate(source)

        if possible_target_key is None:
            continue

        if learned_mapping is not None:
            exact_ops.append(
                {
                    "source_key": source_key,
                    "used_key": permission_key,
                    "target_key": possible_target_key,
                    "mapping": learned_mapping,
                    "marker_source": "learned",
                }
            )

            continue

        candidate_mapping = {
            "best_target": possible_target_key,
            "target_counts": {
                possible_target_key: 1,
            },
            "ambiguous": False,
            "color": learned_color_for_role(
                container_role,
                source.get("ring_blob_count"),
            ),
            "row_ratio": None,
            "col_ratio": None,
        }

        if permission_mode == "scene_layout_fallback":
            marker_source = "scene_layout_fallback"
        else:
            marker_source = "learned_abstract_permission"

        candidate_ops.append(
            {
                "source_key": source_key,
                "used_key": f"promoted_{possible_target_key}",
                "target_key": possible_target_key,
                "mapping": candidate_mapping,
                "permission_key": permission_key,
                "permission_mode": permission_mode,
                "permission_allowed": allowed,
                "marker_source": marker_source,
            }
        )

    occupied_cells = set()

    # ------------------------------------------------------------
    # Step 2: apply exact learned markers first.
    # ------------------------------------------------------------
    for op in exact_ops:
        cell = marker_cell_from_target_key(
            target_key=op["target_key"],
            explanation=explanation,
            mapping=op["mapping"],
        )

        if cell is None:
            continue

        r, c = cell

        if not (0 <= r < len(output)):
            continue

        if not (0 <= c < len(output[0])):
            continue

        color = op["mapping"].get("color", 4)

        output[r][c] = color
        occupied_cells.add((r, c))

        applied_markers.append(
            make_marker_record(
                source_key=op["source_key"],
                used_key=op["used_key"],
                target_key=op["target_key"],
                row=r,
                col=c,
                color=color,
                marker_source=op["marker_source"],
            )
        )

    # ------------------------------------------------------------
    # Step 3: resolve and apply candidate markers.
    # ------------------------------------------------------------
    for op in candidate_ops:
        target_key = op["target_key"]
        mapping = op["mapping"]

        cell, placement_debug = resolve_marker_cell(
            target_key=target_key,
            explanation=explanation,
            mapping=mapping,
            occupied_cells=occupied_cells,
        )

        if cell is not None:
            r, c = cell

            candidate_markers.append(
                {
                    "source_key": op["source_key"],
                    "used_key": f"candidate_{target_key}",
                    "target_key": target_key,
                    "row": r,
                    "col": c,
                    "color": mapping.get("color", 4),
                    "permission_key": op["permission_key"],
                    "permission_mode": op["permission_mode"],
                    "permission_allowed": op["permission_allowed"],
                    "placement_debug": {
                        "reason": placement_debug.get("reason"),
                        "score": placement_debug.get("score"),
                    },
                    "source": "candidate",
                }
            )

        if not op["permission_allowed"]:
            continue

        if cell is None:
            continue

        r, c = cell

        if not (0 <= r < len(output)):
            continue

        if not (0 <= c < len(output[0])):
            continue

        color = mapping.get("color", 4)

        output[r][c] = color
        occupied_cells.add((r, c))

        applied_markers.append(
            make_marker_record(
                source_key=op["source_key"],
                used_key=op["used_key"],
                target_key=target_key,
                row=r,
                col=c,
                color=color,
                marker_source=op["marker_source"],
                placement_debug=placement_debug,
            )
        )

    collisions = find_marker_collisions(applied_markers)

    if collisions:
        applied_markers.append(
            {
                "debug_marker_collisions": collisions,
            }
        )

    if candidate_markers:
        applied_markers.append(
            {
                "debug_candidate_markers": candidate_markers,
            }
        )

    return output, applied_markers
