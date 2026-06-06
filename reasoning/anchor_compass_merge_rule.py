# reasoning/anchor_compass_merge_rule.py

# =============================================================================
# ANCHOR-COMPASS MERGE RULE
# =============================================================================
#
# Extracted from the working visual_story_review.py Section 24 path.
#
# Intended behavior for task 2d0172a1:
#   TEST 0 -> return None, let normal recursive fallback handle it
#   TEST 1 -> return anchor-compass merged grid
#
# Public function:
#   predict_anchor_compass_merge_for_pair(task, pair, test_index=None, debug=False)
#
# =============================================================================


# =============================================================================
# BASIC GRID HELPERS
# =============================================================================

def grid_shape(grid):
    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def in_bounds(grid, row, col):
    h, w = grid_shape(grid)
    return 0 <= row < h and 0 <= col < w


def count_colors(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    return counts


def get_bbox(cells):
    if not cells:
        return {
            "top": 0,
            "left": 0,
            "bottom": -1,
            "right": -1,
            "height": 0,
            "width": 0,
        }

    rows = [
        row
        for row, _ in cells
    ]

    cols = [
        col
        for _, col in cells
    ]

    top = min(rows)
    left = min(cols)
    bottom = max(rows)
    right = max(cols)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
    }


def bbox_area(bbox):
    return bbox["height"] * bbox["width"]


def bbox_strictly_contains(outer_bbox, inner_bbox):
    return (
        outer_bbox["top"] < inner_bbox["top"]
        and outer_bbox["left"] < inner_bbox["left"]
        and outer_bbox["bottom"] > inner_bbox["bottom"]
        and outer_bbox["right"] > inner_bbox["right"]
    )


# =============================================================================
# CONNECTED OBJECT DETECTION
# =============================================================================

def find_connected_components(grid, background_color):
    h, w = grid_shape(grid)

    seen = set()
    components = []

    object_id = 1

    for row in range(h):
        for col in range(w):
            value = grid[row][col]

            if value == background_color:
                continue

            if (row, col) in seen:
                continue

            color = value
            stack = [(row, col)]
            seen.add((row, col))
            cells = []

            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    next_row = current_row + dr
                    next_col = current_col + dc

                    if not in_bounds(grid, next_row, next_col):
                        continue

                    if (next_row, next_col) in seen:
                        continue

                    if grid[next_row][next_col] != color:
                        continue

                    seen.add((next_row, next_col))
                    stack.append((next_row, next_col))

            bbox = get_bbox(cells)

            components.append({
                "id": f"object_{object_id}",
                "color": color,
                "cells": cells,
                "cell_count": len(cells),
                "bbox": bbox,
            })

            object_id += 1

    return components


def analyze_grid(grid):
    color_counts = count_colors(grid)

    candidates = []

    for color, count in sorted(
        color_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        components = find_connected_components(
            grid,
            color,
        )

        candidates.append({
            "background": color,
            "background_count": count,
            "components": components,
            "component_count": len(components),
        })

    return {
        "color_counts": color_counts,
        "candidates": candidates,
    }


def select_display_candidate(analysis):
    candidates = analysis["candidates"]

    if not candidates:
        return {
            "background": 0,
            "background_count": 0,
            "components": [],
            "component_count": 0,
        }

    best_candidate = None
    best_score = -10**9

    for candidate in candidates:
        components = candidate["components"]
        component_count = candidate["component_count"]

        if component_count == 0:
            score = -10**9
        else:
            cell_counts = [
                obj["cell_count"]
                for obj in components
            ]

            largest_component = max(cell_counts) if cell_counts else 0
            total_visible = sum(cell_counts)
            largest_ratio = (
                largest_component / total_visible
                if total_visible
                else 1
            )

            small_object_count = sum(
                1
                for count in cell_counts
                if count <= 12
            )

            score = 0

            # Prefer seeing several meaningful objects.
            score += component_count * 20

            # Small objects matter because leaves/blobs become dots.
            score += small_object_count * 10

            # Avoid one giant inverse object.
            if largest_ratio > 0.90:
                score -= 80

            # Avoid noisy over-splitting.
            if component_count > 12:
                score -= 100

            # Mildly penalize huge visible mass.
            score -= largest_component * 0.05

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


# =============================================================================
# RELATIONSHIP + ROLE TREE
# =============================================================================

def edge_gaps(parent_bbox, child_bbox):
    return {
        "top_gap": child_bbox["top"] - parent_bbox["top"],
        "left_gap": child_bbox["left"] - parent_bbox["left"],
        "bottom_gap": parent_bbox["bottom"] - child_bbox["bottom"],
        "right_gap": parent_bbox["right"] - child_bbox["right"],
    }


def get_relationship_facts(grid):
    if grid is None:
        return {
            "background": 0,
            "objects": [],
            "relationships": [],
        }

    analysis = analyze_grid(grid)
    display_candidate = select_display_candidate(analysis)

    if display_candidate is None:
        color_counts = count_colors(grid)

        if color_counts:
            background = max(color_counts, key=color_counts.get)
        else:
            background = 0

        return {
            "background": background,
            "objects": [],
            "relationships": [],
        }

    objects = display_candidate.get("components", [])

    relationships = []

    for parent in objects:
        parent_bbox = parent["bbox"]

        for child in objects:
            if parent["id"] == child["id"]:
                continue

            child_bbox = child["bbox"]

            if not bbox_strictly_contains(parent_bbox, child_bbox):
                continue

            relationships.append({
                "parent_id": parent["id"],
                "child_id": child["id"],
                "parent_color": parent["color"],
                "child_color": child["color"],
                "parent_bbox": parent_bbox,
                "child_bbox": child_bbox,
                "parent_area": bbox_area(parent_bbox),
                "child_area": bbox_area(child_bbox),
                "gaps": edge_gaps(parent_bbox, child_bbox),
            })

    return {
        "background": display_candidate["background"],
        "objects": objects,
        "relationships": relationships,
    }


def find_immediate_parent(child, objects):
    child_bbox = child["bbox"]

    possible_parents = []

    for parent in objects:
        if parent["id"] == child["id"]:
            continue

        parent_bbox = parent["bbox"]

        if bbox_strictly_contains(parent_bbox, child_bbox):
            possible_parents.append(parent)

    if not possible_parents:
        return None

    possible_parents.sort(
        key=lambda obj: bbox_area(obj["bbox"])
    )

    return possible_parents[0]


def build_role_tree(grid):
    facts = get_relationship_facts(grid)
    objects = facts["objects"]

    nodes_by_id = {}

    for obj in objects:
        nodes_by_id[obj["id"]] = {
            "id": obj["id"],
            "color": obj["color"],
            "cell_count": obj["cell_count"],
            "bbox": obj["bbox"],
            "parent_id": None,
            "children": [],
            "role": "unknown",
        }

    for obj in objects:
        parent = find_immediate_parent(obj, objects)

        if parent is None:
            continue

        child_node = nodes_by_id[obj["id"]]
        parent_node = nodes_by_id[parent["id"]]

        child_node["parent_id"] = parent["id"]
        parent_node["children"].append(child_node)

    roots = []

    for node in nodes_by_id.values():
        if node["parent_id"] is None:
            roots.append(node)

    def assign_roles(node):
        if node["children"]:
            node["role"] = "container"
        else:
            node["role"] = "leaf"

        for child in node["children"]:
            assign_roles(child)

    for root in roots:
        assign_roles(root)

    roots.sort(
        key=lambda node: (
            node["bbox"]["top"],
            node["bbox"]["left"],
        )
    )

    for node in nodes_by_id.values():
        node["children"].sort(
            key=lambda child: (
                child["bbox"]["top"],
                child["bbox"]["left"],
            )
        )

    return {
        "background": facts["background"],
        "objects": objects,
        "roots": roots,
        "nodes_by_id": nodes_by_id,
    }


def role_tree_signature_from_node(node):
    if not node["children"]:
        return "leaf"

    child_signatures = [
        role_tree_signature_from_node(child)
        for child in node["children"]
    ]

    return "container[" + ",".join(child_signatures) + "]"


def collect_role_path_nodes_from_node(node, path, results):
    role = node.get("role", "unknown")
    children = node.get("children", [])

    signature = role_tree_signature_from_node(node)

    results.append({
        "path": path,
        "role": role,
        "signature": signature,
        "color": node.get("color"),
        "bbox": node.get("bbox"),
        "cell_count": node.get("cell_count"),
        "child_count": len(children),
    })

    for child_index, child in enumerate(children):
        child_path = f"{path}.child{child_index}"

        collect_role_path_nodes_from_node(
            child,
            child_path,
            results,
        )


def collect_role_path_nodes(grid):
    tree = build_role_tree(grid)
    results = []

    for root_index, root in enumerate(tree.get("roots", [])):
        path = f"root{root_index}"

        collect_role_path_nodes_from_node(
            root,
            path,
            results,
        )

    return results


# =============================================================================
# COMPASS HELPERS
# =============================================================================

def node_center_from_bbox(bbox):
    return (
        bbox["top"] + (bbox["height"] - 1) / 2,
        bbox["left"] + (bbox["width"] - 1) / 2,
    )


def local_center_from_parent(parent_bbox, child_bbox):
    return (
        child_bbox["top"] - parent_bbox["top"] + (child_bbox["height"] - 1) / 2,
        child_bbox["left"] - parent_bbox["left"] + (child_bbox["width"] - 1) / 2,
    )


def compass_bearing_from_delta(row_delta, col_delta):
    abs_row = abs(row_delta)
    abs_col = abs(col_delta)

    if abs_row == 0 and abs_col == 0:
        return "center"

    if abs_row >= abs_col:
        if row_delta < 0:
            return "north"

        return "south"

    if col_delta < 0:
        return "west"

    return "east"


# =============================================================================
# SECTION 24 EXTRACTED WORKING HELPERS
# =============================================================================

def crop_grid_to_bbox(grid, bbox):
    result = []

    for row in range(bbox["top"], bbox["bottom"] + 1):
        output_row = []

        for col in range(bbox["left"], bbox["right"] + 1):
            output_row.append(grid[row][col])

        result.append(output_row)

    return result


def section24_background_color_from_grid(grid):
    counts = {}

    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    return max(
        counts.items(),
        key=lambda item: item[1],
    )[0]


def section24_make_grid(height, width, background):
    return [
        [
            background
            for _ in range(width)
        ]
        for _ in range(height)
    ]


def section24_draw_cell(grid, row, col, color):
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        grid[row][col] = color


def section24_draw_template(grid, template_info, top, left, target_color):
    mask = template_info["mask"]

    for row_index, row in enumerate(mask):
        for col_index, value in enumerate(row):
            if value == 1:
                grid[top + row_index][left + col_index] = target_color


def section24_draw_container_border(grid, top, left, height, width, color):
    for col in range(left, left + width):
        section24_draw_cell(grid, top, col, color)
        section24_draw_cell(grid, top + height - 1, col, color)

    for row in range(top, top + height):
        section24_draw_cell(grid, row, left, color)
        section24_draw_cell(grid, row, left + width - 1, color)


def section24_learned_unique_template_for_node(task, node):
    signature = role_tree_signature_from_node(node)

    matches = []

    for pair in task.get("train", []):
        output_grid = pair.get("output")
        nodes = collect_role_path_nodes(output_grid)

        for item in nodes:
            if item["signature"] != signature:
                continue

            bbox = item.get("bbox")

            if bbox is None:
                continue

            template_grid = crop_grid_to_bbox(
                output_grid,
                bbox,
            )

            source_color = item.get("color")

            mask = []

            for row in template_grid:
                mask_row = []

                for value in row:
                    if value == source_color:
                        mask_row.append(1)
                    else:
                        mask_row.append(0)

                mask.append(mask_row)

            matches.append({
                "mask": mask,
                "source_color": source_color,
            })

    unique_masks = []

    for match in matches:
        key = tuple(
            tuple(row)
            for row in match["mask"]
        )

        if key not in unique_masks:
            unique_masks.append(key)

    if len(unique_masks) != 1:
        return None

    return {
        "mask": [
            list(row)
            for row in unique_masks[0]
        ],
    }


def section24_find_main_container_root_and_scene_leaves(pair):
    input_grid = pair.get("input")
    tree = build_role_tree(input_grid)

    roots = tree.get("roots", [])

    main_root = None
    outside_leaf_roots = []

    for root in roots:
        if root.get("role") == "container":
            if main_root is None:
                main_root = root
            else:
                current_area = (
                    root["bbox"]["height"]
                    * root["bbox"]["width"]
                )

                main_area = (
                    main_root["bbox"]["height"]
                    * main_root["bbox"]["width"]
                )

                if current_area > main_area:
                    main_root = root

        if root.get("role") == "leaf":
            outside_leaf_roots.append(root)

    return main_root, outside_leaf_roots


def section24_find_anchor_child(parent_node):
    children = parent_node.get("children", [])

    container_children = [
        child
        for child in children
        if child.get("role") == "container"
    ]

    if not container_children:
        return None

    container_children.sort(
        key=lambda child: child["bbox"]["height"] * child["bbox"]["width"],
        reverse=True,
    )

    return container_children[0]


def section24_bearing_from_child_to_anchor(parent_node, child_node, anchor_node):
    parent_bbox = parent_node.get("bbox")
    child_bbox = child_node.get("bbox")
    anchor_bbox = anchor_node.get("bbox")

    child_row, child_col = local_center_from_parent(
        parent_bbox,
        child_bbox,
    )

    anchor_row, anchor_col = local_center_from_parent(
        parent_bbox,
        anchor_bbox,
    )

    row_delta = child_row - anchor_row
    col_delta = child_col - anchor_col

    return compass_bearing_from_delta(
        row_delta,
        col_delta,
    )


def section24_bearing_from_scene_root_to_main(main_root, leaf_root):
    main_bbox = main_root.get("bbox")
    leaf_bbox = leaf_root.get("bbox")

    main_row, main_col = node_center_from_bbox(main_bbox)
    leaf_row, leaf_col = node_center_from_bbox(leaf_bbox)

    row_delta = leaf_row - main_row
    col_delta = leaf_col - main_col

    return compass_bearing_from_delta(
        row_delta,
        col_delta,
    )


def section24_make_anchor_compass_merge_sketch(task, pair):
    input_grid = pair.get("input")

    background = section24_background_color_from_grid(input_grid)

    main_root, outside_leaf_roots = section24_find_main_container_root_and_scene_leaves(
        pair,
    )

    if main_root is None:
        return None

    color = main_root.get("color")

    anchor_child = section24_find_anchor_child(main_root)

    if anchor_child is None:
        return None

    anchor_template = section24_learned_unique_template_for_node(
        task,
        anchor_child,
    )

    if anchor_template is None:
        return None

    anchor_mask = anchor_template["mask"]

    anchor_height = len(anchor_mask)
    anchor_width = len(anchor_mask[0]) if anchor_height else 0

    direct_leaf_children = [
        child
        for child in main_root.get("children", [])
        if child.get("role") == "leaf"
    ]

    bearing_records = []

    for child in direct_leaf_children:
        bearing = section24_bearing_from_child_to_anchor(
            main_root,
            child,
            anchor_child,
        )

        bearing_records.append({
            "source": "inside_parent",
            "bearing": bearing,
        })

    for leaf_root in outside_leaf_roots:
        bearing = section24_bearing_from_scene_root_to_main(
            main_root,
            leaf_root,
        )

        bearing_records.append({
            "source": "outside_scene_root",
            "bearing": bearing,
        })

    edge_gap = 2
    compass_gap = 2

    anchor_center_row = 0
    anchor_center_col = 0

    anchor_top = anchor_center_row - anchor_height // 2
    anchor_left = anchor_center_col - anchor_width // 2

    placements = []

    placements.append({
        "kind": "anchor",
        "top": anchor_top,
        "left": anchor_left,
        "height": anchor_height,
        "width": anchor_width,
    })

    distance_row = anchor_height // 2 + compass_gap
    distance_col = anchor_width // 2 + compass_gap

    for record in bearing_records:
        bearing = record["bearing"]

        leaf_row = anchor_center_row
        leaf_col = anchor_center_col

        if bearing == "north":
            leaf_row = anchor_center_row - distance_row

        elif bearing == "south":
            leaf_row = anchor_center_row + distance_row

        elif bearing == "west":
            leaf_col = anchor_center_col - distance_col

        elif bearing == "east":
            leaf_col = anchor_center_col + distance_col

        placements.append({
            "kind": "leaf",
            "top": leaf_row,
            "left": leaf_col,
            "height": 1,
            "width": 1,
            "bearing": bearing,
            "source": record["source"],
        })

    min_row = min(
        item["top"]
        for item in placements
    )

    min_col = min(
        item["left"]
        for item in placements
    )

    max_row = max(
        item["top"] + item["height"] - 1
        for item in placements
    )

    max_col = max(
        item["left"] + item["width"] - 1
        for item in placements
    )

    output_height = (max_row - min_row + 1) + edge_gap * 2
    output_width = (max_col - min_col + 1) + edge_gap * 2

    # Even-width buffer from the working debug sketch.
    if output_width % 2 == 1:
        output_width += 1

    row_shift = edge_gap - min_row
    col_shift = edge_gap - min_col

    grid = section24_make_grid(
        output_height,
        output_width,
        background,
    )

    section24_draw_container_border(
        grid,
        0,
        0,
        output_height,
        output_width,
        color,
    )

    for placement in placements:
        if placement["kind"] == "anchor":
            section24_draw_template(
                grid,
                anchor_template,
                placement["top"] + row_shift,
                placement["left"] + col_shift,
                color,
            )

        if placement["kind"] == "leaf":
            section24_draw_cell(
                grid,
                placement["top"] + row_shift,
                placement["left"] + col_shift,
                color,
            )

    return {
        "grid": grid,
        "height": output_height,
        "width": output_width,
        "anchor_size": f"{anchor_height}x{anchor_width}",
        "bearings": bearing_records,
    }


# =============================================================================
# PUBLIC RULE API
# =============================================================================

def make_optional_anchor_compass_test_grid(task, pair):
    # Do not use this path for train reconstruction.
    # Train pairs have an expected output.
    if "output" in pair:
        return None

    main_root, outside_leaf_roots = section24_find_main_container_root_and_scene_leaves(
        pair,
    )

    # This merge rule is only for scene-level outside leaves.
    # That protects TEST 0 from being changed by this rule.
    if not outside_leaf_roots:
        return None

    sketch = section24_make_anchor_compass_merge_sketch(
        task,
        pair,
    )

    if sketch is None:
        return None

    return sketch["grid"]


def predict_anchor_compass_merge_for_pair(task, pair, test_index=None, debug=False):
    grid = make_optional_anchor_compass_test_grid(
        task,
        pair,
    )

    if debug and test_index is not None:
        if grid is None:
            print(f"TEST {test_index} DRAW PATH: recursive_fallback")
        else:
            print(f"TEST {test_index} DRAW PATH: anchor_compass_merge")

    return grid


def predict_anchor_compass_merge_for_task(task, debug=False):
    predictions = []

    for test_index, pair in enumerate(task.get("test", [])):
        prediction = predict_anchor_compass_merge_for_pair(
            task=task,
            pair=pair,
            test_index=test_index,
            debug=debug,
        )

        predictions.append(prediction)

    return predictions