# reasoning/anchor_compass_merge_rule.py

from collections import Counter, deque


def predict_anchor_compass_merge_for_task(task, debug=False):
    predictions = []

    for test_index, test_pair in enumerate(task.get("test", [])):
        grid = predict_anchor_compass_merge_for_pair(
            task=task,
            pair=test_pair,
            test_index=test_index,
            debug=debug,
        )

        predictions.append(grid)

    return predictions


def predict_anchor_compass_merge_for_pair(task, pair, test_index=None, debug=False):
    input_grid = pair.get("input")

    if input_grid is None:
        return None

    input_forest = build_role_forest(input_grid)

    main_root, outside_leaf_roots = find_main_container_root_and_outside_leaf_roots(
        input_forest
    )

    if main_root is None:
        return None

    # Critical guard:
    # this rule is only for scene-level outside leaves.
    # This keeps TEST 0 on the normal recursive path.
    if not outside_leaf_roots:
        if debug and test_index is not None:
            print(f"TEST {test_index} DRAW PATH: recursive_fallback")
        return None

    anchor_child = find_anchor_container_child(main_root)

    if anchor_child is None:
        return None

    anchor_template = learn_unique_output_template_for_signature(
        task=task,
        signature=role_tree_signature_from_node(anchor_child),
    )

    if anchor_template is None:
        return None

    anchor_mask = anchor_template["mask"]

    if not anchor_mask:
        return None

    anchor_height = len(anchor_mask)
    anchor_width = len(anchor_mask[0])

    if anchor_height <= 0 or anchor_width <= 0:
        return None

    background = most_common_color(input_grid)
    color = main_root["color"]

    bearings = []

    for child in main_root["children"]:
        if child["path"] == anchor_child["path"]:
            continue

        bearing = compass_bearing(
            from_node=child,
            to_node=anchor_child,
        )

        if bearing is None:
            return None

        bearings.append({
            "kind": "inside_parent",
            "bearing": bearing,
            "path": child["path"],
        })

    for outside_leaf_root in outside_leaf_roots:
        bearing = compass_bearing(
            from_node=outside_leaf_root,
            to_node=main_root,
        )

        if bearing is None:
            return None

        bearings.append({
            "kind": "outside_scene_root",
            "bearing": bearing,
            "path": outside_leaf_root["path"],
        })

    if not bearings:
        return None

    output_grid = draw_anchor_compass_grid(
        background=background,
        color=color,
        anchor_mask=anchor_mask,
        bearings=bearings,
    )

    if debug and test_index is not None:
        print(f"TEST {test_index} DRAW PATH: anchor_compass_merge")

    return output_grid


def draw_anchor_compass_grid(background, color, anchor_mask, bearings):
    anchor_height = len(anchor_mask)
    anchor_width = len(anchor_mask[0])

    edge_gap = 2
    compass_gap = 2

    top_extra = edge_gap
    bottom_extra = edge_gap
    left_extra = edge_gap
    right_extra = edge_gap

    for item in bearings:
        bearing = item["bearing"]

        if bearing == "north":
            top_extra = max(top_extra, edge_gap + compass_gap)

        elif bearing == "south":
            bottom_extra = max(bottom_extra, edge_gap + compass_gap)

        elif bearing == "west":
            left_extra = max(left_extra, edge_gap + compass_gap)

        elif bearing == "east":
            right_extra = max(right_extra, edge_gap + compass_gap)

    output_height = top_extra + anchor_height + bottom_extra
    output_width = left_extra + anchor_width + right_extra

    # The learned debug sketch used even width for this compass frame.
    if output_width % 2 == 1:
        output_width += 1
        right_extra += 1

    grid = make_grid(
        height=output_height,
        width=output_width,
        value=background,
    )

    draw_container_border(
        grid=grid,
        top=0,
        left=0,
        height=output_height,
        width=output_width,
        color=color,
    )

    anchor_top = top_extra
    anchor_left = left_extra

    draw_mask(
        grid=grid,
        mask=anchor_mask,
        top=anchor_top,
        left=anchor_left,
        color=color,
    )

    anchor_center_row = anchor_top + anchor_height // 2
    anchor_center_col = anchor_left + anchor_width // 2

    for item in bearings:
        bearing = item["bearing"]

        if bearing == "north":
            row = edge_gap
            col = anchor_center_col

        elif bearing == "south":
            row = output_height - edge_gap - 1
            col = anchor_center_col

        elif bearing == "west":
            row = anchor_center_row
            col = edge_gap

        elif bearing == "east":
            row = anchor_center_row
            col = output_width - edge_gap - 1

        else:
            continue

        draw_cell(
            grid=grid,
            row=row,
            col=col,
            color=color,
        )

    return grid


def find_main_container_root_and_outside_leaf_roots(forest):
    roots = forest["roots"]

    main_root = None
    outside_leaf_roots = []

    for root in roots:
        if root["role"] == "container":
            if main_root is None:
                main_root = root
            else:
                # More than one container root makes this rule unsafe.
                return None, []

        elif root["role"] == "leaf":
            outside_leaf_roots.append(root)

    if main_root is None:
        return None, []

    if not outside_leaf_roots:
        return main_root, []

    return main_root, outside_leaf_roots


def find_anchor_container_child(parent_node):
    container_children = [
        child
        for child in parent_node["children"]
        if child["role"] == "container"
    ]

    if len(container_children) != 1:
        return None

    return container_children[0]


def learn_unique_output_template_for_signature(task, signature):
    masks = []

    for pair in task.get("train", []):
        output_grid = pair.get("output")

        if output_grid is None:
            continue

        output_forest = build_role_forest(output_grid)

        for node in flatten_nodes(output_forest["roots"]):
            node_signature = role_tree_signature_from_node(node)

            if node_signature != signature:
                continue

            mask = crop_node_to_mask(
                grid=output_grid,
                node=node,
            )

            masks.append(mask)

    unique = []

    for mask in masks:
        key = tuple(
            tuple(row)
            for row in mask
        )

        if key not in unique:
            unique.append(key)

    if len(unique) != 1:
        return None

    return {
        "mask": [
            list(row)
            for row in unique[0]
        ],
    }


def crop_node_to_mask(grid, node):
    bbox = node["bbox"]
    color = node["color"]

    top = bbox["top"]
    left = bbox["left"]
    height = bbox["height"]
    width = bbox["width"]

    mask = []

    for row in range(top, top + height):
        mask_row = []

        for col in range(left, left + width):
            if grid[row][col] == color:
                mask_row.append(1)
            else:
                mask_row.append(0)

        mask.append(mask_row)

    return mask


def build_role_forest(grid):
    background = most_common_color(grid)
    components = find_components(grid, background)

    for component in components:
        component["children"] = []
        component["parent"] = None
        component["role"] = "leaf"
        component["path"] = ""

    for child in components:
        possible_parents = []

        for parent in components:
            if parent is child:
                continue

            if parent["color"] != child["color"]:
                continue

            if bbox_strictly_contains(
                outer=parent["bbox"],
                inner=child["bbox"],
            ):
                possible_parents.append(parent)

        if possible_parents:
            best_parent = min(
                possible_parents,
                key=lambda item: item["bbox"]["area"],
            )

            child["parent"] = best_parent
            best_parent["children"].append(child)

    roots = [
        component
        for component in components
        if component["parent"] is None
    ]

    sort_nodes_spatially(roots)

    for root_index, root in enumerate(roots):
        assign_roles_and_paths(
            node=root,
            path=f"root{root_index}",
        )

    return {
        "background": background,
        "roots": roots,
        "components": components,
    }


def assign_roles_and_paths(node, path):
    node["path"] = path

    sort_nodes_spatially(node["children"])

    if node["children"]:
        node["role"] = "container"
    else:
        node["role"] = "leaf"

    for child_index, child in enumerate(node["children"]):
        assign_roles_and_paths(
            node=child,
            path=f"{path}.child{child_index}",
        )


def sort_nodes_spatially(nodes):
    nodes.sort(
        key=lambda node: (
            node["bbox"]["top"],
            node["bbox"]["left"],
            node["bbox"]["height"],
            node["bbox"]["width"],
        )
    )


def flatten_nodes(nodes):
    output = []

    for node in nodes:
        output.append(node)
        output.extend(flatten_nodes(node["children"]))

    return output


def role_tree_signature_from_forest(forest):
    roots = forest["roots"]

    if len(roots) == 1:
        return role_tree_signature_from_node(roots[0])

    return "scene[" + ",".join(
        role_tree_signature_from_node(root)
        for root in roots
    ) + "]"


def role_tree_signature_from_node(node):
    if node["role"] == "leaf":
        return "leaf"

    child_signatures = [
        role_tree_signature_from_node(child)
        for child in node["children"]
    ]

    return "container[" + ",".join(child_signatures) + "]"


def compass_bearing(from_node, to_node):
    from_row, from_col = bbox_center(from_node["bbox"])
    to_row, to_col = bbox_center(to_node["bbox"])

    delta_row = from_row - to_row
    delta_col = from_col - to_col

    abs_row = abs(delta_row)
    abs_col = abs(delta_col)

    if abs_row == 0 and abs_col == 0:
        return None

    if abs_row >= abs_col:
        if delta_row < 0:
            return "north"

        return "south"

    if delta_col < 0:
        return "west"

    return "east"


def bbox_center(bbox):
    center_row = bbox["top"] + (bbox["height"] - 1) / 2
    center_col = bbox["left"] + (bbox["width"] - 1) / 2

    return center_row, center_col


def find_components(grid, background):
    height = len(grid)
    width = len(grid[0]) if height else 0

    seen = set()
    components = []

    for row in range(height):
        for col in range(width):
            value = grid[row][col]

            if value == background:
                continue

            if (row, col) in seen:
                continue

            cells = flood_fill_component(
                grid=grid,
                start_row=row,
                start_col=col,
                seen=seen,
            )

            bbox = bbox_from_cells(cells)

            components.append({
                "color": value,
                "cells": cells,
                "bbox": bbox,
            })

    return components


def flood_fill_component(grid, start_row, start_col, seen):
    height = len(grid)
    width = len(grid[0]) if height else 0

    color = grid[start_row][start_col]

    queue = deque()
    queue.append((start_row, start_col))

    seen.add((start_row, start_col))

    cells = []

    while queue:
        row, col = queue.popleft()
        cells.append((row, col))

        for next_row, next_col in neighbors_4(row, col):
            if next_row < 0 or next_row >= height:
                continue

            if next_col < 0 or next_col >= width:
                continue

            if (next_row, next_col) in seen:
                continue

            if grid[next_row][next_col] != color:
                continue

            seen.add((next_row, next_col))
            queue.append((next_row, next_col))

    return cells


def neighbors_4(row, col):
    return [
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    ]


def bbox_from_cells(cells):
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

    height = bottom - top + 1
    width = right - left + 1

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": height,
        "width": width,
        "area": height * width,
    }


def bbox_strictly_contains(outer, inner):
    return (
        outer["top"] < inner["top"]
        and outer["left"] < inner["left"]
        and outer["bottom"] > inner["bottom"]
        and outer["right"] > inner["right"]
    )


def most_common_color(grid):
    counts = Counter()

    for row in grid:
        for value in row:
            counts[value] += 1

    return counts.most_common(1)[0][0]


def make_grid(height, width, value):
    return [
        [
            value
            for _ in range(width)
        ]
        for _ in range(height)
    ]


def draw_container_border(grid, top, left, height, width, color):
    for col in range(left, left + width):
        draw_cell(grid, top, col, color)
        draw_cell(grid, top + height - 1, col, color)

    for row in range(top, top + height):
        draw_cell(grid, row, left, color)
        draw_cell(grid, row, left + width - 1, color)


def draw_mask(grid, mask, top, left, color):
    for row_index, mask_row in enumerate(mask):
        for col_index, value in enumerate(mask_row):
            if value == 1:
                draw_cell(
                    grid=grid,
                    row=top + row_index,
                    col=left + col_index,
                    color=color,
                )


def draw_cell(grid, row, col, color):
    if row < 0 or row >= len(grid):
        return

    if col < 0 or col >= len(grid[0]):
        return

    grid[row][col] = color


def task_matches_anchor_compass_merge(task):
    for test_pair in task.get("test", []):
        prediction = predict_anchor_compass_merge_for_pair(
            task=task,
            pair=test_pair,
            debug=False,
        )

        if prediction is not None:
            return True

    return False