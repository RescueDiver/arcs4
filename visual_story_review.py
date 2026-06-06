"""
visual_story_review.py
2d0172a1
Purpose:
    Visual Story Review + Rule Discovery.

Current rule:
    Do NOT use leave-one-out during discovery.

Discovery order:
    1. Observe every input/output.
    2. Find objects.
    3. Find relationships.
    4. Build neutral roles.
    5. Measure what changed.
    6. Find consistent facts across all train pairs.
    7. Build candidate rules only from discovered facts.
    8. Score candidates on all train pairs.
    9. After a complete rule is found, then optional leave-one-out can test it.
    10. Apply best learned rule to test inputs.

"""
import json
import os
import tkinter as tk
from tkinter import Canvas, Frame, Scrollbar


# =============================================================================
# SECTION 0: TASK LOADING
# =============================================================================

def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_loaded_tasks(raw_data):
    """
    Accepts either:
        1. one ARC task:
            {"train": [...], "test": [...]}

        2. many ARC tasks:
            {
                "task_id": {"train": [...], "test": [...]},
                ...
            }

    Returns:
        {
            "task_id": task
        }
    """
    if isinstance(raw_data, dict) and "train" in raw_data and "test" in raw_data:
        return {"single_task": raw_data}

    if isinstance(raw_data, dict):
        tasks = {}

        for key, value in raw_data.items():
            if isinstance(value, dict) and "train" in value and "test" in value:
                tasks[key] = value

        return tasks

    return {}


def resolve_task_file(user_text):
    """
    Lets you type:
        data
        2d0172a1
        2d0172a1.json
        data_failures/extracted_tasks/2d0172a1.json
        full path
    """
    text = user_text.strip().strip('"')

    if text.lower() == "data":
        return os.path.join("data", "data.json")

    if os.path.exists(text):
        return text

    if text.endswith(".json"):
        possible = os.path.join("data_failures", "extracted_tasks", text)
        if os.path.exists(possible):
            return possible

        possible = os.path.join("data", text)
        if os.path.exists(possible):
            return possible

        return text

    possible = os.path.join("data_failures", "extracted_tasks", text + ".json")
    if os.path.exists(possible):
        return possible

    possible = os.path.join("data", text + ".json")
    if os.path.exists(possible):
        return possible

    return text + ".json"

# =============================================================================
# SECTION 1A: BASIC GRID HELPERS
# =============================================================================

def grid_shape(grid):
    if not grid:
        return 0, 0

    return len(grid), len(grid[0])


def in_bounds(grid, row, col):
    h, w = grid_shape(grid)
    return 0 <= row < h and 0 <= col < w


def copy_grid(grid):
    return [row[:] for row in grid]


def make_empty_grid(height, width, fill_value=0):
    return [
        [fill_value for _ in range(width)]
        for _ in range(height)
    ]


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

    rows = [r for r, c in cells]
    cols = [c for r, c in cells]

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


def format_bbox(bbox):
    return (
        f"top={bbox['top']} "
        f"left={bbox['left']} "
        f"h={bbox['height']} "
        f"w={bbox['width']}"
    )


def make_final_test_prediction_grid(task, pair, test_index=None):
    anchor_compass_grid = make_optional_anchor_compass_test_grid(
        task,
        pair,
    )

    if anchor_compass_grid is not None:
        if test_index is not None:
            print(f"TEST {test_index} DRAW PATH: anchor_compass_merge")
        return anchor_compass_grid

    if test_index is not None:
        print(f"TEST {test_index} DRAW PATH: recursive_fallback")

    return make_section_19_whole_test_output_sketch_grid(
        task,
        pair,
    )


# =============================================================================
# SECTION 2B: OBJECT RELATIONSHIP DETECTION
# =============================================================================

def get_relationship_facts(grid):
    """
    Finds parent/child style relationships between visible objects.

    Safe version:
        If no display candidate exists, return an empty object set
        instead of crashing.
    """
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
                "relative_position": relative_position(parent_bbox, child_bbox),
                "gaps": edge_gaps(parent_bbox, child_bbox),
            })

    return {
        "background": display_candidate["background"],
        "objects": objects,
        "relationships": relationships,
    }


def relationship_text_block(grid, title):
    """
    SECTION 2 QUESTION:
        Which objects are related?

    This is still not a rule.
    This is only structure observation.
    """
    facts = get_relationship_facts(grid)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 2: WHICH OBJECTS ARE RELATED?")
    lines.append("")
    lines.append(f"background: {facts['background']}")
    lines.append(f"objects   : {len(facts['objects'])}")
    lines.append(f"relations : {len(facts['relationships'])}")
    lines.append("")

    if not facts["relationships"]:
        lines.append("RELATIONSHIPS")
        lines.append("-" * 60)
        lines.append("no containment relationships found")
        return "\n".join(lines)

    lines.append("RELATIONSHIPS")
    lines.append("-" * 60)

    for rel in facts["relationships"]:
        gaps = rel["gaps"]

        lines.append(
            f"{rel['parent_id']} contains {rel['child_id']} | "
            f"pos={rel['relative_position']} | "
            f"gaps="
            f"T{gaps['top_gap']} "
            f"L{gaps['left_gap']} "
            f"B{gaps['bottom_gap']} "
            f"R{gaps['right_gap']}"
        )

    return "\n".join(lines)


def print_section_2_relationships_for_task(task_id, task):
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("#" * 80)
    print(f"SECTION 2 RELATIONSHIP CHECK — TASK {task_id}")
    print("#" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        print()
        print(relationship_text_block(input_grid, f"TRAIN {pair_index} INPUT"))
        print()
        print(relationship_text_block(output_grid, f"TRAIN {pair_index} EXPECTED"))

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair.get("input")

        print()
        print(relationship_text_block(input_grid, f"TEST {test_index} INPUT"))


# =============================================================================
# SECTION 3A: ROLE TREE BUILDING
# =============================================================================

def find_immediate_parent(child, objects):
    """
    Finds the tightest object that strictly contains this child.

    Why:
        Relationship detection lists all ancestors.
        For a tree, each child should have only one immediate parent.

    Example:
        object_1 contains object_2
        object_1 contains object_4
        object_2 contains object_4

        object_4's immediate parent should be object_2, not object_1.
    """
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
    """
    Builds a neutral object tree.

    It does not call anything a ring, blob, answer, or rule.

    It only says:
        - this object has children
        - this object has a parent
        - this object is a leaf
        - this object is a root
    """
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

    roots.sort(key=lambda node: (node["bbox"]["top"], node["bbox"]["left"]))

    for node in nodes_by_id.values():
        node["children"].sort(key=lambda child: (child["bbox"]["top"], child["bbox"]["left"]))

    return {
        "background": facts["background"],
        "objects": objects,
        "roots": roots,
        "nodes_by_id": nodes_by_id,
    }


def role_tree_signature_from_node(node):
    """
    Creates a simple structure signature.

    Example:
        container[container[leaf],leaf]
    """
    if not node["children"]:
        return "leaf"

    child_signatures = [
        role_tree_signature_from_node(child)
        for child in node["children"]
    ]

    return "container[" + ",".join(child_signatures) + "]"


def role_tree_signature(tree):
    root_signatures = [
        role_tree_signature_from_node(root)
        for root in tree["roots"]
    ]

    if len(root_signatures) == 1:
        return root_signatures[0]

    return "scene[" + ",".join(root_signatures) + "]"


def draw_role_tree_lines(node, lines, depth=0):
    indent = "  " * depth
    bbox = node["bbox"]

    lines.append(
        f"{indent}{node['id']} "
        f"role={node['role']} "
        f"color={node['color']} "
        f"cells={node['cell_count']} "
        f"bbox=({format_bbox(bbox)})"
    )

    for child in node["children"]:
        draw_role_tree_lines(child, lines, depth + 1)


def role_tree_text_block(grid, title):
    """
    SECTION 3 QUESTION:
        What is the clean parent/child tree?

    This removes duplicate ancestor relationships and gives each object
    one immediate parent.
    """
    tree = build_role_tree(grid)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 3: WHAT IS THE ROLE TREE?")
    lines.append("")
    lines.append(f"background: {tree['background']}")
    lines.append(f"objects   : {len(tree['objects'])}")
    lines.append(f"roots     : {len(tree['roots'])}")
    lines.append(f"signature : {role_tree_signature(tree)}")
    lines.append("")
    lines.append("ROLE TREE")
    lines.append("-" * 60)

    if not tree["roots"]:
        lines.append("no roots found")
    else:
        for root in tree["roots"]:
            draw_role_tree_lines(root, lines, depth=0)

    return "\n".join(lines)


def print_section_3_role_trees_for_task(task_id, task):
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("#" * 80)
    print(f"SECTION 3 ROLE TREE CHECK — TASK {task_id}")
    print("#" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        print()
        print(role_tree_text_block(input_grid, f"TRAIN {pair_index} INPUT"))
        print()
        print(role_tree_text_block(output_grid, f"TRAIN {pair_index} EXPECTED"))

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair.get("input")

        print()
        print(role_tree_text_block(input_grid, f"TEST {test_index} INPUT"))


# =============================================================================
# SECTION 4A: PAIR CHANGE MEASUREMENT
# =============================================================================

def sorted_nodes_for_matching(tree):
    """
    Gives a stable node order for rough input/output comparison.

    Important:
        This is still not the final matcher.
        It is a first measurement pass.

    Current ordering:
        role first, then bbox position.

    Later:
        We can replace this with true role/path matching.
    """
    nodes = list(tree["nodes_by_id"].values())

    def role_rank(node):
        if node["role"] == "container":
            return 0
        if node["role"] == "leaf":
            return 1
        return 2

    nodes.sort(
        key=lambda node: (
            role_rank(node),
            node["bbox"]["top"],
            node["bbox"]["left"],
            node["bbox"]["height"],
            node["bbox"]["width"],
        )
    )

    return nodes


def compare_node_size(input_node, output_node):
    input_bbox = input_node["bbox"]
    output_bbox = output_node["bbox"]

    input_area = input_bbox["height"] * input_bbox["width"]
    output_area = output_bbox["height"] * output_bbox["width"]

    became_1x1 = (
        output_bbox["height"] == 1
        and output_bbox["width"] == 1
    )

    became_smaller = output_area < input_area

    return {
        "input_id": input_node["id"],
        "output_id": output_node["id"],
        "input_role": input_node["role"],
        "output_role": output_node["role"],
        "input_cells": input_node["cell_count"],
        "output_cells": output_node["cell_count"],
        "input_bbox": input_bbox,
        "output_bbox": output_bbox,
        "input_bbox_area": input_area,
        "output_bbox_area": output_area,
        "became_smaller": became_smaller,
        "became_1x1": became_1x1,
        "role_preserved": input_node["role"] == output_node["role"],
    }


def measure_pair_changes(input_grid, output_grid):
    """
    SECTION 4 QUESTION:
        What changed from input to expected output?

    This does not decide the rule.
    It only measures possible transformation facts.
    """
    input_tree = build_role_tree(input_grid)
    output_tree = build_role_tree(output_grid)

    input_nodes = sorted_nodes_for_matching(input_tree)
    output_nodes = sorted_nodes_for_matching(output_tree)

    matched_count = min(len(input_nodes), len(output_nodes))

    node_changes = []

    for index in range(matched_count):
        node_changes.append(
            compare_node_size(input_nodes[index], output_nodes[index])
        )

    input_signature = role_tree_signature(input_tree)
    output_signature = role_tree_signature(output_tree)

    return {
        "input_background": input_tree["background"],
        "output_background": output_tree["background"],
        "input_object_count": len(input_tree["objects"]),
        "output_object_count": len(output_tree["objects"]),
        "input_root_count": len(input_tree["roots"]),
        "output_root_count": len(output_tree["roots"]),
        "input_signature": input_signature,
        "output_signature": output_signature,
        "tree_signature_preserved": input_signature == output_signature,
        "object_count_preserved": len(input_tree["objects"]) == len(output_tree["objects"]),
        "root_count_preserved": len(input_tree["roots"]) == len(output_tree["roots"]),
        "background_preserved": input_tree["background"] == output_tree["background"],
        "node_changes": node_changes,
    }


def pair_change_text_block(input_grid, output_grid, title):
    changes = measure_pair_changes(input_grid, output_grid)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 4: WHAT CHANGED?")
    lines.append("")
    lines.append(f"background preserved    : {changes['background_preserved']}")
    lines.append(f"object count preserved  : {changes['object_count_preserved']}")
    lines.append(f"root count preserved    : {changes['root_count_preserved']}")
    lines.append(f"tree signature preserved: {changes['tree_signature_preserved']}")
    lines.append("")
    lines.append(f"input signature : {changes['input_signature']}")
    lines.append(f"output signature: {changes['output_signature']}")
    lines.append("")
    lines.append("NODE CHANGE MEASUREMENTS")
    lines.append("-" * 60)

    if not changes["node_changes"]:
        lines.append("no node changes measured")
        return "\n".join(lines)

    for change in changes["node_changes"]:
        lines.append(
            f"{change['input_id']}({change['input_role']}) -> "
            f"{change['output_id']}({change['output_role']}) | "
            f"role_preserved={change['role_preserved']} | "
            f"bbox_area {change['input_bbox_area']} -> {change['output_bbox_area']} | "
            f"cells {change['input_cells']} -> {change['output_cells']} | "
            f"smaller={change['became_smaller']} | "
            f"became_1x1={change['became_1x1']}"
        )

    return "\n".join(lines)


def print_section_4_pair_changes_for_task(task_id, task):
    train_pairs = task.get("train", [])

    print()
    print("#" * 80)
    print(f"SECTION 4 PAIR CHANGE CHECK — TASK {task_id}")
    print("#" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        print()
        print(pair_change_text_block(
            input_grid,
            output_grid,
            f"TRAIN {pair_index} INPUT -> EXPECTED",
        ))


# =============================================================================
# SECTION 5A: TASK-LEVEL CONSISTENT FACT DISCOVERY
# =============================================================================

def all_true(values):
    return bool(values) and all(values)


def collect_role_changes(pair_changes):
    role_summary = {
        "container": {
            "count": 0,
            "became_smaller": [],
            "became_1x1": [],
            "role_preserved": [],
            "input_bbox_areas": [],
            "output_bbox_areas": [],
            "input_cell_counts": [],
            "output_cell_counts": [],
        },
        "leaf": {
            "count": 0,
            "became_smaller": [],
            "became_1x1": [],
            "role_preserved": [],
            "input_bbox_areas": [],
            "output_bbox_areas": [],
            "input_cell_counts": [],
            "output_cell_counts": [],
        },
    }

    for change in pair_changes["node_changes"]:
        role = change["input_role"]

        if role not in role_summary:
            continue

        role_summary[role]["count"] += 1
        role_summary[role]["became_smaller"].append(change["became_smaller"])
        role_summary[role]["became_1x1"].append(change["became_1x1"])
        role_summary[role]["role_preserved"].append(change["role_preserved"])
        role_summary[role]["input_bbox_areas"].append(change["input_bbox_area"])
        role_summary[role]["output_bbox_areas"].append(change["output_bbox_area"])
        role_summary[role]["input_cell_counts"].append(change["input_cells"])
        role_summary[role]["output_cell_counts"].append(change["output_cells"])

    return role_summary


def unique_values(values):
    result = []

    for value in values:
        if value not in result:
            result.append(value)

    return result


def unique_output_bbox_sizes_for_role(task, role):
    sizes = []

    for pair in task.get("train", []):
        output_grid = pair.get("output")
        tree = build_role_tree(output_grid)

        nodes = []

        for root in tree.get("roots", []):
            collect_nodes_by_role(root, role, nodes)

        for node in nodes:
            size = bbox_size_text(node.get("bbox"))

            if size is not None:
                sizes.append(size)

    return unique_values(sizes)



def collect_nodes_by_role(node, role, results):
    if node.get("role") == role:
        results.append(node)

    for child in node.get("children", []):
        collect_nodes_by_role(child, role, results)


def bbox_sizes_from_pair_changes(pair_change_list, role):
    sizes = []

    for changes in pair_change_list:
        output_tree = changes.get("output_tree")

        if output_tree is None:
            output_grid = changes.get("output_grid")
            if output_grid is None:
                continue
            output_tree = build_role_tree(output_grid)

        nodes = []

        for root in output_tree.get("roots", []):
            collect_nodes_by_role(root, role, nodes)

        for node in nodes:
            bbox = node.get("bbox")
            size = bbox_size_text(bbox)

            if size is not None:
                sizes.append(size)

    return sorted(set(sizes))


def discover_consistent_task_facts(task):
    """
    SECTION 5 QUESTION:
        What facts are true across all train pairs?

    This is the first task-level learning step.

    It does not decide a final rule yet.
    It only promotes measured facts that survive every train pair.
    """
    train_pairs = task.get("train", [])

    pair_change_list = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        changes = measure_pair_changes(input_grid, output_grid)
        changes["pair_index"] = pair_index

        pair_change_list.append(changes)

    task_facts = {
        "pair_count": len(pair_change_list),

        "background_preserved_all": all_true(
            changes["background_preserved"]
            for changes in pair_change_list
        ),

        "object_count_preserved_all": all_true(
            changes["object_count_preserved"]
            for changes in pair_change_list
        ),

        "root_count_preserved_all": all_true(
            changes["root_count_preserved"]
            for changes in pair_change_list
        ),

        "tree_signature_preserved_all": all_true(
            changes["tree_signature_preserved"]
            for changes in pair_change_list
        ),

        "input_signatures": [
            changes["input_signature"]
            for changes in pair_change_list
        ],

        "output_signatures": [
            changes["output_signature"]
            for changes in pair_change_list
        ],

        "pair_changes": pair_change_list,
    }

    combined_role_summary = {
        "container": {
            "count": 0,
            "became_smaller": [],
            "became_1x1": [],
            "role_preserved": [],
            "input_bbox_areas": [],
            "output_bbox_areas": [],
            "input_cell_counts": [],
            "output_cell_counts": [],
        },
        "leaf": {
            "count": 0,
            "became_smaller": [],
            "became_1x1": [],
            "role_preserved": [],
            "input_bbox_areas": [],
            "output_bbox_areas": [],
            "input_cell_counts": [],
            "output_cell_counts": [],
        },
    }

    for changes in pair_change_list:
        role_summary = collect_role_changes(changes)

        for role in ["container", "leaf"]:
            combined_role_summary[role]["count"] += role_summary[role]["count"]

            for key in [
                "became_smaller",
                "became_1x1",
                "role_preserved",
                "input_bbox_areas",
                "output_bbox_areas",
                "input_cell_counts",
                "output_cell_counts",
            ]:
                combined_role_summary[role][key].extend(role_summary[role][key])

    task_facts["role_summary"] = combined_role_summary

    task_facts["containers_preserve_role_all"] = all_true(
        combined_role_summary["container"]["role_preserved"]
    )

    task_facts["containers_become_smaller_all"] = all_true(
        combined_role_summary["container"]["became_smaller"]
    )

    task_facts["containers_become_1x1_all"] = all_true(
        combined_role_summary["container"]["became_1x1"]
    )

    task_facts["leaves_preserve_role_all"] = all_true(
        combined_role_summary["leaf"]["role_preserved"]
    )

    task_facts["leaves_become_smaller_all"] = all_true(
        combined_role_summary["leaf"]["became_smaller"]
    )

    task_facts["leaves_become_1x1_all"] = all_true(
        combined_role_summary["leaf"]["became_1x1"]
    )

    # Size keeps orientation, so 9x11 and 11x9 stay different.
    task_facts["unique_container_output_bbox_sizes"] = unique_output_bbox_sizes_for_role(
        task,
        "container",
    )

    task_facts["unique_leaf_output_bbox_sizes"] = unique_output_bbox_sizes_for_role(
        task,
        "leaf",
    )

    # Area is still useful, but it hides orientation.
    task_facts["unique_container_output_bbox_areas"] = unique_values(
        combined_role_summary["container"]["output_bbox_areas"]
    )

    task_facts["unique_leaf_output_bbox_areas"] = unique_values(
        combined_role_summary["leaf"]["output_bbox_areas"]
    )

    task_facts["unique_container_output_cell_counts"] = unique_values(
        combined_role_summary["container"]["output_cell_counts"]
    )

    task_facts["unique_leaf_output_cell_counts"] = unique_values(
        combined_role_summary["leaf"]["output_cell_counts"]
    )

    return task_facts


def consistent_task_facts_text_block(task, title):
    facts = discover_consistent_task_facts(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 5: WHAT IS CONSISTENT ACROSS ALL TRAIN PAIRS?")
    lines.append("")
    lines.append(f"train pair count                 : {facts['pair_count']}")
    lines.append(f"background preserved all         : {facts['background_preserved_all']}")
    lines.append(f"object count preserved all       : {facts['object_count_preserved_all']}")
    lines.append(f"root count preserved all         : {facts['root_count_preserved_all']}")
    lines.append(f"tree signature preserved all     : {facts['tree_signature_preserved_all']}")
    lines.append("")
    lines.append("ROLE FACTS")
    lines.append("-" * 60)

    container_summary = facts["role_summary"]["container"]
    leaf_summary = facts["role_summary"]["leaf"]

    lines.append(f"container count measured          : {container_summary['count']}")
    lines.append(f"containers preserve role all      : {facts['containers_preserve_role_all']}")
    lines.append(f"containers become smaller all     : {facts['containers_become_smaller_all']}")
    lines.append(f"containers become 1x1 all         : {facts['containers_become_1x1_all']}")
    lines.append(f"container output bbox sizes       : {facts.get('unique_container_output_bbox_sizes', [])}")
    lines.append(f"container output bbox areas       : {facts['unique_container_output_bbox_areas']}")
    lines.append(f"container output cell counts      : {facts['unique_container_output_cell_counts']}")
    lines.append("")

    lines.append(f"leaf count measured               : {leaf_summary['count']}")
    lines.append(f"leaves preserve role all          : {facts['leaves_preserve_role_all']}")
    lines.append(f"leaves become smaller all         : {facts['leaves_become_smaller_all']}")
    lines.append(f"leaves become 1x1 all             : {facts['leaves_become_1x1_all']}")
    lines.append(f"leaf output bbox sizes            : {facts.get('unique_leaf_output_bbox_sizes', [])}")
    lines.append(f"leaf output bbox areas            : {facts['unique_leaf_output_bbox_areas']}")
    lines.append(f"leaf output cell counts           : {facts['unique_leaf_output_cell_counts']}")
    lines.append("")

    lines.append("SIGNATURES BY PAIR")
    lines.append("-" * 60)

    for index, changes in enumerate(facts["pair_changes"]):
        lines.append(
            f"pair {index}: "
            f"{changes['input_signature']} -> {changes['output_signature']} | "
            f"preserved={changes['tree_signature_preserved']}"
        )

    return "\n".join(lines)


def print_section_5_consistent_facts_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 5 CONSISTENT FACT CHECK — TASK {task_id}")
    print("#" * 80)
    print()
    print(consistent_task_facts_text_block(
        task,
        f"TASK {task_id} CONSISTENT FACTS",
    ))


# =============================================================================
# SECTION 6: LEAF PRESERVATION LEARNING
# =============================================================================

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
        collect_role_path_nodes_from_node(child, child_path, results)


def collect_role_path_nodes(grid):
    tree = build_role_tree(grid)
    results = []

    for root_index, root in enumerate(tree.get("roots", [])):
        path = f"root{root_index}"
        collect_role_path_nodes_from_node(root, path, results)

    return results


def collect_leaf_path_nodes(grid):
    nodes = collect_role_path_nodes(grid)
    return [
        node for node in nodes
        if node["role"] == "leaf"
    ]


def bbox_area_from_bbox(bbox):
    if bbox is None:
        return None

    return bbox["height"] * bbox["width"]


def measure_leaf_preservation_for_pair(pair):
    input_grid = pair.get("input")
    output_grid = pair.get("output")

    input_leaves = collect_leaf_path_nodes(input_grid)
    output_leaves = collect_leaf_path_nodes(output_grid)

    input_by_path = {
        leaf["path"]: leaf
        for leaf in input_leaves
    }

    output_by_path = {
        leaf["path"]: leaf
        for leaf in output_leaves
    }

    all_paths = sorted(set(input_by_path.keys()) | set(output_by_path.keys()))

    leaf_facts = []

    for path in all_paths:
        input_leaf = input_by_path.get(path)
        output_leaf = output_by_path.get(path)

        output_bbox_area = None
        output_cell_count = None
        output_color = None

        if output_leaf is not None:
            output_bbox_area = bbox_area_from_bbox(output_leaf.get("bbox"))
            output_cell_count = output_leaf.get("cell_count")
            output_color = output_leaf.get("color")

        input_color = None
        if input_leaf is not None:
            input_color = input_leaf.get("color")

        leaf_facts.append({
            "path": path,
            "input_exists": input_leaf is not None,
            "output_exists": output_leaf is not None,
            "input_color": input_color,
            "output_color": output_color,
            "color_preserved": input_color == output_color,
            "output_bbox_area": output_bbox_area,
            "output_cell_count": output_cell_count,
        })

    return {
        "input_leaf_count": len(input_leaves),
        "output_leaf_count": len(output_leaves),
        "leaf_count_preserved": len(input_leaves) == len(output_leaves),
        "leaf_facts": leaf_facts,
    }


def discover_leaf_preservation_facts(task):
    train_pairs = task.get("train", [])

    pair_measurements = []

    all_leaf_counts_preserved = True
    all_leaf_paths_preserved = True
    all_leaf_colors_preserved = True

    output_bbox_areas = []
    output_cell_counts = []

    for pair_index, pair in enumerate(train_pairs):
        measurement = measure_leaf_preservation_for_pair(pair)
        measurement["pair_index"] = pair_index
        pair_measurements.append(measurement)

        if not measurement["leaf_count_preserved"]:
            all_leaf_counts_preserved = False

        for leaf_fact in measurement["leaf_facts"]:
            if not leaf_fact["input_exists"] or not leaf_fact["output_exists"]:
                all_leaf_paths_preserved = False

            if not leaf_fact["color_preserved"]:
                all_leaf_colors_preserved = False

            if leaf_fact["output_bbox_area"] is not None:
                output_bbox_areas.append(leaf_fact["output_bbox_area"])

            if leaf_fact["output_cell_count"] is not None:
                output_cell_counts.append(leaf_fact["output_cell_count"])

    unique_output_bbox_areas = sorted(set(output_bbox_areas))
    unique_output_cell_counts = sorted(set(output_cell_counts))

    return {
        "train_pair_count": len(train_pairs),
        "pair_measurements": pair_measurements,
        "all_leaf_counts_preserved": all_leaf_counts_preserved,
        "all_leaf_paths_preserved": all_leaf_paths_preserved,
        "all_leaf_colors_preserved": all_leaf_colors_preserved,
        "unique_output_bbox_areas": unique_output_bbox_areas,
        "unique_output_cell_counts": unique_output_cell_counts,
        "consistent_output_bbox_area": (
            unique_output_bbox_areas[0]
            if len(unique_output_bbox_areas) == 1
            else None
        ),
        "consistent_output_cell_count": (
            unique_output_cell_counts[0]
            if len(unique_output_cell_counts) == 1
            else None
        ),
    }


def leaf_preservation_text_block(task, title):
    facts = discover_leaf_preservation_facts(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 6: ARE ALL LEAF OBJECTS PRESERVED?")
    lines.append("")

    lines.append(f"train pair count                  : {facts['train_pair_count']}")
    lines.append(f"leaf count preserved all          : {facts['all_leaf_counts_preserved']}")
    lines.append(f"leaf paths preserved all          : {facts['all_leaf_paths_preserved']}")
    lines.append(f"leaf colors preserved all         : {facts['all_leaf_colors_preserved']}")
    lines.append(f"leaf output bbox areas            : {facts['unique_output_bbox_areas']}")
    lines.append(f"leaf output cell counts           : {facts['unique_output_cell_counts']}")
    lines.append(f"learned output bbox area          : {facts['consistent_output_bbox_area']}")
    lines.append(f"learned output cell count         : {facts['consistent_output_cell_count']}")
    lines.append("")

    lines.append("LEAF PATHS BY PAIR")
    lines.append("-" * 60)

    for measurement in facts["pair_measurements"]:
        pair_index = measurement["pair_index"]

        lines.append(f"pair {pair_index}:")
        lines.append(f"  input leaf count  : {measurement['input_leaf_count']}")
        lines.append(f"  output leaf count : {measurement['output_leaf_count']}")
        lines.append(f"  count preserved   : {measurement['leaf_count_preserved']}")

        for leaf_fact in measurement["leaf_facts"]:
            lines.append(
                "  "
                f"{leaf_fact['path']} | "
                f"exists {leaf_fact['input_exists']} -> {leaf_fact['output_exists']} | "
                f"color {leaf_fact['input_color']} -> {leaf_fact['output_color']} | "
                f"bbox_area={leaf_fact['output_bbox_area']} | "
                f"cell_count={leaf_fact['output_cell_count']}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_6_leaf_preservation_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 6 LEAF PRESERVATION CHECK — TASK {task_id}")
    print("#" * 80)
    print()
    print(leaf_preservation_text_block(
        task,
        f"TASK {task_id} LEAF PRESERVATION",
    ))


# =============================================================================
# SECTION 7: CONTAINER PRESERVATION LEARNING
# =============================================================================

def collect_container_path_nodes(grid):
    nodes = collect_role_path_nodes(grid)
    return [
        node for node in nodes
        if node["role"] == "container"
    ]


def bbox_size_text(bbox):
    if bbox is None:
        return None

    return f"{bbox['height']}x{bbox['width']}"


def measure_container_preservation_for_pair(pair):
    input_grid = pair.get("input")
    output_grid = pair.get("output")

    input_containers = collect_container_path_nodes(input_grid)
    output_containers = collect_container_path_nodes(output_grid)

    input_by_path = {
        container["path"]: container
        for container in input_containers
    }

    output_by_path = {
        container["path"]: container
        for container in output_containers
    }

    all_paths = sorted(set(input_by_path.keys()) | set(output_by_path.keys()))

    container_facts = []

    for path in all_paths:
        input_container = input_by_path.get(path)
        output_container = output_by_path.get(path)

        input_color = None
        output_color = None
        output_bbox_area = None
        output_bbox_size = None
        output_cell_count = None
        output_child_count = None

        if input_container is not None:
            input_color = input_container.get("color")

        if output_container is not None:
            output_color = output_container.get("color")
            output_bbox_area = bbox_area_from_bbox(output_container.get("bbox"))
            output_bbox_size = bbox_size_text(output_container.get("bbox"))
            output_cell_count = output_container.get("cell_count")
            output_child_count = output_container.get("child_count")

        container_facts.append({
            "path": path,
            "input_exists": input_container is not None,
            "output_exists": output_container is not None,
            "input_color": input_color,
            "output_color": output_color,
            "color_preserved": input_color == output_color,
            "output_bbox_area": output_bbox_area,
            "output_bbox_size": output_bbox_size,
            "output_cell_count": output_cell_count,
            "output_child_count": output_child_count,
        })

    return {
        "input_container_count": len(input_containers),
        "output_container_count": len(output_containers),
        "container_count_preserved": len(input_containers) == len(output_containers),
        "container_facts": container_facts,
    }


def discover_container_preservation_facts(task):
    train_pairs = task.get("train", [])

    pair_measurements = []

    all_container_counts_preserved = True
    all_container_paths_preserved = True
    all_container_colors_preserved = True

    output_bbox_areas = []
    output_cell_counts = []

    for pair_index, pair in enumerate(train_pairs):
        measurement = measure_container_preservation_for_pair(pair)
        measurement["pair_index"] = pair_index
        pair_measurements.append(measurement)

        if not measurement["container_count_preserved"]:
            all_container_counts_preserved = False

        for container_fact in measurement["container_facts"]:
            if not container_fact["input_exists"] or not container_fact["output_exists"]:
                all_container_paths_preserved = False

            if not container_fact["color_preserved"]:
                all_container_colors_preserved = False

            if container_fact["output_bbox_area"] is not None:
                output_bbox_areas.append(container_fact["output_bbox_area"])

            if container_fact["output_cell_count"] is not None:
                output_cell_counts.append(container_fact["output_cell_count"])

    return {
        "train_pair_count": len(train_pairs),
        "pair_measurements": pair_measurements,
        "all_container_counts_preserved": all_container_counts_preserved,
        "all_container_paths_preserved": all_container_paths_preserved,
        "all_container_colors_preserved": all_container_colors_preserved,
        "unique_output_bbox_areas": sorted(set(output_bbox_areas)),
        "unique_output_cell_counts": sorted(set(output_cell_counts)),
    }


def container_preservation_text_block(task, title):
    facts = discover_container_preservation_facts(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 7: ARE ALL CONTAINER OBJECTS PRESERVED?")
    lines.append("")

    lines.append(f"train pair count                       : {facts['train_pair_count']}")
    lines.append(f"container count preserved all          : {facts['all_container_counts_preserved']}")
    lines.append(f"container paths preserved all          : {facts['all_container_paths_preserved']}")
    lines.append(f"container colors preserved all         : {facts['all_container_colors_preserved']}")
    lines.append(f"container output bbox areas            : {facts['unique_output_bbox_areas']}")
    lines.append(f"container output cell counts           : {facts['unique_output_cell_counts']}")
    lines.append("")

    lines.append("CONTAINER PATHS BY PAIR")
    lines.append("-" * 60)

    for measurement in facts["pair_measurements"]:
        pair_index = measurement["pair_index"]

        lines.append(f"pair {pair_index}:")
        lines.append(f"  input container count  : {measurement['input_container_count']}")
        lines.append(f"  output container count : {measurement['output_container_count']}")
        lines.append(f"  count preserved        : {measurement['container_count_preserved']}")

        for container_fact in measurement["container_facts"]:
            lines.append(
                "  "
                f"{container_fact['path']} | "
                f"exists {container_fact['input_exists']} -> {container_fact['output_exists']} | "
                f"color {container_fact['input_color']} -> {container_fact['output_color']} | "
                f"bbox={container_fact['output_bbox_size']} | "
                f"bbox_area={container_fact['output_bbox_area']} | "
                f"cell_count={container_fact['output_cell_count']} | "
                f"children={container_fact['output_child_count']}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_7_container_preservation_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 7 CONTAINER PRESERVATION CHECK — TASK {task_id}")
    print("#" * 80)
    print()
    print(container_preservation_text_block(
        task,
        f"TASK {task_id} CONTAINER PRESERVATION",
    ))


# =============================================================================
# SECTION 8: CHILD POSITION INSIDE CONTAINERS
# =============================================================================

def child_position_inside_parent(parent_node, child_node):
    parent_bbox = parent_node.get("bbox")
    child_bbox = child_node.get("bbox")

    if parent_bbox is None or child_bbox is None:
        return None

    parent_top = parent_bbox["top"]
    parent_left = parent_bbox["left"]
    parent_height = parent_bbox["height"]
    parent_width = parent_bbox["width"]

    child_top = child_bbox["top"]
    child_left = child_bbox["left"]
    child_height = child_bbox["height"]
    child_width = child_bbox["width"]

    row_offset = child_top - parent_top
    col_offset = child_left - parent_left

    child_center_row = row_offset + (child_height - 1) / 2
    child_center_col = col_offset + (child_width - 1) / 2

    parent_center_row = (parent_height - 1) / 2
    parent_center_col = (parent_width - 1) / 2

    if child_center_row < parent_center_row:
        vertical = "top"
    elif child_center_row > parent_center_row:
        vertical = "bottom"
    else:
        vertical = "middle"

    if child_center_col < parent_center_col:
        horizontal = "left"
    elif child_center_col > parent_center_col:
        horizontal = "right"
    else:
        horizontal = "center"

    return {
        "row_offset": row_offset,
        "col_offset": col_offset,
        "vertical": vertical,
        "horizontal": horizontal,
        "parent_size": bbox_size_text(parent_bbox),
        "child_size": bbox_size_text(child_bbox),
        "child_center_row": child_center_row,
        "child_center_col": child_center_col,
    }


def collect_child_positions_from_node(node, parent_path, results):
    children = node.get("children", [])

    for child_index, child in enumerate(children):
        child_path = f"{parent_path}.child{child_index}"

        position = child_position_inside_parent(node, child)

        results.append({
            "parent_path": parent_path,
            "child_path": child_path,
            "parent_role": node.get("role"),
            "child_role": child.get("role"),
            "parent_signature": role_tree_signature_from_node(node),
            "child_signature": role_tree_signature_from_node(child),
            "position": position,
        })

        collect_child_positions_from_node(child, child_path, results)


def collect_child_positions(grid):
    tree = build_role_tree(grid)
    results = []

    for root_index, root in enumerate(tree.get("roots", [])):
        root_path = f"root{root_index}"
        collect_child_positions_from_node(root, root_path, results)

    return results


def measure_child_positions_for_pair(pair):
    input_grid = pair.get("input")
    output_grid = pair.get("output")

    input_positions = collect_child_positions(input_grid)
    output_positions = collect_child_positions(output_grid)

    input_by_child_path = {
        item["child_path"]: item
        for item in input_positions
    }

    output_by_child_path = {
        item["child_path"]: item
        for item in output_positions
    }

    all_child_paths = sorted(
        set(input_by_child_path.keys()) |
        set(output_by_child_path.keys())
    )

    child_facts = []

    for child_path in all_child_paths:
        input_item = input_by_child_path.get(child_path)
        output_item = output_by_child_path.get(child_path)

        input_position = None
        output_position = None

        if input_item is not None:
            input_position = input_item.get("position")

        if output_item is not None:
            output_position = output_item.get("position")

        child_facts.append({
            "child_path": child_path,
            "input_exists": input_item is not None,
            "output_exists": output_item is not None,
            "parent_path": output_item.get("parent_path") if output_item else None,
            "child_role": output_item.get("child_role") if output_item else None,
            "parent_signature": output_item.get("parent_signature") if output_item else None,
            "child_signature": output_item.get("child_signature") if output_item else None,
            "input_position": input_position,
            "output_position": output_position,
        })

    return {
        "input_child_relation_count": len(input_positions),
        "output_child_relation_count": len(output_positions),
        "child_relation_count_preserved": len(input_positions) == len(output_positions),
        "child_facts": child_facts,
    }


def discover_child_position_facts(task):
    train_pairs = task.get("train", [])

    pair_measurements = []

    all_child_relation_counts_preserved = True
    all_child_paths_preserved = True

    output_vertical_positions = []
    output_horizontal_positions = []
    output_row_offsets = []
    output_col_offsets = []

    for pair_index, pair in enumerate(train_pairs):
        measurement = measure_child_positions_for_pair(pair)
        measurement["pair_index"] = pair_index
        pair_measurements.append(measurement)

        if not measurement["child_relation_count_preserved"]:
            all_child_relation_counts_preserved = False

        for child_fact in measurement["child_facts"]:
            if not child_fact["input_exists"] or not child_fact["output_exists"]:
                all_child_paths_preserved = False

            output_position = child_fact.get("output_position")

            if output_position is not None:
                output_vertical_positions.append(output_position["vertical"])
                output_horizontal_positions.append(output_position["horizontal"])
                output_row_offsets.append(output_position["row_offset"])
                output_col_offsets.append(output_position["col_offset"])

    return {
        "train_pair_count": len(train_pairs),
        "pair_measurements": pair_measurements,
        "all_child_relation_counts_preserved": all_child_relation_counts_preserved,
        "all_child_paths_preserved": all_child_paths_preserved,
        "unique_output_vertical_positions": unique_values(output_vertical_positions),
        "unique_output_horizontal_positions": unique_values(output_horizontal_positions),
        "unique_output_row_offsets": unique_values(output_row_offsets),
        "unique_output_col_offsets": unique_values(output_col_offsets),
    }


def child_position_text_block(task, title):
    facts = discover_child_position_facts(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 8: WHERE DO CHILDREN SIT INSIDE THEIR CONTAINERS?")
    lines.append("")

    lines.append(f"train pair count                         : {facts['train_pair_count']}")
    lines.append(f"child relation count preserved all       : {facts['all_child_relation_counts_preserved']}")
    lines.append(f"child paths preserved all                : {facts['all_child_paths_preserved']}")
    lines.append(f"output vertical positions                : {facts['unique_output_vertical_positions']}")
    lines.append(f"output horizontal positions              : {facts['unique_output_horizontal_positions']}")
    lines.append(f"output row offsets                       : {facts['unique_output_row_offsets']}")
    lines.append(f"output col offsets                       : {facts['unique_output_col_offsets']}")
    lines.append("")

    lines.append("CHILD POSITIONS BY PAIR")
    lines.append("-" * 60)

    for measurement in facts["pair_measurements"]:
        pair_index = measurement["pair_index"]

        lines.append(f"pair {pair_index}:")
        lines.append(f"  input child relations  : {measurement['input_child_relation_count']}")
        lines.append(f"  output child relations : {measurement['output_child_relation_count']}")
        lines.append(f"  count preserved        : {measurement['child_relation_count_preserved']}")

        for child_fact in measurement["child_facts"]:
            output_position = child_fact["output_position"]

            if output_position is None:
                position_text = "None"
            else:
                position_text = (
                    f"{output_position['vertical']}-{output_position['horizontal']} | "
                    f"offset=({output_position['row_offset']},{output_position['col_offset']}) | "
                    f"parent={output_position['parent_size']} | "
                    f"child={output_position['child_size']}"
                )

            lines.append(
                "  "
                f"{child_fact['child_path']} | "
                f"parent={child_fact['parent_path']} | "
                f"child_role={child_fact['child_role']} | "
                f"position={position_text}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_8_child_positions_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 8 CHILD POSITION CHECK — TASK {task_id}")
    print("#" * 80)
    print()
    print(child_position_text_block(
        task,
        f"TASK {task_id} CHILD POSITIONS",
    ))


# =============================================================================
# SECTION 9: SIMPLE INPUT VS OUTPUT LAYOUT CHECK
# =============================================================================

def child_center_pattern_from_positions(child_positions):
    usable_items = []

    for item in child_positions:
        position = item.get("position")

        if position is None:
            continue

        usable_items.append({
            "child_path": item["child_path"],
            "center_row": position["child_center_row"],
            "center_col": position["child_center_col"],
        })

    if not usable_items:
        return {
            "row_span": 0,
            "col_span": 0,
            "items": [],
        }

    rows = [
        item["center_row"]
        for item in usable_items
    ]

    cols = [
        item["center_col"]
        for item in usable_items
    ]

    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)

    row_span = max_row - min_row
    col_span = max_col - min_col

    normalized_items = []

    for item in usable_items:
        if row_span == 0:
            norm_row = 0.5
        else:
            norm_row = (item["center_row"] - min_row) / row_span

        if col_span == 0:
            norm_col = 0.5
        else:
            norm_col = (item["center_col"] - min_col) / col_span

        normalized_items.append({
            "child_path": item["child_path"],
            "norm_row": norm_row,
            "norm_col": norm_col,
            "source_center_row": item["center_row"],
            "source_center_col": item["center_col"],
        })

    normalized_items.sort(
        key=lambda item: item["child_path"],
    )

    return {
        "row_span": row_span,
        "col_span": col_span,
        "items": normalized_items,
    }


def layout_direction_from_child_positions(child_positions):
    if len(child_positions) < 2:
        return "single_child"

    centers = []

    for item in child_positions:
        position = item.get("position")

        if position is None:
            continue

        centers.append({
            "row": position["child_center_row"],
            "col": position["child_center_col"],
        })

    if len(centers) < 2:
        return "single_child"

    row_values = [
        item["row"]
        for item in centers
    ]

    col_values = [
        item["col"]
        for item in centers
    ]

    row_span = max(row_values) - min(row_values)
    col_span = max(col_values) - min(col_values)

    if col_span > row_span:
        return "horizontal"

    if row_span > col_span:
        return "vertical"

    return "mixed"


def container_layout_directions(grid):
    child_positions = collect_child_positions(grid)

    by_parent = {}

    for item in child_positions:
        parent_path = item["parent_path"]

        if parent_path not in by_parent:
            by_parent[parent_path] = []

        by_parent[parent_path].append(item)

    results = {}

    for parent_path, items in by_parent.items():
        results[parent_path] = {
            "child_count": len(items),
            "direction": layout_direction_from_child_positions(items),
        }

    return results


def measure_input_output_layout_directions_for_pair(pair):
    input_grid = pair.get("input")
    output_grid = pair.get("output")

    input_layouts = container_layout_directions(input_grid)
    output_layouts = container_layout_directions(output_grid)

    all_paths = sorted(set(input_layouts.keys()) | set(output_layouts.keys()))

    facts = []

    for path in all_paths:
        input_info = input_layouts.get(path)
        output_info = output_layouts.get(path)

        input_direction = None
        output_direction = None

        if input_info is not None:
            input_direction = input_info["direction"]

        if output_info is not None:
            output_direction = output_info["direction"]

        facts.append({
            "path": path,
            "input_direction": input_direction,
            "output_direction": output_direction,
            "direction_preserved": input_direction == output_direction,
        })

    return facts


def input_output_layout_direction_text_block(task, title):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 9: DOES INPUT LAYOUT DIRECTION MATCH OUTPUT?")
    lines.append("")

    train_pairs = task.get("train", [])

    all_preserved_values = []

    for pair_index, pair in enumerate(train_pairs):
        facts = measure_input_output_layout_directions_for_pair(pair)

        lines.append(f"pair {pair_index}:")

        for fact in facts:
            all_preserved_values.append(fact["direction_preserved"])

            lines.append(
                "  "
                f"{fact['path']} | "
                f"input={fact['input_direction']} -> "
                f"output={fact['output_direction']} | "
                f"preserved={fact['direction_preserved']}"
            )

        lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(f"layout direction preserved all : {all_true(all_preserved_values)}")

    return "\n".join(lines)


def print_section_9_input_output_layout_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 9 INPUT VS OUTPUT LAYOUT CHECK — TASK {task_id}")
    print("#" * 80)
    print()
    print(input_output_layout_direction_text_block(
        task,
        f"TASK {task_id} INPUT VS OUTPUT LAYOUT",
    ))


# =============================================================================
# SECTION 10: TEST INPUT OBSERVATION
# =============================================================================

def test_input_observation_for_pair(pair):
    input_grid = pair.get("input")

    tree = build_role_tree(input_grid)
    nodes = collect_role_path_nodes(input_grid)

    containers = [
        node for node in nodes
        if node["role"] == "container"
    ]

    leaves = [
        node for node in nodes
        if node["role"] == "leaf"
    ]

    container_colors = unique_values([
        node["color"]
        for node in containers
    ])

    leaf_colors = unique_values([
        node["color"]
        for node in leaves
    ])

    layout_directions = container_layout_directions(input_grid)

    return {
        "signature": role_tree_signature(tree),
        "container_count": len(containers),
        "leaf_count": len(leaves),
        "container_colors": container_colors,
        "leaf_colors": leaf_colors,
        "layout_directions": layout_directions,
    }


def test_input_observation_text_block(task, title):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 10: WHAT DOES THE PROGRAM SEE IN THE TEST INPUT?")
    lines.append("")

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        facts = test_input_observation_for_pair(pair)

        lines.append(f"TEST {test_index}")
        lines.append("-" * 60)
        lines.append(f"signature        : {facts['signature']}")
        lines.append(f"container count  : {facts['container_count']}")
        lines.append(f"leaf count       : {facts['leaf_count']}")
        lines.append(f"container colors : {facts['container_colors']}")
        lines.append(f"leaf colors      : {facts['leaf_colors']}")

        lines.append("layout directions:")

        for path, info in facts["layout_directions"].items():
            lines.append(
                f"  {path} | "
                f"children={info['child_count']} | "
                f"direction={info['direction']}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_10_test_input_observation_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 10 TEST INPUT OBSERVATION — TASK {task_id}")
    print("#" * 80)
    print()
    print(test_input_observation_text_block(
        task,
        f"TASK {task_id} TEST INPUT OBSERVATION",
    ))


# =============================================================================
# SECTION 11: TEST STRUCTURE MATCH CHECK
# =============================================================================

def collect_subtree_signatures_from_node(node, results):
    signature = role_tree_signature_from_node(node)
    results.append(signature)

    for child in node.get("children", []):
        collect_subtree_signatures_from_node(child, results)


def collect_subtree_signatures(grid):
    tree = build_role_tree(grid)
    results = []

    for root in tree.get("roots", []):
        collect_subtree_signatures_from_node(root, results)

    return unique_values(results)


def train_structure_library(task):
    train_pairs = task.get("train", [])

    full_input_signatures = []
    full_output_signatures = []
    subtree_signatures = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        input_tree = build_role_tree(input_grid)
        output_tree = build_role_tree(output_grid)

        input_signature = role_tree_signature(input_tree)
        output_signature = role_tree_signature(output_tree)

        full_input_signatures.append({
            "pair_index": pair_index,
            "signature": input_signature,
        })

        full_output_signatures.append({
            "pair_index": pair_index,
            "signature": output_signature,
        })

        for signature in collect_subtree_signatures(input_grid):
            subtree_signatures.append({
                "pair_index": pair_index,
                "source": "input",
                "signature": signature,
            })

        for signature in collect_subtree_signatures(output_grid):
            subtree_signatures.append({
                "pair_index": pair_index,
                "source": "output",
                "signature": signature,
            })

    return {
        "full_input_signatures": full_input_signatures,
        "full_output_signatures": full_output_signatures,
        "subtree_signatures": subtree_signatures,
    }


def find_signature_matches(signature, signature_items):
    matches = []

    for item in signature_items:
        if item["signature"] == signature:
            matches.append(item)

    return matches


def test_structure_match_for_pair(task, pair):
    input_grid = pair.get("input")

    test_tree = build_role_tree(input_grid)
    test_signature = role_tree_signature(test_tree)
    test_subtrees = collect_subtree_signatures(input_grid)

    library = train_structure_library(task)

    exact_input_matches = find_signature_matches(
        test_signature,
        library["full_input_signatures"],
    )

    exact_output_matches = find_signature_matches(
        test_signature,
        library["full_output_signatures"],
    )

    subtree_results = []

    for signature in test_subtrees:
        matches = find_signature_matches(
            signature,
            library["subtree_signatures"],
        )

        subtree_results.append({
            "signature": signature,
            "known": len(matches) > 0,
            "matches": matches,
        })

    return {
        "test_signature": test_signature,
        "exact_input_matches": exact_input_matches,
        "exact_output_matches": exact_output_matches,
        "subtree_results": subtree_results,
    }


def test_structure_match_text_block(task, title):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 11: HAVE WE SEEN THIS TEST STRUCTURE BEFORE?")
    lines.append("")

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        facts = test_structure_match_for_pair(task, pair)

        lines.append(f"TEST {test_index}")
        lines.append("-" * 60)
        lines.append(f"signature              : {facts['test_signature']}")
        lines.append(f"exact input matches    : {len(facts['exact_input_matches'])}")
        lines.append(f"exact output matches   : {len(facts['exact_output_matches'])}")

        if facts["exact_input_matches"]:
            for match in facts["exact_input_matches"]:
                lines.append(f"  exact input match    : train pair {match['pair_index']}")

        if facts["exact_output_matches"]:
            for match in facts["exact_output_matches"]:
                lines.append(f"  exact output match   : train pair {match['pair_index']}")

        lines.append("")
        lines.append("known substructures:")

        for item in facts["subtree_results"]:
            if item["known"]:
                match_text = ", ".join(
                    f"pair {match['pair_index']} {match['source']}"
                    for match in item["matches"]
                )
            else:
                match_text = "none"

            lines.append(
                f"  {item['signature']} | "
                f"known={item['known']} | "
                f"matches={match_text}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_11_test_structure_match_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 11 TEST STRUCTURE MATCH CHECK — TASK {task_id}")
    print("#" * 80)
    print()
    print(test_structure_match_text_block(
        task,
        f"TASK {task_id} TEST STRUCTURE MATCH",
    ))


def compact_test_structure_summary(task, test_pair):
    observation = test_input_observation_for_pair(test_pair)
    match = test_structure_match_for_pair(task, test_pair)

    known_parts = []
    unknown_parts = []

    for item in match["subtree_results"]:
        signature = item["signature"]

        if item["known"]:
            known_parts.append(signature)
        else:
            unknown_parts.append(signature)

    return {
        "signature": observation["signature"],
        "container_count": observation["container_count"],
        "leaf_count": observation["leaf_count"],
        "container_colors": observation["container_colors"],
        "leaf_colors": observation["leaf_colors"],
        "layout_directions": observation["layout_directions"],
        "exact_input_match_count": len(match["exact_input_matches"]),
        "exact_output_match_count": len(match["exact_output_matches"]),
        "known_parts": unique_values(known_parts),
        "unknown_parts": unique_values(unknown_parts),
    }


# =============================================================================
# SECTION 12: KNOWN PIECE OUTPUT LIBRARY
# =============================================================================

def collect_output_piece_records_for_pair(pair_index, pair):
    output_grid = pair.get("output")
    nodes = collect_role_path_nodes(output_grid)

    records = []

    for node in nodes:
        bbox = node.get("bbox")

        records.append({
            "pair_index": pair_index,
            "path": node["path"],
            "role": node["role"],
            "signature": node["signature"],
            "color": node["color"],
            "bbox_size": bbox_size_text(bbox),
            "bbox_area": bbox_area_from_bbox(bbox),
            "cell_count": node["cell_count"],
            "child_count": node["child_count"],
        })

    return records


def build_known_piece_output_library(task):
    train_pairs = task.get("train", [])

    records = []

    for pair_index, pair in enumerate(train_pairs):
        records.extend(
            collect_output_piece_records_for_pair(pair_index, pair)
        )

    by_signature = {}

    for record in records:
        signature = record["signature"]

        if signature not in by_signature:
            by_signature[signature] = []

        by_signature[signature].append(record)

    library = []

    for signature, signature_records in by_signature.items():
        bbox_sizes = unique_values([
            record["bbox_size"]
            for record in signature_records
        ])

        cell_counts = unique_values([
            record["cell_count"]
            for record in signature_records
        ])

        colors = unique_values([
            record["color"]
            for record in signature_records
        ])

        roles = unique_values([
            record["role"]
            for record in signature_records
        ])

        library.append({
            "signature": signature,
            "seen_count": len(signature_records),
            "roles": roles,
            "colors": colors,
            "bbox_sizes": bbox_sizes,
            "cell_counts": cell_counts,
            "records": signature_records,
        })

    return library


def known_piece_output_library_text_block(task, title):
    library = build_known_piece_output_library(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 12: WHAT OUTPUT FORM DID EACH KNOWN PIECE BECOME?")
    lines.append("")

    for item in library:
        lines.append(f"piece signature : {item['signature']}")
        lines.append(f"seen count      : {item['seen_count']}")
        lines.append(f"roles           : {item['roles']}")
        lines.append(f"colors          : {item['colors']}")
        lines.append(f"output sizes    : {item['bbox_sizes']}")
        lines.append(f"cell counts     : {item['cell_counts']}")

        for record in item["records"]:
            lines.append(
                "  "
                f"pair {record['pair_index']} "
                f"{record['path']} | "
                f"role={record['role']} | "
                f"size={record['bbox_size']} | "
                f"cells={record['cell_count']} | "
                f"children={record['child_count']}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_12_known_piece_output_library_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 12 KNOWN PIECE OUTPUT LIBRARY — TASK {task_id}")
    print("#" * 80)
    print()
    print(known_piece_output_library_text_block(
        task,
        f"TASK {task_id} KNOWN PIECE OUTPUT LIBRARY",
    ))


# =============================================================================
# SECTION 13: APPLY KNOWN PIECES TO TEST STRUCTURE PLAN
# =============================================================================

def known_piece_lookup(task):
    library = build_known_piece_output_library(task)

    lookup = {}

    for item in library:
        lookup[item["signature"]] = item

    return lookup


def collect_test_piece_plan_from_node(node, path, lookup, results):
    signature = role_tree_signature_from_node(node)
    known_item = lookup.get(signature)

    if known_item is None:
        known = False
        output_sizes = []
        cell_counts = []
    else:
        known = True
        output_sizes = known_item["bbox_sizes"]
        cell_counts = known_item["cell_counts"]

    results.append({
        "path": path,
        "role": node.get("role"),
        "signature": signature,
        "known": known,
        "output_sizes": output_sizes,
        "cell_counts": cell_counts,
        "child_count": len(node.get("children", [])),
    })

    for child_index, child in enumerate(node.get("children", [])):
        child_path = f"{path}.child{child_index}"
        collect_test_piece_plan_from_node(child, child_path, lookup, results)


def test_structure_plan_for_pair(task, pair):
    input_grid = pair.get("input")
    tree = build_role_tree(input_grid)
    lookup = known_piece_lookup(task)

    results = []

    for root_index, root in enumerate(tree.get("roots", [])):
        path = f"root{root_index}"
        collect_test_piece_plan_from_node(root, path, lookup, results)

    return results


def test_structure_plan_text_block(task, title):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 13: WHICH TEST PIECES ARE ALREADY KNOWN?")
    lines.append("")

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        plan = test_structure_plan_for_pair(task, pair)

        lines.append(f"TEST {test_index}")
        lines.append("-" * 60)

        for item in plan:
            lines.append(
                f"{item['path']} | "
                f"role={item['role']} | "
                f"known={item['known']} | "
                f"signature={item['signature']} | "
                f"sizes={item['output_sizes']} | "
                f"cells={item['cell_counts']} | "
                f"children={item['child_count']}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_13_test_structure_plan_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 13 TEST STRUCTURE PLAN — TASK {task_id}")
    print("#" * 80)
    print()
    print(test_structure_plan_text_block(
        task,
        f"TASK {task_id} TEST STRUCTURE PLAN",
    ))



# =============================================================================
# SECTION 14: COMPOSITION REQUIREMENTS
# =============================================================================

def composition_requirements_for_test_pair(task, pair):
    plan = test_structure_plan_for_pair(task, pair)
    input_grid = pair.get("input")
    layout_directions = container_layout_directions(input_grid)

    by_path = {
        item["path"]: item
        for item in plan
    }

    requirements = []

    for item in plan:
        if item["role"] != "container":
            continue

        if item["known"]:
            continue

        child_paths = [
            path
            for path in by_path
            if path.startswith(item["path"] + ".child")
            and path.count(".child") == item["path"].count(".child") + 1
        ]

        child_items = [
            by_path[path]
            for path in child_paths
        ]

        all_children_known = all(
            child["known"]
            for child in child_items
        )

        direction = None
        layout_info = layout_directions.get(item["path"])

        if layout_info is not None:
            direction = layout_info["direction"]

        requirements.append({
            "path": item["path"],
            "signature": item["signature"],
            "child_count": len(child_items),
            "direction": direction,
            "all_children_known": all_children_known,
            "children": child_items,
        })

    return requirements


def composition_requirements_text_block(task, title):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 14: CAN UNKNOWN TEST PARENTS BE BUILT FROM KNOWN CHILDREN?")
    lines.append("")

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        requirements = composition_requirements_for_test_pair(task, pair)

        lines.append(f"TEST {test_index}")
        lines.append("-" * 60)

        if not requirements:
            lines.append("no unknown parent containers found")
            lines.append("")
            continue

        for req in requirements:
            lines.append(f"unknown parent : {req['path']}")
            lines.append(f"signature      : {req['signature']}")
            lines.append(f"child count    : {req['child_count']}")
            lines.append(f"direction      : {req['direction']}")
            lines.append(f"children known : {req['all_children_known']}")
            lines.append("children:")

            for child in req["children"]:
                lines.append(
                    "  "
                    f"{child['path']} | "
                    f"role={child['role']} | "
                    f"known={child['known']} | "
                    f"signature={child['signature']} | "
                    f"sizes={child['output_sizes']} | "
                    f"cells={child['cell_counts']}"
                )

            lines.append("")

    return "\n".join(lines)


def print_section_14_composition_requirements_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 14 COMPOSITION REQUIREMENTS — TASK {task_id}")
    print("#" * 80)
    print()
    print(composition_requirements_text_block(
        task,
        f"TASK {task_id} COMPOSITION REQUIREMENTS",
    ))



# =============================================================================
# SECTION 15: COMPOSE UNKNOWN PARENT SIZE PLAN
# =============================================================================

def parse_size_text(size_text):
    if size_text is None:
        return None

    height_text, width_text = size_text.split("x")
    return int(height_text), int(width_text)


def compose_parent_size_plan_for_requirement(req):
    child_sizes = []

    for child in req["children"]:
        if not child["output_sizes"]:
            child_sizes.append(None)
            continue

        size_text = child["output_sizes"][0]
        child_sizes.append(parse_size_text(size_text))

    if any(size is None for size in child_sizes):
        return {
            "can_plan": False,
            "reason": "missing child size",
            "child_sizes": child_sizes,
            "planned_parent_size": None,
        }

    direction = req["direction"]

    # KISS assumption for planning only:
    # known outputs show child offset/border gap of 2
    border_gap = 2
    between_gap = 1

    if direction == "vertical":
        child_heights = [size[0] for size in child_sizes]
        child_widths = [size[1] for size in child_sizes]

        planned_h = border_gap + sum(child_heights) + between_gap * (len(child_sizes) - 1) + border_gap
        planned_w = border_gap + max(child_widths) + border_gap

    elif direction == "horizontal":
        child_heights = [size[0] for size in child_sizes]
        child_widths = [size[1] for size in child_sizes]

        planned_h = border_gap + max(child_heights) + border_gap
        planned_w = border_gap + sum(child_widths) + between_gap * (len(child_sizes) - 1) + border_gap

    else:
        return {
            "can_plan": False,
            "reason": f"unsupported direction {direction}",
            "child_sizes": child_sizes,
            "planned_parent_size": None,
        }

    return {
        "can_plan": True,
        "reason": "ok",
        "direction": direction,
        "child_sizes": child_sizes,
        "border_gap": border_gap,
        "between_gap": between_gap,
        "planned_parent_size": f"{planned_h}x{planned_w}",
    }


def compose_unknown_parent_size_plan_text_block(task, title):
    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 15: WHAT SIZE WOULD UNKNOWN TEST PARENTS NEED?")
    lines.append("")

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        requirements = composition_requirements_for_test_pair(task, pair)

        lines.append(f"TEST {test_index}")
        lines.append("-" * 60)

        for req in requirements:
            plan = compose_parent_size_plan_for_requirement(req)

            lines.append(f"unknown parent : {req['path']}")
            lines.append(f"signature      : {req['signature']}")
            lines.append(f"direction      : {req['direction']}")
            lines.append(f"children known : {req['all_children_known']}")
            lines.append(f"child sizes    : {plan['child_sizes']}")
            lines.append(f"can plan       : {plan['can_plan']}")
            lines.append(f"reason         : {plan['reason']}")

            if plan["can_plan"]:
                lines.append(f"border gap     : {plan['border_gap']}")
                lines.append(f"between gap    : {plan['between_gap']}")
                lines.append(f"planned size   : {plan['planned_parent_size']}")

            lines.append("")

    return "\n".join(lines)


def print_section_15_compose_parent_size_plan_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 15 COMPOSE UNKNOWN PARENT SIZE PLAN — TASK {task_id}")
    print("#" * 80)
    print()
    print(compose_unknown_parent_size_plan_text_block(
        task,
        f"TASK {task_id} COMPOSE UNKNOWN PARENT SIZE PLAN",
    ))


# =============================================================================
# SECTION 16: CHILD PLACEMENT SKETCH
# =============================================================================

def draw_filled_box_on_grid(grid, top, left, height, width, color):
    for c in range(left, left + width):
        grid[top][c] = color
        grid[top + height - 1][c] = color

    for r in range(top, top + height):
        grid[r][left] = color
        grid[r][left + width - 1] = color


def draw_single_cell_on_grid(grid, row, col, color):
    grid[row][col] = color


def make_section_16_child_placement_grid(req):
    plan = compose_parent_size_plan_for_requirement(req)

    if not plan["can_plan"]:
        return None

    parent_h, parent_w = parse_size_text(plan["planned_parent_size"])

    grid = make_empty_color_grid(parent_h, parent_w, 0)

    parent_color = 8
    child_color = 5

    draw_box_on_grid(grid, parent_color)

    child_sizes = plan["child_sizes"]

    border_gap = plan["border_gap"]
    between_gap = plan["between_gap"]

    current_top = border_gap

    if req["direction"] == "vertical":
        for child_h, child_w in child_sizes:
            child_left = (parent_w - child_w) // 2

            if child_h == 1 and child_w == 1:
                draw_single_cell_on_grid(
                    grid,
                    current_top,
                    child_left,
                    child_color,
                )
            else:
                draw_filled_box_on_grid(
                    grid,
                    current_top,
                    child_left,
                    child_h,
                    child_w,
                    child_color,
                )

            current_top += child_h + between_gap

    elif req["direction"] == "horizontal":
        current_left = border_gap

        for child_h, child_w in child_sizes:
            child_top = (parent_h - child_h) // 2

            if child_h == 1 and child_w == 1:
                draw_single_cell_on_grid(
                    grid,
                    child_top,
                    current_left,
                    child_color,
                )
            else:
                draw_filled_box_on_grid(
                    grid,
                    child_top,
                    current_left,
                    child_h,
                    child_w,
                    child_color,
                )

            current_left += child_w + between_gap

    return grid


# =============================================================================
# SECTION 17: RECURSIVE CHILD PLACEMENT SKETCH
# =============================================================================

def find_node_by_path_in_tree(tree, path):
    parts = path.split(".")

    if not parts:
        return None

    root_part = parts[0]

    if not root_part.startswith("root"):
        return None

    root_index = int(root_part.replace("root", ""))
    roots = tree.get("roots", [])

    if root_index < 0 or root_index >= len(roots):
        return None

    node = roots[root_index]

    for part in parts[1:]:
        if not part.startswith("child"):
            return None

        child_index = int(part.replace("child", ""))
        children = node.get("children", [])

        if child_index < 0 or child_index >= len(children):
            return None

        node = children[child_index]

    return node


def first_known_size_for_signature(task, signature):
    lookup = known_piece_lookup(task)
    item = lookup.get(signature)

    if item is None:
        return None

    sizes = item.get("bbox_sizes", [])

    if not sizes:
        return None

    return parse_size_text(sizes[0])


def learned_single_output_size_for_node(task, node, current_direction=None):
    signature = role_tree_signature_from_node(node)

    matches = []

    for pair in task.get("train", []):
        output_grid = pair.get("output")
        output_layouts = container_layout_directions(output_grid)
        nodes = collect_role_path_nodes(output_grid)

        for item in nodes:
            if item["signature"] != signature:
                continue

            bbox = item.get("bbox")

            if bbox is None:
                continue

            source_direction = None

            if item["role"] == "container":
                layout_info = output_layouts.get(item["path"])

                if layout_info is not None:
                    source_direction = layout_info.get("direction")

            if (
                current_direction is not None
                and source_direction is not None
                and source_direction != current_direction
            ):
                continue

            matches.append((
                bbox["height"],
                bbox["width"],
            ))

    unique_matches = []

    for match in matches:
        if match not in unique_matches:
            unique_matches.append(match)

    if len(unique_matches) != 1:
        return None

    return unique_matches[0]


def recursive_sketch_size_for_node(task, node, path, layout_directions):
    role = node.get("role")

    if role == "leaf":
        return 1, 1

    layout_info = layout_directions.get(path)
    current_direction = None

    if layout_info is not None:
        current_direction = layout_info.get("direction")

    learned_size = learned_single_output_size_for_node(
        task,
        node,
        current_direction=current_direction,
    )
    if learned_size is not None:
        return learned_size

    children = node.get("children", [])

    if not children:
        return 1, 1

    child_sizes = []

    for child_index, child in enumerate(children):
        child_path = f"{path}.child{child_index}"

        child_size = recursive_sketch_size_for_node(
            task,
            child,
            child_path,
            layout_directions,
        )

        child_sizes.append(child_size)

    layout_info = layout_directions.get(path)
    direction = None

    if layout_info is not None:
        direction = layout_info["direction"]

    edge_gap = 2
    between_gap = 1

    if direction == "horizontal":
        height = edge_gap + max(size[0] for size in child_sizes) + edge_gap

        width = (
            edge_gap
            + sum(size[1] for size in child_sizes)
            + between_gap * (len(child_sizes) - 1)
            + edge_gap
        )

        return height, width

    if direction == "vertical":
        height = (
            edge_gap
            + sum(size[0] for size in child_sizes)
            + between_gap * (len(child_sizes) - 1)
            + edge_gap
        )

        width = edge_gap + max(size[1] for size in child_sizes) + edge_gap

        return height, width

    if direction == "single_child":
        child_h, child_w = child_sizes[0]

        return (
            edge_gap + child_h + edge_gap,
            edge_gap + child_w + edge_gap,
        )

    height = (
        edge_gap
        + sum(size[0] for size in child_sizes)
        + between_gap * (len(child_sizes) - 1)
        + edge_gap
    )

    width = edge_gap + max(size[1] for size in child_sizes) + edge_gap

    return height, width


def crop_grid_to_bbox(grid, bbox):
    result = []

    for r in range(bbox["top"], bbox["bottom"] + 1):
        row = []

        for c in range(bbox["left"], bbox["right"] + 1):
            row.append(grid[r][c])

        result.append(row)

    return result


def learned_single_output_template_for_node(task, node, current_direction=None):
    signature = role_tree_signature_from_node(node)

    matches = []

    for pair in task.get("train", []):
        output_grid = pair.get("output")
        output_tree = build_role_tree(output_grid)
        output_layouts = container_layout_directions(output_grid)

        nodes = collect_role_path_nodes(output_grid)

        for item in nodes:
            if item["signature"] != signature:
                continue

            bbox = item.get("bbox")

            if bbox is None:
                continue

            source_direction = None

            if item["role"] == "container":
                layout_info = output_layouts.get(item["path"])

                if layout_info is not None:
                    source_direction = layout_info.get("direction")

            if (
                current_direction is not None
                and source_direction is not None
                and source_direction != current_direction
            ):
                continue

            template_grid = crop_grid_to_bbox(
                output_grid,
                bbox,
            )

            matches.append({
                "template": template_grid,
                "source_color": item.get("color"),
                "source_background": output_tree.get("background"),
                "source_direction": source_direction,
            })

    if len(matches) != 1:
        return None

    return matches[0]


def draw_learned_template_on_grid(grid, template_info, top, left, target_color):
    template = template_info["template"]
    source_color = template_info["source_color"]
    source_background = template_info["source_background"]

    for r, row in enumerate(template):
        for c, value in enumerate(row):
            if value == source_background:
                continue

            if value == source_color:
                grid[top + r][left + c] = target_color
            else:
                grid[top + r][left + c] = value


def draw_recursive_sketch_node(grid, task, node, path, top, left,layout_directions,color,):
    layout_info = layout_directions.get(path)
    current_direction = None

    if layout_info is not None:
        current_direction = layout_info.get("direction")

    learned_template = learned_single_output_template_for_node(
        task,
        node,
        current_direction=current_direction,
    )

    if learned_template is not None:
        draw_learned_template_on_grid(
            grid,
            learned_template,
            top,
            left,
            color,
        )
        return
    role = node.get("role")
    height, width = recursive_sketch_size_for_node(
        task,
        node,
        path,
        layout_directions,
    )

    if role == "leaf":
        draw_single_cell_on_grid(
            grid,
            top,
            left,
            color,
        )
        return

    draw_filled_box_on_grid(
        grid,
        top,
        left,
        height,
        width,
        color,
    )

    children = node.get("children", [])

    if not children:
        return

    child_sizes = []

    for child_index, child in enumerate(children):
        child_path = f"{path}.child{child_index}"
        child_sizes.append(
            recursive_sketch_size_for_node(
                task,
                child,
                child_path,
                layout_directions,
            )
        )

    layout_info = layout_directions.get(path)
    direction = None

    if layout_info is not None:
        direction = layout_info["direction"]

    border_gap = 2
    between_gap = 1

    if direction == "horizontal":
        current_left = left + border_gap

        for child_index, child in enumerate(children):
            child_h, child_w = child_sizes[child_index]
            child_top = top + (height - child_h) // 2
            child_path = f"{path}.child{child_index}"

            draw_recursive_sketch_node(
                grid,
                task,
                child,
                child_path,
                child_top,
                current_left,
                layout_directions,
                color,
            )

            current_left += child_w + between_gap

    else:
        current_top = top + border_gap

        for child_index, child in enumerate(children):
            child_h, child_w = child_sizes[child_index]
            child_left = left + (width - child_w) // 2
            child_path = f"{path}.child{child_index}"

            draw_recursive_sketch_node(
                grid,
                task,
                child,
                child_path,
                current_top,
                child_left,
                layout_directions,
                color,
            )

            current_top += child_h + between_gap


def make_section_17_recursive_child_placement_grid(task, pair, req):
    input_grid = pair.get("input")
    tree = build_role_tree(input_grid)

    node = find_node_by_path_in_tree(tree, req["path"])

    if node is None:
        return None

    layout_directions = container_layout_directions(input_grid)

    height, width = recursive_sketch_size_for_node(
        task,
        node,
        req["path"],
        layout_directions,
    )

    background = build_role_tree(input_grid).get("background", 0)

    input_grid = pair.get("input")
    background = build_role_tree(input_grid).get("background", 0)

    grid = make_empty_color_grid(height, width, background)

    color = node.get("color")

    if color is None:
        color = 8

    draw_recursive_sketch_node(
        grid,
        task,
        node,
        req["path"],
        0,
        0,
        layout_directions,
        color,
    )

    return grid



# =============================================================================
# SECTION 18: LEARN GAP VALUES FROM TRAIN OUTPUTS
# =============================================================================

def direct_child_gap_records_from_node(node, path, records):
    children = node.get("children", [])

    if children:
        parent_bbox = node.get("bbox")
        parent_h = parent_bbox["height"]
        parent_w = parent_bbox["width"]

        child_infos = []

        for child_index, child in enumerate(children):
            child_bbox = child.get("bbox")

            row_offset = child_bbox["top"] - parent_bbox["top"]
            col_offset = child_bbox["left"] - parent_bbox["left"]

            child_infos.append({
                "child_index": child_index,
                "path": f"{path}.child{child_index}",
                "role": child.get("role"),
                "signature": role_tree_signature_from_node(child),
                "bbox_size": bbox_size_text(child_bbox),
                "row_offset": row_offset,
                "col_offset": col_offset,
                "height": child_bbox["height"],
                "width": child_bbox["width"],
            })

        layout_direction = gap_layout_direction_from_child_infos(child_infos)

        edge_gaps = []
        center_gaps = []
        between_gaps = []

        if layout_direction == "vertical":
            sorted_children = sorted(
                child_infos,
                key=lambda item: item["row_offset"],
            )

            if sorted_children:
                first = sorted_children[0]
                last = sorted_children[-1]

                edge_gaps.append(first["row_offset"])
                edge_gaps.append(parent_h - last["row_offset"] - last["height"])

            for child in child_infos:
                left_gap = child["col_offset"]
                right_gap = parent_w - child["col_offset"] - child["width"]

                if left_gap == right_gap:
                    center_gaps.append(left_gap)
                else:
                    edge_gaps.append(left_gap)
                    edge_gaps.append(right_gap)

            for index in range(len(sorted_children) - 1):
                first = sorted_children[index]
                second = sorted_children[index + 1]

                gap = second["row_offset"] - (
                    first["row_offset"] + first["height"]
                )
                between_gaps.append(gap)

        elif layout_direction == "horizontal":
            sorted_children = sorted(
                child_infos,
                key=lambda item: item["col_offset"],
            )

            if sorted_children:
                first = sorted_children[0]
                last = sorted_children[-1]

                edge_gaps.append(first["col_offset"])
                edge_gaps.append(parent_w - last["col_offset"] - last["width"])

            for child in child_infos:
                top_gap = child["row_offset"]
                bottom_gap = parent_h - child["row_offset"] - child["height"]

                if top_gap == bottom_gap:
                    center_gaps.append(top_gap)
                else:
                    edge_gaps.append(top_gap)
                    edge_gaps.append(bottom_gap)

            for index in range(len(sorted_children) - 1):
                first = sorted_children[index]
                second = sorted_children[index + 1]

                gap = second["col_offset"] - (
                    first["col_offset"] + first["width"]
                )
                between_gaps.append(gap)

        elif layout_direction == "single_child":
            child = child_infos[0]

            top_gap = child["row_offset"]
            bottom_gap = parent_h - child["row_offset"] - child["height"]
            left_gap = child["col_offset"]
            right_gap = parent_w - child["col_offset"] - child["width"]

            if top_gap == bottom_gap:
                center_gaps.append(top_gap)
            else:
                edge_gaps.append(top_gap)
                edge_gaps.append(bottom_gap)

            if left_gap == right_gap:
                center_gaps.append(left_gap)
            else:
                edge_gaps.append(left_gap)
                edge_gaps.append(right_gap)

        records.append({
            "path": path,
            "signature": role_tree_signature_from_node(node),
            "parent_size": bbox_size_text(parent_bbox),
            "layout_direction": layout_direction,
            "children": child_infos,
            "edge_gaps": edge_gaps,
            "center_gaps": center_gaps,
            "between_gaps": between_gaps,
        })

    for child_index, child in enumerate(children):
        child_path = f"{path}.child{child_index}"
        direct_child_gap_records_from_node(child, child_path, records)


def gap_layout_direction_from_child_infos(child_infos):
    if len(child_infos) == 0:
        return "none"

    if len(child_infos) == 1:
        return "single_child"

    row_centers = unique_values([
        child["row_offset"] + child["height"] / 2
        for child in child_infos
    ])

    col_centers = unique_values([
        child["col_offset"] + child["width"] / 2
        for child in child_infos
    ])

    if len(col_centers) == 1 and len(row_centers) > 1:
        return "vertical"

    if len(row_centers) == 1 and len(col_centers) > 1:
        return "horizontal"

    return "mixed"


def collect_train_output_gap_records(task):
    records = []

    for pair_index, pair in enumerate(task.get("train", [])):
        output_grid = pair.get("output")
        tree = build_role_tree(output_grid)

        for root_index, root in enumerate(tree.get("roots", [])):
            root_path = f"root{root_index}"
            root_records = []

            direct_child_gap_records_from_node(
                root,
                root_path,
                root_records,
            )

            for record in root_records:
                record["pair_index"] = pair_index
                records.append(record)

    return records


def learned_gap_summary(task):
    records = collect_train_output_gap_records(task)

    edge_gaps = []
    center_gaps = []
    between_gaps = []

    for record in records:
        edge_gaps.extend(record["edge_gaps"])
        center_gaps.extend(record["center_gaps"])
        between_gaps.extend(record["between_gaps"])

    return {
        "records": records,
        "edge_gaps": unique_values(edge_gaps),
        "center_gaps": unique_values(center_gaps),
        "between_gaps": unique_values(between_gaps),
    }


def learned_gap_values_text_block(task, title):
    summary = learned_gap_summary(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 18: WHAT GAPS ARE LEARNED FROM TRAIN OUTPUTS?")
    lines.append("")
    lines.append(f"learned edge gaps    : {summary['edge_gaps']}")
    lines.append(f"learned center gaps  : {summary['center_gaps']}")
    lines.append(f"learned between gaps : {summary['between_gaps']}")
    lines.append("")

    for record in summary["records"]:
        lines.append(
            f"pair {record['pair_index']} {record['path']} | "
            f"{record['signature']} | "
            f"parent={record['parent_size']} | "
            f"layout={record['layout_direction']}"
        )

        lines.append(f"  edge gaps    : {record['edge_gaps']}")
        lines.append(f"  center gaps  : {record['center_gaps']}")
        lines.append(f"  between gaps : {record['between_gaps']}")

        for child in record["children"]:
            lines.append(
                "  "
                f"{child['path']} | "
                f"{child['signature']} | "
                f"size={child['bbox_size']} | "
                f"offset=({child['row_offset']},{child['col_offset']})"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_18_learn_gap_values_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 18 LEARN GAP VALUES — TASK {task_id}")
    print("#" * 80)
    print()
    print(learned_gap_values_text_block(
        task,
        f"TASK {task_id} LEARN GAP VALUES",
    ))


# =============================================================================
# SECTION 19: WHOLE TEST OUTPUT SKETCH
# =============================================================================

def root_sketch_size(task, root, root_path, layout_directions):
    return recursive_sketch_size_for_node(
        task,
        root,
        root_path,
        layout_directions,
    )


def root_scene_signature_from_tree(tree):
    parts = []

    for root in tree.get("roots", []):
        parts.append(root.get("role"))

    return "scene_roles[" + ",".join(parts) + "]"


def learn_root_scene_layouts(task):
    layouts = []

    for pair_index, pair in enumerate(task.get("train", [])):
        output_grid = pair.get("output")
        output_tree = build_role_tree(output_grid)

        roots = output_tree.get("roots", [])

        if len(roots) <= 1:
            continue

        output_h, output_w = grid_shape(output_grid)

        base_root = roots[0]
        base_bbox = base_root.get("bbox")

        root_records = []

        for root_index, root in enumerate(roots):
            bbox = root.get("bbox")

            root_records.append({
                "root_index": root_index,
                "role": root.get("role"),
                "signature": role_tree_signature_from_node(root),
                "color": root.get("color"),
                "top_delta": bbox["top"] - base_bbox["top"],
                "left_delta": bbox["left"] - base_bbox["left"],
                "height": bbox["height"],
                "width": bbox["width"],
            })

        layouts.append({
            "pair_index": pair_index,
            "scene_role_signature": root_scene_signature_from_tree(output_tree),
            "output_height": output_h,
            "output_width": output_w,
            "base_root_height": base_bbox["height"],
            "base_root_width": base_bbox["width"],
            "root_records": root_records,
        })

    return layouts


def find_learned_scene_layout_for_roots(task, roots):
    wanted_roles = "scene_roles[" + ",".join(
        root.get("role")
        for root in roots
    ) + "]"

    layouts = learn_root_scene_layouts(task)

    for layout in layouts:
        if layout["scene_role_signature"] == wanted_roles:
            return layout

    return None


def find_learned_scene_layout_for_roots(task, roots):
    wanted_roles = "scene_roles[" + ",".join(
        root.get("role")
        for root in roots
    ) + "]"

    layouts = learn_root_scene_layouts(task)

    for layout in layouts:
        if layout["scene_role_signature"] == wanted_roles:
            return layout

    return None


def root_scene_signature_from_tree(tree):
    parts = []

    for root in tree.get("roots", []):
        parts.append(root.get("role"))

    return "scene_roles[" + ",".join(parts) + "]"


def get_section_19_root_placements(task, pair):
    input_grid = pair.get("input")
    tree = build_role_tree(input_grid)
    layout_directions = container_layout_directions(input_grid)

    roots = tree.get("roots", [])

    if not roots:
        return []

    learned_scene_layout = find_learned_scene_layout_for_roots(
        task,
        roots,
    )

    root_infos = []

    for root_index, root in enumerate(roots):
        root_path = f"root{root_index}"

        root_h, root_w = root_sketch_size(
            task,
            root,
            root_path,
            layout_directions,
        )

        root_bbox = root.get("bbox")

        input_center_col = root_bbox["left"] + root_bbox["width"] / 2

        root_infos.append({
            "root": root,
            "path": root_path,
            "height": root_h,
            "width": root_w,
            "input_center_col": input_center_col,
            "color": root.get("color"),
        })

    root_infos = sorted(
        root_infos,
        key=lambda info: info["input_center_col"],
    )

    placed_infos = []

    if learned_scene_layout is not None:
        learned_records = learned_scene_layout["root_records"]

        for index, info in enumerate(root_infos):
            if index < len(learned_records):
                learned = learned_records[index]

                top = learned["top_delta"]
                left = learned["left_delta"]
            else:
                top = 0
                left = 0

            placed_infos.append({
                **info,
                "top": top,
                "left": left,
            })

        return placed_infos

    between_gap = 1
    current_left = 0

    main_root = root_infos[0]
    main_center_top = (main_root["height"] - 1) // 2

    for index, info in enumerate(root_infos):
        if index == 0:
            top = 0
        else:
            top = main_center_top

        placed_infos.append({
            **info,
            "top": top,
            "left": current_left,
        })

        current_left += info["width"] + between_gap

    return placed_infos


def make_optional_anchor_compass_test_grid(task, pair):
    # Do not use this path for train reconstruction.
    # Train pairs have an expected output.
    if "output" in pair:
        return None

    main_root, outside_leaf_roots = section24_find_main_container_root_and_scene_leaves(
        pair,
    )

    # This merge rule is only for scene-level outside leaves.
    # That protects TEST 0 from being changed by this debug rule.
    if not outside_leaf_roots:
        return None

    sketch = section24_make_anchor_compass_merge_sketch(
        task,
        pair,
    )

    if sketch is None:
        return None

    return sketch["grid"]


def make_section_19_whole_test_output_sketch_grid(task, pair):

    input_grid = pair.get("input")

    tree = build_role_tree(input_grid)
    roots = tree.get("roots", [])

    if not roots:
        return None

    background = tree.get("background", 0)

    placed_infos = get_section_19_root_placements(
        task,
        pair,
    )

    learned_scene_layout = find_learned_scene_layout_for_roots(
        task,
        roots,
    )

    right_pad = 0
    bottom_pad = 0

    if learned_scene_layout is not None:
        learned_extent_h = 0
        learned_extent_w = 0

        for record in learned_scene_layout["root_records"]:
            learned_extent_h = max(
                learned_extent_h,
                record["top_delta"] + record["height"],
            )

            learned_extent_w = max(
                learned_extent_w,
                record["left_delta"] + record["width"],
            )

        bottom_pad = max(
            0,
            learned_scene_layout["output_height"] - learned_extent_h,
        )

        right_pad = max(
            0,
            learned_scene_layout["output_width"] - learned_extent_w,
        )

    output_h = 0
    output_w = 0

    for info in placed_infos:
        output_h = max(
            output_h,
            info["top"] + info["height"],
        )

        output_w = max(
            output_w,
            info["left"] + info["width"],
        )

    output_h += bottom_pad
    output_w += right_pad

    grid = make_empty_color_grid(
        output_h,
        output_w,
        background,
    )

    layout_directions = container_layout_directions(input_grid)

    for info in placed_infos:
        color = info["color"]

        if color is None:
            color = 8

        draw_recursive_sketch_node(
            grid,
            task,
            info["root"],
            info["path"],
            info["top"],
            info["left"],
            layout_directions,
            color,
        )

    return grid


def print_section_19_root_layout_debug_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 19 ROOT LAYOUT DEBUG — TASK {task_id}")
    print("#" * 80)

    for test_index, pair in enumerate(task.get("test", [])):
        print()
        print(f"TEST {test_index}")
        print("-" * 60)

        placed_infos = get_section_19_root_placements(task, pair)

        for info in placed_infos:
            root = info["root"]

            print(
                f"{info['path']} | "
                f"role={root.get('role')} | "
                f"sig={role_tree_signature_from_node(root)} | "
                f"draw_pos=({info['top']},{info['left']}) | "
                f"size={info['height']}x{info['width']} | "
                f"color={info['color']}"
            )


# =============================================================================
# SECTION 20A: TRAIN OUTPUT RECONSTRUCTION CHECK
# =============================================================================

def grids_are_equal(grid_a, grid_b):
    return grid_a == grid_b


def make_whole_output_sketch_grid_from_input(task, input_grid):
    pair = {
        "input": input_grid,
    }

    return make_section_19_whole_test_output_sketch_grid(
        task,
        pair,
    )


def train_reconstruction_records(task):
    records = []

    for pair_index, pair in enumerate(task.get("train", [])):
        input_grid = pair.get("input")
        expected_grid = pair.get("output")

        reconstructed_grid = make_whole_output_sketch_grid_from_input(
            task,
            input_grid,
        )

        exact = grids_are_equal(
            reconstructed_grid,
            expected_grid,
        )

        expected_h, expected_w = grid_shape(expected_grid)
        reconstructed_h, reconstructed_w = grid_shape(reconstructed_grid)

        records.append({
            "pair_index": pair_index,
            "input": input_grid,
            "expected": expected_grid,
            "reconstructed": reconstructed_grid,
            "exact": exact,
            "expected_shape": f"{expected_h}x{expected_w}",
            "reconstructed_shape": f"{reconstructed_h}x{reconstructed_w}",
        })

    return records


def train_reconstruction_text_block(task, title):
    records = train_reconstruction_records(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 20A: CAN THE LEARNED SKETCH REBUILD TRAIN OUTPUTS?")
    lines.append("")

    exact_values = []

    for record in records:
        exact_values.append(record["exact"])

        lines.append(f"TRAIN PAIR {record['pair_index']}")
        lines.append("-" * 60)
        lines.append(f"expected shape      : {record['expected_shape']}")
        lines.append(f"reconstructed shape : {record['reconstructed_shape']}")
        lines.append(f"exact               : {record['exact']}")
        lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(f"all train exact : {all_true(exact_values)}")

    return "\n".join(lines)


def print_section_20a_train_reconstruction_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 20A TRAIN RECONSTRUCTION — TASK {task_id}")
    print("#" * 80)
    print()
    print(train_reconstruction_text_block(
        task,
        f"TASK {task_id} TRAIN RECONSTRUCTION",
    ))


# =============================================================================
# SECTION 20B: TRAIN RECONSTRUCTION DIFF REPORT
# =============================================================================

def grid_diff_summary(expected, reconstructed):
    expected_h, expected_w = grid_shape(expected)
    recon_h, recon_w = grid_shape(reconstructed)

    same_shape = expected_h == recon_h and expected_w == recon_w

    mismatch_cells = []

    max_h = max(expected_h, recon_h)
    max_w = max(expected_w, recon_w)

    for r in range(max_h):
        for c in range(max_w):
            expected_value = None
            recon_value = None

            if 0 <= r < expected_h and 0 <= c < expected_w:
                expected_value = expected[r][c]

            if 0 <= r < recon_h and 0 <= c < recon_w:
                recon_value = reconstructed[r][c]

            if expected_value != recon_value:
                mismatch_cells.append({
                    "row": r,
                    "col": c,
                    "expected": expected_value,
                    "reconstructed": recon_value,
                })

    return {
        "same_shape": same_shape,
        "expected_shape": f"{expected_h}x{expected_w}",
        "reconstructed_shape": f"{recon_h}x{recon_w}",
        "mismatch_count": len(mismatch_cells),
        "first_mismatches": mismatch_cells[:30],
    }


def train_reconstruction_diff_text_block(task, title):
    records = train_reconstruction_records(task)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 20B: WHERE DOES RECONSTRUCTION DIFFER FROM EXPECTED?")
    lines.append("")

    for record in records:
        diff = grid_diff_summary(
            record["expected"],
            record["reconstructed"],
        )

        lines.append(f"TRAIN PAIR {record['pair_index']}")
        lines.append("-" * 60)
        lines.append(f"same shape          : {diff['same_shape']}")
        lines.append(f"expected shape      : {diff['expected_shape']}")
        lines.append(f"reconstructed shape : {diff['reconstructed_shape']}")
        lines.append(f"mismatch count      : {diff['mismatch_count']}")

        lines.append("first mismatches:")
        for item in diff["first_mismatches"]:
            lines.append(
                "  "
                f"r={item['row']} c={item['col']} | "
                f"expected={item['expected']} | "
                f"reconstructed={item['reconstructed']}"
            )

        lines.append("")

    return "\n".join(lines)


def print_section_20b_train_reconstruction_diff_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 20B TRAIN RECONSTRUCTION DIFF — TASK {task_id}")
    print("#" * 80)
    print()
    print(train_reconstruction_diff_text_block(
        task,
        f"TASK {task_id} TRAIN RECONSTRUCTION DIFF",
    ))



# =============================================================================
# SECTION 1B: CONNECTED OBJECT DETECTION
# =============================================================================

def find_connected_components(grid, background_color):
    """
    Finds connected visible objects.

    Important:
        This is still observation only.
        It does not decide what the objects mean.
        It only groups same-colored touching cells.

    Connection:
        4-way connection only:
            up, down, left, right
    """
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
                    nr = current_row + dr
                    nc = current_col + dc

                    if not in_bounds(grid, nr, nc):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] != color:
                        continue

                    seen.add((nr, nc))
                    stack.append((nr, nc))

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
    """
    Looks at each possible background color and asks:

        If this color is the background,
        how many visible objects would I see?

    This is not final learning.
    This is just giving the program possible ways to see the grid.
    """
    color_counts = count_colors(grid)

    candidates = []

    for color, count in sorted(color_counts.items(), key=lambda item: item[1], reverse=True):
        components = find_connected_components(grid, color)

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
    """
    Chooses the best visual background candidate for observation.

    ARC warning:
        The background is not always the most common color in the output.

    Better display rule:
        Prefer the candidate that creates more meaningful visible objects,
        but avoid exploding the grid into tons of noise objects.

    For this visual-story task, the correct background usually produces:
        - multiple visible objects
        - sane connected components
        - not just one giant inverse object
    """
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
            cell_counts = [obj["cell_count"] for obj in components]
            bbox_areas = [
                obj["bbox"]["height"] * obj["bbox"]["width"]
                for obj in components
            ]

            largest_component = max(cell_counts) if cell_counts else 0
            total_visible = sum(cell_counts)
            largest_ratio = largest_component / total_visible if total_visible else 1

            small_object_count = sum(1 for count in cell_counts if count <= 12)

            score = 0

            # Prefer seeing several objects instead of one inverse blob.
            score += component_count * 20

            # Small objects are important because blobs often become dots.
            score += small_object_count * 10

            # Penalize candidates where almost everything becomes one giant object.
            if largest_ratio > 0.90:
                score -= 80

            # Penalize huge messy object splits.
            if component_count > 12:
                score -= 100

            # Mildly prefer candidates whose visible area is not overwhelming.
            score -= largest_component * 0.05

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate


# =============================================================================
# SECTION 2A: RELATIONSHIP HELPERS
# =============================================================================

def bbox_contains(outer_bbox, inner_bbox):
    """
    True when inner_bbox is fully inside outer_bbox.
    Allows touching the boundary.
    """
    return (
        outer_bbox["top"] <= inner_bbox["top"]
        and outer_bbox["left"] <= inner_bbox["left"]
        and outer_bbox["bottom"] >= inner_bbox["bottom"]
        and outer_bbox["right"] >= inner_bbox["right"]
    )


def bbox_strictly_contains(outer_bbox, inner_bbox):
    """
    True when inner_bbox is inside outer_bbox with at least some separation.
    This is useful for parent/child structure.
    """
    return (
        outer_bbox["top"] < inner_bbox["top"]
        and outer_bbox["left"] < inner_bbox["left"]
        and outer_bbox["bottom"] > inner_bbox["bottom"]
        and outer_bbox["right"] > inner_bbox["right"]
    )


def bbox_area(bbox):
    return bbox["height"] * bbox["width"]


def bbox_center(bbox):
    return (
        (bbox["top"] + bbox["bottom"]) / 2,
        (bbox["left"] + bbox["right"]) / 2,
    )


def relative_position(parent_bbox, child_bbox):
    """
    Describes where the child sits inside/near the parent.

    This is still observation.
    It does not force a rule.
    """
    parent_center_row, parent_center_col = bbox_center(parent_bbox)
    child_center_row, child_center_col = bbox_center(child_bbox)

    vertical = "middle"
    horizontal = "center"

    if child_center_row < parent_center_row:
        vertical = "upper"
    elif child_center_row > parent_center_row:
        vertical = "lower"

    if child_center_col < parent_center_col:
        horizontal = "left"
    elif child_center_col > parent_center_col:
        horizontal = "right"

    return f"{vertical}_{horizontal}"


def edge_gaps(parent_bbox, child_bbox):
    """
    Measures empty bbox-space between child bbox and parent bbox.

    This does not assume buffer = 1.
    It only measures the current relationship.
    """
    return {
        "top_gap": child_bbox["top"] - parent_bbox["top"],
        "left_gap": child_bbox["left"] - parent_bbox["left"],
        "bottom_gap": parent_bbox["bottom"] - child_bbox["bottom"],
        "right_gap": parent_bbox["right"] - child_bbox["right"],
    }

# =============================================================================
# SECTION 1C: PURE OBSERVATION REPORT
# =============================================================================

def observation_text_block(grid, title):
    """
    SECTION 1 QUESTION:
        What do I see?

    This does not guess.
    This does not learn a rule.
    This does not assume the final answer.
    This only reports raw visual facts.
    """
    analysis = analyze_grid(grid)
    h, w = grid_shape(grid)

    lines = []
    lines.append(title)
    lines.append("=" * 60)
    lines.append("QUESTION 1: WHAT DO I SEE?")
    lines.append("")
    lines.append(f"grid size: {h}x{w}")
    lines.append(f"colors   : {analysis['color_counts']}")
    lines.append("")
    lines.append("BACKGROUND CANDIDATES")
    lines.append("-" * 60)

    for candidate in analysis["candidates"]:
        lines.append(
            f"if background={candidate['background']}: "
            f"objects={candidate['component_count']}"
        )

    display_candidate = select_display_candidate(analysis)

    lines.append("")
    lines.append("DISPLAY CANDIDATE")
    lines.append("-" * 60)
    lines.append(f"chosen display background: {display_candidate['background']}")
    lines.append(f"visible object count     : {display_candidate['component_count']}")

    lines.append("")
    lines.append("VISIBLE OBJECTS")
    lines.append("-" * 60)

    if not display_candidate["components"]:
        lines.append("no visible objects found")
    else:
        for obj in display_candidate["components"]:
            lines.append(
                f"{obj['id']}: "
                f"color={obj['color']} "
                f"cells={obj['cell_count']} "
                f"bbox=({format_bbox(obj['bbox'])})"
            )

    return "\n".join(lines)


def print_section_1_observations_for_task(task_id, task):
    """
    Prints what the program sees before any rule discovery happens.
    """
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])

    print()
    print("#" * 80)
    print(f"SECTION 1 OBSERVATION CHECK — TASK {task_id}")
    print("#" * 80)

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        print()
        print(observation_text_block(input_grid, f"TRAIN {pair_index} INPUT"))
        print()
        print(observation_text_block(output_grid, f"TRAIN {pair_index} EXPECTED"))

    for test_index, pair in enumerate(test_pairs):
        input_grid = pair.get("input")

        print()
        print(observation_text_block(input_grid, f"TEST {test_index} INPUT"))



# =============================================================================
# VISUAL DRAWING HELPERS
# =============================================================================

ARC_COLORS = {
    0: "#000000",  # black
    1: "#0074D9",  # blue
    2: "#FF4136",  # red
    3: "#2ECC40",  # green
    4: "#FFDC00",  # yellow
    5: "#AAAAAA",  # gray
    6: "#F012BE",  # magenta
    7: "#FF851B",  # orange
    8: "#7FDBFF",  # light blue
    9: "#870C25",  # dark red
}


def color_for_value(value):
    return ARC_COLORS.get(value, "#FFFFFF")


def draw_grid(canvas, grid, x, y, cell_size=18, show_numbers=False):
    """
    Draws a grid on the Tkinter canvas.
    """
    h, w = grid_shape(grid)

    for row in range(h):
        for col in range(w):
            value = grid[row][col]

            x1 = x + col * cell_size
            y1 = y + row * cell_size
            x2 = x1 + cell_size
            y2 = y1 + cell_size

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=color_for_value(value),
                outline="#333333",
                width=1,
            )

            if show_numbers:
                canvas.create_text(
                    x1 + cell_size / 2,
                    y1 + cell_size / 2,
                    text=str(value),
                    fill="#FFFFFF" if value in [0, 9] else "#000000",
                    font=("Arial", max(6, cell_size // 2), "bold"),
                )


def draw_object_highlights(canvas, grid, x, y, cell_size=16):
    """
    Highlights each visible object and labels it with its object number.

    Safe version:
        If object detection finds nothing, draw nothing instead of crashing.
    """
    if grid is None:
        return

    analysis = analyze_grid(grid)
    display_candidate = select_display_candidate(analysis)

    if display_candidate is None:
        return

    for obj in display_candidate.get("components", []):
        bbox = obj["bbox"]

        x1 = x + bbox["left"] * cell_size
        y1 = y + bbox["top"] * cell_size
        x2 = x + (bbox["right"] + 1) * cell_size
        y2 = y + (bbox["bottom"] + 1) * cell_size

        canvas.create_rectangle(
            x1,
            y1,
            x2,
            y2,
            outline="#FFFFFF",
            width=3,
        )

        canvas.create_rectangle(
            x1,
            y1,
            x1 + 28,
            y1 + 18,
            fill="#FFFFFF",
            outline="#000000",
            width=1,
        )

        label = obj["id"].replace("object_", "")

        canvas.create_text(
            x1 + 14,
            y1 + 9,
            text=label,
            fill="#000000",
            font=("Arial", 10, "bold"),
        )


def draw_text_block(canvas, text, x, y, font_size=10):
    """
    Draws multiline text on canvas.
    Returns the approximate bottom y position.
    """
    line_height = font_size + 5
    current_y = y

    for line in text.splitlines():
        canvas.create_text(
            x,
            current_y,
            text=line,
            anchor="nw",
            fill="#000000",
            font=("Consolas", font_size),
        )
        current_y += line_height

    return current_y


def make_empty_color_grid(height, width, color=0):
    return [
        [color for _ in range(width)]
        for _ in range(height)
    ]


def draw_box_on_grid(grid, color):
    height = len(grid)
    width = len(grid[0])

    for c in range(width):
        grid[0][c] = color
        grid[height - 1][c] = color

    for r in range(height):
        grid[r][0] = color
        grid[r][width - 1] = color


def make_section_15_plan_grid(req):
    plan = compose_parent_size_plan_for_requirement(req)

    if not plan["can_plan"]:
        return None

    planned_h, planned_w = parse_size_text(plan["planned_parent_size"])

    color = 8
    if req["children"]:
        # use first child color if available later, fallback to 8
        color = 8

    grid = make_empty_color_grid(planned_h, planned_w, 0)
    draw_box_on_grid(grid, color)

    return grid

# =============================================================================
# SECTION 1E: VISUAL REVIEW WINDOW
# =============================================================================

def draw_train_pair_row(canvas, pair_index, pair, x, y,):
    input_grid = pair.get("input")
    output_grid = pair.get("output")



    title_y = y
    cell_size = 16

    canvas.create_text(
        x,
        title_y,
        text=f"TRAIN PAIR {pair_index}",
        anchor="nw",
        fill="#000000",
        font=("Arial", 14, "bold"),
    )

    grid_y = title_y + 28
    input_y = grid_y + 22

    input_h, input_w = grid_shape(input_grid)
    output_h, output_w = grid_shape(output_grid)

    # INPUT
    input_x = x

    canvas.create_text(
        input_x,
        grid_y,
        text="INPUT",
        anchor="nw",
        fill="#000000",
        font=("Arial", 11, "bold"),
    )

    draw_grid(
        canvas,
        input_grid,
        input_x,
        input_y,
        cell_size=cell_size,
        show_numbers=False,
    )
    draw_object_highlights(
        canvas,
        input_grid,
        input_x,
        input_y,
        cell_size=cell_size,
    )

    # EXPECTED
    expected_x = input_x + input_w * cell_size + 50

    canvas.create_text(
        expected_x,
        grid_y,
        text="EXPECTED",
        anchor="nw",
        fill="#000000",
        font=("Arial", 11, "bold"),
    )

    draw_grid(
        canvas,
        output_grid,
        expected_x,
        input_y,
        cell_size=cell_size,
        show_numbers=False,
    )
    draw_object_highlights(
        canvas,
        output_grid,
        expected_x,
        input_y,
        cell_size=cell_size,
    )

    # GUESS
    guess_x = expected_x + output_w * cell_size + 50
    guess_h = 0
    guess_w = 0



    row_bottom = max(
        input_y + input_h * cell_size,
        input_y + output_h * cell_size,
        input_y + guess_h * cell_size + 40,
    )

    canvas.create_line(
        x,
        row_bottom + 18,
        2600,
        row_bottom + 18,
        fill="#999999",
        width=2,
    )

    return row_bottom + 45


def draw_test_row(canvas, test_index, pair, x, y, task=None):
    input_grid = pair.get("input")


    title_y = y
    cell_size = 16

    canvas.create_text(
        x,
        title_y,
        text=f"TEST INPUT {test_index}",
        anchor="nw",
        fill="#000000",
        font=("Arial", 14, "bold"),
    )

    grid_y = title_y + 28
    input_y = grid_y + 22

    input_h, input_w = grid_shape(input_grid)

    input_x = x

    canvas.create_text(
        input_x,
        grid_y,
        text="INPUT",
        anchor="nw",
        fill="#000000",
        font=("Arial", 11, "bold"),
    )

    draw_grid(
        canvas,
        input_grid,
        input_x,
        input_y,
        cell_size=cell_size,
        show_numbers=False,
    )

    draw_object_highlights(
        canvas,
        input_grid,
        input_x,
        input_y,
        cell_size=cell_size,
    )

    text_x = input_x + input_w * cell_size + 50
    text_y = input_y

    current_sketch_y = input_y

    if task is not None:
        facts = compact_test_structure_summary(task, pair)

        lines = []
        lines.append(f"signature: {facts['signature']}")
        lines.append(f"containers: {facts['container_count']}")
        lines.append(f"leaves: {facts['leaf_count']}")
        lines.append(f"container colors: {facts['container_colors']}")
        lines.append(f"leaf colors: {facts['leaf_colors']}")
        lines.append(
            f"exact matches: "
            f"input={facts['exact_input_match_count']} "
            f"output={facts['exact_output_match_count']}"
        )

        lines.append("layout:")
        for path, info in facts["layout_directions"].items():
            lines.append(f"  {path}: {info['direction']}")

        lines.append("known parts:")
        for signature in facts["known_parts"]:
            lines.append(f"  {signature}")

        lines.append("unknown full/new parts:")
        for signature in facts["unknown_parts"]:
            lines.append(f"  {signature}")

        canvas.create_text(
            text_x,
            text_y,
            text="\n".join(lines),
            anchor="nw",
            fill="#000000",
            font=("Consolas", 10),
        )

        # SECTION 15 PLAN SKETCH
        requirements = composition_requirements_for_test_pair(task, pair)

        sketch_x = text_x + 560
        sketch_y = input_y

        canvas.create_text(
            sketch_x,
            grid_y,
            text="SECTION 17 RECURSIVE SKETCH",
            anchor="nw",
            fill="#000000",
            font=("Arial", 11, "bold"),
        )

        current_sketch_y = sketch_y

        for req in requirements:
            plan_grid = make_section_17_recursive_child_placement_grid(
                task,
                pair,
                req,
            )

            canvas.create_text(
                sketch_x,
                current_sketch_y,
                text=f"{req['path']} {req['signature']}",
                anchor="nw",
                fill="#000000",
                font=("Consolas", 9),
            )

            current_sketch_y += 18

            if plan_grid is not None:
                draw_grid(
                    canvas,
                    plan_grid,
                    sketch_x,
                    current_sketch_y,
                    cell_size=cell_size,
                    show_numbers=False,
                )

                plan_h, plan_w = grid_shape(plan_grid)
                current_sketch_y += plan_h * cell_size + 25

            else:
                canvas.create_text(
                    sketch_x,
                    current_sketch_y,
                    text="NO PLAN",
                    anchor="nw",
                    fill="#990000",
                    font=("Consolas", 10, "bold"),
                )

                current_sketch_y += 40

        # SECTION 19 WHOLE OUTPUT SKETCH
        whole_sketch_x = sketch_x + 360
        whole_sketch_y = input_y

        canvas.create_text(
            whole_sketch_x,
            grid_y,
            text="SECTION 19 WHOLE OUTPUT SKETCH",
            anchor="nw",
            fill="#000000",
            font=("Arial", 11, "bold"),
        )

        whole_grid = make_final_test_prediction_grid(
            task,
            pair,
            test_index,
        )

        if whole_grid is not None:
            draw_grid(
                canvas,
                whole_grid,
                whole_sketch_x,
                whole_sketch_y,
                cell_size=cell_size,
                show_numbers=False,
            )

            whole_h, whole_w = grid_shape(whole_grid)
            current_sketch_y = max(
                current_sketch_y,
                whole_sketch_y + whole_h * cell_size + 25,
            )
        else:
            canvas.create_text(
                whole_sketch_x,
                whole_sketch_y,
                text="NO WHOLE SKETCH",
                anchor="nw",
                fill="#990000",
                font=("Consolas", 10, "bold"),
            )
    row_bottom = max(
        input_y + input_h * cell_size,
        text_y + 260,
        current_sketch_y,
    )

    canvas.create_line(
        x,
        row_bottom + 18,
        2600,
        row_bottom + 18,
        fill="#999999",
        width=2,
    )

    return row_bottom + 45


def draw_train_reconstruction_row(canvas, pair_index, record, x, y):
    cell_size = 16

    input_grid = record["input"]
    expected_grid = record["expected"]
    reconstructed_grid = record["reconstructed"]

    input_h, input_w = grid_shape(input_grid)
    expected_h, expected_w = grid_shape(expected_grid)
    reconstructed_h, reconstructed_w = grid_shape(reconstructed_grid)

    title_y = y
    header_y = title_y + 28
    grid_y = header_y + 24

    canvas.create_text(
        x,
        title_y,
        text=(
            f"TRAIN {pair_index} RECONSTRUCTION CHECK | "
            f"exact={record['exact']} | "
            f"expected={record['expected_shape']} | "
            f"reconstructed={record['reconstructed_shape']}"
        ),
        anchor="nw",
        fill="#000000",
        font=("Arial", 14, "bold"),
    )

    input_x = x
    expected_x = input_x + input_w * cell_size + 60
    reconstructed_x = expected_x + expected_w * cell_size + 60

    canvas.create_text(
        input_x,
        header_y,
        text="INPUT",
        anchor="nw",
        fill="#000000",
        font=("Arial", 11, "bold"),
    )

    canvas.create_text(
        expected_x,
        header_y,
        text="EXPECTED",
        anchor="nw",
        fill="#000000",
        font=("Arial", 11, "bold"),
    )

    canvas.create_text(
        reconstructed_x,
        header_y,
        text="RECONSTRUCTED",
        anchor="nw",
        fill="#000000",
        font=("Arial", 11, "bold"),
    )

    draw_grid(
        canvas,
        input_grid,
        input_x,
        grid_y,
        cell_size=cell_size,
        show_numbers=False,
    )

    draw_grid(
        canvas,
        expected_grid,
        expected_x,
        grid_y,
        cell_size=cell_size,
        show_numbers=False,
    )

    draw_grid(
        canvas,
        reconstructed_grid,
        reconstructed_x,
        grid_y,
        cell_size=cell_size,
        show_numbers=False,
    )

    status_y = grid_y + max(
        input_h,
        expected_h,
        reconstructed_h,
    ) * cell_size + 10

    if record["exact"]:
        status_text = "PASS: reconstructed output exactly matches expected output"
        status_color = "#006600"
    else:
        diff = grid_diff_summary(
            expected_grid,
            reconstructed_grid,
        )

        status_text = (
            "FAIL: "
            f"same_shape={diff['same_shape']} | "
            f"mismatches={diff['mismatch_count']}"
        )
        status_color = "#990000"

    canvas.create_text(
        x,
        status_y,
        text=status_text,
        anchor="nw",
        fill=status_color,
        font=("Consolas", 11, "bold"),
    )

    row_bottom = status_y + 30

    canvas.create_line(
        x,
        row_bottom + 14,
        2600,
        row_bottom + 14,
        fill="#999999",
        width=2,
    )

    return row_bottom + 40


def create_scrollable_task_window(root, task_id, task, x=40, y=40):
    window = tk.Toplevel(root)
    window.title(f"VISUAL STORY REVIEW — {task_id}")
    window.geometry(f"1400x900+{x}+{y}")

    outer = Frame(window)
    outer.pack(fill="both", expand=True)

    canvas = Canvas(outer, bg="#F5F5F5")
    canvas.pack(side="left", fill="both", expand=True)

    y_scrollbar = Scrollbar(outer, orient="vertical", command=canvas.yview)
    y_scrollbar.pack(side="right", fill="y")

    x_scrollbar = Scrollbar(window, orient="horizontal", command=canvas.xview)
    x_scrollbar.pack(side="bottom", fill="x")

    canvas.configure(
        yscrollcommand=y_scrollbar.set,
        xscrollcommand=x_scrollbar.set,
    )

    current_y = 25
    start_x = 25

    canvas.create_text(
        start_x,
        current_y,
        text=f"VISUAL STORY REVIEW — {task_id}",
        anchor="nw",
        fill="#000000",
        font=("Arial", 18, "bold"),
    )

    current_y += 45

    canvas.create_text(
        start_x,
        current_y,
        text="KNOWN-ANSWER SANITY CHECK: TRAIN INPUT | EXPECTED | RECONSTRUCTED",
        anchor="nw",
        fill="#000000",
        font=("Arial", 12, "bold"),
    )

    current_y += 35

    train_records = train_reconstruction_records(task)

    all_exact = all_true(
        record["exact"]
        for record in train_records
    )

    canvas.create_text(
        start_x,
        current_y,
        text=f"ALL TRAIN EXACT: {all_exact}",
        anchor="nw",
        fill="#006600" if all_exact else "#990000",
        font=("Arial", 14, "bold"),
    )

    current_y += 40

    for record in train_records:
        current_y = draw_train_reconstruction_row(
            canvas=canvas,
            pair_index=record["pair_index"],
            record=record,
            x=start_x,
            y=current_y,
        )

    current_y += 30

    canvas.create_line(
        start_x,
        current_y,
        2600,
        current_y,
        fill="#000000",
        width=4,
    )

    current_y += 30

    canvas.create_text(
        start_x,
        current_y,
        text="UNKNOWN-ANSWER INSPECTION: TEST INPUT | PREDICTED",
        anchor="nw",
        fill="#000000",
        font=("Arial", 14, "bold"),
    )

    current_y += 40

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        current_y = draw_test_row(
            canvas=canvas,
            test_index=test_index,
            pair=pair,
            x=start_x,
            y=current_y,
            task=task,
        )

    canvas.configure(scrollregion=(0, 0, 2600, current_y + 100))

    return window


# =============================================================================
# SECTION 22: CHILD CENTER PATTERN LEARNING CHECK
# =============================================================================

def center_of_bbox_relative_to_parent(parent_bbox, child_bbox):
    parent_top = parent_bbox["top"]
    parent_left = parent_bbox["left"]

    child_center_row = (
        child_bbox["top"]
        - parent_top
        + (child_bbox["height"] - 1) / 2
    )

    child_center_col = (
        child_bbox["left"]
        - parent_left
        + (child_bbox["width"] - 1) / 2
    )

    return child_center_row, child_center_col


def collect_child_center_pattern_for_grid(grid):
    tree = build_role_tree(grid)
    results = {}

    def walk(node, path):
        children = node.get("children", [])

        if children:
            parent_bbox = node.get("bbox")

            child_records = []

            for child_index, child in enumerate(children):
                child_path = f"{path}.child{child_index}"
                child_bbox = child.get("bbox")

                center_row, center_col = center_of_bbox_relative_to_parent(
                    parent_bbox,
                    child_bbox,
                )

                child_records.append({
                    "child_path": child_path,
                    "role": child.get("role"),
                    "signature": role_tree_signature_from_node(child),
                    "center_row": center_row,
                    "center_col": center_col,
                    "bbox_size": bbox_size_text(child_bbox),
                })

            rows = [
                record["center_row"]
                for record in child_records
            ]

            cols = [
                record["center_col"]
                for record in child_records
            ]

            row_span = max(rows) - min(rows) if rows else 0
            col_span = max(cols) - min(cols) if cols else 0

            if row_span > col_span:
                spread_direction = "row_spread"
            elif col_span > row_span:
                spread_direction = "col_spread"
            else:
                spread_direction = "same_spread"

            results[path] = {
                "parent_signature": role_tree_signature_from_node(node),
                "parent_size": bbox_size_text(parent_bbox),
                "child_count": len(children),
                "row_span": row_span,
                "col_span": col_span,
                "spread_direction": spread_direction,
                "children": child_records,
            }

        for child_index, child in enumerate(children):
            child_path = f"{path}.child{child_index}"
            walk(child, child_path)

    for root_index, root in enumerate(tree.get("roots", [])):
        walk(root, f"root{root_index}")

    return results


def compare_child_center_patterns(input_pattern, output_pattern):
    all_paths = sorted(
        set(input_pattern.keys()) |
        set(output_pattern.keys())
    )

    comparisons = []

    for path in all_paths:
        input_item = input_pattern.get(path)
        output_item = output_pattern.get(path)

        if input_item is None or output_item is None:
            comparisons.append({
                "path": path,
                "input_exists": input_item is not None,
                "output_exists": output_item is not None,
                "same_child_count": False,
                "same_spread_direction": False,
                "input": input_item,
                "output": output_item,
            })
            continue

        comparisons.append({
            "path": path,
            "input_exists": True,
            "output_exists": True,
            "same_child_count": (
                input_item["child_count"] == output_item["child_count"]
            ),
            "same_spread_direction": (
                input_item["spread_direction"] == output_item["spread_direction"]
            ),
            "input": input_item,
            "output": output_item,
        })

    return comparisons


def print_child_center_pattern_block(label, pattern):
    print(label)
    print("-" * 60)

    if not pattern:
        print("no parent-child patterns found")
        return

    for path, item in pattern.items():
        print(
            f"{path} | "
            f"parent={item['parent_signature']} | "
            f"size={item['parent_size']} | "
            f"children={item['child_count']} | "
            f"row_span={item['row_span']} | "
            f"col_span={item['col_span']} | "
            f"spread={item['spread_direction']}"
        )

        for child in item["children"]:
            print(
                "  "
                f"{child['child_path']} | "
                f"role={child['role']} | "
                f"sig={child['signature']} | "
                f"center=({child['center_row']},{child['center_col']}) | "
                f"size={child['bbox_size']}"
            )


def print_section_22_child_center_pattern_learning_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 22 CHILD CENTER PATTERN LEARNING — TASK {task_id}")
    print("#" * 80)
    print()
    print("QUESTION 22: DOES INPUT CHILD-CENTER PATTERN PRESERVE INTO OUTPUT?")
    print("This section only measures. It does not draw or predict.")
    print()

    train_pairs = task.get("train", [])

    all_same_child_count = []
    all_same_spread_direction = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        input_pattern = collect_child_center_pattern_for_grid(input_grid)
        output_pattern = collect_child_center_pattern_for_grid(output_grid)

        comparisons = compare_child_center_patterns(
            input_pattern,
            output_pattern,
        )

        print()
        print("=" * 80)
        print(f"TRAIN PAIR {pair_index}")
        print("=" * 80)

        print()
        print_child_center_pattern_block(
            "INPUT CHILD-CENTER PATTERN",
            input_pattern,
        )

        print()
        print_child_center_pattern_block(
            "OUTPUT CHILD-CENTER PATTERN",
            output_pattern,
        )

        print()
        print("INPUT -> OUTPUT PATTERN COMPARISON")
        print("-" * 60)

        for comparison in comparisons:
            all_same_child_count.append(
                comparison["same_child_count"]
            )

            all_same_spread_direction.append(
                comparison["same_spread_direction"]
            )

            input_item = comparison["input"]
            output_item = comparison["output"]

            if input_item is None:
                input_spread = "missing"
            else:
                input_spread = input_item["spread_direction"]

            if output_item is None:
                output_spread = "missing"
            else:
                output_spread = output_item["spread_direction"]

            print(
                f"{comparison['path']} | "
                f"exists={comparison['input_exists']}->{comparison['output_exists']} | "
                f"child_count_same={comparison['same_child_count']} | "
                f"spread={input_spread}->{output_spread} | "
                f"spread_same={comparison['same_spread_direction']}"
            )

    print()
    print("=" * 80)
    print("SECTION 22 SUMMARY")
    print("=" * 80)
    print(f"child count same all      : {all_true(all_same_child_count)}")
    print(f"spread direction same all : {all_true(all_same_spread_direction)}")


# =============================================================================
# SECTION 23: ANCHOR-COMPASS LEARNING CHECK
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


def anchor_child_for_parent(children):
    container_children = [
        child
        for child in children
        if child.get("role") == "container"
    ]

    if not container_children:
        return None

    container_children = sorted(
        container_children,
        key=lambda child: child["bbox"]["height"] * child["bbox"]["width"],
        reverse=True,
    )

    return container_children[0]


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


def collect_anchor_compass_facts_for_grid(grid):
    tree = build_role_tree(grid)
    results = {}

    def walk(node, path):
        children = node.get("children", [])

        if children:
            anchor = anchor_child_for_parent(children)

            if anchor is not None:
                parent_bbox = node.get("bbox")
                anchor_bbox = anchor.get("bbox")

                anchor_path = None

                for child_index, child in enumerate(children):
                    if child is anchor:
                        anchor_path = f"{path}.child{child_index}"
                        break

                anchor_center_row, anchor_center_col = local_center_from_parent(
                    parent_bbox,
                    anchor_bbox,
                )

                child_facts = []

                for child_index, child in enumerate(children):
                    child_path = f"{path}.child{child_index}"

                    if child_path == anchor_path:
                        continue

                    child_bbox = child.get("bbox")

                    child_center_row, child_center_col = local_center_from_parent(
                        parent_bbox,
                        child_bbox,
                    )

                    row_delta = child_center_row - anchor_center_row
                    col_delta = child_center_col - anchor_center_col

                    bearing = compass_bearing_from_delta(
                        row_delta,
                        col_delta,
                    )

                    child_facts.append({
                        "child_path": child_path,
                        "role": child.get("role"),
                        "signature": role_tree_signature_from_node(child),
                        "row_delta": row_delta,
                        "col_delta": col_delta,
                        "bearing": bearing,
                        "child_size": bbox_size_text(child_bbox),
                    })

                results[path] = {
                    "parent_signature": role_tree_signature_from_node(node),
                    "parent_size": bbox_size_text(parent_bbox),
                    "anchor_path": anchor_path,
                    "anchor_signature": role_tree_signature_from_node(anchor),
                    "anchor_size": bbox_size_text(anchor_bbox),
                    "children": child_facts,
                }

        for child_index, child in enumerate(children):
            walk(
                child,
                f"{path}.child{child_index}",
            )

    for root_index, root in enumerate(tree.get("roots", [])):
        walk(
            root,
            f"root{root_index}",
        )

    return results


def compare_anchor_compass_facts(input_facts, output_facts):
    all_paths = sorted(
        set(input_facts.keys()) |
        set(output_facts.keys())
    )

    comparisons = []

    for path in all_paths:
        input_item = input_facts.get(path)
        output_item = output_facts.get(path)

        if input_item is None or output_item is None:
            comparisons.append({
                "path": path,
                "input_exists": input_item is not None,
                "output_exists": output_item is not None,
                "same_anchor_signature": False,
                "same_child_bearings": False,
                "input": input_item,
                "output": output_item,
            })
            continue

        input_bearings = {
            child["child_path"]: child["bearing"]
            for child in input_item["children"]
        }

        output_bearings = {
            child["child_path"]: child["bearing"]
            for child in output_item["children"]
        }

        comparisons.append({
            "path": path,
            "input_exists": True,
            "output_exists": True,
            "same_anchor_signature": (
                input_item["anchor_signature"] == output_item["anchor_signature"]
            ),
            "same_child_bearings": (
                input_bearings == output_bearings
            ),
            "input_bearings": input_bearings,
            "output_bearings": output_bearings,
            "input": input_item,
            "output": output_item,
        })

    return comparisons


def print_anchor_compass_block(label, facts):
    print(label)
    print("-" * 60)

    if not facts:
        print("no anchor-container parent found")
        return

    for path, item in facts.items():
        print(
            f"{path} | "
            f"parent={item['parent_signature']} | "
            f"size={item['parent_size']} | "
            f"anchor={item['anchor_path']} | "
            f"anchor_sig={item['anchor_signature']} | "
            f"anchor_size={item['anchor_size']}"
        )

        for child in item["children"]:
            print(
                "  "
                f"{child['child_path']} | "
                f"role={child['role']} | "
                f"sig={child['signature']} | "
                f"bearing={child['bearing']} | "
                f"delta=({child['row_delta']},{child['col_delta']}) | "
                f"size={child['child_size']}"
            )


def print_section_23_anchor_compass_learning_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 23 ANCHOR-COMPASS LEARNING — TASK {task_id}")
    print("#" * 80)
    print()
    print("QUESTION 23: DO CHILD BEARINGS AROUND ANCHOR CONTAINER PRESERVE?")
    print("This section only measures. It does not draw or predict.")
    print()

    train_pairs = task.get("train", [])

    all_same_anchor_signature = []
    all_same_child_bearings = []

    for pair_index, pair in enumerate(train_pairs):
        input_grid = pair.get("input")
        output_grid = pair.get("output")

        input_facts = collect_anchor_compass_facts_for_grid(input_grid)
        output_facts = collect_anchor_compass_facts_for_grid(output_grid)

        comparisons = compare_anchor_compass_facts(
            input_facts,
            output_facts,
        )

        print()
        print("=" * 80)
        print(f"TRAIN PAIR {pair_index}")
        print("=" * 80)

        print()
        print_anchor_compass_block(
            "INPUT ANCHOR-COMPASS FACTS",
            input_facts,
        )

        print()
        print_anchor_compass_block(
            "OUTPUT ANCHOR-COMPASS FACTS",
            output_facts,
        )

        print()
        print("INPUT -> OUTPUT ANCHOR-COMPASS COMPARISON")
        print("-" * 60)

        for comparison in comparisons:
            all_same_anchor_signature.append(
                comparison["same_anchor_signature"]
            )

            all_same_child_bearings.append(
                comparison["same_child_bearings"]
            )

            print(
                f"{comparison['path']} | "
                f"exists={comparison['input_exists']}->{comparison['output_exists']} | "
                f"anchor_same={comparison['same_anchor_signature']} | "
                f"bearings={comparison.get('input_bearings')}->{comparison.get('output_bearings')} | "
                f"bearings_same={comparison['same_child_bearings']}"
            )

    print()
    print("=" * 80)
    print("SECTION 23 SUMMARY")
    print("=" * 80)
    print(f"anchor signature same all : {all_true(all_same_anchor_signature)}")
    print(f"child bearings same all   : {all_true(all_same_child_bearings)}")


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


def section24_draw_container_border(grid, top, left, height, width, color):
    for col in range(left, left + width):
        section24_draw_cell(grid, top, col, color)
        section24_draw_cell(grid, top + height - 1, col, color)

    for row in range(top, top + height):
        section24_draw_cell(grid, row, left, color)
        section24_draw_cell(grid, row, left + width - 1, color)


# =============================================================================
# SECTION 24: TEST ANCHOR-COMPASS MERGE SKETCH
# =============================================================================

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


def section24_draw_filled_box(grid, top, left, height, width, color):
    grid_height = len(grid)
    grid_width = len(grid[0]) if grid_height else 0

    for row in range(top, top + height):
        for col in range(left, left + width):
            if 0 <= row < grid_height and 0 <= col < grid_width:
                grid[row][col] = color


def section24_draw_cell(grid, row, col, color):
    if 0 <= row < len(grid) and 0 <= col < len(grid[0]):
        grid[row][col] = color


def section24_draw_template(grid, template_info, top, left, target_color):
    mask = template_info["mask"]

    for row_index, row in enumerate(mask):
        for col_index, value in enumerate(row):
            if value == 1:
                grid[top + row_index][left + col_index] = target_color


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

    # Even-width buffer: this matches the visual target where the 9-o'clock
    # leaf needs room, but the right side still keeps buffer.
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


def print_grid_as_python_list(grid):
    print("[")

    for row in grid:
        print(f"    {row},")

    print("]")


def print_section_24_anchor_compass_merge_sketch_for_task(task_id, task):
    print()
    print("#" * 80)
    print(f"SECTION 24 ANCHOR-COMPASS MERGE SKETCH — TASK {task_id}")
    print("#" * 80)
    print()
    print("QUESTION 24: WHAT DOES AN ANCHOR-COMPASS MERGED TEST SKETCH LOOK LIKE?")
    print("This section is debug only. It does not replace the main prediction.")
    print()

    test_pairs = task.get("test", [])

    for test_index, pair in enumerate(test_pairs):
        sketch = section24_make_anchor_compass_merge_sketch(
            task,
            pair,
        )

        print()
        print("=" * 80)
        print(f"TEST {test_index}")
        print("=" * 80)

        if sketch is None:
            print("no anchor-compass merge sketch available")
            continue

        print(f"sketch size : {sketch['height']}x{sketch['width']}")
        print(f"anchor size : {sketch['anchor_size']}")
        print("bearings:")
        for record in sketch["bearings"]:
            print(
                f"  {record['source']} -> {record['bearing']}"
            )

        print()
        print("NUMBER GRID")
        print("-" * 60)
        print_grid_as_python_list(
            sketch["grid"],
        )


# =============================================================================
# SECTION MAIN : MAIN
# =============================================================================


def main():
    user_text = input("Enter task file, task json, or data: ").strip()
    task_file = resolve_task_file(user_text)

    if not os.path.exists(task_file):
        print(f"File not found: {task_file}")
        return

    raw_data = load_json_file(task_file)
    tasks = normalize_loaded_tasks(raw_data)

    if not tasks:
        print("No tasks found.")
        return

    print(f"Loaded: {task_file}")
    print(f"Task count: {len(tasks)}")
    print()
    print("visual_story_review.py")
    print("SECTION 1: pure observation")
    print("Discovery uses all train pairs.")
    print("No guessing yet.")
    print("No rule search yet.")
    print("No leave-one-out during discovery.")

    for task_id, task in tasks.items():
        print_section_2_relationships_for_task(task_id, task)
        print_section_3_role_trees_for_task(task_id, task)
        print_section_4_pair_changes_for_task(task_id, task)
        print_section_5_consistent_facts_for_task(task_id, task)
        print_section_6_leaf_preservation_for_task(task_id, task)
        print_section_7_container_preservation_for_task(task_id, task)
        print_section_8_child_positions_for_task(task_id, task)
        print_section_9_input_output_layout_for_task(task_id, task)

        print_section_10_test_input_observation_for_task(task_id, task)
        print_section_11_test_structure_match_for_task(task_id, task)
        print_section_12_known_piece_output_library_for_task(task_id, task)
        print_section_13_test_structure_plan_for_task(task_id, task)
        print_section_14_composition_requirements_for_task(task_id, task)
        print_section_15_compose_parent_size_plan_for_task(task_id, task)

        print_section_18_learn_gap_values_for_task(task_id, task)
        print_section_19_root_layout_debug_for_task(task_id, task)

        print_section_20a_train_reconstruction_for_task(task_id, task)
        print_section_20b_train_reconstruction_diff_for_task(task_id, task)
        print_section_22_child_center_pattern_learning_for_task(task_id, task)

        print_section_23_anchor_compass_learning_for_task(task_id, task)
        print_section_24_anchor_compass_merge_sketch_for_task(task_id,task,)


    root = tk.Tk()
    root.withdraw()

    start_x = 30
    start_y = 30
    step_x = 35
    step_y = 35

    for index, (task_id, task) in enumerate(tasks.items()):
        x = start_x + (index % 8) * step_x
        y = start_y + (index % 8) * step_y

        create_scrollable_task_window(
            root=root,
            task_id=task_id,
            task=task,
            x=x,
            y=y,
        )

    root.mainloop()


if __name__ == "__main__":
    main()