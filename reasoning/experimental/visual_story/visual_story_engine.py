from collections import deque, Counter


def grid_shape(grid):
    return len(grid), len(grid[0]) if grid else 0


def in_bounds(r, c, h, w):
    return 0 <= r < h and 0 <= c < w


def neighbors4(r, c):
    return [
        (r - 1, c),
        (r + 1, c),
        (r, c - 1),
        (r, c + 1),
    ]


def detect_background_color(grid):
    values = []
    for row in grid:
        values.extend(row)

    if not values:
        return 0

    counts = Counter(values)
    return counts.most_common(1)[0][0]


def component_bbox(cells):
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    top = min(rows)
    bottom = max(rows)
    left = min(cols)
    right = max(cols)

    return {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "height": bottom - top + 1,
        "width": right - left + 1,
    }


def extract_connected_components(grid, background):
    h, w = grid_shape(grid)
    seen = set()
    objects = []

    object_id = 1

    for r in range(h):
        for c in range(w):
            color = grid[r][c]

            if color == background:
                continue

            if (r, c) in seen:
                continue

            queue = deque([(r, c)])
            seen.add((r, c))
            cells = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(nr, nc, h, w):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] != color:
                        continue

                    seen.add((nr, nc))
                    queue.append((nr, nc))

            bbox = component_bbox(cells)

            touches_border = any(
                rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1
                for rr, cc in cells
            )

            objects.append({
                "id": f"object_{object_id}",
                "color": color,
                "cell_count": len(cells),
                "bbox": bbox,
                "touches_border": touches_border,
                "cells": cells,
            })

            object_id += 1

    return objects


def extract_background_regions(grid, background):
    h, w = grid_shape(grid)
    seen = set()
    regions = []

    region_id = 1

    for r in range(h):
        for c in range(w):
            if grid[r][c] != background:
                continue

            if (r, c) in seen:
                continue

            queue = deque([(r, c)])
            seen.add((r, c))
            cells = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for nr, nc in neighbors4(cr, cc):
                    if not in_bounds(nr, nc, h, w):
                        continue

                    if (nr, nc) in seen:
                        continue

                    if grid[nr][nc] != background:
                        continue

                    seen.add((nr, nc))
                    queue.append((nr, nc))

            bbox = component_bbox(cells)

            touches_border = any(
                rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1
                for rr, cc in cells
            )

            regions.append({
                "id": f"background_region_{region_id}",
                "cell_count": len(cells),
                "bbox": bbox,
                "touches_border": touches_border,
                "cells": cells,
            })

            region_id += 1

    return regions


def bbox_contains(outer, inner):
    return (
        outer["top"] <= inner["top"]
        and outer["left"] <= inner["left"]
        and outer["bottom"] >= inner["bottom"]
        and outer["right"] >= inner["right"]
    )


def bbox_strictly_contains(outer, inner):
    return (
        outer["top"] < inner["top"]
        and outer["left"] < inner["left"]
        and outer["bottom"] > inner["bottom"]
        and outer["right"] > inner["right"]
    )


def object_center(obj):
    bbox = obj["bbox"]
    return (
        (bbox["top"] + bbox["bottom"]) / 2,
        (bbox["left"] + bbox["right"]) / 2,
    )


def classify_shape_family(obj):
    bbox = obj["bbox"]
    cell_count = obj["cell_count"]
    area = bbox["height"] * bbox["width"]

    if cell_count == area:
        if bbox["height"] == bbox["width"]:
            return "solid_square"
        return "solid_rectangle"

    fill_ratio = cell_count / area if area else 0

    if fill_ratio < 0.45:
        return "sparse_shape"

    if bbox["height"] == bbox["width"]:
        return "irregular_square_like"

    return "irregular_shape"


def detect_inside_relationships(objects):
    relations = []

    # Sort parents from smallest to biggest bbox area.
    # This makes each object pick the tightest container first.
    possible_parents = sorted(
        objects,
        key=lambda obj: obj["bbox"]["height"] * obj["bbox"]["width"]
    )

    for child in objects:
        child_bbox = child["bbox"]

        best_parent = None
        best_parent_area = None

        for parent in possible_parents:
            if child["id"] == parent["id"]:
                continue

            parent_bbox = parent["bbox"]

            # Parent must be meaningfully bigger.
            parent_area = parent_bbox["height"] * parent_bbox["width"]
            child_area = child_bbox["height"] * child_bbox["width"]

            if parent_area <= child_area:
                continue

            # For now, use bbox containment.
            # Later we will upgrade this to true enclosed-background-pocket containment.
            if not bbox_strictly_contains(parent_bbox, child_bbox):
                continue

            if best_parent is None or parent_area < best_parent_area:
                best_parent = parent
                best_parent_area = parent_area

        if best_parent is not None:
            relations.append({
                "type": "inside",
                "child": child["id"],
                "parent": best_parent["id"],
                "method": "bbox_containment",
            })

    return relations


def detect_position_relationships(objects):
    relations = []

    for a in objects:
        ar, ac = object_center(a)

        for b in objects:
            if a["id"] == b["id"]:
                continue

            br, bc = object_center(b)

            if ac < bc:
                relations.append({
                    "type": "left_of",
                    "a": a["id"],
                    "b": b["id"],
                })

            if ar < br:
                relations.append({
                    "type": "above",
                    "a": a["id"],
                    "b": b["id"],
                })

    return relations


def enrich_objects(objects):
    enriched = []

    for obj in objects:
        new_obj = dict(obj)
        new_obj["shape_family"] = classify_shape_family(obj)
        new_obj["center"] = object_center(obj)

        # Do not keep raw cells in printed story by default.
        new_obj["cells"] = obj["cells"]

        enriched.append(new_obj)

    return enriched


def build_scene_tree(objects, inside_relations):
    children_by_parent = {}

    all_children = set()

    for relation in inside_relations:
        parent = relation["parent"]
        child = relation["child"]

        children_by_parent.setdefault(parent, []).append(child)
        all_children.add(child)

    root_children = [
        obj["id"]
        for obj in objects
        if obj["id"] not in all_children
    ]

    return {
        "root": root_children,
        "children_by_parent": children_by_parent,
    }


def assign_object_roles(objects, inside_relations):
    for obj in objects:
        obj["roles"] = []

    if not objects:
        return

    largest = max(objects, key=lambda obj: obj["cell_count"])
    largest["roles"].append("largest_object")

    children_by_parent = {}
    parent_by_child = {}

    for rel in inside_relations:
        if rel["type"] != "inside":
            continue

        child_id = rel["child"]
        parent_id = rel["parent"]

        children_by_parent.setdefault(parent_id, []).append(child_id)
        parent_by_child[child_id] = parent_id

    objects_by_id = {
        obj["id"]: obj
        for obj in objects
    }

    for obj in objects:
        obj_id = obj["id"]
        child_ids = children_by_parent.get(obj_id, [])
        parent_id = parent_by_child.get(obj_id)

        if child_ids:
            obj["roles"].append("enclosure_candidate")

        if obj_id == largest["id"] and child_ids:
            obj["roles"].append("outer_enclosure_candidate")

        if parent_id is not None and child_ids:
            obj["roles"].append("inner_enclosure_candidate")

        if parent_id is not None and not child_ids:
            obj["roles"].append("contained_blob_candidate")

        if parent_id is None and obj_id != largest["id"]:
            obj["roles"].append("outside_blob_candidate")

        if obj["touches_border"]:
            obj["roles"].append("touches_border")

        if not obj["roles"]:
            obj["roles"].append("unclassified_visual_object")


def build_structure_signature(story):
    objects = story["objects"]
    tree = story["scene_tree"]

    root_ids = tree["root"]
    children_by_parent = tree["children_by_parent"]

    enclosure_ids = []
    outer_enclosure_ids = []
    inner_enclosure_ids = []
    contained_blob_ids = []
    outside_blob_ids = []

    for obj in objects:
        obj_id = obj["id"]
        roles = obj.get("roles", [])

        if "enclosure_candidate" in roles:
            enclosure_ids.append(obj_id)

        if "outer_enclosure_candidate" in roles:
            outer_enclosure_ids.append(obj_id)

        if "inner_enclosure_candidate" in roles:
            inner_enclosure_ids.append(obj_id)

        if "contained_blob_candidate" in roles:
            contained_blob_ids.append(obj_id)

        if "outside_blob_candidate" in roles:
            outside_blob_ids.append(obj_id)

    def depth(node_id):
        children = children_by_parent.get(node_id, [])

        if not children:
            return 1

        return 1 + max(depth(child_id) for child_id in children)

    if root_ids:
        max_depth = max(depth(root_id) for root_id in root_ids)
    else:
        max_depth = 0

    return {
        "root_object_count": len(root_ids),
        "enclosure_count": len(enclosure_ids),
        "outer_enclosures": outer_enclosure_ids,
        "inner_enclosures": inner_enclosure_ids,
        "contained_blobs": contained_blob_ids,
        "outside_blobs": outside_blob_ids,
        "max_depth": max_depth,
    }


def summarize_story(story):
    lines = []

    lines.append("VISUAL STORY")
    lines.append(f"- grid size: {story['height']}x{story['width']}")
    lines.append(f"- background color: {story['background']}")
    lines.append(f"- object count: {len(story['objects'])}")
    lines.append(f"- background region count: {len(story['background_regions'])}")

    signature = build_structure_signature(story)

    lines.append("")
    lines.append("STRUCTURE SIGNATURE")
    lines.append(f"- root objects: {signature['root_object_count']}")
    lines.append(f"- enclosure count: {signature['enclosure_count']}")
    lines.append(f"- outer enclosures: {signature['outer_enclosures']}")
    lines.append(f"- inner enclosures: {signature['inner_enclosures']}")
    lines.append(f"- contained blobs: {signature['contained_blobs']}")
    lines.append(f"- outside blobs: {signature['outside_blobs']}")
    lines.append(f"- max depth: {signature['max_depth']}")


    lines.append("")
    lines.append("LARGEST OBJECT")

    if story["objects"]:
        largest = max(
            story["objects"],
            key=lambda obj: obj["cell_count"]
        )

        bbox = largest["bbox"]
        lines.append(
            f"- {largest['id']}: "
            f"color={largest['color']} "
            f"shape={largest['shape_family']} "
            f"cells={largest['cell_count']} "
            f"bbox=top{bbox['top']},left{bbox['left']},"
            f"h{bbox['height']},w{bbox['width']}"
        )
    else:
        lines.append("- none")

    lines.append("")
    lines.append("OBJECT ROLES")

    for obj in story["objects"]:
        role_text = ", ".join(obj.get("roles", []))
        lines.append(f"- {obj['id']}: {role_text}")
    lines.append("")
    lines.append("OBJECTS")

    for obj in story["objects"]:
        bbox = obj["bbox"]
        lines.append(
            f"- {obj['id']}: "
            f"color={obj['color']} "
            f"shape={obj['shape_family']} "
            f"cells={obj['cell_count']} "
            f"bbox=top{bbox['top']},left{bbox['left']},"
            f"h{bbox['height']},w{bbox['width']} "
            f"touches_border={obj['touches_border']}"
        )

    lines.append("")
    lines.append("INSIDE RELATIONS")

    inside = [
        rel for rel in story["relations"]
        if rel["type"] == "inside"
    ]

    if not inside:
        lines.append("- none")
    else:
        for rel in inside:
            method = rel.get("method", "unknown")
            lines.append(
                f"- {rel['child']} inside {rel['parent']} "
                f"method={method}"
            )

    lines.append("")
    lines.append("CONTAINMENT CHAINS")
    chains = format_containment_chains(story)

    if not chains:
        lines.append("- none")
    else:
        for chain in chains:
            lines.append(f"- {chain}")

    lines.append("")
    lines.append("SCENE TREE")
    lines.extend(format_scene_tree(story))

    return "\n".join(lines)


def format_containment_chains(story):
    tree = story["scene_tree"]
    root_ids = tree["root"]
    children_by_parent = tree["children_by_parent"]

    chains = []

    def walk(node_id, path):
        children = children_by_parent.get(node_id, [])

        if not children:
            chains.append(" > ".join(path))
            return

        for child_id in children:
            walk(child_id, path + [child_id])

    for root_id in root_ids:
        walk(root_id, [root_id])

    return chains


def format_scene_tree(story):
    tree = story["scene_tree"]
    children_by_parent = tree["children_by_parent"]

    lines = ["ROOT"]

    def add_node(node_id, indent):
        lines.append(" " * indent + f"└── {node_id}")

        for child_id in children_by_parent.get(node_id, []):
            add_node(child_id, indent + 4)

    for root_id in tree["root"]:
        add_node(root_id, 4)

    return lines


def build_visual_story(grid, forced_background=None):
    h, w = grid_shape(grid)

    if forced_background is None:
        background = detect_background_color(grid)
    else:
        background = forced_background

    objects = extract_connected_components(grid, background)
    objects = enrich_objects(objects)

    background_regions = extract_background_regions(grid, background)

    inside_relations = detect_inside_relationships(objects)
    assign_object_roles(objects, inside_relations)
    position_relations = detect_position_relationships(objects)

    relations = inside_relations + position_relations

    scene_tree = build_scene_tree(objects, inside_relations)

    story = {
        "height": h,
        "width": w,
        "background": background,
        "objects": objects,
        "background_regions": background_regions,
        "relations": relations,
        "scene_tree": scene_tree,
    }

    story["summary"] = summarize_story(story)

    return story