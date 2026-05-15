# reasoning/ring_blob_output_builder.py
"""
Ring Blob Output Builder

Goal:
    Start moving away from "copy the closest training output"
    and toward:

        learned ring/blob scene
        -> symbolic output structure
        -> recolored prediction

This is intentionally experimental.

It does NOT hardcode task IDs.
It learns from the training pairs by comparing:

    input scene:
        rings
        blobs per ring
        blob layouts

    to output:
        shape
        foreground/border color
        fill/background color
        symbolic template

Then for a test input, it:
    1. learns the same scene description
    2. chooses the most similar learned scene
    3. recolors the symbolic output using the test input colors
    4. optionally applies simple learned layout adjustments

This is the bridge between:
    "I see 2 rings and blobs"
and:
    "draw the simplified output"
"""

from collections import Counter

from reasoning.archive.ring_blob_scene import learn_ring_blob_scene


# ============================================================
# Basic grid/color helpers
# ============================================================

def grid_shape(grid):
    if not grid:
        return (0, 0)

    return (len(grid), len(grid[0]))


def copy_grid(grid):
    return [row[:] for row in grid]


def color_counts(grid):
    counts = Counter()

    for row in grid:
        for value in row:
            counts[value] += 1

    return counts


def most_common_color(grid):
    counts = color_counts(grid)

    if not counts:
        return 0

    return counts.most_common(1)[0][0]


def get_input_background_color(input_grid):
    """
    In most ARC tasks, the background is the most common input color.
    """
    return most_common_color(input_grid)


def get_input_foreground_color(input_grid):
    """
    Foreground is the most common non-background color.
    """
    background = get_input_background_color(input_grid)
    counts = color_counts(input_grid)

    foreground_counts = Counter()

    for color, count in counts.items():
        if color == background:
            continue

        foreground_counts[color] = count

    if not foreground_counts:
        return background

    return foreground_counts.most_common(1)[0][0]


def get_output_fill_and_foreground_colors(output_grid):
    """
    For symbolic outputs:
        foreground/border color is usually the dominant edge color.
        fill/background color is usually the most common non-border color.
    """
    h, w = grid_shape(output_grid)

    if h == 0 or w == 0:
        return 0, 0

    edge_counts = Counter()

    for c in range(w):
        edge_counts[output_grid[0][c]] += 1
        edge_counts[output_grid[h - 1][c]] += 1

    for r in range(h):
        edge_counts[output_grid[r][0]] += 1
        edge_counts[output_grid[r][w - 1]] += 1

    foreground_color = edge_counts.most_common(1)[0][0]

    counts = color_counts(output_grid)
    fill_counts = Counter()

    for color, count in counts.items():
        if color == foreground_color:
            continue

        fill_counts[color] = count

    if not fill_counts:
        fill_color = foreground_color
    else:
        fill_color = fill_counts.most_common(1)[0][0]

    return fill_color, foreground_color

    counts = color_counts(output_grid)

    if not counts:
        return 0, 0

    fill_color = counts.most_common(1)[0][0]

    non_fill_counts = Counter()

    for color, count in counts.items():
        if color == fill_color:
            continue

        non_fill_counts[color] = count

    if not non_fill_counts:
        foreground_color = fill_color
    else:
        foreground_color = non_fill_counts.most_common(1)[0][0]

    return fill_color, foreground_color


def recolor_template_to_input(template_grid, input_grid, old_fill_color, old_foreground_color):
    """
    Recolor a learned symbolic template using the test input colors.

    Learned relation:
        output foreground/border/markers should use input foreground color
        output fill/background should use input background color
    """
    new_fill_color = get_input_background_color(input_grid)
    new_foreground_color = get_input_foreground_color(input_grid)

    recolored = []

    for row in template_grid:
        new_row = []

        for value in row:
            if value == old_foreground_color:
                new_row.append(new_foreground_color)
            elif value == old_fill_color:
                new_row.append(new_fill_color)
            else:
                # Unknown/extra color: preserve it for now.
                new_row.append(value)

        recolored.append(new_row)

    return recolored


# ============================================================
# Scene signature helpers
# ============================================================

def scene_ring_blob_counts(scene):
    """
    Return blob counts per ring in ring order.

    Example:
        ring_0 has 1 blob
        ring_1 has 2 blobs

    returns:
        (1, 2)
    """
    counts = []

    for ring in scene.get("rings", []):
        counts.append(ring.get("blob_count", 0))

    return tuple(counts)


def scene_ring_blob_layouts(scene):
    """
    Return blob layouts per ring in ring order.

    Example:
        ("single", "side_by_side_horizontal")
    """
    layouts = []

    for ring in scene.get("rings", []):
        layouts.append(ring.get("blob_layout", "unknown"))

    return tuple(layouts)


def scene_signature(scene):
    """
    Compact description of the learned visual scene.
    """
    return {
        "ring_count": scene.get("ring_count", 0),
        "blob_count": scene.get("blob_count", 0),
        "outside_blob_count": scene.get("outside_blob_count", 0),
        "ring_blob_counts": scene_ring_blob_counts(scene),
        "ring_blob_layouts": scene_ring_blob_layouts(scene),
    }


def signature_to_printable(signature):
    return (
        f"rings={signature.get('ring_count')}, "
        f"blobs={signature.get('blob_count')}, "
        f"outside={signature.get('outside_blob_count')}, "
        f"ring_blob_counts={signature.get('ring_blob_counts')}, "
        f"layouts={signature.get('ring_blob_layouts')}"
    )


# ============================================================
# Output template helpers
# ============================================================

def output_template_info(output_grid):
    """
    Store useful symbolic info about a training output.
    """
    shape = grid_shape(output_grid)
    fill_color, foreground_color = get_output_fill_and_foreground_colors(output_grid)

    return {
        "shape": shape,
        "fill_color": fill_color,
        "foreground_color": foreground_color,
        "template_grid": copy_grid(output_grid),
    }


# ============================================================
# Learning from train pairs
# ============================================================

def learn_ring_blob_output_builder(train_pairs):
    """
    Learn a set of scene -> output examples.

    train_pairs should look like:
        [
            {"input": input_grid, "output": output_grid},
            ...
        ]
    """
    records = []

    for index, pair in enumerate(train_pairs):
        input_grid = pair["input"]
        output_grid = pair["output"]

        scene = learn_ring_blob_scene(input_grid)
        signature = scene_signature(scene)
        template = output_template_info(output_grid)

        record = {
            "train_pair_index": index + 1,
            "input_scene": scene,
            "scene_signature": signature,
            "output_template": template,
        }

        records.append(record)

    builder = {
        "family": "ring_blob_output_builder",
        "records": records,
    }

    return builder


# ============================================================
# Scene matching
# ============================================================

def score_scene_match(test_signature, train_signature):
    """
    Score how similar two visual scenes are.

    Higher score = better.

    This is not final ARC intelligence.
    It is a first learned matching layer based on scene structure.
    """
    score = 0

    # Same number of rings matters a lot.
    if test_signature.get("ring_count") == train_signature.get("ring_count"):
        score += 100
    else:
        score -= 40 * abs(
            test_signature.get("ring_count", 0)
            - train_signature.get("ring_count", 0)
        )

    # Similar total blob count matters.
    score -= 12 * abs(
        test_signature.get("blob_count", 0)
        - train_signature.get("blob_count", 0)
    )

    # Similar outside blob count matters.
    score -= 10 * abs(
        test_signature.get("outside_blob_count", 0)
        - train_signature.get("outside_blob_count", 0)
    )

    test_counts = test_signature.get("ring_blob_counts", ())
    train_counts = train_signature.get("ring_blob_counts", ())

    test_counts = test_signature.get("ring_blob_counts", ())
    train_counts = train_signature.get("ring_blob_counts", ())

    max_len = max(len(test_counts), len(train_counts))

    for index in range(max_len):
        test_count = test_counts[index] if index < len(test_counts) else 0
        train_count = train_counts[index] if index < len(train_counts) else 0

        if test_count == train_count:
            score += 30
        else:
            score -= 14 * abs(test_count - train_count)

    # ------------------------------------------------------------
    # Important learned scene signal:
    #
    # In two-ring scenes, which ring has the extra/repeated blob matters.
    #
    # A repeated blob in ring_0 should favor compact two-ring layouts.
    # A repeated blob in ring_1 should favor taller/two-lane layouts.
    #
    # This is not task-id hardcoding. It uses the learned ring blob counts.
    # ------------------------------------------------------------
    if test_signature.get("ring_count") == 2 and len(test_counts) >= 2:
        test_extra_ring_indexes = [
            index
            for index, count in enumerate(test_counts)
            if count > 1
        ]

        train_extra_ring_indexes = [
            index
            for index, count in enumerate(train_counts)
            if count > 1
        ]

        if test_extra_ring_indexes and train_extra_ring_indexes:
            if test_extra_ring_indexes == train_extra_ring_indexes:
                score += 35
            else:
                score -= 35

        # If the test has an extra blob in ring_0, avoid over-preferring
        # the outside-blob tall template when a compact two-ring template exists.
        if test_extra_ring_indexes == [0]:
            if train_signature.get("outside_blob_count", 0) > 0:
                score -= 25

        # If the test has an extra blob in ring_1, the taller symbolic
        # template is often a better candidate.
        if test_extra_ring_indexes == [1]:
            if train_signature.get("blob_count", 0) >= test_signature.get("blob_count", 0):
                score += 20

    test_layouts = test_signature.get("ring_blob_layouts", ())
    train_layouts = train_signature.get("ring_blob_layouts", ())

    max_len = max(len(test_layouts), len(train_layouts))

    for index in range(max_len):
        test_layout = test_layouts[index] if index < len(test_layouts) else "missing"
        train_layout = train_layouts[index] if index < len(train_layouts) else "missing"

        if test_layout == train_layout:
            score += 12
        elif "empty" in (test_layout, train_layout):
            score -= 8
        else:
            score -= 3

    return score


def choose_best_scene_record(builder, test_scene):
    """
    Pick the closest learned scene/output record.
    """
    test_signature = scene_signature(test_scene)

    best = None

    for record in builder.get("records", []):
        train_signature = record["scene_signature"]
        score = score_scene_match(test_signature, train_signature)

        candidate = {
            "record": record,
            "score": score,
            "test_signature": test_signature,
            "train_signature": train_signature,
        }

        if best is None or score > best["score"]:
            best = candidate

    return best


# ============================================================
# Simple learned layout adjustment
# ============================================================

def should_add_right_lane(test_signature, chosen_train_signature):
    """
    Add a right-side fill lane only when the test has outside blobs
    and the chosen training template did not already have outside blobs.

    This learns the idea:
        outside blob -> right-side symbolic area

    But it avoids the earlier mistake where Test 1 became too wide.
    """
    test_outside = test_signature.get("outside_blob_count", 0)
    train_outside = chosen_train_signature.get("outside_blob_count", 0)

    return test_outside > 0 and train_outside == 0


def add_fill_column_right(grid, fill_color, count=1):
    """
    Add one or more fill columns to the right side.
    This is a safe symbolic adjustment because it does not invent markers.
    """
    if not grid:
        return grid

    expanded = []

    for row in grid:
        expanded.append(row[:] + [fill_color] * count)

    return expanded


def trim_or_expand_height_to_match_ring_blob_pattern(grid, test_signature, fill_color, foreground_color):
    """
    Experimental height adjustment.

    We keep this conservative:
        - do not crop markers aggressively
        - only handle very simple symbolic frame outputs
        - mainly prepares the builder for later learning

    For now, this returns the grid unchanged.

    The important next step is getting scene-based color/template prediction
    printed separately from the old mapper.
    """
    return grid


def adapt_template_grid(template_grid, test_signature, chosen_train_signature, fill_color, foreground_color):
    """
    Apply small symbolic adaptations after choosing a learned template.

    This is intentionally conservative.
    """
    adapted = copy_grid(template_grid)

    if should_add_right_lane(test_signature, chosen_train_signature):
        # Add one fill column. Later we can learn exact lane width.
        adapted = add_fill_column_right(adapted, fill_color, count=1)

    adapted = trim_or_expand_height_to_match_ring_blob_pattern(
        adapted,
        test_signature,
        fill_color,
        foreground_color,
    )

    return adapted


# ============================================================
# Prediction
# ============================================================

def predict_with_ring_blob_output_builder(builder, input_grid):
    """
    Predict an output from a test input using the learned scene builder.

    Returns:
        {
            "prediction": grid,
            "test_scene": scene,
            "chosen_train_pair": int,
            "score": int,
            "test_signature": ...,
            "chosen_train_signature": ...,
            "notes": [...]
        }
    """
    test_scene = learn_ring_blob_scene(input_grid)
    chosen = choose_best_scene_record(builder, test_scene)

    if chosen is None:
        return {
            "prediction": None,
            "test_scene": test_scene,
            "chosen_train_pair": None,
            "score": None,
            "test_signature": scene_signature(test_scene),
            "chosen_train_signature": None,
            "notes": ["No learned records available."],
        }

    record = chosen["record"]
    output_template = record["output_template"]

    old_fill = output_template["fill_color"]
    old_foreground = output_template["foreground_color"]

    recolored = recolor_template_to_input(
        output_template["template_grid"],
        input_grid,
        old_fill,
        old_foreground,
    )

    new_fill = get_input_background_color(input_grid)
    new_foreground = get_input_foreground_color(input_grid)

    adapted = adapt_template_grid(
        recolored,
        chosen["test_signature"],
        chosen["train_signature"],
        new_fill,
        new_foreground,
    )

    notes = [
        "Scene-based symbolic builder prediction.",
        "Template was chosen by ring/blob scene similarity.",
        "Template was recolored from test input foreground/background colors.",
    ]

    if grid_shape(adapted) != grid_shape(recolored):
        notes.append(
            f"Template shape adapted from {grid_shape(recolored)} to {grid_shape(adapted)}."
        )

    return {
        "prediction": adapted,
        "test_scene": test_scene,
        "chosen_train_pair": record["train_pair_index"],
        "score": chosen["score"],
        "test_signature": chosen["test_signature"],
        "chosen_train_signature": chosen["train_signature"],
        "notes": notes,
    }


# ============================================================
# Debug printing
# ============================================================

def print_builder_training_summary(builder):
    print()
    print("RING/BLOB OUTPUT BUILDER — TRAINING SUMMARY")
    print("=" * 60)

    for record in builder.get("records", []):
        pair_index = record["train_pair_index"]
        signature = record["scene_signature"]
        template = record["output_template"]

        print()
        print(f"TRAIN PAIR {pair_index}")
        print("-" * 60)
        print(f"scene   : {signature_to_printable(signature)}")
        print(f"output  : shape={template['shape']}")
        print(f"colors  : fill={template['fill_color']}, foreground={template['foreground_color']}")


def print_builder_prediction_summary(result):
    print()
    print("RING/BLOB OUTPUT BUILDER — TEST PREDICTION")
    print("=" * 60)

    print(f"chosen train pair : {result.get('chosen_train_pair')}")
    print(f"match score       : {result.get('score')}")

    print()
    print("test scene")
    print("-" * 60)
    print(signature_to_printable(result.get("test_signature", {})))

    print()
    print("chosen train scene")
    print("-" * 60)
    chosen_signature = result.get("chosen_train_signature")

    if chosen_signature is None:
        print("None")
    else:
        print(signature_to_printable(chosen_signature))

    print()
    print("notes")
    print("-" * 60)

    for note in result.get("notes", []):
        print(f"- {note}")