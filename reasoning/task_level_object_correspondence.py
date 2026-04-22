from vision.object_finder import find_objects
from core.transforms import apply_transform


TRANSFORMS = [
    "identity",
    "rotate_90",
    "rotate_180",
    "rotate_270",
    "flip_horizontal",
    "flip_vertical",
]

SELECTORS = [
    "largest_object",
    "smallest_object",
    "most_colors",
    "least_colors",
]


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def exact_match(a, b):
    if len(a) != len(b):
        return False
    if not a and not b:
        return True
    if not a or not b:
        return False
    if len(a[0]) != len(b[0]):
        return False

    for r in range(len(a)):
        for c in range(len(a[0])):
            if a[r][c] != b[r][c]:
                return False

    return True


def normalize(grid):
    return [[1 if cell != 0 else 0 for cell in row] for row in grid]


def same_shape(a, b):
    return exact_match(normalize(a), normalize(b))


def infer_recolor(src, dst):
    mapping = {}

    for r in range(len(src)):
        for c in range(len(src[0])):
            s = src[r][c]
            d = dst[r][c]

            if s == 0 and d != 0:
                return None

            if s != 0:
                if s in mapping and mapping[s] != d:
                    return None
                mapping[s] = d

    return mapping


def apply_recolor(grid, mapping):
    if mapping is None:
        return grid

    out = []
    for row in grid:
        new_row = []
        for cell in row:
            if cell == 0:
                new_row.append(0)
            else:
                new_row.append(mapping.get(cell, cell))
        out.append(new_row)

    return out


def select_object(objects, selector):
    if not objects:
        return None

    if selector == "largest_object":
        return max(objects, key=lambda o: o["area"])

    if selector == "smallest_object":
        return min(objects, key=lambda o: o["area"])

    if selector == "most_colors":
        return max(objects, key=lambda o: o["color_count"])

    if selector == "least_colors":
        return min(objects, key=lambda o: o["color_count"])

    return None


# ------------------------------------------------------------
# MAIN DISCOVERY
# ------------------------------------------------------------

def discover_task_level_rule(train_pairs, debug=False):
    if not train_pairs:
        return None

    for selector in SELECTORS:
        for transform in TRANSFORMS:

            learned_recolor = None
            all_pass = True

            if debug:
                print("\nTrying:", selector, transform)

            for i, pair in enumerate(train_pairs):

                inp = pair["input"]
                out = pair["output"]

                objects = find_objects(inp)
                obj = select_object(objects, selector)

                if obj is None:
                    all_pass = False
                    break

                patch = obj["patch"]
                transformed = apply_transform(patch, transform)

                # Exact match
                if exact_match(transformed, out):
                    continue

                # Shape match + recolor
                if not same_shape(transformed, out):
                    all_pass = False
                    break

                recolor = infer_recolor(transformed, out)
                if recolor is None:
                    all_pass = False
                    break

                recolored = apply_recolor(transformed, recolor)

                if not exact_match(recolored, out):
                    all_pass = False
                    break

                if learned_recolor is None:
                    learned_recolor = recolor
                else:
                    if learned_recolor != recolor:
                        all_pass = False
                        break

            if all_pass:
                rule = {
                    "selector": selector,
                    "transform": transform,
                    "recolor_map": learned_recolor
                }

                if debug:
                    print("\nFOUND TASK RULE:", rule)

                return rule

    return None


# ------------------------------------------------------------
# APPLY TO TEST INPUT
# ------------------------------------------------------------

def apply_task_rule(test_input, rule):
    objects = find_objects(test_input)
    obj = select_object(objects, rule["selector"])

    if obj is None:
        return None

    patch = obj["patch"]
    transformed = apply_transform(patch, rule["transform"])
    result = apply_recolor(transformed, rule["recolor_map"])

    return result