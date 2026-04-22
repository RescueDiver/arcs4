from collections import Counter
from vision.global_pattern_reader import (
    read_global_pattern,
    grid_h,
    grid_w,
    crop,
)


def rect_area(rect):
    return rect["height"] * rect["width"]


def rect_center_distance(rect, h, w):
    rect_center_r = (rect["top"] + rect["bottom"]) / 2.0
    rect_center_c = (rect["left"] + rect["right"]) / 2.0
    grid_center_r = (h - 1) / 2.0
    grid_center_c = (w - 1) / 2.0
    return abs(rect_center_r - grid_center_r) + abs(rect_center_c - grid_center_c)


def border_touch_count(rect, h, w):
    count = 0
    if rect["top"] == 0:
        count += 1
    if rect["left"] == 0:
        count += 1
    if rect["bottom"] == h - 1:
        count += 1
    if rect["right"] == w - 1:
        count += 1
    return count


def repeated_neighbor_score(grid, rect):
    """
    Checks whether rows/cols just outside the rectangle look more patterned
    than the rectangle itself. Higher means 'this block interrupts structure'.
    """
    h = grid_h(grid)
    w = grid_w(grid)

    top = rect["top"]
    left = rect["left"]
    bottom = rect["bottom"]
    right = rect["right"]

    score = 0

    # Compare row above and below if both exist
    if top - 1 >= 0 and bottom + 1 < h:
        above = grid[top - 1][left:right + 1]
        below = grid[bottom + 1][left:right + 1]
        score += sum(1 for a, b in zip(above, below) if a == b)

    # Compare col left and right if both exist
    if left - 1 >= 0 and right + 1 < w:
        col_left = [grid[r][left - 1] for r in range(top, bottom + 1)]
        col_right = [grid[r][right + 1] for r in range(top, bottom + 1)]
        score += sum(1 for a, b in zip(col_left, col_right) if a == b)

    return score


def uniformity_score(grid, rect):
    patch = crop(grid, rect["top"], rect["left"], rect["bottom"], rect["right"])
    vals = [v for row in patch for v in row]
    if not vals:
        return 0.0
    ctr = Counter(vals)
    return max(ctr.values()) / len(vals)


def suspicious_rect_score(grid, rect, summary):
    """
    Higher = more likely to be a broken/overwritten region.
    """
    h = summary["height"]
    w = summary["width"]

    score = 0.0

    area = rect_area(rect)
    uniform = uniformity_score(grid, rect)
    center_dist = rect_center_distance(rect, h, w)
    border_touches = border_touch_count(rect, h, w)
    neighbor_repeat = repeated_neighbor_score(grid, rect)

    # Prefer meaningful, not tiny
    score += area * 2.0

    # Prefer very uniform blocks as corruption candidates
    score += uniform * 20.0

    # Prefer blocks near center
    score -= center_dist * 0.75

    # Prefer internal blocks, not edge wrappers
    score -= border_touches * 3.0

    # Prefer regions that interrupt surrounding repetition
    score += neighbor_repeat * 3.0

    # Slight boost if block color is not dominant grid color
    dom = summary["dominant_color"]
    if rect["color"] != dom:
        score += 2.0

    return score


def find_suspect_regions(grid, top_k=5):
    """
    Human-style guess:
    what area looks 'wrong', overwritten, or out of place?
    """
    summary = read_global_pattern(grid)
    rects = summary["uniform_rectangles"]

    ranked = []
    for rect in rects:
        score = suspicious_rect_score(grid, rect, summary)
        ranked.append({
            "top": rect["top"],
            "left": rect["left"],
            "bottom": rect["bottom"],
            "right": rect["right"],
            "height": rect["height"],
            "width": rect["width"],
            "color": rect["color"],
            "area": rect["area"],
            "score": score,
            "reason": "uniform_block_breaks_pattern",
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]


def detect_pattern_break(grid, top_k=5):
    """
    Main entry point for other engines.
    """
    summary = read_global_pattern(grid)
    suspects = find_suspect_regions(grid, top_k=top_k)

    return {
        "summary": summary,
        "suspect_regions": suspects,
    }


def print_pattern_break_report(report):
    print("PATTERN BREAK REPORT:")

    summary = report["summary"]
    print(f"  grid_size              : {summary['height']}x{summary['width']}")
    print(f"  dominant_color         : {summary['dominant_color']}")
    print(f"  nonzero_bbox           : {summary['nonzero_bbox']}")
    print(f"  horizontal_symmetry    : {summary['horizontal_symmetry_score']:.3f}")
    print(f"  vertical_symmetry      : {summary['vertical_symmetry_score']:.3f}")

    print("  suspect_regions:")
    if not report["suspect_regions"]:
        print("    None")
        return

    for i, rect in enumerate(report["suspect_regions"], start=1):
        print(
            f"    {i}. "
            f"color={rect['color']} "
            f"top={rect['top']} left={rect['left']} "
            f"bottom={rect['bottom']} right={rect['right']} "
            f"shape={rect['height']}x{rect['width']} "
            f"score={rect['score']:.2f} "
            f"reason={rect['reason']}"
        )