# ============================================================
# BASIC HELPERS
# ============================================================

def grid_shape(grid):
    h = len(grid) if grid is not None else 0
    w = len(grid[0]) if h else 0
    return h, w


# ============================================================
# OBJECT DEBUG (CLEANED)
# ============================================================

def debug_selected_object(label, obj):
    print(f"\n[{label}]")

    if obj is None:
        print("  None")
        return

    bbox = obj.get("bbox")
    area = obj.get("area")
    colors = obj.get("color_count")
    h = obj.get("height")
    w = obj.get("width")

    print(f"  bbox   : {bbox}")
    print(f"  area   : {area}")
    print(f"  colors : {colors}")
    print(f"  shape  : {h}x{w}")


def debug_object_list(label, objects, max_show=5):
    """
    Only show top few objects to avoid spam.
    """
    print(f"\n[{label}] (showing top {max_show})")

    if not objects:
        print("  None")
        return

    for i, obj in enumerate(objects[:max_show]):
        bbox = obj.get("bbox")
        area = obj.get("area")
        colors = obj.get("color_count")
        h = obj.get("height")
        w = obj.get("width")

        print(
            f"  {i+1}. bbox={bbox} area={area} colors={colors} shape={h}x{w}"
        )

    if len(objects) > max_show:
        print(f"  ... ({len(objects) - max_show} more not shown)")


# ============================================================
# STRATEGY SCORE DEBUG (CLEAN TABLE)
# ============================================================

def debug_strategy_scores(
    result_fc7=None,
    result_v2=None,
    result_object_correspondence=None,
    result_object_projection=None,
    result_multi_object_projection=None,
    result_pattern=None,
    result_region=None,
    result_partition=None,
    result_region_alignment=None,
    result_mirror_repair=None,
    result_region_alignment_v2=None,
):
    print("\n=== STRATEGY SCORES ===")

    def row(name, r):
        if r is None:
            print(f"{name:<28} : None")
            return

        score = r.get("score")
        print(f"{name:<28} : {score}")

    row("object_fc7", result_fc7)
    row("object_v2", result_v2)
    row("object_correspondence", result_object_correspondence)
    row("object_projection", result_object_projection)
    row("multi_object_projection", result_multi_object_projection)
    row("pattern", result_pattern)
    row("region", result_region)
    row("partition", result_partition)
    row("region_alignment", result_region_alignment)
    row("mirror_repair", result_mirror_repair)
    row("region_alignment_v2", result_region_alignment_v2)

def print_grid(grid, title="GRID"):
    print(f"\n{title}")
    if grid is None:
        print("None")
        return

    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    print(f"(h={h}, w={w})")

    for row in grid:
        print(" ".join(str(x) for x in row))


def show_three_grids(inp, pred, exp):
    print_grid(inp, "INPUT")
    print_grid(pred, "PREDICTED")
    print_grid(exp, "EXPECTED")

# ============================================================
# ROUTER DEBUG (CLEAN + NON-RECURSIVE)
# ============================================================

def debug_router_adjustments(
    result_fc7=None,
    result_v2=None,
    result_object_correspondence=None,
    result_object_projection=None,
    result_multi_object_projection=None,
    result_pattern=None,
    result_region=None,
    result_partition=None,
    result_region_alignment=None,
    result_mirror_repair=None,
    result_region_alignment_v2=None,
):
    print("\n=== ROUTER DECISION TABLE ===")

    def show(name, r):
        if r is None:
            print(f"{name:<28} : None")
            return

        raw = r.get("raw_score", r.get("score"))
        adj = r.get("adjusted_score")
        sp = r.get("shape_penalty")
        fg = r.get("full_grid_penalty")

        pred = r.get("predicted")
        if pred is None:
            shape = None
        else:
            h, w = grid_shape(pred)
            shape = f"{h}x{w}"

        print(
            f"{name:<28} "
            f"raw={raw:<4} "
            f"adj={adj:<4} "
            f"shape_pen={sp:<4} "
            f"full_pen={fg:<4} "
            f"out={shape}"
        )

    show("object_fc7", result_fc7)
    show("object_v2", result_v2)
    show("object_correspondence", result_object_correspondence)
    show("object_projection", result_object_projection)
    show("multi_object_projection", result_multi_object_projection)
    show("pattern", result_pattern)
    show("region", result_region)
    show("partition", result_partition)
    show("region_alignment", result_region_alignment)
    show("mirror_repair", result_mirror_repair)
    show("region_alignment_v2", result_region_alignment_v2)


# ============================================================
# FINAL RESULT DEBUG
# ============================================================

def debug_final_choice(result):
    print("\n=== FINAL CHOICE ===")

    if result is None:
        print("No result selected")
        return

    print(f"Strategy : {result.get('strategy')}")
    print(f"Score    : {result.get('score')}")
    print(f"Exact    : {result.get('exact')}")

    if result.get("predicted") is not None:
        h, w = grid_shape(result["predicted"])
        print(f"Output   : {h}x{w}")