def select_largest_object(objects):
    if not objects:
        return None
    return max(objects, key=lambda o: o["area"])


def select_smallest_object(objects):
    if not objects:
        return None
    return min(objects, key=lambda o: o["area"])


def select_most_rectangular_object(objects):
    if not objects:
        return None
    return max(objects, key=lambda o: (o["height"] * o["width"]) - o["area"])


def select_most_dense_object(objects):
    if not objects:
        return None
    return max(objects, key=lambda o: o["area"] / (o["height"] * o["width"]))


def select_center_object(objects):
    if not objects:
        return None

    def center_distance(o):
        r = (o["bbox"]["min_r"] + o["bbox"]["max_r"]) / 2
        c = (o["bbox"]["min_c"] + o["bbox"]["max_c"]) / 2
        return abs(r - 11) + abs(c - 11)

    return min(objects, key=center_distance)


def select_most_multicolor_object(objects):
    if not objects:
        return None
    return max(objects, key=lambda o: (o["color_count"], o["area"]))


def touches_grid_border(obj, grid_h=30, grid_w=30):
    bbox = obj["bbox"]
    return (
        bbox["min_r"] == 0 or
        bbox["min_c"] == 0 or
        bbox["max_r"] == grid_h - 1 or
        bbox["max_c"] == grid_w - 1
    )


def select_most_multicolor_nonframe_object(objects, grid_h, grid_w):
    if not objects:
        return None

    nonframe = [
        o for o in objects
        if not (
            o["bbox"]["min_r"] == 0 and
            o["bbox"]["min_c"] == 0 and
            o["bbox"]["max_r"] == grid_h - 1 and
            o["bbox"]["max_c"] == grid_w - 1
        )
    ]

    pool = nonframe if nonframe else objects
    return max(pool, key=lambda o: (o["color_count"], o["area"]))