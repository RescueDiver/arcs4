import numpy as np


class MotifEngine:
    """
    Motif-based solver for tasks like 136b0064.

    Main idea:
    1. Find the vertical divider color (usually 4).
    2. Split the grid into left and right content areas.
    3. Extract source blocks from each side, top-to-bottom.
    4. Convert each block color into a canonical output motif.
    5. Place motifs into a fixed-width output canvas using a simple policy.
    """

    def __init__(self):
        # Canonical motifs by color.
        self.motif_map = {
            1: [(0, 0), (0, 1), (0, 2)],              # horizontal length 3
            2: [(0, 0), (0, 1)],                      # horizontal length 2
            3: [(0, 0), (0, 1), (0, 2), (0, 3)],      # horizontal length 4
            6: [(0, 0), (1, 0)],                      # vertical length 2
            5: [(0, 0)],                              # single marker
        }

    # ============================================================
    # BASIC HELPERS
    # ============================================================

    def to_array(self, grid):
        if isinstance(grid, np.ndarray):
            return grid.copy()
        return np.array(grid, dtype=int)

    def to_list(self, grid):
        if isinstance(grid, np.ndarray):
            return grid.tolist()
        return grid

    def blank_grid(self, h, w, fill=0):
        return np.full((h, w), fill, dtype=int)

    # ============================================================
    # DIVIDER / SPLIT LOGIC
    # ============================================================

    def find_divider_column(self, grid, divider_color=4):
        h, w = grid.shape

        best_col = None
        best_score = -1

        for c in range(w):
            col = grid[:, c]
            score = int(np.sum(col == divider_color))
            nonzero_nondivider = int(np.sum((col != 0) & (col != divider_color)))
            adjusted = score - nonzero_nondivider

            if adjusted > best_score:
                best_score = adjusted
                best_col = c

        if best_col is None:
            return None

        if np.sum(grid[:, best_col] == divider_color) < max(2, h // 2):
            return None

        return best_col

    def split_sides(self, input_grid, divider_color=4):
        grid = self.to_array(input_grid)
        divider_col = self.find_divider_column(grid, divider_color=divider_color)

        if divider_col is None:
            mid = grid.shape[1] // 2
            return grid[:, :mid], None, grid[:, mid:]

        left = grid[:, :divider_col]
        right = grid[:, divider_col + 1:]
        return left, divider_col, right

    # ============================================================
    # BLOCK EXTRACTION
    # ============================================================

    def get_blocks(self, side_grid):
        grid = self.to_array(side_grid)
        h, w = grid.shape

        blocks = []
        r = 0

        while r < h:
            chunk = grid[r:r + 3, :]
            coords = np.argwhere(chunk > 0)

            if coords.size > 0:
                nonzero_vals = chunk[chunk > 0]
                colors, counts = np.unique(nonzero_vals, return_counts=True)
                color = int(colors[np.argmax(counts)])
                local_x = int(np.min(coords[:, 1]))

                blocks.append({
                    "color": color,
                    "colors_present": [int(c) for c in colors.tolist()],
                    "local_x": local_x,
                    "min_r": r,
                    "patch": chunk.copy(),
                })

            r += 4

        return blocks

    def classify_blocks(self, blocks):
        groups = {
            "top": [],
            "middle": [],
            "vertical": [],
            "bottom": [],
        }

        for b in blocks:
            colors_present = b.get("colors_present", [b["color"]])

            if 2 in colors_present:
                groups["top"].append({"color": 2, "local_x": b["local_x"], "min_r": b["min_r"], "patch": b["patch"]})

            if 1 in colors_present:
                groups["middle"].append({"color": 1, "local_x": b["local_x"], "min_r": b["min_r"], "patch": b["patch"]})

            if 6 in colors_present:
                groups["vertical"].append({"color": 6, "local_x": b["local_x"], "min_r": b["min_r"], "patch": b["patch"]})

            if 3 in colors_present:
                groups["bottom"].append({"color": 3, "local_x": b["local_x"], "min_r": b["min_r"], "patch": b["patch"]})

        return groups

    # ============================================================
    # MOTIF LOGIC
    # ============================================================

    def get_motif_cells(self, color):
        return self.motif_map.get(color, [(0, 0)])

    def motif_size(self, color):
        cells = self.get_motif_cells(color)
        max_r = max(r for r, c in cells)
        max_c = max(c for r, c in cells)
        return max_r + 1, max_c + 1

    def choose_anchor_x(self, color, local_x, output_width, side):
        """
        Placement based mainly on side and motif role.
        Tuned to match the current task family.
        """
        motif_h, motif_w = self.motif_size(color)

        if color == 5:
            base_x = 1

        elif side == "left":
            # Left side semantic placement.
            if color == 2:
                base_x = 0
            elif color == 1:
                base_x = 0
            elif color == 6:
                base_x = 4
            elif color == 3:
                base_x = 1
            else:
                base_x = 0

        elif side == "right":
            # Right side semantic placement.
            if color == 1:
                base_x = 2
            elif color == 2:
                base_x = 4
            elif color == 6:
                base_x = 5
            elif color == 3:
                base_x = 1
            else:
                base_x = output_width // 2

        else:
            base_x = 0

        max_x = max(0, output_width - motif_w)
        return max(0, min(base_x, max_x))

    def draw_motif(self, canvas, color, start_y, start_x):
        cells = self.get_motif_cells(color)
        h, w = canvas.shape

        for dy, dx in cells:
            y = start_y + dy
            x = start_x + dx
            if 0 <= y < h and 0 <= x < w:
                canvas[y, x] = color

    # ============================================================
    # CURSOR / PLACEMENT
    # ============================================================

    def advance_cursor_y(self, color, side="left"):
        """
        Vertical spacing rule after placing a motif.
        """
        if color == 6 and side == "left":
            # In this task family, left-side 6 often acts like a taller vertical run.
            return 4
        motif_h, _ = self.motif_size(color)
        return motif_h

    def place_blocks_sequence(self, canvas, blocks, start_y=0, debug=False, side="left"):
        """
        Place a sequence of blocks top-to-bottom.
        Returns updated cursor_y.
        """
        cursor_y = start_y
        output_h, output_w = canvas.shape

        for block in blocks:
            color = block["color"]

            if color == 5:
                # 5 is handled separately as the marker
                continue

            if color not in self.motif_map:
                continue

            x = self.choose_anchor_x(color, block["local_x"], output_w, side)
            motif_h, _ = self.motif_size(color)
            advance = self.advance_cursor_y(color, side=side)

            if cursor_y + motif_h > output_h:
                break

            self.draw_motif(canvas, color, cursor_y, x)

            if debug:
                print(
                    f"  placing color={color} side={side} "
                    f"at y={cursor_y}, x={x}, motif_h={motif_h}, advance={advance}"
                )

            cursor_y += advance

        return cursor_y

    # ============================================================
    # OUTPUT SHAPE
    # ============================================================

    def infer_output_width(self, input_grid, divider_col):
        return 7

    def infer_output_height(self, input_grid):
        grid = self.to_array(input_grid)
        return grid.shape[0]

    def extract_marker_five(self, input_grid, divider_col, output_width):
        grid = self.to_array(input_grid)
        coords = np.argwhere(grid == 5)
        if coords.size == 0:
            return None

        r, c = coords[0]
        col = min(1, output_width - 1)
        return int(r), int(col)

    # ============================================================
    # MAIN SOLVE
    # ============================================================

    def solve(self, input_grid, debug=False):
        """
        Solve one input grid.
        Returns list-of-lists output grid.
        """
        grid = self.to_array(input_grid)
        input_h, input_w = grid.shape

        left_grid, divider_col, right_grid = self.split_sides(grid, divider_color=4)

        left_blocks = self.get_blocks(left_grid)
        right_blocks = self.get_blocks(right_grid)

        output_h = self.infer_output_height(grid)
        output_w = self.infer_output_width(grid, divider_col)
        output = self.blank_grid(output_h, output_w, fill=0)

        # Place the 5 marker first if present.
        five_pos = self.extract_marker_five(grid, divider_col, output_w)
        if five_pos is not None:
            fy, fx = five_pos
            if 0 <= fy < output_h and 0 <= fx < output_w:
                output[fy, fx] = 5

        # Start below the 5 if the 5 is on the first row.
        cursor_y = 1 if five_pos is not None and five_pos[0] == 0 else 0

        # Place left blocks in semantic order.
        groups = self.classify_blocks(left_blocks)

        cursor_y = self.place_blocks_sequence(
            output, groups["top"], start_y=cursor_y, debug=debug, side="left"
        )
        cursor_y = self.place_blocks_sequence(
            output, groups["middle"], start_y=cursor_y, debug=debug, side="left"
        )
        cursor_y = self.place_blocks_sequence(
            output, groups["vertical"], start_y=cursor_y, debug=debug, side="left"
        )
        cursor_y = self.place_blocks_sequence(
            output, groups["bottom"], start_y=cursor_y, debug=debug, side="left"
        )

        # Usually do not place right-side 5 again.
        right_non_five = [b for b in right_blocks if b["color"] != 5]
        cursor_y = self.place_blocks_sequence(
            output, right_non_five, start_y=cursor_y, debug=debug, side="right"
        )

        if debug:
            print("=== MOTIF ENGINE DEBUG ===")
            print(f"Input shape: {input_h}x{input_w}")
            print(f"Divider col: {divider_col}")
            print(f"Left blocks: {[(b['color'], b['local_x'], b['min_r']) for b in left_blocks]}")
            print(f"Right blocks: {[(b['color'], b['local_x'], b['min_r']) for b in right_blocks]}")
            print(f"Grouped left blocks: {{")
            print(f"  'top': {[b['color'] for b in groups['top']]},")
            print(f"  'middle': {[b['color'] for b in groups['middle']]},")
            print(f"  'vertical': {[b['color'] for b in groups['vertical']]},")
            print(f"  'bottom': {[b['color'] for b in groups['bottom']]}")
            print(f"}}")
            print(f"Output shape: {output_h}x{output_w}")

        return self.to_list(output)