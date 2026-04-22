import numpy as np
import matplotlib.pyplot as plt


COLOR_MAP = {
    0: (1.0, 1.0, 1.0),  # white background
    1: (0.0, 0.0, 1.0),  # blue
    2: (1.0, 0.0, 0.0),  # red
    3: (0.0, 0.8, 0.0),  # green
    4: (1.0, 1.0, 0.0),  # yellow
    5: (1.0, 0.0, 1.0),  # magenta
    6: (0.0, 1.0, 1.0),  # cyan
    7: (1.0, 0.5, 0.0),  # orange
    8: (0.5, 0.0, 1.0),  # purple
    9: (0.5, 0.5, 0.5),  # gray
}


def grid_to_image(grid):
    h = len(grid)
    w = len(grid[0]) if h else 0

    img = np.ones((h, w, 3), dtype=float)

    for r in range(h):
        for c in range(w):
            value = grid[r][c]
            img[r, c] = COLOR_MAP.get(value, (0.0, 0.0, 0.0))

    return img


def show_grid(grid, title="GRID"):
    img = grid_to_image(grid)

    plt.figure(figsize=(4, 4))
    plt.imshow(img, interpolation="nearest")
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def show_three_grids(a, b, c, title_a="INPUT", title_b="SELECTED OBJECT", title_c="EXPECTED"):
    img_a = grid_to_image(a)
    img_b = grid_to_image(b)
    img_c = grid_to_image(c)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(img_a, interpolation="nearest")
    axs[0].set_title(title_a)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(img_b, interpolation="nearest")
    axs[1].set_title(title_b)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    axs[2].imshow(img_c, interpolation="nearest")
    axs[2].set_title(title_c)
    axs[2].set_xticks([])
    axs[2].set_yticks([])

    plt.tight_layout()
    plt.show()