import json
import os
import tkinter as tk


TASK_ID = "0934a4d8"
CELL = 24

COLORS = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def load_task(task_id):
    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, "data", "data.json")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if task_id not in data:
        raise KeyError(f"Task ID not found: {task_id}")

    return data[task_id]


def draw_grid(parent, grid, title):
    frame = tk.Frame(parent)
    frame.pack(side=tk.LEFT, padx=12, pady=12, anchor="n")

    label = tk.Label(frame, text=title, font=("Arial", 12, "bold"))
    label.pack()

    h = len(grid)
    w = len(grid[0]) if h else 0

    canvas = tk.Canvas(
        frame,
        width=w * CELL,
        height=h * CELL,
        bg="white",
        highlightthickness=1,
        highlightbackground="black",
    )
    canvas.pack()

    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            x1 = c * CELL
            y1 = r * CELL
            x2 = x1 + CELL
            y2 = y1 + CELL

            canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                fill=COLORS.get(value, "#FFFFFF"),
                outline="#444444",
            )


def main():
    task = load_task(TASK_ID)

    root = tk.Tk()
    root.title(f"ARC Task Viewer - {TASK_ID}")

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer)
    scrollbar_y = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_x = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)

    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    for i, pair in enumerate(task.get("train", []), start=1):
        row = tk.Frame(scroll_frame)
        row.pack(anchor="w", pady=10)

        draw_grid(row, pair["input"], f"TRAIN {i} INPUT")
        draw_grid(row, pair["output"], f"TRAIN {i} OUTPUT")

    for i, pair in enumerate(task.get("test", []), start=1):
        row = tk.Frame(scroll_frame)
        row.pack(anchor="w", pady=10)

        draw_grid(row, pair["input"], f"TEST {i} INPUT")

    root.mainloop()


if __name__ == "__main__":
    main()