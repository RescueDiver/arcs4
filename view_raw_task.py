import json
import os
import tkinter as tk


CELL = 28

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

    places = [
        os.path.join(base_dir, "data", "data.json"),
        os.path.join(base_dir, "data_failures", "extracted_tasks", task_id + ".json"),
    ]

    # full data file
    data_path = places[0]
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if task_id in data:
            return task_id, data[task_id]

    # extracted failure file
    failure_path = places[1]
    if os.path.exists(failure_path):
        with open(failure_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if "train" in raw:
            return task_id, raw

        if task_id in raw:
            return task_id, raw[task_id]

        if isinstance(raw, dict) and len(raw) == 1:
            k = next(iter(raw))
            return k, raw[k]

    raise FileNotFoundError(f"Could not find task: {task_id}")


def shape_text(grid):
    if grid is None:
        return "None"
    h = len(grid)
    w = len(grid[0]) if h else 0
    return f"{h}x{w}"


def draw_grid(parent, grid, title):
    box = tk.Frame(parent)
    box.pack(side=tk.LEFT, padx=12, pady=8, anchor="n")

    tk.Label(
        box,
        text=f"{title}  {shape_text(grid)}",
        font=("Arial", 12, "bold"),
    ).pack()

    h = len(grid)
    w = len(grid[0]) if h else 0

    canvas = tk.Canvas(
        box,
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
                outline="#555555",
            )


def main():
    task_id = input("Enter task id: ").strip()
    task_id, task = load_task(task_id)

    root = tk.Tk()
    root.title(f"RAW TASK VIEWER - {task_id}")

    tk.Label(
        root,
        text=f"RAW TASK: {task_id}",
        font=("Arial", 16, "bold"),
    ).pack(pady=8)

    outer = tk.Frame(root)
    outer.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(outer)
    y_scroll = tk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
    x_scroll = tk.Scrollbar(outer, orient=tk.HORIZONTAL, command=canvas.xview)

    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(
        yscrollcommand=y_scroll.set,
        xscrollcommand=x_scroll.set,
    )

    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)

    for i, pair in enumerate(task.get("train", [])):
        tk.Label(
            scroll_frame,
            text=f"TRAIN PAIR {i + 1}",
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(14, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, pair["input"], "INPUT")
        draw_grid(row, pair["output"], "EXPECTED")

    for i, pair in enumerate(task.get("test", [])):
        tk.Label(
            scroll_frame,
            text=f"TEST PAIR {i}",
            font=("Arial", 14, "bold"),
            anchor="w",
        ).pack(fill=tk.X, padx=10, pady=(20, 0))

        row = tk.Frame(scroll_frame)
        row.pack(anchor="w")

        draw_grid(row, pair["input"], "TEST INPUT")

    root.mainloop()


if __name__ == "__main__":
    main()