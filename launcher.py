import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import sorts
from visualizer import run

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HERE = Path(__file__).parent


def ask_settings():
    """Show the launcher window; returns (img_path, algo_name, swaps, shuffle, realtime, compiled) or None if closed."""
    root = tk.Tk()
    root.title("Image sort visualizer")
    root.resizable(False, False)
    frame = ttk.Frame(root, padding=12)
    frame.grid()

    images = sorted(p.name for p in HERE.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    img_var = tk.StringVar(value=images[0] if images else "")
    algo_var = tk.StringVar(value=next(iter(sorts.ALGORITHMS)))
    swaps_var = tk.IntVar(value=sorts.SWAPS_PER_STEP)
    shuffle_var = tk.BooleanVar(value=True)
    realtime_var = tk.BooleanVar(value=True)
    compiled_var = tk.BooleanVar(value=True)

    def browse():
        path = filedialog.askopenfilename(
            initialdir=HERE,
            filetypes=[("Images", " ".join(f"*{e}" for e in IMAGE_EXTS)), ("All files", "*")],
        )
        if path:
            img_var.set(path)

    ttk.Label(frame, text="Image").grid(row=0, column=0, sticky="w", pady=4)
    ttk.Combobox(frame, textvariable=img_var, values=images, width=30).grid(row=0, column=1, pady=4)
    ttk.Button(frame, text="Browse…", command=browse).grid(row=0, column=2, padx=(6, 0))

    ttk.Label(frame, text="Algorithm").grid(row=1, column=0, sticky="w", pady=4)
    ttk.Combobox(frame, textvariable=algo_var, values=list(sorts.ALGORITHMS),
                 state="readonly", width=30).grid(row=1, column=1, pady=4)

    ttk.Checkbutton(frame, text="Real-time (sort at full speed)", variable=realtime_var).grid(
        row=2, column=1, sticky="w", pady=4)

    compiled_box = ttk.Checkbutton(frame, text="Compiled (Numba, native speed)", variable=compiled_var)
    compiled_box.grid(row=3, column=1, sticky="w", pady=4)

    ttk.Label(frame, text="Swaps per step").grid(row=4, column=0, sticky="w", pady=4)
    swaps_box = ttk.Spinbox(frame, textvariable=swaps_var, from_=1, to=1_000_000,
                            increment=1, width=10)
    swaps_box.grid(row=4, column=1, sticky="w", pady=4)

    # the pace setting only matters when sorting and drawing take turns,
    # and compiled sorts only exist for the real-time worker
    def update_states(*_):
        realtime = realtime_var.get()
        swaps_box.state(["disabled"] if realtime else ["!disabled"])
        compiled_box.state(["!disabled"] if realtime else ["disabled"])

    realtime_var.trace_add("write", update_states)
    update_states()

    ttk.Checkbutton(frame, text="Shuffle first", variable=shuffle_var).grid(
        row=5, column=1, sticky="w", pady=4)

    result = None

    def start(_event=None):
        nonlocal result
        try:
            swaps = max(1, swaps_var.get())
        except tk.TclError:  # non-numeric input in the spinbox
            swaps = 1
        img = Path(img_var.get())
        if not img.is_absolute():
            img = HERE / img
        if not img.is_file():
            return
        result = (str(img), algo_var.get(), swaps, shuffle_var.get(), realtime_var.get(), compiled_var.get())
        root.destroy()

    ttk.Button(frame, text="Start", command=start).grid(row=6, column=1, sticky="e", pady=(8, 0))
    root.bind("<Return>", start)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return result


def main():
    # loop back to the launcher after the visualizer window is closed
    while (settings := ask_settings()) is not None:
        img_path, algo_name, swaps, do_shuffle, realtime, compiled = settings
        sorts.SWAPS_PER_STEP = swaps  # sorts read this global at runtime (stepped mode)
        run(img_path, algo_name, do_shuffle, realtime, compiled)


if __name__ == "__main__":
    main()
