import tkinter as tk
from tkinter import ttk
import random
import time
import numpy as np
import pygame
import algorithms
from algorithm_metadata import ALGORITHMS
from colors import COLOR_ACTIVE, COLOR_PASSIVE, COLOR_SORTED, CANVAS_DARK, TEXT_LIGHT,FRAME_DARK, BG_DARK
from utils import play_tone

def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Draw bars and overlay
def draw_data(data, color_array):
    canvas.delete("all")
    c_height = 380
    c_width = 600
    x_width = c_width / (len(data) + 1)
    spacing = 5

    # Filter out padding values
    visible_data = [i for i in data if i != float('inf')]
    if not visible_data:
        return

    max_val = max(visible_data)
    normalized_data = [i / max_val if i != float('inf') else 0 for i in data]

    for i, height in enumerate(normalized_data):
        if data[i] == float('inf'):
            continue  # skip drawing padding bars

        x0 = i * x_width + spacing
        y0 = c_height - height * 340
        x1 = (i + 1) * x_width
        y1 = c_height
        canvas.create_rectangle(x0, y0, x1, y1, fill=color_array[i])
        canvas.create_text(x0 + 2, y0, anchor=tk.SW, text=str(data[i]), font=("Helvetica", 8), fill=TEXT_LIGHT)

    root.update_idletasks()


    # Overlay
    algo = algorithm_var.get()
    meta = ALGORITHMS[algo]
    canvas.create_rectangle(10, 10, 190, 110, fill=FRAME_DARK, outline="")
    canvas.create_text(20, 20, anchor=tk.NW, text=f"Algorithm: {algo}", font=("Helvetica", 10, "bold"), fill=TEXT_LIGHT)
    canvas.create_text(20, 40, anchor=tk.NW, text=f"Best: {meta['Best']}", font=("Helvetica", 9), fill=TEXT_LIGHT)
    canvas.create_text(20, 55, anchor=tk.NW, text=f"Average: {meta['Average']}", font=("Helvetica", 9), fill=TEXT_LIGHT)
    canvas.create_text(20, 70, anchor=tk.NW, text=f"Worst: {meta['Worst']}", font=("Helvetica", 9), fill=TEXT_LIGHT)
    canvas.create_text(20, 85, anchor=tk.NW, text=f"Space: {meta['Space']}", font=("Helvetica", 9), fill=TEXT_LIGHT)
    canvas.create_text(20, 100, anchor=tk.NW, text=f"Stable: {meta['Stable']}", font=("Helvetica", 9), fill=TEXT_LIGHT)

    root.update_idletasks()

def refresh_colors():
    global COLOR_ACTIVE, COLOR_PASSIVE, COLOR_SORTED
    COLOR_ACTIVE = random_color()
    COLOR_PASSIVE = random_color()
    COLOR_SORTED = random_color()

# Start sorting
def start_sort():
    global data
    speed = speed_scale.get()
    algo = algorithm_var.get()
    if rand_colors.get():
        refresh_colors()
    sort_fn = getattr(algorithms, ALGORITHMS[algo]["function"])
    sort_fn(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE)

# Autoplay
def autoplay():
    for algo in ALGORITHMS.keys():
        algorithm_var.set(algo)
        generate_data()
        start_sort()
        time.sleep(2)

# Generate data
def generate_data():
    global data
    size = int(size_entry.get())
    data = [random.randint(10, 100) for _ in range(size)]
    draw_data(data, [COLOR_PASSIVE for _ in range(len(data))])

# GUI setup
root = tk.Tk()
root.title("Sorting Algorithm Visualizer")
root.geometry("1000x500")
root.config(bg=BG_DARK)

frame = tk.Frame(root, bg=FRAME_DARK)
frame.pack(pady=10)

tk.Label(frame, text="Array Size:", bg=FRAME_DARK, fg=TEXT_LIGHT).grid(row=0, column=0)
size_entry = tk.Entry(frame, bg=BG_DARK, fg=TEXT_LIGHT, insertbackground=TEXT_LIGHT)
size_entry.grid(row=0, column=1)
size_entry.insert(0, "30")

tk.Label(frame, text="Speed (sec):", bg=FRAME_DARK, fg=TEXT_LIGHT).grid(row=0, column=2)
speed_scale = ttk.Scale(frame, from_=0.01, to=0.2, length=150, orient=tk.HORIZONTAL)
speed_scale.set(0.05)
speed_scale.grid(row=0, column=3)

rand_colors = tk.BooleanVar(value=True)

tk.Button(frame, text="Generate Array", command=generate_data, bg="#444", fg=TEXT_LIGHT).grid(row=0, column=4, padx=5)
tk.Button(frame, text="Start", command=start_sort, bg="#444", fg=TEXT_LIGHT).grid(row=0, column=5, padx=5)
tk.Button(frame, text="Autoplay", command=autoplay, bg="#444", fg=TEXT_LIGHT).grid(row=0, column=6, padx=5)
# tk.Checkbutton(
#     frame,
#     text="Random Colors",        # label next to the box
#     variable=rand_colors,
#     onvalue=True,
#     offvalue=False,
#     bg=FRAME_DARK,
#     fg=TEXT_LIGHT,
#     selectcolor=BG_DARK   # dark theme
# ).grid(row=0, column=9, padx=5)


algorithm_var = tk.StringVar()
algorithm_menu = ttk.Combobox(frame, textvariable=algorithm_var, values=list(ALGORITHMS.keys()), state="readonly")
algorithm_menu.grid(row=0, column=7)
algorithm_menu.set("Bubble Sort")
# rand_colors = tk.BooleanVar(value=True)  # starts checked

canvas = tk.Canvas(root, width=600, height=380, bg=CANVAS_DARK, highlightthickness=0)
canvas.pack(pady=20)

data = []
generate_data()
root.mainloop()