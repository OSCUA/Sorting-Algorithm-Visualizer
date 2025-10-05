import tkinter as tk
from tkinter import ttk
import random
import time
import numpy as np
import pygame

def random_color():
    return "#%06x" % random.randint(0, 0xFFFFFF)

# Initialize pygame mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Sorting metadata
ALGORITHMS = {
    "Bubble Sort": {"function": "bubble_sort", "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)", "Space": "O(1)", "Stable": "Yes"},
    "Insertion Sort": {"function": "insertion_sort", "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)", "Space": "O(1)", "Stable": "Yes"},
    "Selection Sort": {"function": "selection_sort", "Best": "O(n²)", "Average": "O(n²)", "Worst": "O(n²)", "Space": "O(1)", "Stable": "No"},
    "Quick Sort": {"function": "quick_sort", "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n²)", "Space": "O(log n)", "Stable": "No"},
    "Merge Sort": {"function": "merge_sort", "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)", "Space": "O(n)", "Stable": "Yes"},
    "Heap Sort": {"function": "heap_sort", "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n log n)", "Space": "O(1)", "Stable": "No"},
    "Shell Sort": {"function": "shell_sort", "Best": "O(n log n)", "Average": "O(n log² n)", "Worst": "O(n²)", "Space": "O(1)", "Stable": "No"},
    "Tim Sort": {"function": "tim_sort", "Best": "O(n)", "Average": "O(n log n)", "Worst": "O(n log n)", "Space": "O(n)", "Stable": "Yes"},
    "Counting Sort": {"function": "counting_sort", "Best": "O(n + k)", "Average": "O(n + k)", "Worst": "O(n + k)", "Space": "O(k)", "Stable": "Yes"},
    "Radix Sort": {"function": "radix_sort", "Best": "O(nk)", "Average": "O(nk)", "Worst": "O(nk)", "Space": "O(n + k)", "Stable": "Yes"},
    "Bucket Sort": {"function": "bucket_sort", "Best": "O(n + k)", "Average": "O(n + k)", "Worst": "O(n²)", "Space": "O(n)", "Stable": "Yes"},
    "Cocktail Sort": {"function": "cocktail_sort", "Best": "O(n)", "Average": "O(n²)", "Worst": "O(n²)", "Space": "O(1)", "Stable": "Yes"},
    "Tree Sort": {"function": "tree_sort", "Best": "O(n log n)", "Average": "O(n log n)", "Worst": "O(n²)", "Space": "O(n)", "Stable": "No"},
    "Gnome Sort": {"function": "gnome_sort","Best": "O(n)","Average": "O(n²)","Worst": "O(n²)","Space": "O(1)","Stable": "Yes"},
    "Bitonic Sort": {"function": "bitonic_sort","Best": "O(log² n)","Average": "O(log² n)","Worst": "O(log² n)","Space": "O(n)","Stable": "No"},
    "Cycle Sort": {"function": "cycle_sort","Best": "O(n²)","Average": "O(n²)","Worst": "O(n²)","Space": "O(1)","Stable": "No"},
    "Odd-Even Sort": {"function": "odd_even_sort","Best": "O(n)","Average": "O(n²)","Worst": "O(n²)","Space": "O(1)","Stable": "Yes"},
    "Comb Sort": {"function": "comb_sort","Best": "O(n log n)","Average": "O(n²)","Worst": "O(n²)","Space": "O(1)","Stable": "No"},
}


# Dark Mode Colors
BG_DARK = "#1e1e1e"
FRAME_DARK = "#2e2e2e"
TEXT_LIGHT = "#dcdcdc"
CANVAS_DARK = "#121212"

COLOR_ACTIVE = "#ff6f61"
COLOR_PASSIVE = "#6fa8dc"
COLOR_SORTED = "#93c47d"

# Sound effect
def play_tone(position, max_value, duration=0.05):
    freq = 200 + int((position / max_value) * 1800)
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(freq * t * 2 * np.pi)
    volume = 0.2  # Scale from 0.0 (silent) to 1.0 (max)
    audio = (wave * 32767 * volume).astype(np.int16).tobytes()
    sound = pygame.mixer.Sound(buffer=audio)
    sound.play()

# Completion sweep animation
def completion_sweep(data, draw_data, speed):
    for i in range(len(data)):
        color_array = []
        for x in range(len(data)):
            if x < i:
                color_array.append(COLOR_SORTED)
            elif x == i:
                color_array.append(COLOR_ACTIVE)
            else:
                color_array.append(COLOR_PASSIVE)
        draw_data(data, color_array)
        play_tone(i, len(data))
        time.sleep(speed / 2)
    draw_data(data, [COLOR_SORTED for _ in range(len(data))])

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

# Sorting algorithms
def bubble_sort(data, draw_data, speed):
    for i in range(len(data)-1):
        for j in range(len(data)-i-1):
            draw_data(data, [COLOR_ACTIVE if x == j or x == j+1 else COLOR_PASSIVE for x in range(len(data))])
            play_tone(j, len(data))
            time.sleep(speed)
            if data[j] > data[j+1]:
                data[j], data[j+1] = data[j+1], data[j]
                draw_data(data, [COLOR_ACTIVE if x == j or x == j+1 else COLOR_PASSIVE for x in range(len(data))])
                play_tone(j+1, len(data))
                time.sleep(speed)
    completion_sweep(data, draw_data, speed)

def insertion_sort(data, draw_data, speed):
    for i in range(1, len(data)):
        key = data[i]
        j = i - 1
        while j >= 0 and data[j] > key:
            data[j + 1] = data[j]
            draw_data(data, [COLOR_ACTIVE if x == j or x == j+1 else COLOR_PASSIVE for x in range(len(data))])
            play_tone(j, len(data))
            time.sleep(speed)
            j -= 1
        data[j + 1] = key
    completion_sweep(data, draw_data, speed)

def selection_sort(data, draw_data, speed):
    for i in range(len(data)):
        min_idx = i
        for j in range(i+1, len(data)):
            draw_data(data, [COLOR_ACTIVE if x == j or x == min_idx else COLOR_PASSIVE for x in range(len(data))])
            play_tone(j, len(data))
            time.sleep(speed)
            if data[j] < data[min_idx]:
                min_idx = j
        data[i], data[min_idx] = data[min_idx], data[i]
        draw_data(data, [COLOR_ACTIVE if x == i or x == min_idx else COLOR_PASSIVE for x in range(len(data))])
        play_tone(i, len(data))
        time.sleep(speed)
    completion_sweep(data, draw_data, speed)

def quick_sort(data, draw_data, speed):
    def partition(low, high):
        pivot = data[high]
        i = low - 1
        for j in range(low, high):
            draw_data(data, [COLOR_ACTIVE if x == j or x == high else COLOR_PASSIVE for x in range(len(data))])
            play_tone(j, len(data))
            time.sleep(speed)
            if data[j] < pivot:
                i += 1
                data[i], data[j] = data[j], data[i]
                draw_data(data, [COLOR_ACTIVE if x in (i, j) else COLOR_PASSIVE for x in range(len(data))])
                play_tone(i, len(data))
                time.sleep(speed)
        data[i+1], data[high] = data[high], data[i+1]
        return i + 1

    def quick_sort_recursive(low, high):
        if low < high:
            pi = partition(low, high)
            quick_sort_recursive(low, pi - 1)
            quick_sort_recursive(pi + 1, high)

    quick_sort_recursive(0, len(data) - 1)
    completion_sweep(data, draw_data, speed)

def merge_sort(data, draw_data, speed):
    def merge_sort_recursive(arr, l, r):
        if l < r:
            m = (l + r) // 2
            merge_sort_recursive(arr, l, m)
            merge_sort_recursive(arr, m+1, r)
            merge(arr, l, m, r)

    def merge(arr, l, m, r):
        left = arr[l:m+1]
        right = arr[m+1:r+1]
        i = j = 0
        k = l
        while i < len(left) and j < len(right):
            draw_data(data, [COLOR_ACTIVE if x == k else COLOR_PASSIVE for x in range(len(data))])
            play_tone(k, len(data))
            time.sleep(speed)
            if left[i] <= right[j]:
                arr[k] = left[i]
                i += 1
            else:
                arr[k] = right[j]
                j += 1
            k += 1
        while i < len(left):
            arr[k] = left[i]
            i += 1
            k += 1
        while j < len(right):
            arr[k] = right[j]
            j += 1
            k += 1

    merge_sort_recursive(data, 0, len(data)-1)
    completion_sweep(data, draw_data, speed)

def heap_sort(data, draw_data, speed):
    def heapify(n, i):
        largest = i
        l = 2*i + 1
        r = 2*i + 2

        if l < n and data[l] > data[largest]:
            largest = l
        if r < n and data[r] > data[largest]:
            largest = r

        if largest != i:
            data[i], data[largest] = data[largest], data[i]
            draw_data(data, [COLOR_ACTIVE if x in (i, largest) else COLOR_PASSIVE for x in range(len(data))])
            play_tone(i, len(data))
            time.sleep(speed)
            heapify(n, largest)

    n = len(data)
    # Build max heap
    for i in range(n//2 - 1, -1, -1):
        heapify(n, i)
    # Extract elements
    for i in range(n-1, 0, -1):
        data[i], data[0] = data[0], data[i]
        draw_data(data, [COLOR_ACTIVE if x in (0, i) else COLOR_PASSIVE for x in range(len(data))])
        play_tone(i, len(data))
        time.sleep(speed)
        heapify(i, 0)

    completion_sweep(data, draw_data, speed)

def shell_sort(data, draw_data, speed):
    n = len(data)
    gap = n // 2
    while gap > 0:
        for i in range(gap, n):
            temp = data[i]
            j = i
            while j >= gap and data[j - gap] > temp:
                data[j] = data[j - gap]
                draw_data(data, [COLOR_ACTIVE if x in (j, j-gap) else COLOR_PASSIVE for x in range(n)])
                play_tone(j, n)
                time.sleep(speed)
                j -= gap
            data[j] = temp
        gap //= 2
    completion_sweep(data, draw_data, speed)


def tim_sort(data, draw_data, speed):
    # Python's built-in sort is TimSort, so we can leverage it
    # For visualization, we just simulate comparisons and then call sorted()
    n = len(data)
    # Simulate scanning
    for i in range(n):
        draw_data(data, [COLOR_ACTIVE if x == i else COLOR_PASSIVE for x in range(n)])
        play_tone(i, n)
        time.sleep(speed / 2)
    # Actually sort
    data[:] = sorted(data)
    completion_sweep(data, draw_data, speed)


def counting_sort(data, draw_data, speed):
    n = len(data)
    max_val = max(data)
    count = [0] * (max_val + 1)

    # Count occurrences
    for num in data:
        count[num] += 1

    # Rebuild array
    idx = 0
    for i, c in enumerate(count):
        for _ in range(c):
            data[idx] = i
            draw_data(data, [COLOR_ACTIVE if x == idx else COLOR_PASSIVE for x in range(n)])
            play_tone(idx, n)
            time.sleep(speed / 2)
            idx += 1

    completion_sweep(data, draw_data, speed)

def radix_sort(data, draw_data, speed):
    def counting_sort_exp(exp):
        n = len(data)
        output = [0] * n
        count = [0] * 10

        for i in range(n):
            index = (data[i] // exp) % 10
            count[index] += 1

        for i in range(1, 10):
            count[i] += count[i - 1]

        i = n - 1
        while i >= 0:
            index = (data[i] // exp) % 10
            output[count[index] - 1] = data[i]
            count[index] -= 1
            i -= 1

        for i in range(n):
            data[i] = output[i]
            draw_data(data, [COLOR_ACTIVE if x == i else COLOR_PASSIVE for x in range(n)])
            play_tone(i, n)
            time.sleep(speed / 2)

    max_val = max(data)
    exp = 1
    while max_val // exp > 0:
        counting_sort_exp(exp)
        exp *= 10

    completion_sweep(data, draw_data, speed)


def bucket_sort(data, draw_data, speed):
    n = len(data)
    max_val = max(data)
    size = max_val / n

    buckets = [[] for _ in range(n)]
    for num in data:
        index = int(num / size)
        if index != n:
            buckets[index].append(num)
        else:
            buckets[n - 1].append(num)

    for bucket in buckets:
        bucket.sort()

    idx = 0
    for b in buckets:
        for num in b:
            data[idx] = num
            draw_data(data, [COLOR_ACTIVE if x == idx else COLOR_PASSIVE for x in range(n)])
            play_tone(idx, n)
            time.sleep(speed / 2)
            idx += 1

    completion_sweep(data, draw_data, speed)


def cocktail_sort(data, draw_data, speed):
    n = len(data)
    swapped = True
    start = 0
    end = n - 1
    while swapped:
        swapped = False
        for i in range(start, end):
            draw_data(data, [COLOR_ACTIVE if x in (i, i+1) else COLOR_PASSIVE for x in range(n)])
            play_tone(i, n)
            time.sleep(speed)
            if data[i] > data[i+1]:
                data[i], data[i+1] = data[i+1], data[i]
                swapped = True
        if not swapped:
            break
        swapped = False
        end -= 1
        for i in range(end-1, start-1, -1):
            draw_data(data, [COLOR_ACTIVE if x in (i, i+1) else COLOR_PASSIVE for x in range(n)])
            play_tone(i, n)
            time.sleep(speed)
            if data[i] > data[i+1]:
                data[i], data[i+1] = data[i+1], data[i]
                swapped = True
        start += 1
    completion_sweep(data, draw_data, speed)

def tree_sort(data, draw_data, speed):
    class Node:
        def __init__(self, val):
            self.val = val
            self.left = None
            self.right = None

    def insert(root, val):
        if root is None:
            return Node(val)
        if val < root.val:
            root.left = insert(root.left, val)
        else:
            root.right = insert(root.right, val)
        return root

    def inorder(root, arr):
        if root:
            inorder(root.left, arr)
            arr.append(root.val)
            inorder(root.right, arr)

    # Build BST
    root = None
    for val in data:
        root = insert(root, val)

    # Extract sorted values
    sorted_data = []
    inorder(root, sorted_data)

    for i in range(len(data)):
        data[i] = sorted_data[i]
        draw_data(data, [COLOR_ACTIVE if x == i else COLOR_PASSIVE for x in range(len(data))])
        play_tone(i, len(data))
        time.sleep(speed / 2)

    completion_sweep(data, draw_data, speed)

def gnome_sort(data, draw_data, speed):
    index = 0
    while index < len(data):
        if index == 0 or data[index] >= data[index - 1]:
            index += 1
        else:
            data[index], data[index - 1] = data[index - 1], data[index]
            draw_data(data, [COLOR_ACTIVE if x in (index, index - 1) else COLOR_PASSIVE for x in range(len(data))])
            play_tone(index, len(data))
            time.sleep(speed)
            index -= 1
    completion_sweep(data, draw_data, speed)

def bitonic_sort(data, draw_data, speed):
    def compare_and_swap(up, i, j):
        if (up and data[i] > data[j]) or (not up and data[i] < data[j]):
            data[i], data[j] = data[j], data[i]
            draw_data(data, [COLOR_ACTIVE if x in (i, j) else COLOR_PASSIVE for x in range(len(data))])
            play_tone(i, len(data))
            time.sleep(speed)

    def bitonic_merge(low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                compare_and_swap(up, i, i + k)
            bitonic_merge(low, k, up)
            bitonic_merge(low + k, k, up)

def bitonic_sort(data, draw_data, speed):
    def compare_and_swap(arr, i, j, up):
        if (up and arr[i] > arr[j]) or (not up and arr[i] < arr[j]):
            arr[i], arr[j] = arr[j], arr[i]
            draw_data(arr, [COLOR_ACTIVE if x in (i, j) else COLOR_PASSIVE for x in range(len(arr))])
            play_tone(i, len(arr))
            time.sleep(speed)

    def bitonic_merge(arr, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            for i in range(low, low + k):
                compare_and_swap(arr, i, i + k, up)
            bitonic_merge(arr, low, k, up)
            bitonic_merge(arr, low + k, k, up)

    def bitonic_sort_rec(arr, low, cnt, up):
        if cnt > 1:
            k = cnt // 2
            bitonic_sort_rec(arr, low, k, True)
            bitonic_sort_rec(arr, low + k, k, False)
            bitonic_merge(arr, low, cnt, up)

    # Pad to next power of two
    n = len(data)
    next_pow2 = 1 << (n - 1).bit_length()
    padded_data = data + [float('inf')] * (next_pow2 - n)

    bitonic_sort_rec(padded_data, 0, len(padded_data), True)

    # Remove padding
    data[:] = [x for x in padded_data if x != float('inf')]
    completion_sweep(data, draw_data, speed)

def cycle_sort(data, draw_data, speed):
    n = len(data)
    for cycle_start in range(n - 1):
        item = data[cycle_start]
        pos = cycle_start
        for i in range(cycle_start + 1, n):
            if data[i] < item:
                pos += 1
        if pos == cycle_start:
            continue
        while item == data[pos]:
            pos += 1
        data[pos], item = item, data[pos]
        draw_data(data, [COLOR_ACTIVE if x == pos else COLOR_PASSIVE for x in range(n)])
        play_tone(pos, n)
        time.sleep(speed)
        while pos != cycle_start:
            pos = cycle_start
            for i in range(cycle_start + 1, n):
                if data[i] < item:
                    pos += 1
            while item == data[pos]:
                pos += 1
            data[pos], item = item, data[pos]
            draw_data(data, [COLOR_ACTIVE if x == pos else COLOR_PASSIVE for x in range(n)])
            play_tone(pos, n)
            time.sleep(speed)
    completion_sweep(data, draw_data, speed)

def odd_even_sort(data, draw_data, speed):
    n = len(data)
    sorted = False
    while not sorted:
        sorted = True
        for i in range(1, n - 1, 2):
            if data[i] > data[i + 1]:
                data[i], data[i + 1] = data[i + 1], data[i]
                sorted = False
                draw_data(data, [COLOR_ACTIVE if x in (i, i + 1) else COLOR_PASSIVE for x in range(n)])
                play_tone(i, n)
                time.sleep(speed)
        for i in range(0, n - 1, 2):
            if data[i] > data[i + 1]:
                data[i], data[i + 1] = data[i + 1], data[i]
                sorted = False
                draw_data(data, [COLOR_ACTIVE if x in (i, i + 1) else COLOR_PASSIVE for x in range(n)])
                play_tone(i, n)
                time.sleep(speed)
    completion_sweep(data, draw_data, speed)

def comb_sort(data, draw_data, speed):
    n = len(data)
    gap = n
    shrink = 1.3
    sorted = False
    while not sorted:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted = True
        i = 0
        while i + gap < n:
            if data[i] > data[i + gap]:
                data[i], data[i + gap] = data[i + gap], data[i]
                sorted = False
                draw_data(data, [COLOR_ACTIVE if x in (i, i + gap) else COLOR_PASSIVE for x in range(n)])
                play_tone(i, n)
                time.sleep(speed)
            i += 1
    completion_sweep(data, draw_data, speed)




# Start sorting
def start_sort():
    global data
    speed = speed_scale.get()
    algo = algorithm_var.get()
    if rand_colors.get() == True:
        refresh_colors()
    sort_fn = globals()[ALGORITHMS[algo]["function"]]
    sort_fn(data, draw_data, speed)

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
tk.Checkbutton(
    frame,
    text="Random Colors",        # label next to the box
    variable=rand_colors,
    onvalue=True,
    offvalue=False,
    bg=FRAME_DARK,
    fg=TEXT_LIGHT,
    selectcolor=BG_DARK   # dark theme
).grid(row=0, column=7, padx=5)


algorithm_var = tk.StringVar()
algorithm_menu = ttk.Combobox(frame, textvariable=algorithm_var, values=list(ALGORITHMS.keys()), state="readonly")
algorithm_menu.grid(row=0, column=7)
algorithm_menu.set("Bubble Sort")
rand_colors = tk.BooleanVar(value=True)  # starts checked

canvas = tk.Canvas(root, width=600, height=380, bg=CANVAS_DARK, highlightthickness=0)
canvas.pack(pady=20)

data = []
generate_data()
root.mainloop()
