from colors import COLOR_ACTIVE, COLOR_PASSIVE, COLOR_SORTED
from utils import play_tone, completion_sweep
import time

def bubble_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def insertion_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def selection_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def quick_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def merge_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def heap_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def shell_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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


def tim_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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


def counting_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def radix_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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


def bucket_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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


def cocktail_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def tree_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def gnome_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def bitonic_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def bitonic_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def cycle_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def odd_even_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def comb_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
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

def stalin_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
    sorted_data = [data[0]]
    draw_data(data, [COLOR_ACTIVE if i == 0 else COLOR_PASSIVE for i in range(len(data))])
    play_tone(0, len(data))
    time.sleep(speed)

    for i in range(1, len(data)):
        if data[i] >= sorted_data[-1]:
            sorted_data.append(data[i])
            draw_data(sorted_data + data[i+1:], [COLOR_ACTIVE if x == i else COLOR_PASSIVE for x in range(len(sorted_data + data[i+1:]))])
            play_tone(i, len(data))
            time.sleep(speed)

    # Pad with removed elements for visual consistency
    while len(sorted_data) < len(data):
        sorted_data.append(0)

    data[:] = sorted_data
    completion_sweep(data, draw_data, speed)

def gravity_sort(data, draw_data, speed, play_tone, COLOR_ACTIVE, COLOR_PASSIVE):
    max_val = max(data)
    beads = [[0] * len(data) for _ in range(max_val)]

    # Drop beads
    for i, val in enumerate(data):
        for j in range(val):
            beads[j][i] = 1

    # Gravity pull
    for row in beads:
        count = sum(row)
        for i in range(len(data)):
            row[i] = 1 if i < count else 0

    # Count beads per column
    for i in range(len(data)):
        data[i] = sum(row[i] for row in beads)
        draw_data(data, [COLOR_ACTIVE if x == i else COLOR_PASSIVE for x in range(len(data))])
        play_tone(i, len(data))
        time.sleep(speed)

    completion_sweep(data, draw_data, speed)
