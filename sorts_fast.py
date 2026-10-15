"""Numba-compiled versions of the sorts in sorts.py, for real-time mode.

Same algorithms, same in-place access pattern (so they look the same on screen),
but no yields: they run start to finish at native speed on a numpy int array.
cache=True stores the machine code in __pycache__ so only the first run compiles.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def shuffle(order):
    """Fisher-Yates shuffle."""
    n = len(order)
    for i in range(n):
        j = np.random.randint(i, n)
        order[i], order[j] = order[j], order[i]


@njit(cache=True)
def quicksort(order):
    """In-place quicksort (Hoare partition) with an explicit stack."""
    # only ranges of 2+ elements are pushed and pending ranges never overlap,
    # so at most n/2 (lo, hi) pairs are ever on the stack
    stack = np.empty(len(order) + 4, dtype=np.int64)
    stack[0], stack[1] = 0, len(order) - 1
    top = 2
    while top:
        top -= 2
        lo, hi = stack[top], stack[top + 1]

        pivot = order[(lo + hi) // 2]
        i, j = lo, hi
        while i <= j:
            while order[i] < pivot:
                i += 1
            while order[j] > pivot:
                j -= 1
            if i <= j:
                order[i], order[j] = order[j], order[i]
                i += 1
                j -= 1

        # left part is pushed last so it gets sorted first
        if i < hi:
            stack[top], stack[top + 1] = i, hi
            top += 2
        if lo < j:
            stack[top], stack[top + 1] = lo, j
            top += 2


@njit(cache=True)
def bubblesort(order):
    done = False
    while not done:
        done = True
        for i in range(len(order) - 1):
            if order[i + 1] < order[i]:
                order[i], order[i + 1] = order[i + 1], order[i]
                done = False


@njit(cache=True)
def radixsort(order, base=10):
    """LSD radix sort; like sorts.radixsort it writes each pass back front to back, bucket by bucket."""
    n = len(order)
    out = np.empty_like(order)
    m = order.max()
    div = 1
    while m // div > 0:
        counts = np.zeros(base + 1, dtype=np.int64)
        for i in range(n):
            counts[(order[i] // div) % base + 1] += 1
        for b in range(base):
            counts[b + 1] += counts[b]
        for i in range(n):
            c = (order[i] // div) % base
            out[counts[c]] = order[i]
            counts[c] += 1
        for i in range(n):
            order[i] = out[i]
        div *= base


@njit  # not cached: numba segfaults loading cached recursive functions
def bitonicmerge(a, low, count, direction):
    if count > 1:
        k = 1
        while k * 2 < count:
            k *= 2
        for i in range(low, low + count - k):
            if (direction == 1 and a[i] > a[i + k]) or (direction == 0 and a[i] < a[i + k]):
                a[i], a[i + k] = a[i + k], a[i]
        bitonicmerge(a, low, k, direction)
        bitonicmerge(a, low + k, count - k, direction)


@njit
def bitonicsort(a, low, count, direction):
    if count > 1:
        k = count // 2
        bitonicsort(a, low, k, 1 - direction)
        bitonicsort(a, low + k, count - k, direction)
        bitonicmerge(a, low, count, direction)


@njit
def bitonic(order):
    bitonicsort(order, 0, len(order), 1)


@njit(cache=True)
def cocktailsort(order):
    lo, hi = 0, len(order) - 1
    changed = True
    while changed:
        changed = False
        for i in range(lo, hi):
            if order[i + 1] < order[i]:
                order[i], order[i + 1] = order[i + 1], order[i]
                changed = True
        if not changed:
            break
        hi -= 1
        changed = False
        for i in range(hi, lo, -1):
            if order[i - 1] > order[i]:
                order[i], order[i - 1] = order[i - 1], order[i]
                changed = True
        lo += 1


@njit(cache=True)
def gnomesort(order):
    pos = 1
    while pos < len(order):
        if pos == 0 or order[pos] >= order[pos - 1]:
            pos += 1
        else:
            order[pos], order[pos - 1] = order[pos - 1], order[pos]
            pos -= 1


@njit(cache=True)
def _sift_down(order, root, end):
    while 2 * root + 1 < end:
        child = 2 * root + 1
        if child + 1 < end and order[child] < order[child + 1]:
            child += 1
        if order[root] >= order[child]:
            return
        order[root], order[child] = order[child], order[root]
        root = child


@njit(cache=True)
def heapsort(order):
    """In-place heap sort (max-heap)."""
    n = len(order)
    for start in range(n // 2 - 1, -1, -1):
        _sift_down(order, start, n)
    for end in range(n - 1, 0, -1):
        order[0], order[end] = order[end], order[0]
        _sift_down(order, 0, end)


# keyed like sorts.ALGORITHMS; anything missing here (bogosort) falls back to the Python version
ALGORITHMS = {
    "Gnome sort": gnomesort,
    "Quicksort": quicksort,
    "Heap sort": heapsort,
    "Bubble sort": bubblesort,
    "Cocktail sort": cocktailsort,
    "Radix sort": radixsort,
    "Bitonic sort": bitonic,
}
