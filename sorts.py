import time
from collections import deque
from multiprocessing.shared_memory import SharedMemory

import numpy as np

SWAPS_PER_STEP = 2 # swaps between two yields of a generator


def shuffle(order):
    """Fisher-Yates shuffle, yields every SWAPS_PER_STEP swaps."""
    n = len(order)
    targets = np.random.default_rng().integers(np.arange(n), n).tolist()
    for i, j in enumerate(targets):
        order[i], order[j] = order[j], order[i]
        if i % SWAPS_PER_STEP == 0:
            yield


def quicksort(order):
    """In-place quicksort (Hoare partition), yields every SWAPS_PER_STEP swaps."""
    swaps = 0
    stack = [(0, len(order) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue

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
                swaps += 1
                if swaps % SWAPS_PER_STEP == 0:
                    yield
                    #sleep(0)

        # left part is pushed last so it gets sorted first
        stack.append((i, hi))
        stack.append((lo, j))

def bubblesort(order):
    swaps = 0
    std = False
    while std is False:
        std = True
        for i in range(len(order) - 1):
            if order[i + 1] < order[i]:
                t = order[i]
                order[i] = order[i + 1]
                order[i + 1] = t
                swaps += 1
                std = False
                if swaps % SWAPS_PER_STEP == 0:
                    yield

def radixsort(order, base=10):
    m = max(order)
    div = 1
    steps = 0
    while m // div > 0:
        hashmap = [[] for x in range(base)]
        for i in range(len(order)):
            c = (order[i] // div) % base
            hashmap[c].append(order[i])
        order_index = 0
        for i in range(base):
            for e in hashmap[i]:
                order[order_index] = e
                order_index += 1
                steps += 1
                if steps % SWAPS_PER_STEP == 0:
                    yield
        div *= base

def bitonicmerge(a, low, count, direction):
    #sleep(0.001)
    if count > 1:
        k = 1 << ((count - 1).bit_length() - 1)
        for i in range(low, low + count - k):
            if (direction == 1 and a[i] > a[i + k]) or (direction == 0 and a[i] < a[i + k]):
                t = a[i + k]
                a[i + k] = a[i]
                a[i] = t
                yield
        yield from bitonicmerge(a, low, k, direction)
        yield from bitonicmerge(a, low + k, count - k, direction)

def bitonicsort(a, low, count, direction):
    #sleep(0.001)
    yield
    if count > 1:
        k = count // 2

        yield from bitonicsort(a, low, k, 1 - direction)
        yield from bitonicsort(a, low + k, count - k, direction)
        yield from bitonicmerge(a, low, count, direction)

def swap(arr, a, b):
    t = arr[a]
    arr[a] = arr[b]
    arr[b] = t

def cocktailsort(order):
    dir = 1
    swaps = 0
    i = 1
    s = False
    changed = False
    while s is False:
        if i < 1 or i >= len(order) - 1:
            dir = dir * -1
            if changed == False:
                s = True
            else:
                changed = False
        if dir == 1:
            if order[i + 1] < order[i]:
                swap(order, i, i + 1)
                changed = True
                swaps += 1
                if swaps % SWAPS_PER_STEP == 0:
                    yield
            i += 1
        else:
            if order[i - 1] > order[i]:
                swap(order, i, i - 1)
                changed = True
                swaps += 1
                if swaps % SWAPS_PER_STEP == 0:
                    yield
            i -= 1

def is_sorted(order):
    for i in range(len(order) - 1):
        if order[i + 1] < order[i]:
            return False
    return True

def bogosort(order):
    while is_sorted(order) is False:
        yield from shuffle(order)

def gnomesort(order):
    pos = 1
    swaps = 0
    while pos < len(order):
        if pos == 0 or order[pos] >= order[pos-1]:
            pos += 1
        else:
            swap(order, pos, pos - 1)
            pos -= 1
            swaps += 1
            if swaps % SWAPS_PER_STEP == 0:
                yield

def heapsort(order):
    """In-place heap sort (max-heap), yields every SWAPS_PER_STEP swaps."""
    n = len(order)
    swaps = 0

    def sift_down(root, end):
        nonlocal swaps
        while 2 * root + 1 < end:
            child = 2 * root + 1
            if child + 1 < end and order[child] < order[child + 1]:
                child += 1
            if order[root] >= order[child]:
                return
            swap(order, root, child)
            root = child
            swaps += 1
            if swaps % SWAPS_PER_STEP == 0:
                yield

    for start in range(n // 2 - 1, -1, -1):
        yield from sift_down(start, n)
    for end in range(n - 1, 0, -1):
        swap(order, 0, end)
        swaps += 1
        if swaps % SWAPS_PER_STEP == 0:
            yield
        yield from sift_down(0, end)


def bitonic(order):
    yield from bitonicsort(order, 0, len(order), 1)


# looked up by name so a worker process can pick the algorithm (lambdas/closures can't be pickled)
ALGORITHMS = {
    "Gnome sort": gnomesort,
    "Quicksort": quicksort,
    "Heap sort": heapsort,
    "Bubble sort": bubblesort,
    "Cocktail sort": cocktailsort,
    "Radix sort": radixsort,
    "Bitonic sort": bitonic,
    "Bogosort": bogosort,
}


def animation(algo_name, do_shuffle, order):
    """Optionally shuffle, then run the named algorithm on order."""
    if do_shuffle:
        yield from shuffle(order)
    yield from ALGORITHMS[algo_name](order)


def sort_worker(shm_name, n, algo_name, do_shuffle, compiled, elapsed):
    """Process entry point: sort the n ints in shared memory at full speed, store the time taken in elapsed.

    compiled=True uses the Numba versions from sorts_fast when there is one for algo_name.
    """
    global SWAPS_PER_STEP
    SWAPS_PER_STEP = 10_000  # nobody waits on the yields here, so make them rare
    shm = SharedMemory(name=shm_name)
    # the opened block can be bigger than requested (Windows rounds up to whole pages), so cut it to n
    order = shm.buf[:n * 4].cast("i")
    try:
        if compiled:
            import sorts_fast  # numba is slow to import, only pay for it when needed
            compiled = algo_name in sorts_fast.ALGORITHMS
        if compiled:
            arr = np.ndarray((len(order),), dtype=np.intc, buffer=order)
            fast_sort = sorts_fast.ALGORITHMS[algo_name]
            # compile (or load from cache) on a throwaway array so it isn't counted in elapsed
            sorts_fast.shuffle(arr[:0].copy())
            fast_sort(np.arange(2, dtype=np.intc))
            start = time.perf_counter()
            if do_shuffle:
                sorts_fast.shuffle(arr)
            fast_sort(arr)
            elapsed.value = time.perf_counter() - start
            del arr
        else:
            start = time.perf_counter()
            deque(animation(algo_name, do_shuffle, order), maxlen=0)  # drain the generator
            elapsed.value = time.perf_counter() - start
    finally:
        order.release()
        shm.close()
