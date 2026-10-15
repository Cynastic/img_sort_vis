import multiprocessing as mp
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pygame
from PIL import Image

import sorts

FRAME_BUDGET = 1 / 30  # seconds of sorting work per rendered frame


def hilbert_path(w, h):
    """Row-major slot indices of a w x h image, in Hilbert-curve order.

    Works for any size: walks the Hilbert curve of the smallest power-of-2
    square covering the image and keeps only the slots inside it.
    Vectorized, so it's fast even for multi-megapixel images.
    """
    n = 1 << max(w - 1, h - 1, 1).bit_length()
    y, x = np.divmod(np.arange(w * h, dtype=np.int64), w)
    d = np.zeros(w * h, dtype=np.int64)
    s = n // 2
    while s > 0:
        rx = (x & s) > 0
        ry = (y & s) > 0
        d += s * s * ((3 * rx.astype(np.int64)) ^ ry.astype(np.int64))
        # rotate the quadrant so the sub-curve is oriented correctly
        flip = ~ry & rx
        x = np.where(flip, n - 1 - x, x)
        y = np.where(flip, n - 1 - y, y)
        swap = ~ry
        x, y = np.where(swap, y, x), np.where(swap, x, y)
        s //= 2
    return np.argsort(d, kind="stable")


def run(img_path, algo_name, do_shuffle=True, realtime=True, compiled=True):
    """Show img_path and animate sorts.ALGORITHMS[algo_name] on its pixels.

    realtime=True sorts in a separate process at full speed while this window
    just samples the shared state each frame; realtime=False interleaves sorting
    and drawing on this thread (sorts.SWAPS_PER_STEP sets the pace).
    compiled=True (real-time only) runs the Numba versions from sorts_fast.
    """
    img = np.array(Image.open(img_path).convert("RGBX"))
    h, w, _ = img.shape
    pixels = img.view(np.uint32).ravel()   # one packed value per pixel, cheap to gather

    # Hilbert layout:
    #   path[k]     = row-major screen slot at step k of the curve
    #   path_pos[s] = curve step of row-major screen slot s (inverse of path)
    path = hilbert_path(w, h)
    path_pos = np.empty_like(path)
    path_pos[path] = np.arange(w * h)
    pixels_on_path = pixels[path]          # pixels re-read in curve order, done once

    # order[k] is the original curve position of the pixel shown at curve position k;
    # it lives in shared memory so a sorter process can permute it while we draw it
    shm = SharedMemory(create=True, size=w * h * np.dtype(np.intc).itemsize)
    order_view = np.ndarray((w * h,), dtype=np.intc, buffer=shm.buf)
    order_view[:] = np.arange(w * h)
    order = None   # memoryview for sorting on this thread (stepped mode only)
    proc = None
    try:
        pygame.init()
        desk_w, desk_h = pygame.display.get_desktop_sizes()[0]
        scale = min(0.8 * desk_w / w, 0.8 * desk_h / h)
        screen = pygame.display.set_mode((int(w * scale), int(h * scale)), pygame.RESIZABLE)
        title = f"Image sort visualizer - {algo_name}"
        pygame.display.set_caption(title)

        if realtime:
            elapsed = mp.Value("d", -1.0)
            proc = mp.Process(target=sorts.sort_worker, daemon=True,
                              args=(shm.name, w * h, algo_name, do_shuffle, compiled, elapsed))
            proc.start()
            steps = None
        else:
            order = shm.buf[:w * h * 4].cast("i")
            steps = sorts.animation(algo_name, do_shuffle, order)

        clock = pygame.time.Clock()
        caption = None
        sw, sh, visible = 0, 0, None
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                if event.type == pygame.QUIT:
                    running = False

            if steps is not None:
                deadline = time.perf_counter() + FRAME_BUDGET
                while time.perf_counter() < deadline:
                    if next(steps, StopIteration) is StopIteration:
                        steps = None
                        break
            else:
                clock.tick(60)  # nothing to step here, so don't spin a core redrawing

            if proc is not None:
                if proc.is_alive():
                    status = "sorting..."
                elif elapsed.value >= 0:
                    status = f"done in {elapsed.value:.2f} s"
                else:
                    status = "sorter crashed"
                if status != caption:
                    caption = status
                    pygame.display.set_caption(f"{title} - {status}")

            # only look up the pixels that are actually visible at the current window size;
            # visible holds the curve positions of those screen slots
            if screen.get_size() != (sw, sh):
                sw, sh = screen.get_size()
                slots = ((np.arange(sh) * h // sh)[:, None] * w + np.arange(sw) * w // sw).ravel()
                visible = path_pos[slots]
            frame = pygame.image.frombuffer(pixels_on_path[order_view[visible]], (sw, sh), "RGBX")
            screen.blit(frame, (0, 0))
            pygame.display.flip()
    finally:
        pygame.quit()
        if proc is not None:
            proc.terminate()
            proc.join()
        # every view into the buffer must be gone before the shared memory can be closed
        steps = None
        if order is not None:
            order.release()
        del order_view
        shm.close()
        shm.unlink()
