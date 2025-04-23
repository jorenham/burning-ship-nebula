from pathlib import Path

import numba
import numpy as np
import numpy.typing as npt
from PIL import Image


@numba.jit(parallel=True, fastmath=True, error_model="numpy", cache=True)
def ship_escape(
    w: int = 1920,
    h: int = 1080,
    aa: int = 4,
    z0: float = -2.5 - 2j,
    z1: float = 1.5 + 0.75j,
    maxiter: int = 100,
    clip: int = 42,
    bail: float = 4.0,
) -> npt.NDArray[np.uint8]:
    """Burning Ship fractal."""
    assert w > 0
    assert w > 0
    assert aa > 0

    assert maxiter > 0
    assert clip > 0
    assert bail > 0

    x0, y0 = z0.real, z0.imag
    x1, y1 = z1.real, z1.imag

    assert x0 < x1
    assert y0 < y1

    dx = (x1 - x0) / w
    dy = (y1 - y0) / h

    # sub-pixel spacing
    ddx = dx / aa
    ddy = dy / aa

    # sub-pixel offset
    dx0 = ddx / 2
    dy0 = ddy / 2

    out = np.zeros((h, w), dtype=np.uint8)

    for i in numba.prange(h):
        yi = y0 + dy0 + dy * i

        for j in range(w):
            xi = x0 + dx0 + dx * j

            steps = 0
            count = 0
            for ki in range(aa):
                y = yi + ddy * ki

                for kj in range(aa):
                    x = xi + ddx * kj

                    zx = x
                    zy = y
                    zx2 = zx * zx
                    zy2 = zy * zy

                    step = 0
                    while zx2 + zy2 < bail:
                        zx, zy = x + zx2 - zy2, y + 2 * abs(zx * zy)

                        step += 1
                        if step < maxiter:
                            zx2 = zx * zx
                            zy2 = zy * zy
                        else:
                            # no escape
                            break

                    if step < maxiter:
                        count += 1
                        steps += step

            if count:
                nmax = clip * count
                out[i, j] = int(255 * min(nmax, steps) / nmax)

    return out


@numba.jit(parallel=True, fastmath=True, error_model="numpy", cache=True)
def ship_path(
    rng: np.random.Generator,
    w: int,
    h: int,
    max_iter: int = 200,  # max steps before escape
    n_reps: int = 100,  # number of warmup pulls
    bail: float = 42.0,
    z0: float = -2.4 - 1.5j,
    z1: float = 2.4 + 1.2j,
    phi: float = np.pi / 6,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    # npt.NDArray[np.complex128],
]:
    """Buddha Ship fractal."""
    x0, y0 = z0.real, z0.imag
    x1, y1 = z1.real, z1.imag

    dx = (x1 - x0) / w
    dy = (y1 - y0) / h

    # number of escaped iterations per pixel (total reward per arm)
    n_success = np.zeros((h, w), dtype=np.uint32)

    # traced path histogram
    hist_path = np.zeros((h, w), dtype=np.float64)
    # traced path origin histogram
    # hist_esc = np.zeros((h, w), dtype=np.complex128)

    cos_phi = np.cos(-phi)
    sin_phi = np.sin(-phi)

    ij2xy = np.empty((h, w, 2), np.float64)
    for i in numba.prange(h):
        for j in range(w):
            ijx = x0 + dx * j
            ijy = y0 + dy * i
            ij2xy[i, j] = ijx * cos_phi - ijy * sin_phi, ijx * sin_phi + ijy * cos_phi

    for _ in range(n_reps):
        # pre-calculate the sub-pixel offsets (`rng` is not threadsafe)
        ddx = rng.uniform(0, dx)
        ddy = rng.uniform(0, dy)
        ddx, ddy = ddx * cos_phi - ddy * sin_phi, ddx * sin_phi + ddy * cos_phi

        for i in numba.prange(h):
            for j in range(w):
                x, y = ij2xy[i, j]
                x = x + ddx
                y = y + ddy

                zs = [(x, y)]
                zx = x
                zy = y
                zx2 = zx * zx
                zy2 = zy * zy

                step = 0
                while zx2 + zy2 < bail and step < max_iter:
                    zx_next = x + zx2 - zy2
                    zy_next = y + 2 * abs(zx * zy)

                    if zx_next == zx and zy_next == zy:
                        # fast forward if stuck
                        step = max_iter
                        break

                    zx = zx_next
                    zy = zy_next
                    zs.append((zx, zy))
                    zx2 = zx * zx
                    zy2 = zy * zy

                    step += 1

                if step == max_iter:
                    continue

                # rasterize the points on the path (Wu's algorithm)
                success = False

                for px, py in zs[1:]:
                    # position relevive to `c` (escape direction)
                    # dc = complex(x - px, y - py)

                    # undo rotation, and map back to sub-pixel coordinates
                    pj = (px * cos_phi + py * sin_phi - x0) / dx
                    pi = (-px * sin_phi + py * cos_phi - y0) / dy

                    if not (0 <= pi < h and 0 <= pj < w):
                        continue

                    # floor
                    pi0 = int(pi)
                    pj0 = int(pj)

                    # floor error
                    di0 = pi - pi0
                    dj0 = pj - pj0

                    # ceil
                    pi1 = pi0 + 1
                    pj1 = pj0 + 1

                    # ceil error
                    di1 = 1 - di0
                    dj1 = 1 - dj0

                    if 0 <= pi0 < h:
                        # bottom-left
                        if 0 <= pj0 < w:
                            hist_path[pi0, pj0] += di1 * dj1
                            # hist_esc[pi0, pj0] += r00 * dc

                        # bottom-right
                        if 0 <= pj1 < w:
                            hist_path[pi0, pj1] += di0 * dj1
                            # hist_esc[pi0, pj1] += r10 * dc

                    if 0 <= pi1 < h:
                        # top-left
                        if 0 <= pj0 < w:
                            hist_path[pi1, pj0] += di1 * dj0
                            # hist_esc[pi1, pj0] += r01 * dc

                        # top-right
                        if 0 <= pj1 < w:
                            hist_path[pi1, pj1] += di0 * dj0
                            # hist_esc[pi1, pj1] += r11 * dc

                    success = True

                if success:
                    n_success[i, j] += 1

    # return (n_success / n_reps, hist_path, hist_esc)
    return n_success / n_reps, hist_path


@numba.jit(parallel=True, cache=True)
def color_path(hist_path: npt.NDArray[np.float64]) -> npt.NDArray[np.uint8]:
    """histogram coloroing"""

    values = np.unique(np.sort(np.floor(hist_path).astype(np.int64).ravel()))
    nvals = len(values)
    cmap = {v: i / nvals for i, v in enumerate(values)}

    h, w = hist_path.shape
    out = np.empty((h, w), dtype=np.uint8)
    for i in numba.prange(h):
        for j in range(w):
            c = cmap[int(hist_path[i, j])]
            out[i, j] = int(255 * c)
    return out


@numba.jit(parallel=True, cache=True)
def color_esc(
    hist_path: npt.NDArray[np.float64],
    hist_esc: npt.NDArray[np.complex128],
    saturation: int = 255,
) -> npt.NDArray[np.uint8]:
    tau = np.pi * 2
    h, w = hist_esc.shape

    # maxr = np.amax(np.abs(hist_esc))

    out_hsv = np.empty((h, w, 3), dtype=np.uint8)

    for i in numba.prange(h):
        for j in range(w):
            z = hist_esc[i, j]
            out_hsv[i, j, 0] = (int(np.angle(z) % tau / tau * 256) - 64) % 256
            # out_hsv[i, j, 1] = saturation
            # out_hsv[i, j, 2] = np.abs(z) / maxr

    # out_hsv[:, :, 0] = color_path(np.angle(hist_esc, deg=True) % 360)
    # out_hsv[:, :, 0] = (color_path(np.angle(hist_esc, deg=True) % 360) + 128) % 255

    # v1 = color_path(np.abs(hist_esc))
    # v2 = color_path(hist_path)
    out_hsv[:, :, 1] = saturation
    out_hsv[:, :, 2] = color_path(hist_path)
    # out_hsv[:, :, 2] = np.minimum(v1, v2)
    return out_hsv


def main() -> None:
    out = Path("./out")
    out.mkdir(exist_ok=True)

    seed = +31263274188

    K = 4
    # K = 1
    w, h = K * 960, K * 540

    n_rgb = 10_000, 500, 100
    t = 4000

    img_hist = np.zeros((h, w, 3), dtype=np.uint8)

    for k, n in enumerate(n_rgb):
        rng = np.random.default_rng(seed)
        # success, hist_path, _ = ship_path(rng, w, h, max_iter=n, n_reps=t)
        success, hist_path = ship_path(rng, w, h, max_iter=n, n_reps=t)

        fname_k = f"n{n}_t{t}_{K}k"

        img_hist[:, :, k] = img_path_k = color_path(hist_path)
        pil_path_k = Image.fromarray(img_path_k, mode="L")
        pil_path_k.save(out / f"{fname_k}_path.png", "png", optimize=True)
        pil_path_k.save(out / f"{fname_k}_path.webp", "webp", lossless=True)
        del pil_path_k

        np.savez_compressed(
            out / f"{fname_k}.npz",
            success=success,
            hist_path=hist_path,
            img_path=img_path_k,
        )

        del success
        del hist_path
        del img_path_k

    n_str = "-".join(map(str, n_rgb))
    fname = f"rgb_{n_str}_t{t}_{K}k"

    np.savez_compressed(out / f"{fname}.npz", img=img_hist)

    im = Image.fromarray(img_hist, mode="RGB")
    im.save(out / f"{fname}.png", "png", optimize=True)
    im.save(out / f"{fname}.webp", "webp", lossless=True)
    im.show()


# def main() -> None:
#     out = Path("./out_esc")
#     out.mkdir(exist_ok=True)

#     seed = +31263274188

#     # K = 4
#     K = 1
#     w, h = K * 960, K * 540

#     n = 1000
#     t = 1000

#     rng = np.random.default_rng(seed)
#     success, hist_path, hist_esc = ship_path(rng, w, h, max_iter=n, n_reps=t)

#     fname_k = f"n{n}_t{t}_{K}k"

#     img_esc_k = color_esc(hist_path, hist_esc)

#     np.savez_compressed(
#         out / f"{fname_k}.npz",
#         success=success,
#         hist_path=hist_esc,
#         hist_esc=hist_esc,
#         img_hsv=img_esc_k,
#     )

#     pil_esc_k = Image.fromarray(img_esc_k, mode="HSV").convert("RGB")
#     pil_esc_k.save(out / f"{fname_k}.png", "png", optimize=True)
#     pil_esc_k.save(out / f"{fname_k}.webp", "webp", lossless=True)
#     pil_esc_k.show()


if __name__ == "__main__":
    main()
