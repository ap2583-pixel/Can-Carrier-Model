"""Synthetic edge-on carrier stacks with known counts, used to tune and test
the counter. Rendered at 4x, then blurred/downsampled/noised to match the
real photo's ~4-5 px per carrier."""
import numpy as np

PALETTE = {  # typical RGB measured on the real photo
    'white': (240, 232, 220), 'black': (38, 33, 30), 'silver': (160, 150, 142),
    'green': (112, 158, 52), 'light green': (188, 210, 90), 'dark green': (62, 70, 50),
    'orange': (232, 128, 50), 'amber': (225, 165, 60), 'yellow': (240, 195, 70),
    'tan': (160, 125, 80), 'pink': (210, 100, 115), 'blue': (70, 120, 190),
    'navy': (50, 58, 82), 'red': (135, 40, 40),
}
NAMES = list(PALETTE)

def make_stack(rng, W=60):
    P = rng.uniform(3.6, 5.6)          # px per carrier at final scale
    n = int(rng.integers(15, 41))
    k = 4
    H = int(np.ceil((n * P + 8) * k))
    img = np.zeros((H, W * k, 3)) + rng.uniform(20, 90)   # background
    slot_cols = np.arange(8 * k, 14 * k)
    solid_cols = np.arange(24 * k, 40 * k)
    colours = []
    c = rng.choice(NAMES)
    for i in range(n):
        if rng.random() < 0.45:  # colours come in runs, like the real stacks
            c = rng.choice(NAMES, p=None if rng.random() < 0.5 else _black_heavy())
        colours.append(c)
    top = 4 * k
    for i, c in enumerate(colours):
        a = int(round(top + i * P * k)); b = int(round(top + (i + 1) * P * k))
        base = np.array(PALETTE[c], float) * rng.uniform(0.85, 1.12) + rng.normal(0, 8, 3)
        img[a:b, 2 * k:56 * k] = base
        sh = max(1, int(round(rng.uniform(0.25, 0.45) * (b - a))))
        img[b - sh:b, slot_cols] = rng.uniform(0, 25)               # slot opening
        img[b - 1:b, 2 * k:56 * k] *= rng.uniform(0.6, 0.95)        # seam
        if rng.random() < 0.08:                                      # occluded slot
            img[b - sh:b, slot_cols] = base
    img *= np.linspace(rng.uniform(0.8, 1.0), rng.uniform(0.9, 1.15), img.shape[0])[:, None, None]
    # blur + area downsample
    from numpy.lib.stride_tricks import sliding_window_view
    s = rng.uniform(0.6, 1.6) * k
    g = np.exp(-0.5 * (np.arange(-3 * int(s) - 1, 3 * int(s) + 2) / s) ** 2); g /= g.sum()
    img = np.apply_along_axis(lambda v: np.convolve(v, g, 'same'), 0, img)
    off = int(rng.integers(0, k))
    img = img[off:]
    h = img.shape[0] // k
    img = img[:h * k].reshape(h, k, W, k, 3).mean((1, 3))
    img += rng.normal(0, rng.uniform(2, 8), img.shape)
    img = np.clip(img, 0, 255)
    t = (top - off) / k
    return dict(img=img, n=n, colours=colours, P=P,
                slot=list(range(8, 14)), solid=list(range(24, 40)),
                top=int(round(t)), bottom=int(round(t + n * P)) - 1)

def _black_heavy():
    p = np.ones(len(NAMES)); p[NAMES.index('black')] = 6; p[NAMES.index('white')] = 3
    return p / p.sum()
