"""Find carrier stacks automatically and work out, for each one, the slot
columns, the plain-wall columns and the top/bottom rows that counter.py needs."""
import numpy as np
from scipy import ndimage as nd

def ridge_density(L):
    """How many dark horizontal ridges (slot rows) per 31 rows, per pixel."""
    bp = nd.gaussian_filter1d(L, 0.8, axis=0) - nd.gaussian_filter1d(L, 3, axis=0)
    mn = (bp <= nd.minimum_filter1d(bp, 3, axis=0)) & (bp < -8)
    return nd.uniform_filter(mn.astype(float), (31, 9)) * 31

def _runs(mask, min_len):
    out, start = [], None
    for i, v in enumerate(list(mask) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    return out

def find_boxes(L, thresh=3.5):
    """Rough stack boxes (x0, y0, x1, y1): shelves are row bands of high ridge
    density, stacks are column runs inside each band."""
    D = ridge_density(L)
    m = nd.binary_opening(D > thresh, np.ones((5, 5)))
    H, W = L.shape
    boxes = []
    for y0, y1 in _runs(m.mean(1) > 0.15, 40):
        col = m[y0:y1].mean(0)
        col = nd.uniform_filter1d(col, 5)
        for x0, x1 in _runs(col > 0.35 * col.max(), 30):
            boxes.append((x0, y0, x1, y1))
    return boxes

def refine(L, box):
    """From a box drawn around one stack (top edge on the top carrier, bottom
    edge on the bottom carrier; the sides can be loose), pick the slot columns
    and the plain-wall columns."""
    x0, y0, x1, y1 = box
    sub = L[y0:y1, x0:x1]
    hp = sub - nd.uniform_filter1d(sub, 9, axis=0)
    score = np.abs(hp[5:-5]).mean(0)                      # per-column texture
    inner = np.arange(x1 - x0)
    lo, hi = int(0.08 * len(inner)), int(0.92 * len(inner))
    s_in = score[lo:hi]
    # slot columns: the strongest-textured run in the left half of the stack
    half = lo + (hi - lo) // 2
    c = lo + int(np.argmax(score[lo:half]))
    slot = [x for x in range(c - 3, c + 4) if lo <= x < hi and score[x] > 0.8 * score[c]]
    # wall columns: longest run of medium texture (plain plastic, not slots or background)
    med = (score > 0.25 * s_in.max()) & (score < 0.65 * s_in.max())
    med[:lo] = False; med[hi:] = False
    runs = _runs(med, 4) or [(lo, lo + 4)]
    a, b = max(runs, key=lambda r: r[1] - r[0])
    solid = list(range(a + 1, b - 1)) or list(range(a, b))
    top, bottom = y0, y1 - 1
    return dict(slot=[x0 + x for x in slot], solid=[x0 + x for x in solid],
                top=int(top), bottom=int(bottom), box=box)
