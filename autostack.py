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
    # slots are the columns with the most dark ridges (colour-band edges give texture too)
    bp = nd.gaussian_filter1d(sub, 0.8, axis=0) - nd.gaussian_filter1d(sub, 3, axis=0)
    rid = ((bp <= nd.minimum_filter1d(bp, 3, axis=0)) & (bp < -8))[5:-5].mean(0)
    rid = nd.uniform_filter1d(rid, 3)
    c = lo + int(np.argmax(rid[lo:half]))
    slot = [x for x in range(c - 3, c + 4) if lo <= x < hi and rid[x] > 0.7 * rid[c]]
    # wall columns: longest run of medium texture (plain plastic, not slots or background)
    med = (score > 0.25 * s_in.max()) & (score < 0.65 * s_in.max())
    med[:lo] = False; med[hi:] = False
    runs = _runs(med, 4) or [(lo, lo + 4)]
    a, b = max(runs, key=lambda r: r[1] - r[0])
    solid = list(range(a + 1, b - 1)) or list(range(a, b))
    top, bottom = y0, y1 - 1
    return dict(slot=[x0 + x for x in slot], solid=[x0 + x for x in solid],
                top=int(top), bottom=int(bottom), box=box)

def _ridge_boxes(L, thresh=3.0, frac=0.5):
    """Stack boxes with no hand input: find_boxes gives the shelf bands and
    column runs, then each box is narrowed to its stack's columns and to the
    longest run of rows that still has dense slot ridges."""
    D = ridge_density(L)
    out = []
    for x0, y0, x1, y1 in find_boxes(L, thresh):
        mask = nd.binary_opening(D[y0:y1, x0:x1] > thresh, np.ones((5, 5)))
        col = nd.uniform_filter1d(mask.mean(0), 5)
        xs = np.where(col > 0.35 * col.max())[0]
        prof = nd.uniform_filter1d(mask[:, xs[0]:xs[-1] + 1].mean(1), 7)
        runs = _runs(prof > frac * prof.max(), 20)
        if not runs:
            continue
        r = max(runs, key=lambda r: r[1] - r[0])
        out.append((int(x0 + xs[0]), int(y0 + r[0]), int(x0 + xs[-1] + 1), int(y0 + r[1])))
    return out

def _blob_boxes(rgb, dist=35, min_area=600):
    """Stacks as foreground blobs: pixels far from their row's median colour
    (the wall/shelf behind). Works when the background is mostly plain."""
    d = np.sqrt(((rgb - np.median(rgb, axis=1, keepdims=True)) ** 2).sum(2))
    m = nd.binary_opening(nd.binary_closing(d > dist, np.ones((9, 9))), np.ones((7, 7)))
    lab, _ = nd.label(m)
    boxes = [(s[1].start, s[0].start, s[1].stop, s[0].stop) for s in nd.find_objects(lab)]
    # a carrier close to the wall's colour splits its stack: rejoin pieces that
    # share their columns and sit within `gap` rows of each other
    gap, merged = 35, True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ov = min(a[2], b[2]) - max(a[0], b[0])
                if ov >= 0.7 * min(a[2] - a[0], b[2] - b[0]) and max(a[1], b[1]) - min(a[3], b[3]) <= gap:
                    boxes[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= min_area]

def auto_boxes(rgb):
    """rgb: HxWx3 float. Returns (boxes, loose): loose boxes (ridge method) may
    spill past the stack's ends and need trim_ends. Foreground blobs when they look like separate stacks
    (none covers over a third of the photo); otherwise slot-ridge density."""
    H, W = rgb.shape[:2]
    blobs = _blob_boxes(rgb)
    if blobs and all((x1 - x0) * (y1 - y0) < H * W / 3 for x0, y0, x1, y1 in blobs):
        return blobs, False
    return _ridge_boxes(rgb.mean(2)), True

def trim_ends(L, slot_cols, slots, P, k=0.35, span=3):
    """(lo, hi) slice of `slots` to keep. A slot at either end that is barely
    darker than its surroundings is the shelf or background, not a carrier."""
    p = L[:, slot_cols].mean(1)
    q = np.array([p[r] for r in slots]) - np.array([nd.uniform_filter1d(p, int(2 * round(P)) | 1)[r] for r in slots])
    neg = q[q < 0]
    if not len(neg):
        return 0, len(slots)
    weak = q > -k * np.median(-neg)
    n = len(slots)
    hi = n
    for i in range(max(0, n - span), n):
        if weak[i] and (i == n - 1 or weak[i + 1]):
            hi = i
            break
    lo = 0
    for i in range(min(span, n) - 1, -1, -1):
        if weak[i] and (i == 0 or weak[i - 1]):
            lo = i + 1
            break
    return lo, max(hi, lo)
