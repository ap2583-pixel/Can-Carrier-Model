"""Can-carrier stack counter.

Carriers sit in stacks seen edge-on. Every carrier shows one dark "slot" row
(the gap through its side openings) at its bottom edge, so counting carriers is
counting slot rows. Pipeline per stack:
  1. take the vertical brightness profile through the slot columns
  2. estimate the carrier pitch (px per carrier) from the profile spectrum
  3. dynamic programming picks the slot rows: dark rows, spaced about one pitch
  4. each carrier's colour = median of the solid (non-slot) columns between slots
  5. a k-NN colour classifier (trained on hand-labelled carriers) names the colour
"""
import numpy as np

def period(p, lo=3.2, hi=6.5, w=5, fp=2):
    """Carrier pitch from the profile spectrum. w: detrend window; the power is
    weighted by freq**fp so slow colour changes do not win over the pitch."""
    h = w // 2
    q = p - np.convolve(p, np.ones(w) / w, 'same')
    q = q[h:-h] * np.hanning(len(p) - 2 * h)
    N = 4096
    fr = np.fft.rfftfreq(N)
    f = np.abs(np.fft.rfft(q, N)) ** 2 * fr ** fp
    m = (fr >= 1 / hi) & (fr <= 1 / lo)
    return 1 / fr[m][f[m].argmax()]

def dp_slots(p, P, lam=3.0, lo=0.6, hi=1.6):
    """p: profile from the stack's top edge to its bottom edge (low = slot).
    Returns slot row indices; the first slot is >= 0.5P below the top edge
    and the last one within 0.6P of the bottom edge."""
    n = len(p)
    # local contrast: a slot in a run of white carriers is still darker than
    # its neighbours even though it is brighter than the stack average
    w = int(2 * round(P)) | 1
    pad = np.pad(p, w, mode='reflect')
    q = p - np.convolve(pad, np.ones(w) / w, 'same')[w:-w]
    z = q / (q.std() + 1e-9)
    dmin = max(2, int(np.floor(lo * P)))
    dmax = int(np.ceil(hi * P))
    INF = 1e18
    cost = np.full(n, INF)
    prev = -np.ones(n, int)
    s0, s1 = int(round(0.5 * P)), int(round(1.6 * P))
    for i in range(n):
        if s0 <= i <= s1:
            cost[i] = z[i]
        for d in range(dmin, dmax + 1):
            j = i - d
            if j < 0:
                break
            if cost[j] >= INF:
                continue
            c = cost[j] + z[i] + lam * ((d - P) / P) ** 2
            if c < cost[i]:
                cost[i], prev[i] = c, j
    e = min(range(max(0, n - int(np.ceil(0.6 * P)) - 1), n), key=lambda i: cost[i])
    out = []
    while e >= 0:
        out.append(e)
        e = prev[e]
    return out[::-1]

def analyse_stack(rgb, slot_cols, solid_cols, top, bottom, lam=8.0, w=5, fp=2):
    """rgb: HxWx3 float image. Returns (pitch, slot_rows, carrier_colours)."""
    L = rgb.mean(2)
    p = L[top:bottom + 1][:, slot_cols].mean(1)
    P = period(p, w=w, fp=fp)
    slots = [top + i for i in dp_slots(p, P, lam)]
    cols = []
    prev = top - 1
    for s in slots:
        c = (prev + s) / 2
        r0, r1 = int(np.floor(c - 0.3 * P)), int(np.ceil(c + 0.3 * P))
        r0, r1 = max(r0, prev + 1), min(r1, s - 1)
        if r1 < r0:
            r0 = r1 = s - 1
        cols.append(np.median(rgb[r0:r1 + 1][:, solid_cols].reshape(-1, 3), 0))
        prev = s
    return P, slots, np.array(cols)

def rgb_to_lab(c):
    c = np.asarray(c, float) / 255
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)

class KNNColour:
    def __init__(self, k=3):
        self.k = k
    def fit(self, rgbs, labels):
        self.X = rgb_to_lab(rgbs)
        self.y = np.array(labels)
        return self
    def predict(self, rgbs):
        X = rgb_to_lab(rgbs)
        d = ((X[:, None, :] - self.X[None]) ** 2).sum(-1)
        out = []
        for row in d:
            nn = self.y[np.argsort(row)[:self.k]]
            vals, cnt = np.unique(nn, return_counts=True)
            out.append(vals[cnt.argmax()] if cnt.max() > 1 else nn[0])
        return np.array(out)
