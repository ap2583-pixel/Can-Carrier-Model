"""Held-out check of the whole auto pipeline: render unseen synthetic shelf scenes
(gen_test.py seeds nobody tuned on), count them with no hints, compare to truth.
  python3 eval_scenes.py [first_seed] [n_scenes]"""
import sys, json, subprocess, os
import numpy as np
from PIL import Image
from autostack import auto_boxes, refine, trim_ends
from counter import analyse_stack

m = json.load(open('model.json'))
first = int(sys.argv[1]) if len(sys.argv) > 1 else 2001
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0]); h = min(a[3], b[3]) - max(a[1], b[1])
    i = max(w, 0) * max(h, 0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

errs, missed, extra, tot = [], 0, 0, 0
for seed in range(first, first + N):
    subprocess.run(['python3', 'gen_test.py', str(seed)], check=True, capture_output=True)
    rgb = np.asarray(Image.open(f'test_scene_{seed}.png').convert('RGB')).astype(float); L = rgb.mean(2)
    truth = json.load(open(f'test_scene_{seed}.json'))
    found = []
    boxes, loose = auto_boxes(rgb)
    for b in boxes:
        try:
            st = refine(L, b)
            P, slots, _ = analyse_stack(rgb, st['slot'], st['solid'], st['top'], st['bottom'], *m['params'])
            lo, hi = trim_ends(L, st['slot'], slots, P) if loose else (0, len(slots))
            found.append((b, hi - lo))
        except Exception:
            found.append((b, 0))
    used = set()
    for t in truth:
        tot += 1
        best = max(((iou(b, t['box']), i) for i, (b, _) in enumerate(found) if i not in used), default=(0, -1))
        if best[0] < 0.5:
            missed += 1; continue
        used.add(best[1]); errs.append(found[best[1]][1] - t['n'])
    extra += len(found) - len(used)
    os.remove(f'test_scene_{seed}.png'); os.remove(f'test_scene_{seed}.json')
e = np.array(errs)
print(f'{N} scenes, {tot} stacks: found {len(e)}, missed {missed}, spurious boxes {extra}')
print(f'exact {np.mean(e == 0):.0%}  within1 {np.mean(abs(e) <= 1):.0%}  within3 {np.mean(abs(e) <= 3):.0%}  mean abs err {abs(e).mean():.2f}  bias {e.mean():+.2f}')
