"""Compose a synthetic shelf photo from unseen synthetic stacks, with known counts.
  python3 gen_test.py [seed]  ->  test_scene_<seed>.png + test_scene_<seed>.json"""
import sys, json
import numpy as np
from PIL import Image
from synth import make_stack

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1001
rng = np.random.default_rng(seed)
W, H = 420, 430
bg = np.zeros((H, W, 3)) + np.array([150, 140, 125]) * rng.uniform(0.8, 1.1)
bg += rng.normal(0, 4, bg.shape)
layout = [(3, 40, 205), (2, 250, 410)]   # (stacks on shelf, band top, shelf plank y)
truth = []
for ns, ytop, yplank in layout:
    x = int(rng.integers(10, 30))
    for _ in range(ns):
        while True:
            s = make_stack(rng)
            if s['bottom'] - s['top'] <= yplank - ytop - 8:
                break
        img = s['img'][s['top']:s['bottom'] + 1, 2:56]   # carriers only, no dark frame
        h, w = img.shape[:2]
        y = yplank - h
        if x + w > W:
            break
        bg[y:yplank, x:x + w] = img
        truth.append(dict(n=s['n'], box=[x, y, x + w, yplank]))
        x += w + int(rng.integers(15, 45))
    bg[yplank:yplank + 10] = np.array([95, 70, 45]) + rng.normal(0, 5, (10, W, 3))   # plank
bg *= np.linspace(1.08, 0.9, H)[:, None, None]                                      # light falloff
out = f'test_scene_{seed}'
Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8)).save(out + '.png')
json.dump(truth, open(out + '.json', 'w'), indent=1)
print(out, [t['n'] for t in truth], 'total', sum(t['n'] for t in truth))
