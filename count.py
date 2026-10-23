"""Count can carriers in a new shelf photo.

  python3 count.py PHOTO                     # step 1: writes PHOTO_grid.png with
                                             #   a pixel ruler and suggested boxes
  python3 count.py PHOTO --box x0,y0,x1,y1 [--box ...]   # step 2: count

One --box per stack, in the photo's own pixel coordinates. Put the top edge on
the top carrier and the bottom edge on the bottom carrier; the sides can be
loose. Writes PHOTO_counted.png and prints counts per stack and per colour.
"""
import argparse, json, os
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw
from counter import analyse_stack, KNNColour
from autostack import find_boxes, refine
from synth import PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))

def load(path, height):
    img = Image.open(path).convert('RGB')
    s = height / img.height if img.height > height else 1.0
    small = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s != 1 else img
    return img, small, s

def grid(path, img, boxes, s):
    g = img.copy(); d = ImageDraw.Draw(g)
    step = 100 if max(img.size) > 1500 else 50 if max(img.size) > 600 else 20
    for x in range(0, img.width, step):
        d.line([(x, 0), (x, img.height)], fill=(0, 255, 255), width=1); d.text((x + 2, 2), str(x), fill=(255, 0, 255))
    for y in range(0, img.height, step):
        d.line([(0, y), (img.width, y)], fill=(0, 255, 255), width=1); d.text((2, y + 2), str(y), fill=(255, 0, 255))
    for i, b in enumerate(boxes):
        B = [round(v / s) for v in b]
        d.rectangle(B, outline=(255, 0, 0), width=max(2, img.width // 300))
        d.text((B[0] + 4, B[1] + 4), f'suggested {i + 1}: {B[0]},{B[1]},{B[2]},{B[3]}', fill=(255, 0, 0))
    out = os.path.splitext(path)[0] + '_grid.png'; g.save(out); return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('photo')
    ap.add_argument('--box', action='append', default=[], help='x0,y0,x1,y1 around one stack')
    ap.add_argument('--height', type=int, default=418,
                    help='photo is shrunk to this height first (the model was built at ~4-5 px per carrier)')
    a = ap.parse_args()
    img, small, s = load(a.photo, a.height)
    if not a.box:
        L = np.asarray(small).astype(float).mean(2)
        out = grid(a.photo, img, find_boxes(L, 3.0), s)
        print(f'No --box given. Open {out}, check or fix the red suggestions '
              '(top edge on the top carrier, bottom edge on the bottom carrier), then rerun with --box.')
        return
    m = json.load(open(os.path.join(HERE, 'model.json')))
    knn = KNNColour(3).fit(np.array(m['rgb']), m['label'])
    rgb = np.asarray(small).astype(float); L = rgb.mean(2)
    out = img.copy(); d = ImageDraw.Draw(out)
    total = Counter()
    for i, b in enumerate(a.box):
        x0, y0, x1, y1 = [int(v) for v in b.split(',')]
        st = refine(L, tuple(round(v * s) for v in (x0, y0, x1, y1)))
        _, slots, cols = analyse_stack(rgb, st['slot'], st['solid'], st['top'], st['bottom'], *m['params'])
        names = knn.predict(cols) if len(cols) else []
        c = Counter(list(names)); total.update(c)
        print(f'stack {i + 1}: {len(slots)} carriers  ' + ', '.join(f'{k} {v}' for k, v in c.most_common()))
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 255), width=max(2, img.width // 400))
        xs = st['slot'][0] / s
        for y, n in zip(slots, names):
            Y = (y + 0.5) / s; h = max(3, 2 / s)
            d.rectangle([xs - 6 * h, Y - h, xs - 2 * h, Y + h], fill=PALETTE[n], outline=(0, 0, 0))
        d.text((x0 + 4, y0 - 14), f'{len(slots)}', fill=(255, 0, 255))
    n = sum(total.values())
    print(f'TOTAL: {n} carriers (room for {4 * n} cans)  ' + ', '.join(f'{k} {v}' for k, v in total.most_common()))
    p = os.path.splitext(a.photo)[0] + '_counted.png'; out.save(p); print('marked-up photo:', p)

if __name__ == '__main__':
    main()
