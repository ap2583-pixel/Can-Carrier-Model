"""One call from photo to counts + annotated picture (used by server.py).

analyse(PIL image) finds the stacks, counts the carriers in each, names each
carrier's colour, and returns the marked-up photo."""
import json, os
from collections import Counter
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from counter import analyse_stack, KNNColour
from autostack import auto_boxes, refine, trim_ends
from synth import PALETTE

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = json.load(open(os.path.join(HERE, 'model.json')))
KNN = KNNColour(3).fit(np.array(MODEL['rgb']), MODEL['label'])
HEIGHT = 418          # the model was tuned at ~4-5 px per carrier at this photo height
MAX_OUT = 1600        # longest side of the annotated picture

def _font(size):
    for f in ('DejaVuSans-Bold.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()

def _reading_order(boxes):
    """Shelf by shelf, left to right. Boxes share a shelf when their row spans overlap."""
    shelves = []
    for b in sorted(boxes, key=lambda b: b[1]):
        for sh in shelves:
            lo, hi = min(x[1] for x in sh), max(x[3] for x in sh)
            if min(hi, b[3]) - max(lo, b[1]) > 0.3 * (b[3] - b[1]):
                sh.append(b); break
        else:
            shelves.append([b])
    return [b for sh in sorted(shelves, key=lambda sh: min(x[1] for x in sh)) for b in sorted(sh, key=lambda b: b[0])]

def analyse(img, height=HEIGHT):
    img = ImageOps.exif_transpose(img).convert('RGB')
    if max(img.size) > MAX_OUT:
        k = MAX_OUT / max(img.size)
        img = img.resize((round(img.width * k), round(img.height * k)), Image.LANCZOS)
    s = height / img.height if img.height > height else 1.0     # photo -> model scale
    small = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s != 1 else img
    rgb = np.asarray(small).astype(float); L = rgb.mean(2)
    boxes, loose = auto_boxes(rgb)
    boxes = _reading_order(boxes)

    out = img.copy().convert('RGBA'); ov = Image.new('RGBA', out.size, (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    fs = max(14, img.width // 40); font = _font(fs); lw = max(2, img.width // 350)
    stacks, total = [], Counter()
    for i, b in enumerate(boxes):
        try:
            st = refine(L, tuple(int(v) for v in b))
            P, slots, cols = analyse_stack(rgb, st['slot'], st['solid'], st['top'], st['bottom'], *MODEL['params'])
            if loose:
                lo, hi = trim_ends(L, st['slot'], slots, P)
                slots, cols = slots[lo:hi], cols[lo:hi]
        except Exception:
            continue
        if not len(slots):
            continue
        names = list(KNN.predict(cols))
        c = Counter(names); total.update(c)
        x0, y0, x1, y1 = [v / s for v in b]
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 255, 255), width=lw)
        for y, n in zip(slots, names):
            Y = (y + 0.5) / s; h = max(2, 1.6 / s)
            d.line([(x0, Y), (x1, Y)], fill=(255, 255, 255, 110), width=1)
            d.rectangle([x0 - 7 * h, Y - h, x0 - 2 * h, Y + h], fill=PALETTE[n] + (255,), outline=(0, 0, 0, 255))
        label = f'#{len(stacks) + 1}  {len(slots)}'
        tw = d.textlength(label, font=font)
        ty = max(0, y0 - fs - 8)
        d.rectangle([x0, ty, x0 + tw + 12, ty + fs + 6], fill=(255, 0, 255, 235))
        d.text((x0 + 6, ty + 2), label, fill=(255, 255, 255, 255), font=font)
        stacks.append(dict(id=len(stacks) + 1, count=len(slots), colours=dict(c.most_common())))
    n = sum(total.values())
    return dict(image=Image.alpha_composite(out, ov).convert('RGB'), stacks=stacks, total=n,
                cans=4 * n, colours=dict(total.most_common()),
                colour_rgb={k: PALETTE[k] for k in total})
