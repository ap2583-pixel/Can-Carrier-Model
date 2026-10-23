"""Draw every detected carrier on the photo, coloured by the k-NN prediction."""
import json, numpy as np
from collections import Counter
from PIL import Image, ImageDraw
from counter import analyse_stack, KNNColour
from stacks import STACKS
from reference import REFERENCE
from synth import PALETTE

best = tuple(json.load(open('results_count.json'))['best'])
img = Image.open('shelf_carriers.png').convert('RGB'); rgb = np.asarray(img).astype(float)
k = 3
out = img.resize((img.width * k, img.height * k), Image.BICUBIC)
canvas = Image.new('RGB', (out.width + 330, out.height), (250, 250, 250)); canvas.paste(out, (0, 0))
d = ImageDraw.Draw(canvas)
total = Counter(); y_txt = 10
for name, s in STACKS.items():
    _, sl, cols = analyse_stack(rgb, s['slot'], s['solid'], s['top'], s['bottom'], *best)
    # classifier trained on the other three stacks only, so this stack is unseen
    feats, labs = [], []
    for n in STACKS:
        if n == name: continue
        _, sl2, c2 = analyse_stack(rgb, STACKS[n]['slot'], STACKS[n]['solid'], STACKS[n]['top'], STACKS[n]['bottom'], *best)
        ref = dict(REFERENCE[n])
        for y, c in zip(sl2, c2):
            m = [r for r in ref if abs(r - y) <= 2]
            if m: feats.append(c); labs.append(ref[m[0]])
    pred = KNNColour(3).fit(np.array(feats), labs).predict(cols)
    x0 = min(s['slot']) * k - 30
    for y, p in zip(sl, pred):
        Y = y * k + 1
        d.line([(x0, Y), (x0 + 22, Y)], fill=(255, 0, 255), width=2)
        d.rectangle([x0 - 12, Y - 3 * k + 2, x0 - 2, Y - 1], fill=PALETTE[p], outline=(0, 0, 0))
    cnt = Counter(pred.tolist()); total.update(cnt)
    d.text((out.width + 10, y_txt), f'{name}: {len(sl)} carriers', fill=(0, 0, 0)); y_txt += 16
    for c, v in cnt.most_common():
        d.rectangle([out.width + 18, y_txt + 2, out.width + 28, y_txt + 11], fill=PALETTE[c], outline=(0, 0, 0))
        d.text((out.width + 34, y_txt), f'{c}: {v}', fill=(0, 0, 0)); y_txt += 14
    y_txt += 8
d.text((out.width + 10, y_txt), f'TOTAL: {sum(total.values())} carriers', fill=(0, 0, 0))
canvas.save('annotated.png')
print(sum(total.values()), total.most_common())
