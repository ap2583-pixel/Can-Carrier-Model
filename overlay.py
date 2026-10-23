import numpy as np, sys
from PIL import Image, ImageDraw
from counter import analyse_stack
from stacks import STACKS

def draw(img_path, name, out_path, lam=3.0, labels=None, k=8):
    img = Image.open(img_path).convert('RGB')
    rgb = np.asarray(img).astype(float)
    s = STACKS[name]
    P, slots, cols = analyse_stack(rgb, s['slot'], s['solid'], s['top'], s['bottom'], lam)
    x0 = min(s['slot'] + s['solid']) - 25; x1 = max(s['slot'] + s['solid']) + 6
    y0 = s['top'] - 4; y1 = s['bottom'] + 4
    c = img.crop((x0, y0, x1, y1)).resize(((x1 - x0) * k, (y1 - y0) * k), Image.BICUBIC)
    out = Image.new('RGB', (c.width + 190, c.height), (35, 35, 35)); out.paste(c, (190, 0))
    d = ImageDraw.Draw(out)
    prev = s['top'] - 1
    for j, (y, col) in enumerate(zip(slots, cols)):
        Ya, Yb = (prev + 1 - y0) * k, (y + 1 - y0) * k
        d.rectangle([150, Ya + 1, 185, Yb - 1], fill=tuple(int(v) for v in col))
        d.line([(150, Yb), (230, Yb)], fill=(255, 0, 255), width=2)
        txt = f'{j + 1} y{y}' + (f' {labels[j]}' if labels is not None else '')
        d.text((4, (Ya + Yb) // 2 - 5), txt, fill=(255, 255, 0))
        prev = y
    out.save(out_path)
    return P, slots, cols

if __name__ == '__main__':
    draw(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 3.0)
