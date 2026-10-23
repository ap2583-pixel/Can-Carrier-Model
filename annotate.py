"""Draw every detected carrier on a photo, coloured by the k-NN prediction."""
import argparse
import json
import os
import numpy as np
from collections import Counter
from PIL import Image, ImageDraw
from counter import analyse_stack, KNNColour
from synth import PALETTE

# Assuming these might still be needed if you don't pass external boxes, 
# but we will make them optional fallbacks.
try:
    from stacks import STACKS
    from reference import REFERENCE
except ImportError:
    STACKS, REFERENCE = {}, {}

def main():
    ap = argparse.ArgumentParser(description="Annotate carrier stacks dynamically.")
    ap.add_argument('photo', help="Path to the shelf image (e.g., photo.png)")
    ap.add_argument('--model', default='results_count.json', help="Path to the results/model JSON file")
    ap.add_argument('--scale', type=int, default=3, help="Upscale factor for the output canvas visualization")
    a = ap.parse_args()

    if not os.path.exists(a.photo):
        print(f"Error: Photo path '{a.photo}' does not exist.")
        return

    # 1. Load the dynamic photo and best parameters
    best = tuple(json.load(open(a.model))['best'])
    img = Image.open(a.photo).convert('RGB')
    rgb = np.asarray(img).astype(float)
    
    k = a.scale
    out = img.resize((img.width * k, img.height * k), Image.BICUBIC)
    canvas = Image.new('RGB', (out.width + 330, out.height), (250, 250, 250))
    canvas.paste(out, (0, 0))
    d = ImageDraw.Draw(canvas)
    total = Counter()
    y_txt = 10

    # 2. Iterate through your stacks dynamically
    for name, s in STACKS.items():
        _, sl, cols = analyse_stack(rgb, s['slot'], s['solid'], s['top'], s['bottom'], *best)
        
        # Classifier cross-validation training logic on remaining stacks
        feats, labs = [], []
        for n in STACKS:
            if n == name: 
                continue
            _, sl2, c2 = analyse_stack(rgb, STACKS[n]['slot'], STACKS[n]['solid'], STACKS[n]['top'], STACKS[n]['bottom'], *best)
            ref = dict(REFERENCE[n])
            for y, c in zip(sl2, c2):
                m = [r for r in ref if abs(r - y) <= 2]
                if m: 
                    feats.append(c)
                    labs.append(ref[m[0]])
                    
        if not feats:
            print(f"Skipping stack {name}: No training features found from other stacks.")
            continue

        pred = KNNColour(3).fit(np.array(feats), labs).predict(cols)
        x0 = min(s['slot']) * k - 30
        
        for y, p in zip(sl, pred):
            Y = y * k + 1
            d.line([(x0, Y), (x0 + 22, Y)], fill=(255, 0, 255), width=2)
            d.rectangle([x0 - 12, Y - 3 * k + 2, x0 - 2, Y - 1], fill=PALETTE[p], outline=(0, 0, 0))
            
        cnt = Counter(pred.tolist())
        total.update(cnt)
        
        # Draw text labels on the right-side panel
        d.text((out.width + 10, y_txt), f'{name}: {len(sl)} carriers', fill=(0, 0, 0))
        y_txt += 16
        for c, v in cnt.most_common():
            d.rectangle([out.width + 18, y_txt + 2, out.width + 28, y_txt + 11], fill=PALETTE[c], outline=(0, 0, 0))
            d.text((out.width + 34, y_txt), f'{c}: {v}', fill=(0, 0, 0))
            y_txt += 14
        y_txt += 8
        
    d.text((out.width + 10, y_txt), f'TOTAL: {sum(total.values())} carriers', fill=(0, 0, 0))
    
    # 3. Save output relative to input name
    output_name = os.path.splitext(a.photo)[0] + '_annotated.png'
    canvas.save(output_name)
    print(f"Saved annotation map to: {output_name}")
    print(sum(total.values()), total.most_common())

if __name__ == '__main__':
    main()
