"""Leave-one-stack-out test of the k-NN colour classifier on the real photo.
Detected carriers are matched to the hand reference by slot row (<=2 px)."""
import json, numpy as np
from collections import Counter
from PIL import Image
from counter import analyse_stack, KNNColour
from stacks import STACKS
from reference import REFERENCE

best = tuple(json.load(open('results_count.json'))['best'])
rgb = np.asarray(Image.open('shelf_carriers.png').convert('RGB')).astype(float)
det = {}
for name, s in STACKS.items():
    _, sl, cols = analyse_stack(rgb, s['slot'], s['solid'], s['top'], s['bottom'], *best)
    ref = REFERENCE[name]
    used, rows = set(), []
    for y, c in zip(sl, cols):
        j = min(range(len(ref)), key=lambda j: abs(ref[j][0] - y))
        if abs(ref[j][0] - y) <= 2 and j not in used:
            used.add(j); rows.append((y, c, ref[j][1]))
        else:
            rows.append((y, c, None))           # extra detection
    det[name] = dict(rows=rows, missed=[ref[j] for j in range(len(ref)) if j not in used])

report = {}
allp, allt = [], []
for test in STACKS:
    X = [c for n in STACKS if n != test for _, c, l in det[n]['rows'] if l]
    y = [l for n in STACKS if n != test for _, c, l in det[n]['rows'] if l]
    m = KNNColour(k=3).fit(np.array(X), y)
    rows = det[test]['rows']
    pred = m.predict(np.array([c for _, c, _ in rows]))
    pairs = [(p, l) for p, (_, _, l) in zip(pred, rows) if l]
    acc = np.mean([p == l for p, l in pairs])
    allp += [p for p, _ in pairs]; allt += [l for _, l in pairs]
    report[test] = dict(detected=len(rows), reference=len(REFERENCE[test]),
                        matched=len(pairs), extra=sum(l is None for _, _, l in rows),
                        missed=len(det[test]['missed']), colour_acc=round(float(acc), 3),
                        pred_counts=dict(Counter(pred.tolist())),
                        ref_counts=dict(Counter(c for _, c in REFERENCE[test])))
    print(test, {k: v for k, v in report[test].items() if 'counts' not in k})
    for p, l in pairs:
        if p != l:
            print('   ref', l, '-> pred', p)
report['overall_colour_acc'] = round(float(np.mean(np.array(allp) == np.array(allt))), 3)
print('overall LOSO colour accuracy', report['overall_colour_acc'], 'on', len(allt), 'carriers')
json.dump(report, open('results_colour.json', 'w'), indent=1)
