"""Tune the counter on synthetic stacks, test on held-out synthetic stacks,
then score it on the real photo against the hand reference."""
import json, numpy as np
from PIL import Image
from counter import analyse_stack, KNNColour
from synth import make_stack
from stacks import STACKS
from reference import REFERENCE

def run(stacks, lam, w, fp):
    err = []
    for s in stacks:
        _, sl, _ = analyse_stack(s['img'], s['slot'], s['solid'], s['top'], s['bottom'], lam, w, fp)
        err.append(len(sl) - s['n'])
    e = np.array(err)
    return dict(exact=float((e == 0).mean()), within1=float((abs(e) <= 1).mean()),
                mae=float(abs(e).mean()), mape=float(np.mean([abs(x) / s['n'] for x, s in zip(e, stacks)])))

if __name__ == '__main__':
    train = [make_stack(np.random.default_rng(i)) for i in range(300)]
    test = [make_stack(np.random.default_rng(10_000 + i)) for i in range(300)]
    grid = [(lam, w, fp) for lam in (1, 2, 3, 5, 8, 12, 20) for w in (5, 7, 9) for fp in (0, 2, 4)]
    scores = {g: run(train, *g) for g in grid}
    best = min(scores, key=lambda g: (scores[g]['mae'], -scores[g]['exact']))
    print('best', best)
    test_sc = run(test, *best)
    print('test', test_sc)

    rgb = np.asarray(Image.open('shelf_carriers.png').convert('RGB')).astype(float)
    real = {}
    for name, s in STACKS.items():
        _, sl, cols = analyse_stack(rgb, s['slot'], s['solid'], s['top'], s['bottom'], *best)
        real[name] = dict(slots=[int(v) for v in sl], cols=cols.tolist(), ref=len(REFERENCE[name]))
        print(name, 'model', len(sl), 'reference', len(REFERENCE[name]))
    json.dump(dict(best=best, train=scores[best], test=test_sc, real=real),
              open('results_count.json', 'w'), indent=1)
