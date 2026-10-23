# Can carrier counter: shelf_carriers.png

## Result for this photo

| Stack | Model count | Hand reference |
|---|---|---|
| Top-left | 34 | 34 |
| Top-right | 31 | 31 |
| Bottom-left (mostly black) | 24 | 25 |
| Bottom-right | 31 | 31 |
| **Total** | **120** | **121** |

121 carriers × 4 = room for about 484 cans.

Per colour (hand reference, all 4 stacks): black 43, white 24, pink 9, orange 8,
silver 7, yellow 6, blue 5, tan 4, amber 4, green 3, dark green 3, navy 2, red 2,
light green 1.

Per colour (model, each stack predicted by a classifier that never saw that
stack): black 45, white 23, pink 10, orange 9, silver 8, yellow 7, green 4, tan 4,
blue 4, dark green 2, navy 2, amber 2.

`annotated.png` marks every detected carrier and its predicted colour.

## How it works

Each carrier, seen edge-on, shows one dark slot row along its bottom edge, so
the model counts slot rows. Per stack: brightness profile through the slot
columns, carrier pitch from the profile spectrum, then dynamic programming picks
the slot rows (dark, about one pitch apart). Colour = median of the plain wall
pixels between slots, named by a 3-nearest-neighbour classifier in Lab colour
space. The four stack regions are drawn by hand in `stacks.py`.

## How accurate it is

- **Counting, synthetic test set** (300 rendered stacks never used for tuning,
  15-40 carriers, 3.6-5.6 px per carrier): 91% exact, 98.7% within ±1,
  mean error 0.12 carriers per stack (0.5%).
- **Counting, this photo**: 120 vs 121. Treat this as a sanity check, not a
  test: the hand reference was made by reviewing the detector's own marks on
  8x zooms, and at ~4-5 px per carrier the reference is itself uncertain by
  about ±1-2 per stack (±3 for the black bottom-left stack).
- **Colour, leave-one-stack-out on this photo**: 85.8% (103/120). Errors are
  near shades: navy or dark green read as black, amber read as yellow/orange,
  and light green, which appears only once, cannot be learned from the others.

## Limits

One low-resolution photo (374×418) cannot train a model that generalises.
The counter's settings were tuned on synthetic stacks; colours were learned
from 120 carriers in this one picture. To use this on new photos you need a
stack detector (instead of hand-drawn regions) and more labelled photos;
30-100 shelf photos at full phone resolution would allow a proper detector and
a real held-out accuracy number.

## Using it on a new photo

1. Install once: `pip install numpy pillow scipy`
2. Take the photo straight on, with each stack fully in frame, like the first one.
3. `python3 count.py my_photo.jpg` writes `my_photo_grid.png`: the photo with a
   pixel ruler and red suggested boxes around the stacks.
4. Check the boxes. Each one needs its top edge on the top carrier and its
   bottom edge on the bottom carrier; the sides can be loose. Fix any that are
   off by reading the ruler (or the pixel position your image viewer shows).
5. `python3 count.py my_photo.jpg --box x0,y0,x1,y1 --box x0,y0,x1,y1 ...`
   (one `--box` per stack) prints counts per stack and colour, and writes
   `my_photo_counted.png` with every carrier marked.

Notes: the photo is shrunk to 418 px tall first, because the model was tuned
at 4-5 px per carrier; change it with `--height` if the stacks fill much more
or less of the frame. Boxes off by ±5 px on the sides and ±2 px at top/bottom
changed the counts on the test photo by at most 1. Colours are learned from
the first photo only, so different lighting will mostly hurt the colour split,
not the count.

## Files

`counter.py` pipeline · `stacks.py` stack regions · `reference.py` hand labels ·
`synth.py` synthetic data · `train_eval.py` tuning + count scores ·
`colour_eval.py` colour scores · `annotate.py` draws `annotated.png` · `count.py` command line for new photos ·
`autostack.py` box suggestions and slot/wall column finder · `model.json` tuned settings + colour examples.
Run: `pip install numpy pillow`, then `python3 train_eval.py && python3 colour_eval.py && python3 annotate.py`.
