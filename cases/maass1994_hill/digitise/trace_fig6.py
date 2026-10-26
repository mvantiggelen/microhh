#!/usr/bin/env python3
"""Redigitise Maass & Schumann (1994) Fig. 6 (page 295) from a 300 dpi
rendering: solid line = case 4 DNS, dashed = Hudson's LDV (time mean).
    pdftoppm -r 300 -png -f 9 -l 9 maass1994.pdf hi9
    python3 trace_fig6.py hi9-09.png out_dir
Frames and ticks were located in the 300 dpi image (pixel rows/columns of
the long frame lines; tick marks checked: uu 0.05 at col 651.5, ww 0.005 /
0.010 at 1220 / 1396.5, uw -0.01 / 0.00 at 1180.5 / 1444)."""
import json, sys
import numpy as np
from PIL import Image
from scipy import ndimage

PANELS = {   # name: (left, right, top, bottom) frame pixels, x range at left/right
    'uu': ((383, 913, 466, 1003), (0.0, 0.10)),
    'ww': ((1042, 1574, 466, 1003), (0.0, 0.015)),
    'vv': ((390, 919, 1207, 1732), (0.0, 0.10)),
    'uw': ((1048, 1577, 1207, 1732), (-0.015, 0.0050)),
}


def trace(img, frame, xr, name):
    L, R, T, B = frame
    ins = 6
    sub = img[T + ins:B - 2, L + ins:R - ins].copy()
    if name == 'uw':                       # the zero line of the u'w' panel
        c0 = 1444 - (L + ins)
        sub[:, c0 - 4:c0 + 5] = False
    lab, n = ndimage.label(sub, structure=np.ones((3, 3)))
    sizes = ndimage.sum(sub, lab, range(1, n + 1))
    bbox = ndimage.find_objects(lab)
    ext = np.array([max(b[0].stop - b[0].start, b[1].stop - b[1].start) for b in bbox])
    solid_ids = [i + 1 for i in range(n) if ext[i] > 60]
    dash_ids = [i + 1 for i in range(n) if 6 <= ext[i] <= 60 and sizes[i] >= 12]
    solid = np.isin(lab, solid_ids)
    dash = np.isin(lab, dash_ids)
    X = lambda c: xr[0] + (c + L + ins - L) / (R - L) * (xr[1] - xr[0])
    E = lambda r: (B - (r + T + ins)) / (B - T)
    out = {}
    for key, m in (('sim', solid), ('exp', dash)):
        pts = []
        for r in range(m.shape[0]):
            cs = np.nonzero(m[r])[0]
            if cs.size:
                # several crossings in one row: keep each run's centre
                runs = np.split(cs, np.nonzero(np.diff(cs) > 2)[0] + 1)
                for rr in runs:
                    pts.append((E(r), X(rr.mean())))
        out[key] = pts
    return out, solid, dash


def main():
    img = np.array(Image.open(sys.argv[1]).convert('L')) < 140
    od = sys.argv[2]
    res = {}
    rgb = np.stack([np.where(img, 0, 255)] * 3, -1).astype(np.uint8)
    for name, (frame, xr) in PANELS.items():
        o, solid, dash = trace(img, frame, xr, name)
        res[name] = o
        L, R, T, B = frame
        ins = 6
        sl = (slice(T + ins, B - 2), slice(L + ins, R - ins))
        rgb[sl][solid] = (40, 120, 220)
        rgb[sl][dash] = (235, 104, 52)
    Image.fromarray(rgb).crop((300, 400, 1650, 1800)).save(f'{od}/fig6_trace_overlay.png')
    json.dump(res, open(f'{od}/fig6_raw.json', 'w'))
    for k, v in res.items():
        print(k, 'sim', len(v['sim']), 'exp', len(v['exp']))


if __name__ == '__main__':
    main()
