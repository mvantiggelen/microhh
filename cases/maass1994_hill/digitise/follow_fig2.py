#!/usr/bin/env python3
"""Fig. 2 DNS profiles by curve following: start on each profile's clean
upper part (z = 0.45) and walk down the dark pixels, 3 px per step,
choosing the continuation with the smallest change of direction (the
circles of Hudson's data branch off at large angles). Stops at the wavy
surface. Writes fig2_sim.json and fig2_sim_check.png."""
import json, sys
import numpy as np
from PIL import Image, ImageDraw

X0, SX, Z0, SZ, SKEW = 512.5, 1015.5, 962.5, 1025., 0.0045
img = np.array(Image.open(sys.argv[1]).convert('L')) < 150
xc = lambda c: (c - X0) / SX
zc = lambda c, r: (Z0 + SKEW * (c - 470) - r) / SZ
col = lambda x: X0 + x * SX
row = lambda c, z: Z0 + SKEW * (c - 470) - z * SZ
h = lambda x: 0.05 * np.cos(2 * np.pi * x)


def dark(r, c):
    r, c = int(round(r)), int(round(c))
    return img[r - 1:r + 2, c - 1:c + 2].sum() >= 2


def start(n):
    """the line's column at z = 0.45: the dark run in [x_n + 0.06, x_n + 0.16]
    that continues downward (rows 6 and 12 below are dark within 4 px)."""
    z = 0.45
    c0, c1 = int(col(0.1 * n + 0.06)), int(col(0.1 * n + 0.16))
    r = int(row(c0, z))
    cand = [c for c in range(c0, c1) if img[r, c]]
    best = None
    for c in cand:
        ok = sum(img[r + k, c - 5:c + 3].any() for k in (6, 12, 18))
        if ok == 3:
            best = c if best is None else best
    return r, best


def follow(n):
    r, c = start(n)
    pts = [(r, c)]
    d = np.array([1.0, -0.15]); d /= np.linalg.norm(d)        # downward, slightly left
    misses = 0
    for _ in range(2000):
        best, bscore = None, None
        for step in (3, 5, 8, 12, 16):
            for ang in np.radians(np.arange(-60, 61, 5)):
                ca, sa = np.cos(ang), np.sin(ang)
                v = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
                rr, cc = r + step * v[0], c + step * v[1]
                if dark(rr, cc):
                    s = abs(ang) + 0.02 * step
                    # near the wall the DNS line lies left of Hudson's circles
                    # (the paper: computed u too small near the wavy wall),
                    # so at a branch prefer the left one
                    if zc(c, r) < 0.15 and abs(ang) <= np.radians(45):
                        s += 0.03 * (cc - c)
                    if bscore is None or s < bscore:
                        best, bscore = (rr, cc, v), s
            if best:
                break
        if not best:
            break
        rr, cc, v = best
        d = 0.6 * d + 0.4 * v; d /= np.linalg.norm(d)
        r, c = rr, cc
        pts.append((r, c))
        x, z = xc(c), zc(c, r)
        if z <= h(x) + 0.002 or z < -0.06 or x < 0.1 * n - 0.03:
            break
    return [(xc(c), zc(c, r)) for r, c in pts]


out = {}
rgb = Image.open(sys.argv[1]).convert('RGB'); dr = ImageDraw.Draw(rgb)
for n in range(10):
    p = follow(n)
    out['%.1f' % (0.1 * n)] = [[round(z, 4), round((x - 0.1 * n) / 0.1, 3)] for x, z in p]
    for x, z in p:
        c = col(x); r = row(c, z)
        dr.ellipse((c - 2, r - 2, c + 2, r + 2), fill=(30, 170, 30))
    print('%.1f: %d pts, z %.3f..%.3f' % (0.1 * n, len(p), p[-1][1], p[0][1]))
rgb.crop((460, 430, 1600, 1030)).save('fig2_sim_check.png')
json.dump(out, open('fig2_sim.json', 'w'))
