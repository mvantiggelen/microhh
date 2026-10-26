#!/usr/bin/env python3
"""Reduce the raw Fig. 6 traces (trace_fig6.py) to profiles: one x per
eta step (0.01 H), following the curve from row to row (nearest run to
the previous one), and drop tick marks from the dashed set."""
import json, sys
import numpy as np

raw = json.load(open('fig6_raw.json'))
XR = {'uu': (0, .10), 'ww': (0, .015), 'vv': (0, .10), 'uw': (-.015, .005)}
out = {}
for k, d in raw.items():
    span = XR[k][1] - XR[k][0]
    res = {}
    for key in ('sim', 'exp'):
        p = np.array(d[key])
        if not p.size:
            continue
        if key == 'exp':      # ticks: next to the left frame, on the bottom axis, the 0.05 tick
            keep = (p[:, 1] - XR[k][0] > 0.03 * span) & (p[:, 0] > 0.02) & (p[:, 0] < 0.5)
            p = p[keep]
        p[:, 0] = np.round(p[:, 0], 4)
        etas = np.unique(p[:, 0])
        prof, prev = [], None
        for e in sorted(etas, reverse=(k == 'uw' or True)):
            xs = p[p[:, 0] == e, 1]
            x = xs[np.argmin(np.abs(xs - prev))] if prev is not None else np.median(xs)
            prof.append((e, x)); prev = x
        prof = np.array(sorted(prof))
        grid = np.round(np.arange(0.0, 1.0001, 0.01), 3)
        g = grid[(grid >= prof[0, 0]) & (grid <= prof[-1, 0])]
        if key == 'exp':
            g = g[(g >= prof[0, 0] + 0.005) & (g <= prof[-1, 0] - 0.005)]
        # bin mean within +-0.005 of each grid eta
        vals = []
        for e in g:
            s = np.abs(prof[:, 0] - e) <= 0.005
            if s.any():
                vals.append([float(e), round(float(prof[s, 1].mean()), 5)])
        res['sim_case4' if key == 'sim' else 'exp_hudson'] = vals
    out[k] = res
json.dump(out, open('fig6_profiles.json', 'w'), indent=0)
for k, v in out.items():
    s = np.array(v['sim_case4'])
    i = np.argmax(np.abs(s[:, 1]))
    print(k, 'sim eta %.2f..%.2f, |max| %.4f at eta %.2f' % (s[0, 0], s[-1, 0], s[i, 1], s[i, 0]),
          'exp n=%d' % len(v.get('exp_hudson', [])))
