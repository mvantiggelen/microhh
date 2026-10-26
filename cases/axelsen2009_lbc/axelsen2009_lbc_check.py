#!/usr/bin/env python3
"""
Quick look at an axelsen2009_lbc run from its binary crosses:

    python3 axelsen2009_lbc_check.py --rundir runs/P1_lbc [--time 3600] [--out check.png]

Panels
  a  xz section (middle of the domain): thl - thl_a(z), the departure from the
     resting environment, with u contours, at --time (default: the last one)
  b  along-slope wind u_s = (u cos a + ... ) ~ u/cos(alpha) at heights h above
     the local surface (u_tf<h> crosses, mean over y) along x, at --time
  c  the same for thl - thl_a(z_s + h)
  d  time series over the middle half of the slope: max over h of mean u_tf,
     and thl deficit at the lowest tf height; reference: the paper's LES u_max = 16 V
     (Axelsen and van Dop 2009, P1)
Also printed: the sponge budget (u and thl departure in the west/east
sponges), and the y-uniformity of the near-surface flow.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--rundir', default='.')
ap.add_argument('--time', type=int, default=None)
ap.add_argument('--out', default=None)
a = ap.parse_args()
rd = a.rundir
p = json.load(open(os.path.join(rd, 'case_parameters.json')))
it, jt, kt, dx, dz = p['itot'], p['jtot'], p['ktot'], p['dx'], p['dz']
alpha = np.deg2rad(p['alpha_deg'])
x = (np.arange(it) + 0.5) * dx
xh = np.arange(it) * dx
z = (np.arange(kt) + 0.5) * dz
dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jt, it)
zs = dem.mean(axis=0)
thl_a = lambda zz: p['theta_a0'] + p['gamma'] * (zz - p['z_ref'])
# The paper's LES jet maximum (reference_axelsen2009.json: P1 u_max = 16 V).
UMAX_REF = 16.
for d in (os.path.dirname(os.path.abspath(__file__)),
          os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'axelsen2009')):
    f = os.path.join(d, 'reference_axelsen2009.json')
    if os.path.exists(f):
        UMAX_REF = json.load(open(f))['cases'][p.get('case', 'P1')]['les']['umax_over_V']
        break


def times(pattern):
    fs = glob.glob(os.path.join(rd, pattern))
    return sorted({int(f.rsplit('.', 1)[1]) for f in fs if f.rsplit('.', 1)[1].isdigit()})


def xz(var, t):
    f = glob.glob(os.path.join(rd, f'{var}.xz.*.%07d' % t))
    return np.fromfile(sorted(f)[0]).reshape(kt, it) if f else None


def tf(var, h, t):
    f = os.path.join(rd, f'{var}_tf{h}.xy.000.%07d' % t)
    return np.fromfile(f).reshape(jt, it) if os.path.exists(f) else None


ts = times('thl.xz.*')
if not ts:
    raise SystemExit(f'no thl.xz crosses in {rd}')
t = a.time if a.time is not None else ts[-1]
hs = [h for h in p['tf_heights'] if tf('u', h, t) is not None]

fig, ax = plt.subplots(2, 2, figsize=(13, 8.5), constrained_layout=True)

# a: xz section.
thl = xz('thl', t)
u = xz('u', t)
solid = z[:, None] <= zs[None, :]
dth = np.ma.masked_where(solid, thl - thl_a(z)[:, None])
ztop = min(p['z_top_terrain'] + 300., kt * dz)
lim = max(0.5, float(np.abs(dth).max()))
pc = ax[0, 0].pcolormesh(x / 1e3, z, dth, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
cs = ax[0, 0].contour(xh / 1e3, z, np.ma.masked_where(solid, u), levels=[-1, -0.5, -0.25, 0.25, 0.5, 1, 2],
                      colors='k', linewidths=0.6)
ax[0, 0].clabel(cs, fmt='%g', fontsize=7)
ax[0, 0].fill_between(x / 1e3, 0, zs, color='0.6', step='mid')
for xs in (p['n_sponge'] * dx, p['lx'] - p['n_sponge'] * dx):
    ax[0, 0].axvline(xs / 1e3, color='m', ls=':', lw=1)
ax[0, 0].set_ylim(0, ztop)
ax[0, 0].set_xlabel('x (km)')
ax[0, 0].set_ylabel('z (m)')
ax[0, 0].set_title(f'a  t = {t} s: thl - thl_a(z) (colour, K), u (contours, m/s); sponges dotted')
fig.colorbar(pc, ax=ax[0, 0], shrink=0.8)

# b, c: along x at tf heights.
cmap = plt.get_cmap('viridis')
for n, h in enumerate(hs):
    col = cmap(n / max(1, len(hs) - 1))
    us = tf('u', h, t).mean(axis=0) / np.cos(alpha)
    th = tf('thl', h, t).mean(axis=0) - thl_a(zs + h)
    ax[0, 1].plot(x / 1e3, us, color=col, label=f'{h} m')
    ax[1, 0].plot(x / 1e3, th, color=col, label=f'{h} m')
for axx, lab in ((ax[0, 1], 'u/cos(alpha) (m/s)'), (ax[1, 0], 'thl - thl_a(z_s + h) (K)')):
    for xs in (p['x_top'], p['x_foot']):
        axx.axvline(xs / 1e3, color='0.5', lw=0.8)
    for xs in (p['n_sponge'] * dx, p['lx'] - p['n_sponge'] * dx):
        axx.axvline(xs / 1e3, color='m', ls=':', lw=1)
    axx.set_xlabel('x (km)')
    axx.set_ylabel(lab)
    axx.axhline(0, color='k', lw=0.5)
ax[0, 1].axhline(UMAX_REF * p['V'], color='r', ls='--', lw=0.8, label=f'paper LES u_max = {UMAX_REF:g} V')
ax[0, 1].legend(fontsize=7, ncol=2)
ax[0, 1].set_title(f'b  t = {t} s: wind at h above the surface (mean over y)')
ax[1, 0].set_title('c  temperature departure at h above the surface')

# d: time series over the middle half of the slope.
mid = (x > p['x_top'] + 0.25 * p['slope_length']) & (x < p['x_foot'] - 0.25 * p['slope_length'])
tts = times(f'u_tf{hs[0]}.xy.*')
umax, dth0, hmax = [], [], []
for tt in tts:
    prof = [tf('u', h, tt)[:, mid].mean() / np.cos(alpha) for h in hs]
    k = int(np.argmax(prof))
    umax.append(prof[k])
    hmax.append(hs[k])
    dth0.append((tf('thl', hs[0], tt)[:, mid] - thl_a(zs[mid] + hs[0])).mean())
ax[1, 1].plot(np.array(tts) / 3600., umax, 'b-o', ms=3, label='max_h mean u_s (m/s)')
ax[1, 1].axhline(UMAX_REF * p['V'], color='b', ls='--', lw=0.8)
ax2 = ax[1, 1].twinx()
ax2.plot(np.array(tts) / 3600., dth0, 'r-s', ms=3, label=f'thl deficit at {hs[0]} m (K)')
ax2.set_ylabel('K', color='r')
ax[1, 1].set_xlabel('t (h)')
ax[1, 1].set_ylabel('m/s', color='b')
ax[1, 1].set_title(f'd  middle half of the slope; tau_osc = {p["tau_osc"]/3600.:.2f} h')
ax[1, 1].legend(loc='upper left', fontsize=8)
ax2.legend(loc='lower left', fontsize=8)

out = a.out or os.path.join(rd, f'check_{t:07d}.png')
fig.savefig(out, dpi=110)
print(f'wrote {out}')

# Printed diagnostics.
print(f't = {t} s')
for tt, um, hm, d0 in zip(tts, umax, hmax, dth0):
    if tt % 1800 == 0 or tt == tts[-1]:
        print(f'  t {tt:6d} s: mid-slope max mean u_s {um:6.3f} m/s at h = {hm:3d} m, '
              f'thl deficit at {hs[0]} m {d0:7.3f} K')
h0 = hs[0]
u0 = tf('u', h0, t)
spread = np.abs(u0 - u0.mean(axis=0)).max()
print(f'  y-spread of u_tf{h0} (max |u - mean_y u|): {spread:.3f} m/s')
ns = p['n_sponge']
for name, sl in (('west sponge', np.s_[:ns]), ('east sponge', np.s_[it - ns:]),
                 ('valley (outside sponge)', np.s_[int(p['x_foot'] / dx) + 5: it - ns])):
    print(f'  {name:24s}: mean u_tf{h0} {u0[:, sl].mean():6.3f} m/s, '
          f'thl - thl_a {(tf("thl", h0, t)[:, sl] - thl_a(zs[sl] + h0)).mean():7.3f} K')
