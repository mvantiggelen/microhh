#!/usr/bin/env python3
r"""
Along-slope momentum and heat budget of an axelsen2009_lbc run: where the
buoyancy force goes (acceleration, layer-thickening pressure, drag) and how
much of the surface cooling the flow has carried downslope.

    python3 axelsen2009_lbc_drag_budget.py --rundir RUN [--rundir RUN2 ...]
            [--label ...] [--times 6600,7200] [--ztop 40] [--out figs/drag_budget.png]

Needs, per run: case_parameters.json, dem.0000000, the time-averaged
terrain-following planes u_tavg<h>.xy.000.<t> and thl_tavg<h>.xy.000.<t>
([IB] tavg_heights) at the --times (the END times of the averaging
intervals), and taux_form_ib / taux_skin_ib xy crosses at the same times for
the direct drag diagnostics. Everything is averaged over y (periodic).

Method (README "Diagnosing the drag on the IB terrain"):

  b(x, h)  = g/theta_a0 (thl - thl_a(z_s + h)),  thl_a the initial ambient
  p_h(x,h) = -int_h^Z b dh'       hydrostatic pressure of the cold layer [m2/s2]

Column budget per unit ground area, steady, from the surface to Z (--ztop):

  d/dx int u^2 dz = -p_h,s dz_s/dx  - d/dx int p_h dz  - D
                    \____ drive ___/  \__ thickening __/  \ drag (skin + dynamic form)

  drive:  the cold-air pressure on the sloping surface; = tan(a) int(-b) dz,
          the katabatic force b sin(a) of the slope frame (Mahrt 1982).
  D:      the residual. Compare with the direct diagnostics:
          dyn. form = taux_form_ib/rho - p_h,s dz_s/dx   (the hydrostatic part
          taken out: on a slope the riser pressure contains the DRIVE)
          skin      = taux_skin_ib/rho

Heat: Q(x) = int u b dz, the downslope flux of buoyancy deficit. With no
adiabatic warming and no storage, Q = F_s (x - x_top); the infinite-slope
(Prandtl) state of Axelsen and van Dop (2009) needs Q_eq = 140 V B H
(4.4 m3/s3 for P1), i.e. a fetch of at least Q_eq/|F_s| = 3.6 km.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAV = 9.81
RHO = 1.226                  # rhoref near the floor at pbot = 1000 hPa, 284 K
Q_EQ_VBH = 140.1             # int u b dn of the paper's LES (Fig. 3.7) in V B H
TRAP = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


def budget(rd, times, ztop):
    p = json.load(open(os.path.join(rd, 'case_parameters.json')))
    it, jt, dx = p['itot'], p['jtot'], p['dx']
    hts = np.array(p['tavg_heights'], float)
    x = (np.arange(it) + 0.5) * dx
    zs = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jt, it).mean(0)

    def plane(name, t):
        return np.fromfile(os.path.join(rd, f'{name}.xy.000.{t:07d}')).reshape(jt, it).mean(0)

    u = np.array([np.mean([plane(f'u_tavg{int(h)}', t) for t in times], 0) for h in hts])
    th = np.array([np.mean([plane(f'thl_tavg{int(h)}', t) for t in times], 0) for h in hts])
    thl_a = p['theta_a0'] + p['gamma'] * (zs[None, :] + hts[:, None] - p['z_ref'])
    b = GRAV / p['theta_a0'] * (th - thl_a)

    m = hts <= ztop
    h = np.r_[0., hts[m]]
    uu = np.vstack([np.zeros(it), u[m]])                 # no slip at the surface
    bb = np.vstack([b[0], b[m]])                         # b(0-1 m) = b(1 m): a lower bound
    ph = np.array([[-TRAP(bb[k:, i], h[k:]) for k in range(len(h))] for i in range(it)]).T
    dzs = np.gradient(zs, dx)
    r = dict(p=p, x=x, zs=zs, u2=u[list(hts).index(2.)], umax=u.max(0),
             drive=-ph[0] * dzs,
             thick=-np.gradient(TRAP(ph, h, axis=0), dx),
             adv=np.gradient(TRAP(uu**2, h, axis=0), dx),
             Q=TRAP(uu * bb, h, axis=0), M=TRAP(uu, h, axis=0))
    r['drag'] = r['drive'] + r['thick'] - r['adv']
    try:
        dt = [t for t in times]
        r['form'] = np.mean([plane('taux_form_ib', t) for t in dt], 0) / RHO
        r['skin'] = np.mean([plane('taux_skin_ib', t) for t in dt], 0) / RHO
        r['dynform'] = r['form'] - ph[0] * dzs
    except FileNotFoundError:
        r['form'] = r['skin'] = r['dynform'] = None
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', nargs='+', required=True)
    ap.add_argument('--label', nargs='+', default=None)
    ap.add_argument('--times', default='6600,7200', help='end times of the tavg intervals to average')
    ap.add_argument('--ztop', type=float, default=40., help='top of the budget column [m above the surface]')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    times = [int(t) for t in a.times.split(',')]
    labs = a.label or [os.path.basename(os.path.normpath(d)) for d in a.rundir]
    runs = [(l, budget(d, times, a.ztop)) for l, d in zip(labs, a.rundir)]

    for lab, r in runs:
        p = r['p']
        s = (r['x'] > p['x_top'] + 50) & (r['x'] < p['x_foot'] - 40)
        H, V, B = p['H'], p['V'], p['B']
        q_eq = Q_EQ_VBH * V * B * H
        i = int((p['x_foot'] - 40) / p['dx'])
        print(f'== {lab} ({p["sgs"]}, dz {p["dz"]:g} m, slope {p["slope_length"]:g} m), '
              f'tavg intervals ending {times}')
        print('   slope mean [1e-3 m2/s2]: drive {:.1f}, thickening {:.1f}, acceleration {:.1f}, '
              'drag (residual) {:.1f}'.format(*(1e3 * r[k][s].mean() for k in ('drive', 'thick', 'adv', 'drag'))))
        if r['form'] is not None:
            print('   direct: taux_form/rho {:.1f} = hydrostatic {:.1f} + dynamic form {:.1f}; skin {:.2f}'.format(
                1e3 * r['form'][s].mean(), 1e3 * (r['form'] - r['dynform'])[s].mean(),
                1e3 * r['dynform'][s].mean(), 1e3 * r['skin'][s].mean()))
        print(f'   near the foot: u_max {r["umax"][i]:.2f} m/s ({r["umax"][i] / V:.1f} V), '
              f'-int u b dz {-r["Q"][i]:.3f} m3/s3 = {100 * abs(r["Q"][i]) / q_eq:.1f} % of the '
              f'infinite-slope state ({q_eq:.2f}); |F_s| x = {abs(p["Fs"]) * (r["x"][i] - p["x_top"]):.3f}; '
              f'fetch for the infinite-slope state >= {q_eq / abs(p["Fs"]) / 1e3:.1f} km')

    if a.out:
        cols = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
        for c, (lab, r) in zip(cols, runs):
            p = r['p']
            s = (r['x'] > p['x_top']) & (r['x'] < p['x_foot'])
            d = r['x'][s] - p['x_top']
            ax[0].plot(d, r['u2'][s], color=c, lw=2, label=lab)
            ax[1].plot(d, -r['Q'][s], color=c, lw=2, label=lab)
        p = runs[0][1]['p']
        q_eq = Q_EQ_VBH * p['V'] * p['B'] * p['H']
        ax[0].axhline(16 * p['V'], color='0.35', ls='--', lw=1.2, label='paper LES jet max (16 V)')
        xx = np.linspace(0, 1.25 * q_eq / abs(p['Fs']), 50)
        ax[1].plot(xx, abs(p['Fs']) * xx, ':', color='0.35', label='|F_s| x')
        ax[1].axhline(q_eq, color='0.35', ls='--', lw=1.2, label='infinite-slope state (paper LES)')
        r = runs[0][1]
        s = (r['x'] > p['x_top'] + 20) & (r['x'] < p['x_foot'] - 30)
        k = np.ones(5) / 5
        for key, lab, c, sg in (('drive', 'buoyancy drive', cols[0], 1), ('adv', 'acceleration', cols[2], 1),
                                ('thick', 'thickening pressure (opposes)', cols[3], -1),
                                ('drag', 'drag (residual)', cols[1], 1)):
            ax[2].plot(r['x'][s] - p['x_top'], 1e3 * sg * np.convolve(r[key][s], k, 'same'), color=c, lw=2, label=lab)
        for i, (xl, yl, tl) in enumerate([('distance down the slope (m)', 'u at 2 m (m/s)', '(a) wind at 2 m'),
                                          ('distance down the slope (m)', '-int u b dz (m3/s3)', '(b) cold-air flux'),
                                          ('distance down the slope (m)', '1e-3 m2/s2 per ground area',
                                           f'(c) momentum budget, {runs[0][0]}')]):
            ax[i].set_xlabel(xl); ax[i].set_ylabel(yl); ax[i].set_title(tl, fontsize=10, loc='left')
            ax[i].grid(alpha=0.25); ax[i].legend(fontsize=7.5)
        fig.tight_layout()
        fig.savefig(a.out, dpi=130)
        print('wrote', a.out)


if __name__ == '__main__':
    main()
