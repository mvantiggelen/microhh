#!/usr/bin/env python3
"""
Several axelsen2009_lbc runs (e.g. --res coarse and --res fine) against the
Pasterze observations and the paper's LES, from their slope-averaged
profiles (<rundir>/figs/P1_lbc_metrics.json, written by
axelsen2009_lbc_analysis.py):

    python3 axelsen2009_lbc_compare.py --rundir runs/P1_lbc runs/P1_lbc_fine [--label "dz 10 m" "dz 1 m"]
            [--out figs/P1_lbc_compare.png]

Left block, as Axelsen and van Dop (2009) Fig. 7 (= Axelsen 2010 Fig. 3.7):
u/V, tau/V^2, b/B and F/(VB) against z/H <= 1.5 with the JD 212 mast (+- sd),
the JD 212 10:25 balloon ascent (ADAS) and the paper's LES. Right block:
the same wind and potential temperature in m/s and deg C against a log
height, with the Smeets et al. (1998) A1 mean profiles.
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAV = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
COLS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', nargs='+', required=True)
    ap.add_argument('--label', nargs='+', default=None)
    ap.add_argument('--out', default=os.path.join(HERE, 'figs', 'P1_lbc_compare.png'))
    a = ap.parse_args()
    O = json.load(open(os.path.join(HERE, 'observations_P1.json')))
    BAL = json.load(open(os.path.join(HERE, 'pasterze_balloon_1994.json')))
    f37 = O['axelsen']['fig3_7']
    runs = []
    for k, rd in enumerate(a.rundir):
        p = json.load(open(os.path.join(rd, 'case_parameters.json')))
        m = json.load(open(os.path.join(rd, 'figs', 'P1_lbc_metrics.json')))
        lab = a.label[k] if a.label else f"dx {p['dx']:g} m, dz {p['dz']:g} m"
        runs.append((p, m, lab + f" ({m['window'].split(' (')[0]}, {m['from_s']:.0f}-{m['to_s']:.0f} s)"))
    p0 = runs[0][0]
    V, B, H, th0, gam = p0['V'], p0['B'], p0['H'], p0['theta_a0'], p0['gamma']

    q = next(s for s in BAL['soundings'] if s['doy'] == 212 and s['launch'] == '10:25')
    up = np.array(q['ascent'])
    bz, bu, bth = (np.array(q[k])[up] for k in ('z_m', 'u_slope', 'theta_C'))
    m_ = (bz >= 70.) & (bz <= 550.)
    cf = np.polyfit(bz[m_], bth[m_], 1)
    bb = GRAV / th0 * (bth - np.polyval(cf, bz))

    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1.1, 1.1])
    ax = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    keys = [('u_over_V', 'u_s', V, 'u / V', '(a) downslope velocity'),
            ('tau_over_V2', 'tau', V**2, 'tau / V^2', '(b) momentum flux'),
            ('b_over_B', 'b', B, 'b / B', '(c) buoyancy'),
            ('F_over_VB', 'F', V * B, 'F / (VB)', '(d) buoyancy flux')]
    for axx, (key, mk, sc, xl, tl) in zip(ax, keys):
        e = f37[key]
        les = np.array(e['les'])
        axx.plot(les[:, 0], les[:, 1], '-', color='0.35', lw=1.3, label='Axelsen & van Dop (2009) LES')
        mm = e['mast']
        axx.errorbar([r['value'] for r in mm], [r['z_over_H'] for r in mm], xerr=[r['sd'] or 0. for r in mm],
                     fmt='s', ms=5, mfc='white', color='k', lw=0.8, label='Mast JD 212', zorder=6)
        if key == 'u_over_V':
            axx.plot(bu / V, bz / H, '+', color='k', ms=5, label='Balloon JD 212 10:25 (ascent)')
        if key == 'b_over_B':
            axx.plot(bb / B, bz / H, '+', color='k', ms=5, label='Balloon JD 212 10:25 (ascent)')
        for k, (p, m, lab) in enumerate(runs):
            if mk in ('u_s', 'b'):
                n, v = np.array(m['n']), np.array(m[mk], float)
            else:
                n = np.array(m['n_flux'])
                v = np.array([np.nan if x is None else x for x in m[mk]], float)
            g = np.isfinite(v) & (n / H <= 1.55)
            axx.plot(v[g] / sc, n[g] / H, 'o-' if mk in ('tau', 'F') else '-', ms=3, color=COLS[k], lw=2,
                     label=f'MicroHH {lab}')
            surf = {'tau': m.get('tau_s_over_V2'), 'F': m.get('Fs_over_VB')}.get(mk)
            if surf is not None:
                axx.plot(surf, 0., '>', color=COLS[k], ms=8, clip_on=False)
        axx.set_ylim(0, 1.5); axx.set_xlabel(xl); axx.set_ylabel('z / H'); axx.set_title(tl, fontsize=10)
        axx.grid(alpha=0.25)
    ax[2].set_xlim(-80, 10)
    ax[0].legend(fontsize=6.5, loc='upper right')

    # Dimensional, log height.
    a1 = fig.add_subplot(gs[:, 2])
    a2 = fig.add_subplot(gs[:, 3], sharey=a1)
    mm = f37['u_over_V']['mast']
    a1.errorbar([r['value'] * V for r in mm], [r['z_over_H'] * H for r in mm], xerr=[(r['sd'] or 0.) * V for r in mm],
                fmt='s', ms=5, mfc='white', color='k', lw=0.8, label='Mast JD 212', zorder=6)
    mm = f37['b_over_B']['mast']
    a2.errorbar([th0 - 273.15 + gam * r['z_over_H'] * H + r['value'] * B * th0 / GRAV for r in mm],
                [r['z_over_H'] * H for r in mm], xerr=[(r['sd'] or 0.) * B * th0 / GRAV for r in mm],
                fmt='s', ms=5, mfc='white', color='k', lw=0.8, label='Mast JD 212 (from b/B)', zorder=6)
    a1.plot(bu, bz, '+', color='k', ms=5, label='Balloon JD 212 10:25')
    a2.plot(bth, bz, '+', color='k', ms=5, label='Balloon JD 212 10:25')
    s = O['smeets']
    c = np.array(s['fig4d']['z_m_U_ms']); a1.plot(c[:, 1], c[:, 0], 'o-', color='tab:green', mfc='none', ms=4, lw=0.8,
                                                  label='Smeets et al. (1998) mast mean*')
    c = np.array(s['fig4d']['z_m_T_C']); a2.plot(c[:, 1], c[:, 0], 'o-', color='tab:green', mfc='none', ms=4, lw=0.8,
                                                 label='Smeets et al. (1998) mast mean (T)*')
    for k, (p, m, lab) in enumerate(runs):
        n, u, b = np.array(m['n']), np.array(m['u_s']), np.array(m['b'])
        g = n >= 0.2
        a1.plot(u[g], n[g], '-', color=COLS[k], lw=2, label=f'MicroHH {lab}')
        a2.plot(th0 - 273.15 + gam * n[g] + b[g] * th0 / GRAV, n[g], '-', color=COLS[k], lw=2)
    for axx in (a1, a2):
        axx.set_yscale('log'); axx.set_ylim(0.2, 300); axx.grid(alpha=0.25, which='both')
        axx.legend(fontsize=6.5, loc='upper left')
    a1.set_xlabel('along-slope wind (m/s)'); a2.set_xlabel('potential temperature (deg C)')
    a1.set_ylabel('height above the surface (m)')
    fig.suptitle('Axelsen P1 with open LBCs: MicroHH slope averages against Pasterze A1, 31 July 1994 (JD 212)\n'
                 '* Smeets et al. (1998): means over runs with no wind maximum below 13 m, not the P1 day',
                 fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=120)
    print('wrote', a.out)
    for p, m, lab in runs:
        print(f"  {lab}: u_max {m['umax_over_V']:.1f} V at {m['zj_over_H']:.2f} H, b lowest {m['b_lowest_over_B']:.0f} B, "
              f"z_inv {m['zinv_over_H']:.2f} H, b_s {m['bs_over_B']:.0f} B, tau_s {m['tau_s_over_V2']:.2f} V^2")


if __name__ == '__main__':
    main()
