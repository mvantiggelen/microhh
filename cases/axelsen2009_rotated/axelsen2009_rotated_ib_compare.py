#!/usr/bin/env python3
"""
Test T1: the rotated-frame P1 twins (native surface vs flat IB, written by
axelsen2009_rotated_ib_input.py) against each other, the paper's LES and,
optionally, the open-LBC run's slope average.

    python3 axelsen2009_rotated_ib_compare.py --native T1/native --ib T1/ib [T1_alloff/ib ...] [--label ...]
            [--lbc-metrics ../axelsen2009_lbc/runs/P1_lbc/figs/P1_lbc_metrics.json] [--tavg 1800]
            [--out T1/T1_compare.png]

From the [stats] profiles (horizontal means; ib/ level k_ib + k is native/
level k). In the rotated frame of [thermo] swthermo=buoy the buoyancy
force on u is +sin(alpha) b, so a cooled slope drives u < 0: the downslope
wind is u_s = -u. Scales as Axelsen and van Dop (2009): V = (|Fs|/N)^1/2,
B = |Fs|/V, H = V/N, tau_osc = 2 pi/(N sin alpha).

Panels: (a) u_s/V, (b) b/B, (c) total momentum flux (resolved + SFS)/V^2,
(d) SFS TKE and resolved TKE, all over the last --tavg s against n/H; (e)
the jet maximum u_max/V and (f) the first-level b/B against t/tau_osc.
Prints the numbers that matter for the IB question. With --momentum vr
(velocity reconstruction, patch 59) the [stats] flux AT the surface face is not the
wall stress: it is the SFS flux between the reconstructed first air point
and the zeroed point below the surface. Compare the flux one dz higher
(printed as "at n = dz"), where both runs carry the real flux.
"""
import argparse
import json
import os

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def load(rd):
    p = json.load(open(os.path.join(rd, 'case_parameters.json')))
    f = os.path.join(rd, 'axelsen2009_rotated.default.0000000.nc')
    d = nc.Dataset(f)
    k0 = p['k_ib']
    g, th = d.groups['default'], d.groups['thermo']
    S = dict(t=d['time'][:], z=d['z'][k0:] - k0 * p['dz'], zh=d['zh'][k0:] - k0 * p['dz'],
             u=-g['u'][:, k0:], b=th['b'][:, k0:], u2=g['u_2'][:, k0:], v2=g['v_2'][:, k0:],
             w2=g['w_2'][:, k0:], uflux=-g['u_flux'][:, k0:])
    if 'sgstke' in d.groups:                   # tke2: the prognostic SFS TKE
        S['sgstke'] = d.groups['sgstke']['sgstke'][:, k0:]
        S['sfs_how'] = 'SFS'
    else:
        # smag2 has no SFS TKE. Estimate it from the eddy viscosity as tke2
        # would relate them, e = (K / (c_m l))^2 with c_m = 0.1 (Deardorff
        # 1980) and l the Mason-damped length (n = 2) with the height above
        # the surface: l^-2 = Delta^-2 + (kappa z)^-2. An estimate only.
        delta = (p['dx'] * p['dx'] * p['dz'])**(1. / 3.)
        l = (1. / delta**2 + 1. / (0.4 * S['z'])**2)**-0.5
        S['sgstke'] = (np.asarray(g['evisc'][:, k0:]) / (0.1 * l[None, :]))**2
        S['sfs_how'] = 'SFS (estimate from evisc)'
    return p, S


def tmean(S, key, t0, t1=np.inf):
    m = (S['t'] >= t0) & (S['t'] <= t1 + 1e-6)
    return np.asarray(S[key][m]).mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--native', required=True)
    ap.add_argument('--ib', required=True, nargs='+', help='one or more flat-IB runs')
    ap.add_argument('--label', nargs='+', default=None, help='labels of the --ib runs')
    ap.add_argument('--lbc-metrics', default=None)
    ap.add_argument('--tavg', type=float, default=1800.)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    labs = a.label or ['flat IB, slope-aware switches'] + [os.path.basename(os.path.dirname(os.path.abspath(d))) for d in a.ib[1:]]
    runs = [('native surface',) + load(a.native)] + [(l,) + load(d) for l, d in zip(labs, a.ib)]
    p = runs[0][1]
    Fs, N, al = abs(p['Fs']), p['N'], np.deg2rad(p['alpha_deg'])
    V = np.sqrt(Fs / N); B = Fs / V; H = V / N
    tau = 2 * np.pi / (N * np.sin(al))
    tend = min(float(S['t'][-1]) for _, _, S in runs)
    t0 = tend - a.tavg
    # the digitised paper profiles live with the open-LBC case next door
    O = json.load(open(os.path.join(HERE, '..', 'axelsen2009_lbc', 'observations_P1.json')))['axelsen']['fig3_7']

    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    ax = ax.ravel()
    cols = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    for c, (lab, pp, S) in zip(cols, runs):
        z, zh = S['z'], S['zh']
        ax[0].plot(tmean(S, 'u', t0, tend) / V, z / H, color=c, lw=2, label=lab)
        ax[1].plot(tmean(S, 'b', t0, tend) / B, z / H, color=c, lw=2, label=lab)
        ax[2].plot(tmean(S, 'uflux', t0, tend) / V**2, zh / H, color=c, lw=2, label=lab)
        res = 0.5 * (tmean(S, 'u2', t0, tend) + tmean(S, 'v2', t0, tend) + 0.5 * (tmean(S, 'w2', t0, tend)[:-1] + tmean(S, 'w2', t0, tend)[1:]))
        ax[3].plot(res, z / H, color=c, lw=2, label=f'{lab}: resolved')
        ax[3].plot(tmean(S, 'sgstke', t0, tend), z / H, color=c, lw=1.2, ls='--', label=f"{lab}: {S['sfs_how']}")
        um = np.asarray(S['u']).max(axis=1)
        ax[4].plot(S['t'] / tau, um / V, color=c, lw=2, label=lab)
        ax[5].plot(S['t'] / tau, np.asarray(S['b'])[:, 0] / B, color=c, lw=2, label=lab)
    for i, key in ((0, 'u_over_V'), (1, 'b_over_B'), (2, 'tau_over_V2')):
        les = np.array(O[key]['les'])
        ax[i].plot(les[:, 0], les[:, 1], color='0.4', lw=1.2,
                   label='Axelsen & van Dop (2009) LES')
    if a.lbc_metrics:
        m = json.load(open(a.lbc_metrics))
        n = np.array(m['n'])
        ax[0].plot(np.array(m['u_s']) / V, n / H, 'k:', lw=2, label='open-LBC IB staircase, slope average')
        ax[1].plot(np.array(m['b']) / B, n / H, 'k:', lw=2, label='open-LBC IB staircase, slope average')
    for i, (xl, tl) in enumerate([('u_s / V', '(a) downslope wind'), ('b / B', '(b) buoyancy'),
                                  ('(u_s\'w\' + SFS) / V^2', '(c) momentum flux of u_s'),
                                  ('TKE (m2/s2)', '(d) resolved and SFS TKE')]):
        ax[i].set_ylim(0, 2.); ax[i].set_xlabel(xl); ax[i].set_ylabel('n / H'); ax[i].set_title(tl, fontsize=10)
        ax[i].grid(alpha=0.3)
    ax[3].set_xscale('log'); ax[3].set_xlim(1e-5, 2)
    ax[4].set_xlabel('t / tau_osc'); ax[4].set_ylabel('u_max / V'); ax[4].set_title('(e) jet maximum', fontsize=10)
    ax[5].set_xlabel('t / tau_osc'); ax[5].set_ylabel('b(first level) / B'); ax[5].set_title('(f) first-level buoyancy', fontsize=10)
    for i in (4, 5):
        ax[i].grid(alpha=0.3)
    ax[0].legend(fontsize=7); ax[3].legend(fontsize=6.5); ax[4].legend(fontsize=7)
    fig.suptitle(f'T1: Axelsen P1 in the rotated frame (alpha {p["alpha_deg"]:g} deg), native surface vs flat IB, '
                 f'dx {p["dx"]:g} m, dz {p["dz"]:g} m; profiles {t0:.0f}-{tend:.0f} s '
                 f'({t0 / tau:.2f}-{tend / tau:.2f} tau_osc)', fontsize=10)
    fig.tight_layout()
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.native)), 'T1_compare.png')
    fig.savefig(out, dpi=110)
    print('wrote', out)
    print(f'V {V:.3f} m/s, B {B:.5f} m/s2, H {H:.1f} m, tau_osc {tau:.0f} s; window {t0:.0f}-{tend:.0f} s')
    for lab, pp, S in runs:
        u = tmean(S, 'u', t0, tend); b = tmean(S, 'b', t0, tend); z = S['z']
        k = int(np.argmax(u))
        res = 0.5 * (tmean(S, 'u2', t0, tend) + tmean(S, 'v2', t0, tend))
        print(f'  {lab:32s} u_max {u[k] / V:5.1f} V at n = {z[k]:4.1f} m ({z[k] / H:.2f} H), '
              f'b(0.5 m) {b[0] / B:6.1f} B, momentum flux at the wall {tmean(S, "uflux", t0, tend)[0] / V**2:6.2f} V^2 '
              f'(at n = dz {tmean(S, "uflux", t0, tend)[1] / V**2:6.2f} V^2), '
              f'resolved/SFS TKE at 2.5 m {res[2]:.2e}/{tmean(S, "sgstke", t0, tend)[2]:.2e} m2/s2')


if __name__ == '__main__':
    main()
