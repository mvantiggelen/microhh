#!/usr/bin/env python3
"""
Analysis of the IB version of Fedorovich et al. (2017) case B13.

Reads the xz cross-sections (u, v, w, thl) written by the run, maps every
column to height ABOVE ITS OWN SURFACE (h = z - z_s), and averages over the
cross-section planes and over the interior columns only, i.e. at least
--margin metres away from the west and east edges (the x-sponges and the
patch-28 seam). Perturbations for the TKE are taken against that interior
plane mean.

Produces, in --outdir:
  b13_wind_ht.png     |V|/|Vg| height-time        (paper Fig. 12a)
  b13_tke_ht.png      TKE/|Vg|^2 height-time      (paper Fig. 10a)
  b13_profiles.png    b/g, u/|Vg|, v/|Vg| at the paper's times (Fig. 11a,c)
  b13_hodograph.png   hodographs at 100-500 m AGL (Fig. 13a)
  b13_edges.png       interior mean vs the columns next to each edge
  b13_metrics.json    the numbers compared against reference_b13.json

Usage:
  python3 fedorovich2017_b13_analysis.py [--rundir .] [--margin 200]
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
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
GRAV = 9.81
PAPER_TIMES = [10800, 18000, 25200, 32400, 39600, 46800, 54000]
HODO_HEIGHTS = [100, 200, 300, 400, 500]


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--rundir', default='.')
    p.add_argument('--outdir', default=None)
    p.add_argument('--margin', type=float, default=None,
                   help='distance from the x-edges excluded from the averages [m] '
                        '(default: 3 * n_sponge * dx, at least 15 %% of Lx)')
    p.add_argument('--reference', default=os.path.join(HERE, 'reference_b13.json'))
    return p.parse_args()


def read_crosses(rundir, var, ktot, itot):
    """{time: [array (ktot, itot) per plane]} for an xz cross variable."""
    files = sorted(glob.glob(os.path.join(rundir, f'{var}.xz.*.*')))
    out = {}
    for f in files:
        m = re.match(rf'{var}\.xz\.\d+\.(\d+)\.(\d+)$', os.path.basename(f))
        if not m:
            continue
        t = int(m.group(2))
        out.setdefault(t, []).append(np.fromfile(f, dtype=np.float64).reshape(ktot, itot))
    return out


def to_agl(fld, zl, zs, h):
    """Interpolate every column of fld (k, i) onto heights h above zs[i]."""
    out = np.full((h.size, fld.shape[1]), np.nan)
    for i in range(fld.shape[1]):
        zt = h + zs[i]
        air = zl > zs[i]
        out[:, i] = np.interp(zt, zl[air], fld[air, i], left=np.nan, right=np.nan)
    return out


def main():
    a = parse()
    rd = a.rundir
    od = a.outdir or rd
    os.makedirs(od, exist_ok=True)
    P = json.load(open(os.path.join(rd, 'case_parameters.json')))
    itot, ktot, dx, dz = P['itot'], P['ktot'], P['dx'], P['dz']
    vg, gam, thr = P['vg'], P['gamma'], P['theta_r']
    lx = itot * dx

    z = (np.arange(ktot) + 0.5) * dz
    zh = np.arange(ktot) * dz
    xc = (np.arange(itot) + 0.5) * dx
    zs = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(-1, itot)[0]

    margin = a.margin if a.margin is not None else max(3*P['n_sponge']*dx, 0.15*lx)
    inner = (xc > margin) & (xc < lx - margin)
    edge_w = (xc > P['n_sponge']*dx) & (xc < P['n_sponge']*dx + 0.1*lx)
    edge_e = (xc < lx - P['n_sponge']*dx) & (xc > lx - P['n_sponge']*dx - 0.1*lx)

    h = np.arange(0.5*dz, P['lz'], dz)

    raw = {v: read_crosses(rd, v, ktot, itot) for v in ['u', 'v', 'w', 'thl']}
    times = sorted(set.intersection(*[set(raw[v]) for v in raw]))
    if not times:
        raise SystemExit('no complete set of u, v, w, thl xz crosses found in ' + rd)

    nt = len(times)
    U, V, B, TKE = (np.full((nt, h.size), np.nan) for _ in range(4))
    Ue, Ve = {}, {}
    for n, t in enumerate(times):
        tke = []
        for p in range(len(raw['u'][t])):
            # u and w to cell centres, so all four share the column positions.
            u = 0.5*(raw['u'][t][p] + np.roll(raw['u'][t][p], -1, axis=1))
            w = np.vstack([0.5*(raw['w'][t][p][:-1] + raw['w'][t][p][1:]), raw['w'][t][p][-1:]])
            v = raw['v'][t][p]
            th = raw['thl'][t][p]
            b = GRAV/thr * (th - (thr + gam*z)[:, None])      # buoyancy w.r.t. the environment
            ua, va, wa, ba = (to_agl(f, z, zs, h) for f in (u, v, w, b))
            for name, f in [('u', ua), ('v', va), ('b', ba)]:
                raw.setdefault('agl_'+name, {}).setdefault(t, []).append(f)
            um, vm, wm = (np.nanmean(f[:, inner], axis=1) for f in (ua, va, wa))
            tke.append(0.5*np.nanmean((ua[:, inner]-um[:, None])**2
                                     + (va[:, inner]-vm[:, None])**2
                                     + (wa[:, inner]-wm[:, None])**2, axis=1))
        stack = lambda name: np.stack(raw['agl_'+name][t])
        U[n] = np.nanmean(stack('u')[:, :, inner], axis=(0, 2))
        V[n] = np.nanmean(stack('v')[:, :, inner], axis=(0, 2))
        B[n] = np.nanmean(stack('b')[:, :, inner], axis=(0, 2))
        TKE[n] = np.mean(tke, axis=0)
        for side, mask in [('west', edge_w), ('east', edge_e)]:
            Ue.setdefault(side, []).append(np.nanmean(stack('u')[:, :, mask], axis=(0, 2)))
            Ve.setdefault(side, []).append(np.nanmean(stack('v')[:, :, mask], axis=(0, 2)))

    th_h = np.array(times) / 3600.
    S = np.sqrt(U**2 + V**2) / vg

    # --- height-time panels -------------------------------------------------
    for name, data, kw, label in [
            ('b13_wind_ht.png', S, dict(cmap='magma_r', vmin=0, vmax=1.5), r'$|V|/|V_g|$'),
            ('b13_tke_ht.png', np.maximum(TKE/vg**2, 1e-6),
             dict(cmap='magma', norm=LogNorm(1e-4, 1e-1)), r'TKE$/|V_g|^2$')]:
        fig, ax = plt.subplots(figsize=(7, 3))
        pc = ax.pcolormesh(th_h, h, data.T, shading='nearest', **kw)
        fig.colorbar(pc, ax=ax, label=label)
        ax.set_xlabel('t (h)'); ax.set_ylabel('height above surface (m)')
        ax.set_ylim(0, min(1500, h[-1]))
        ax.axvline(P['t_sunset']/3600., color='0.5', lw=0.8, ls='--')
        ax.set_title(f"IB B13, alpha={P['alpha']:g} deg, dx={dx:g} m (interior mean)", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(od, name), dpi=150); plt.close(fig)

    # --- profiles at the paper's times ---------------------------------------
    sel = [t for t in PAPER_TIMES if t in times] or times[::max(1, nt//6)]
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(sel)))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)
    for c, t in zip(cols, sel):
        n = times.index(t)
        axs[0].plot(B[n]/GRAV, h, color=c, lw=1.5, label=f'{t} s')
        axs[1].plot(U[n]/vg, h, color=c, lw=1.5, ls='--')
        axs[1].plot(V[n]/vg, h, color=c, lw=1.5)
    axs[0].set_xlabel('b/g'); axs[1].set_xlabel(r'$u/|V_g|$ (dashed), $v/|V_g|$ (solid)')
    axs[0].set_ylabel('height above surface (m)'); axs[0].set_ylim(0, min(1500, h[-1]))
    axs[0].legend(fontsize=7, frameon=False)
    for ax in axs:
        ax.grid(color='0.9', lw=0.6)
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    fig.tight_layout(); fig.savefig(os.path.join(od, 'b13_profiles.png'), dpi=150); plt.close(fig)

    # --- hodographs after sunset ---------------------------------------------
    night = th_h >= P['t_sunset']/3600.
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, len(HODO_HEIGHTS)))
    for c, hh in zip(cols, HODO_HEIGHTS):
        k = np.argmin(abs(h-hh))
        ax.plot(U[night, k]/vg, V[night, k]/vg, color=c, lw=1.5, label=f'{hh} m')
    ax.set_xlim(-1, 0.5); ax.set_ylim(0, 1.5); ax.set_aspect('equal')
    ax.set_xlabel(r'$u/|V_g|$'); ax.set_ylabel(r'$v/|V_g|$'); ax.grid(color='0.9', lw=0.6)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(od, 'b13_hodograph.png'), dpi=150); plt.close(fig)

    # --- boundary influence: interior vs the bands next to each edge ---------
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
    n = nt - 1
    for ax, (lab, I, E) in zip(axs, [('u', U, Ue), ('v', V, Ve)]):
        ax.plot(I[n]/vg, h, 'k', lw=2, label='interior')
        for side, ls in [('west', '--'), ('east', ':')]:
            ax.plot(np.array(E[side])[n]/vg, h, 'k', ls=ls, lw=1.2, label=f'next to {side} sponge')
        ax.set_xlabel(f'{lab}/|Vg| at t = {times[n]} s'); ax.grid(color='0.9', lw=0.6)
    axs[0].set_ylabel('height above surface (m)'); axs[0].legend(fontsize=7, frameon=False)
    axs[0].set_ylim(0, min(1500, h[-1]))
    fig.tight_layout(); fig.savefig(os.path.join(od, 'b13_edges.png'), dpi=150); plt.close(fig)

    # --- metrics ---------------------------------------------------------------
    met = {'times_s': [int(t) for t in times], 'margin_m': margin,
           'n_interior_columns': int(inner.sum()), 'staircase_steps': P['n_steps']}
    i_max = np.unravel_index(np.nanargmax(np.where(night[:, None], S, -1)), S.shape) if night.any() else None
    if i_max is not None:
        met['jet_max_V_over_Vg'] = float(S[i_max])
        met['jet_height_m'] = float(h[i_max[1]])
        met['jet_time_h'] = float(th_h[i_max[0]])
    low = h < 500
    tke_low = np.nanmean(TKE[:, low], axis=1) / vg**2
    met['tke_low_over_Vg2'] = [float(v) for v in tke_low]
    after = np.where(night & (tke_low < 1e-3))[0]
    if after.size:
        met['tke_collapse_h'] = float(th_h[after[0]])
        later = np.where((th_h > th_h[after[0]]) & (tke_low > 3e-3))[0]
        if later.size:
            met['tke_reemergence_h'] = float(th_h[later[0]])
    ks = np.argmin(abs(h - 250))
    k_sun = np.argmin(abs(th_h - P['t_sunset']/3600.))
    met['u_over_Vg_250m_sunset'] = float(U[k_sun, ks]/vg)
    met['v_over_Vg_250m_sunset'] = float(V[k_sun, ks]/vg)
    edge_dev = max(np.nanmax(abs(np.array(Ve[s])[-1] - V[-1])) for s in Ve) / vg
    met['max_edge_minus_interior_v_over_Vg'] = float(edge_dev)
    json.dump(met, open(os.path.join(od, 'b13_metrics.json'), 'w'), indent=1)

    if os.path.exists(a.reference):
        ref = json.load(open(a.reference))
        print(f"{'metric':34s} {'this run':>10s} {'paper (B13)':>12s}")
        for k, r in ref['metrics'].items():
            v = met.get(k)
            vs = f'{v:10.3g}' if isinstance(v, float) else f'{str(v):>10s}'
            print(f"{k:34s} {vs} {r['value']:>12} {r.get('note', '')}")
    print('figures and b13_metrics.json written to', od)


if __name__ == '__main__':
    main()
