#!/usr/bin/env python3
"""
Turbulence diagnostics of an axelsen2009_lbc run, on the straight part of
the slope (middle half by default): resolved vs SGS TKE, and energy spectra
along the periodic y direction at fixed heights above the surface.

    python3 axelsen2009_lbc_turbulence.py --rundir runs/P1_lbc_fine [--tavg 3600] [--segment 0.25,0.75]

1. TKE (figs/P1_lbc_tke.png, figs/P1_lbc_tke.nc)
   - From the time-averaged planes (<q>_tavg<h>, written when
     [IB] tavg_interval > 0): tke_res = 1/2 (u'u' + v'v' + w'w') over each
     interval and tke_sgs = the interval mean of the SGS TKE, at every
     tavg height; plus the slope-rotated resolved fluxes
     u_s'w_n' = (u'u' - w'w') sin b cos b + u'w' (cos^2 b - sin^2 b),
     w_n'thl' = u'thl' sin b + w'thl' cos b.
   - Without tavg planes: from the instantaneous xz crosses (u, v, w with
     departures from each cell's time mean over the window) and the
     sgstke xz crosses.
   The resolved fraction tke_res/(tke_res + tke_sgs) should be >= 0.8 for
   a well-resolved LES (Pope 2000); Axelsen (2010) Figs. 2.4, 3.12 show the
   SGS share for the paper's runs.

2. Spectra (figs/P1_lbc_spectra.png, figs/P1_lbc_spectra.nc)
   One-sided power spectra E(k) along y of u_s, v and w_n from the
   instantaneous terrain-following planes <var>_tf<h> (every cross output
   in the window), per column on the slope (mean along y removed, Hann
   window), averaged over columns and times. Normalised so that
   sum E dk = the variance along y. The inertial subrange shows as a
   k^(-5/3) slope (Kolmogorov 1941) between the energy-containing scales
   and the dissipation by the SGS model and the numerics, which in MicroHH
   (2i5 advection) starts at about 4-6 dy (k ~ 2 pi / (4..6 dy)); the
   Nyquist wavenumber is pi/dy. The SGS TKE at the same height is printed
   and stored for comparison.
"""
import argparse
import glob
import json
import os

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAV = 9.81


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--tavg', type=float, default=None, help='window at the end of the run [s] (default: last tau_osc, '
                                                            'or the last third if the run is shorter than 1.5 tau_osc)')
    ap.add_argument('--segment', default='0.25,0.75')
    ap.add_argument('--heights', default=None, help='tf heights for the spectra (default: all, up to 2 H)')
    a = ap.parse_args()
    rd = a.rundir
    p = json.load(open(os.path.join(rd, 'case_parameters.json')))
    it, jt, kt, dx, dz = p['itot'], p['jtot'], p['ktot'], p['dx'], p['dz']
    H, V, tau = p['H'], p['V'], p['tau_osc']
    outdir = a.outdir or os.path.join(rd, 'figs')
    os.makedirs(outdir, exist_ok=True)
    x = (np.arange(it) + 0.5) * dx
    dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jt, it)
    zs = dem.mean(axis=0)
    beta = np.arctan(-np.gradient(zs, dx))
    f0, f1 = (float(s) for s in a.segment.split(','))
    xa = p['x_top'] + 0.5 * p['break_width'] + dx
    xb = p['x_foot'] - 0.5 * p['break_width'] - dx
    cols = np.where((x >= xa + f0 * (xb - xa)) & (x <= xa + f1 * (xb - xa)))[0]
    cb, sb = np.cos(beta[cols]), np.sin(beta[cols])

    def times_of(pattern):
        return np.array(sorted({int(f.rsplit('.', 1)[1]) for f in glob.glob(os.path.join(rd, pattern))
                                if f.rsplit('.', 1)[1].isdigit()}))   # skip cross_to_nc's .nc

    def window(ts):
        ts = ts[ts > 0]
        if len(ts) == 0:
            return ts
        if a.tavg is not None:
            return ts[ts >= ts[-1] - a.tavg]
        if ts[-1] >= 1.5 * tau:
            return ts[ts >= ts[-1] - tau]
        return ts[ts >= ts[-1] * 2. / 3.]

    def xy(name, t):
        f = os.path.join(rd, '%s.xy.000.%07d' % (name, t))
        return np.fromfile(f).reshape(jt, it) if os.path.exists(f) else None

    # ---------------------------------------------------------------- 1. TKE
    tk = {}
    src = None
    hts = p.get('tavg_heights') or []
    tt = window(times_of(f'tke_res_tavg{int(round(hts[0]))}.xy.000.*')) if hts else np.array([])
    if len(tt):
        src = f'tavg planes, intervals ending {tt[0]}-{tt[-1]} s'
        rows = []
        for h in hts:
            sh = f'_tavg{int(round(h))}'
            acc = {k: [] for k in ('res', 'sgs', 'uw', 'wt', 'uu', 'vv', 'ww')}
            for t in tt:
                g = {q: xy(q + sh, t) for q in ('tke_res', 'tke_sgs', 'u2', 'v2', 'w2', 'uw', 'uthl', 'wthl')}
                if any(v is None for v in g.values()):
                    continue
                c = {q: v[:, cols] for q, v in g.items()}
                acc['res'].append(c['tke_res'].mean()); acc['sgs'].append(c['tke_sgs'].mean())
                acc['uu'].append(c['u2'].mean()); acc['vv'].append(c['v2'].mean()); acc['ww'].append(c['w2'].mean())
                acc['uw'].append(((c['u2'] - c['w2']) * sb * cb + c['uw'] * (cb**2 - sb**2)).mean())
                acc['wt'].append((c['uthl'] * sb + c['wthl'] * cb).mean())
            if acc['res']:
                rows.append([h] + [float(np.mean(acc[k])) for k in ('res', 'sgs', 'uw', 'wt', 'uu', 'vv', 'ww')])
        R = np.array(rows)
        tk = dict(h=R[:, 0], res=R[:, 1], sgs=R[:, 2], uw=R[:, 3], wt=R[:, 4], uu=R[:, 5], vv=R[:, 6], ww=R[:, 7])
    else:
        tx = window(times_of('u.xz.*'))
        planes = sorted({f.split('.')[-2] for f in glob.glob(os.path.join(rd, 'u.xz.100.*')) if f.split('.')[-1].isdigit()})
        if len(tx) and planes and glob.glob(os.path.join(rd, 'sgstke.xz.*')):
            src = f'instantaneous xz crosses, {tx[0]}-{tx[-1]} s (no tavg planes in this run)'
            z = (np.arange(kt) + 0.5) * dz
            h = z[:, None] - zs[None, cols]
            air = h > 0.
            Q = {k: [] for k in ('u', 'v', 'w', 'e')}
            for t in tx:
                for ip, j in enumerate(planes):
                    def rdp(nm, loc):
                        # v sits on the y faces: its plane index differs from the
                        # cell-centred ones; take the ip-th plane of that variable.
                        pl = sorted({f_.split('.')[-2] for f_ in glob.glob(os.path.join(rd, f'{nm}.xz.*'))
                                     if f_.split('.')[-1].isdigit()})
                        jj = pl[min(ip, len(pl) - 1)]   # a v face at y = ysize is not written
                        f_ = glob.glob(os.path.join(rd, '%s.xz.*.%s.%07d' % (nm, jj, t)))
                        return np.fromfile(f_[0]).reshape(kt, it)
                    u = rdp('u', '100'); w = rdp('w', '001'); v = rdp('v', '000'); e = rdp('sgstke', '000')
                    uc = 0.5 * (u + np.roll(u, -1, axis=1)); wc = 0.5 * (w + np.vstack([w[1:], 0 * w[:1]]))
                    Q['u'].append((uc[:, cols] * cb - wc[:, cols] * sb)); Q['w'].append(uc[:, cols] * sb + wc[:, cols] * cb)
                    Q['v'].append(v[:, cols]); Q['e'].append(e[:, cols])
            npl = len(planes)
            arr = {k: np.array(v).reshape(len(tx), npl, kt, len(cols)) for k, v in Q.items()}
            var = sum(arr[k].var(axis=0) for k in ('u', 'v', 'w'))      # per plane, cell
            res = 0.5 * var.mean(axis=0)
            sgs = arr['e'].mean(axis=(0, 1))
            n = h * cb[None, :]
            edges = np.arange(0., 2. * H + dz, max(dz, 1.))
            idx = np.digitize(n[air], edges) - 1
            ok = (idx >= 0) & (idx < len(edges) - 1)
            cnt = np.bincount(idx[ok], minlength=len(edges) - 1)
            with np.errstate(all='ignore'):
                r_ = np.bincount(idx[ok], res[air][ok], minlength=len(edges) - 1) / cnt
                s_ = np.bincount(idx[ok], sgs[air][ok], minlength=len(edges) - 1) / cnt
            g = cnt > 0
            tk = dict(h=(0.5 * (edges[1:] + edges[:-1]))[g], res=r_[g], sgs=s_[g])

    if tk:
        frac = tk['sgs'] / (tk['res'] + tk['sgs'])
        fig, ax = plt.subplots(1, 3 if 'uw' in tk else 2, figsize=(14 if 'uw' in tk else 10, 5.5), sharey=True)
        y = tk['h'] / H
        ax[0].plot(tk['res'] / V**2, y, 'o-', color='tab:blue', label='resolved')
        ax[0].plot(tk['sgs'] / V**2, y, 's-', color='tab:orange', label='SGS')
        ax[0].plot((tk['res'] + tk['sgs']) / V**2, y, 'k--', label='total')
        ax[0].set_xlabel('TKE / V^2'); ax[0].set_ylabel('h / H (height above the surface)'); ax[0].legend(fontsize=8)
        ax[1].plot(frac, y, 'o-', color='k')
        ax[1].axvline(0.2, color='r', ls=':', lw=1, label='SGS share 0.2 (Pope 2000)')
        ax[1].set_xlim(0, 1); ax[1].set_xlabel('SGS TKE / total TKE'); ax[1].legend(fontsize=8)
        if 'uw' in tk:
            ax[2].plot(tk['uw'] / V**2, y, 'o-', color='tab:blue', label="resolved u_s'w_n' / V^2")
            ax[2].plot(tk['wt'] * GRAV / p['theta_a0'] / (V * p['B']), y, 's-', color='tab:red',
                       label="resolved w_n'b' / (VB)")
            ax[2].axvline(0, color='0.6', lw=0.5); ax[2].legend(fontsize=8); ax[2].set_xlabel('resolved fluxes')
        for axx in ax:
            axx.grid(alpha=0.25); axx.set_ylim(0, min(4., y.max() * 1.05))
            axx.axhline(0.5 * dz / H, color='0.6', ls=':', lw=0.8)
        fig.suptitle(f"axelsen2009_lbc (dx {dx:g} m, dz {dz:g} m): slope x = {x[cols[0]]/1e3:.2f}-{x[cols[-1]]/1e3:.2f} km; "
                     f"{src}", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, 'P1_lbc_tke.png'), dpi=130)
        plt.close(fig)
        with nc.Dataset(os.path.join(outdir, 'P1_lbc_tke.nc'), 'w') as f:
            f.source = src
            f.createDimension('h', len(tk['h']))
            for k, v in tk.items():
                var = f.createVariable(k if k != 'h' else 'h', 'f8', ('h',))
                var[:] = v
            f.createVariable('sgs_fraction', 'f8', ('h',))[:] = frac
        print(f'TKE ({src}):')
        for hh, r, s_, fr in zip(tk['h'], tk['res'], tk['sgs'], frac):
            print(f'  h = {hh:6.1f} m: resolved {r:.4f}, SGS {s_:.4f} m2/s2, SGS share {fr:.2f}')
    else:
        print('TKE: neither tavg planes nor xz u/v/w + sgstke crosses in this run')

    # ------------------------------------------------------------- 2. spectra
    tfh = [h for h in p['tf_heights'] if glob.glob(os.path.join(rd, f'v_tf{h}.xy.000.*'))]
    if a.heights:
        tfh = [h for h in tfh if h in [int(v) for v in a.heights.split(',')]]
    else:
        tfh = [h for h in tfh if h <= 2. * H]
    ts = window(times_of(f'u_tf{tfh[0]}.xy.000.*')) if tfh else np.array([])
    if not len(ts) or jt < 8:
        print('spectra: no tf planes in the window, or jtot < 8')
        return
    k = 2. * np.pi * np.fft.rfftfreq(jt, d=dx)
    win = np.hanning(jt)
    wnorm = np.sum(win**2) / jt
    dk = k[1]
    S = {}
    for h in tfh:
        acc = {q: np.zeros(len(k)) for q in ('us', 'v', 'wn')}
        nsp = 0
        for t in ts:
            u, v, w = xy(f'u_tf{h}', t), xy(f'v_tf{h}', t), xy(f'w_tf{h}', t)
            if u is None or v is None or w is None:
                continue
            us = u[:, cols] * cb - w[:, cols] * sb
            wn = u[:, cols] * sb + w[:, cols] * cb
            for q, f in (('us', us), ('v', v[:, cols]), ('wn', wn)):
                d = (f - f.mean(axis=0)) * win[:, None]
                F = np.fft.rfft(d, axis=0)
                e = (np.abs(F)**2) / (jt**2 * wnorm)
                e[1:-1] *= 2.                                # one-sided
                acc[q] += (e / dk).mean(axis=1)
            nsp += 1
        if nsp:
            S[h] = {q: v / nsp for q, v in acc.items()}
    with nc.Dataset(os.path.join(outdir, 'P1_lbc_spectra.nc'), 'w') as f:
        f.description = ('One-sided power spectra along y (periodic) of u_s, v, w_n on terrain-following planes '
                         f'h above the surface, slope x = {x[cols[0]]:.0f}-{x[cols[-1]]:.0f} m, mean over columns '
                         f'and {len(ts)} times {ts[0]}-{ts[-1]} s. sum(E) dk = variance along y.')
        f.createDimension('k', len(k)); f.createDimension('h', len(S))
        f.createVariable('k', 'f8', ('k',))[:] = k
        f['k'].units = 'rad m-1'
        f.createVariable('h', 'f8', ('h',))[:] = list(S)
        for q in ('us', 'v', 'wn'):
            vv = f.createVariable(f'E_{q}', 'f8', ('h', 'k'))
            vv[:] = np.array([S[h][q] for h in S])
            vv.units = 'm3 s-2'
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    cmap = plt.get_cmap('viridis')
    for i, h in enumerate(S):
        col = cmap(i / max(1, len(S) - 1))
        for axx, q in zip(ax, ('us', 'v', 'wn')):
            axx.loglog(k[1:], S[h][q][1:], color=col, lw=1.2, label=f'h = {h} m ({h / H:.2f} H)')
    for axx, q in zip(ax, ('u_s (along slope)', 'v (cross slope)', 'w_n (slope normal)')):
        kk = np.array([k[2], k[-1]])
        ref = S[list(S)[len(S) // 2]]['wn'][len(k) // 4] if len(S) else 1.
        k0 = k[len(k) // 4]
        axx.loglog(kk, ref * (kk / k0)**(-5. / 3.), 'k--', lw=1, label='k^-5/3')
        axx.axvline(np.pi / dx, color='r', ls=':', lw=1, label='Nyquist pi/dy')
        axx.axvline(2. * np.pi / (6. * dx), color='0.5', ls=':', lw=1, label='6 dy (numerical dissipation)')
        axx.set_xlabel('k (rad/m)'); axx.set_title(q, fontsize=10); axx.grid(alpha=0.25, which='both')
    ax[0].set_ylabel('E(k) (m3/s2)')
    ax[0].legend(fontsize=7)
    fig.suptitle(f'axelsen2009_lbc (dx {dx:g} m, dz {dz:g} m): spectra along y on the slope, {len(ts)} snapshots '
                 f'{ts[0]}-{ts[-1]} s', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'P1_lbc_spectra.png'), dpi=130)
    plt.close(fig)
    # Inertial-subrange slope fitted between 4 dy and 1/4 of the domain.
    kfit = (k > 2. * np.pi / (jt * dx / 4.)) & (k < 2. * np.pi / (6. * dx))
    print(f'spectra ({len(ts)} snapshots, {ts[0]}-{ts[-1]} s): log-log slope for 6 dy < wavelength < Ly/4 '
          f'(-5/3 = -1.67):')
    for h in S:
        sl = [np.polyfit(np.log(k[kfit]), np.log(S[h][q][kfit]), 1)[0] if kfit.sum() > 2 else np.nan
              for q in ('us', 'v', 'wn')]
        print(f'  h = {h:4d} m: u_s {sl[0]:5.2f}, v {sl[1]:5.2f}, w_n {sl[2]:5.2f}')
    print(f'figures and netCDF in {outdir}')


if __name__ == '__main__':
    main()
