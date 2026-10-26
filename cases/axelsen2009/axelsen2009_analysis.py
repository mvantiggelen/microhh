#!/usr/bin/env python3
"""
Analysis of the Axelsen and van Dop (2009) runs made by axelsen2009_input.py.

    python3 axelsen2009_analysis.py --rundir run_P1_flux [--outdir figs]
    python3 axelsen2009_analysis.py --rundir runs/*_flux runs/*_temp --outdir figs

Works on the xz crosses (u, w, thl; 4 planes between the y sponges) when
the run has them, else on the terrain-following crosses. The
xz route gives every model level above the local surface, which the
paper-resolution runs need (the jet at 3-6 m, the tf planes are whole
metres); the IB surface flux comes from the thlfluxbot_ib crosses.

  along-slope (downslope) velocity  u_s = u cos(alpha) - w sin(alpha)
                                         (the terrain falls towards +x)
  buoyancy                           b = g/theta_a0 (thl - thl_a(z_s + h)),
                                         thl_a(z) = theta_a0 + G (z - z_mid)
  slope-normal height                n = h cos(alpha)

averaged over the interior columns (n_sponge + 2 cells away from every
edge) and, for the steady state, over the last tau_osc, as in the paper.
Scaled with V = (|Fs|/N)^1/2, B = |Fs|/V, H = V/N (their eq. 9), also for
--bc temp (whose own flux is reported as Fs_model / Fs).

Writes, per run, <outdir>/<case>_<bc>_profiles.png, _timeseries.png and
_metrics.json; with several runs also summary.md (flux vs temp per case,
against the paper's LES values in reference_axelsen2009.json).
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRAV = 9.81
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
REF = json.load(open(os.path.join(HERE, 'reference_axelsen2009.json')))['cases']


def load_run(rd):
    """Profiles from the xz crosses when there are any (any resolution,
    sub-metre heights), else from the terrain-following crosses."""
    p = json.load(open(os.path.join(rd, 'case_parameters.json')))
    if glob.glob(os.path.join(rd, 'thl.xz.000.*')):
        return load_run_xz(rd, p)
    return load_run_tf(rd, p)


def surface_flux(rd, p, times, mask):
    itot, jtot = p['itot'], p['jtot']
    flux = np.full(len(times), np.nan)
    for it, t in enumerate(times):
        fn = os.path.join(rd, 'thl_fluxbot_ib.xy.000.%07d' % t)    # crosslist thlfluxbot_ib
        if not os.path.exists(fn):
            fn = os.path.join(rd, 'thlfluxbot_ib.xy.000.%07d' % t)
        if os.path.exists(fn):
            flux[it] = np.nanmean(np.fromfile(fn).reshape(jtot, itot)[mask])
    return flux


def interior_mask(p):
    itot, jtot = p['itot'], p['jtot']
    ns = p['n_sponge'] + 2
    mask = np.zeros((jtot, itot), bool)
    mask[ns:jtot-ns, ns:itot-ns] = True
    if not mask.any():                       # narrow domain: x only
        mask[:, ns:itot-ns] = True
    return mask


def load_run_xz(rd, p):
    """Slope-normal profiles from the xz crosses: per column, the values at
    the cell centres above the local surface are interpolated to heights
    h = dz (0.5, 1.5, ...) above it (u averaged to the centres from the
    faces, w from the half levels), then averaged over the interior columns
    of every plane. n = h cos(alpha)."""
    itot, ktot, dx, dz = p['itot'], p['ktot'], p['dx'], p['dz']
    files = sorted(glob.glob(os.path.join(rd, 'thl.xz.000.*')))
    planes = sorted({f.split('.')[-2] for f in files})
    times = np.array(sorted({int(f.split('.')[-1]) for f in files}))
    times = times[times > 0]

    ca, sa = np.cos(np.deg2rad(p['alpha_deg'])), np.sin(np.deg2rad(p['alpha_deg']))
    th0, gam, zmid = p['theta_a0'], p['gamma'], p['z_mid']
    tana = np.tan(np.deg2rad(p['alpha_deg']))
    x = (np.arange(itot) + 0.5) * dx
    zs = p['z_base'] + tana * (p['lx'] - x)
    # Uniform levels, or those of a stretched grid (--stretch-above).
    z = np.array(p['z_full']) if 'z_full' in p else (np.arange(ktot) + 0.5) * dz
    ns = p['n_sponge'] + 2
    cols = np.arange(ns, itot - ns)
    h_t = dz * (0.5 + np.arange(int(0.8 * p.get('lz', 0.8 * ktot * dz) / dz)))

    def plane(name, loc, j, t):
        return np.fromfile(os.path.join(rd, '%s.xz.%s.%s.%07d' % (name, loc, j, t))).reshape(ktot, itot)

    us = np.zeros((len(times), len(h_t)))
    bb = np.zeros_like(us)
    for it, t in enumerate(times):
        acc_u = np.zeros(len(h_t)); acc_b = np.zeros(len(h_t)); cnt = np.zeros(len(h_t))
        for j in planes:
            u = plane('u', '100', j, t)
            w = plane('w', '001', j, t)
            th = plane('thl', '000', j, t)
            uc = 0.5 * (u + np.roll(u, -1, axis=1))
            wc = 0.5 * (w + np.vstack([w[1:], np.zeros((1, itot))]))
            for i in cols:
                h = z - zs[i]
                k = h > 0
                ok = h_t <= h[k][-1]
                usl = uc[k, i] * ca - wc[k, i] * sa
                b = GRAV / th0 * (th[k, i] - (th0 + gam * (zs[i] + h[k] - zmid)))
                acc_u[ok] += np.interp(h_t[ok], h[k], usl)
                acc_b[ok] += np.interp(h_t[ok], h[k], b)
                cnt[ok] += 1
        us[it] = acc_u / np.maximum(cnt, 1)
        bb[it] = acc_b / np.maximum(cnt, 1)
    flux = surface_flux(rd, p, times, interior_mask(p))
    return p, times, h_t * ca, us, bb, flux


def load_run_tf(rd, p):
    itot, jtot = p['itot'], p['jtot']
    dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jtot, itot)
    heights = [h for h in p['tf_heights']
               if glob.glob(os.path.join(rd, f'u_tf{h}.xy.000.*'))]
    files = sorted(glob.glob(os.path.join(rd, f'u_tf{heights[0]}.xy.000.*')))
    times = np.array([int(f.split('.')[-1]) for f in files])
    times = times[times > 0]

    mask = interior_mask(p)

    ca, sa = np.cos(np.deg2rad(p['alpha_deg'])), np.sin(np.deg2rad(p['alpha_deg']))
    th0, gam, zmid = p['theta_a0'], p['gamma'], p['z_mid']

    def plane(name, t):
        return np.fromfile(os.path.join(rd, '%s.xy.000.%07d' % (name, t))).reshape(jtot, itot)

    us = np.zeros((len(times), len(heights)))
    bb = np.zeros_like(us)
    for it, t in enumerate(times):
        for ih, h in enumerate(heights):
            u = plane(f'u_tf{h}', t)
            w = plane(f'w_tf{h}', t)
            th = plane(f'thl_tf{h}', t)
            us[it, ih] = np.mean((u*ca - w*sa)[mask])
            tha = th0 + gam * (dem + h - zmid)
            bb[it, ih] = np.mean((GRAV / th0 * (th - tha))[mask])
    flux = surface_flux(rd, p, times, mask)
    n = np.array(heights, float) * ca
    return p, times, n, us, bb, flux


def metrics(p, times, n, us, bb, flux):
    V, B, H, tau = p['V'], p['B'], p['H'], p['tau_osc']
    # The paper averages over the last tau_osc of a 3 tau_osc run. A run that
    # stopped before 1.5 tau_osc has no steady last tau_osc; then only its
    # last third is averaged (flagged), so the spin-up does not bias it low.
    if times[-1] >= 1.5 * tau:
        last = times >= times[-1] - tau
        window = 'last tau_osc'
    else:
        last = times >= times[-1] * 2. / 3.
        window = 'last third (run shorter than 1.5 tau_osc: NOT steady)'
    u_m = us[last].mean(0)
    b_m = bb[last].mean(0)
    k = int(np.argmax(u_m))
    zj = n[k]
    if 0 < k < len(n) - 1:                    # parabola through the maximum
        x, y = n[k-1:k+2], u_m[k-1:k+2]
        c = np.polyfit(x, y, 2)
        if c[0] < 0:
            zj = -c[1] / (2*c[0])
    neg = np.where(b_m >= 0)[0] if p['Fs'] < 0 else np.where(b_m <= 0)[0]
    zinv = n[neg[0]] if len(neg) else np.nan
    Fs_model = GRAV / p['theta_a0'] * np.nanmean(flux[last])
    # vertical integrals with u = 0 at the surface (no slip), b constant below n[0]
    nn = np.concatenate([[0.], n])
    UD = _trapz(np.concatenate([np.zeros((len(times), 1)), us], 1), nn, axis=1)
    BD = _trapz(np.concatenate([bb[:, :1], bb], 1), nn, axis=1)
    return dict(umax_over_V=float(u_m[k] / V), zj_over_H=float(zj / H),
                b1_over_B=float(b_m[0] / B), zinv_over_H=float(zinv / H),
                Fs_model_over_Fs=float(Fs_model / p['Fs']),
                n_first=float(n[0]), steady_from_s=float(times[last][0]),
                window=window, t=[float(x) for x in times],
                umax_over_V_t=(us.max(axis=1) / V).tolist(),
                UD=UD.tolist(), BD=BD.tolist(),
                u_profile=u_m.tolist(), b_profile=b_m.tolist(), n=n.tolist())


def plots(p, times, n, us, bb, m, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    V, B, H, tau = p['V'], p['B'], p['H'], p['tau_osc']
    tag = f"{p['case']}_{p['bc']}"
    ref = REF[p['case']]['les']

    fig, ax = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    ax[0].plot(np.array(m['u_profile']) / V, n / H, 'k.-', label=f"MicroHH ({p.get('res', 'coarse')}, IB)")
    ax[0].plot(ref['umax_over_V'], ref['zj_over_H'], 'rs', ms=9, label='paper LES: jet maximum')
    ax[0].axhline(0.5 * p['dz'] / H, color='0.6', ls=':', label='first model level')
    ax[0].set_xlabel('u_s / V'); ax[0].set_ylabel('n / H'); ax[0].legend(fontsize=8)
    ax[1].plot(np.array(m['b_profile']) / B, n / H, 'k.-')
    if 'b1_over_B' in ref:
        ax[1].plot(ref['b1_over_B'], 0.02, 'rs', ms=9, label='paper LES, lowest level')
        ax[1].legend(fontsize=8)
    ax[1].axvline(0, color='0.6', lw=0.5)
    ax[1].set_xlabel('b / B')
    for a in ax:
        a.set_ylim(0, max(3., 1.5 * m['zj_over_H']) if p['Fs'] < 0 else 8.)
        a.grid(alpha=0.3)
    fig.suptitle(f"{tag}: mean over the {m['window']} (from {m['steady_from_s']:.0f} s)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_profiles.png'), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times / tau, np.array(m['UD']) / (V*H), 'k-', label='UD / (V H)')
    ax.plot(times / tau, np.array(m['BD']) / (B*H), 'k--', label='BD / (B H)')
    ax.plot(times / tau, m['umax_over_V_t'], 'b-',
            label='u_s,max / V (paper: %g)' % REF[p['case']]['les']['umax_over_V'])
    ax.set_xlabel('t / tau_osc'); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title(f'{tag}: vertically integrated flow (cf. paper Fig. 5a)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_timeseries.png'), dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', nargs='+', required=True)
    ap.add_argument('--outdir', default='figs')
    ap.add_argument('--no-plots', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    rows = []
    for rd in a.rundir:
        p, times, n, us, bb, flux = load_run(rd)
        if len(times) == 0:
            print(f'{rd}: no terrain-following crosses yet'); continue
        m = metrics(p, times, n, us, bb, flux)
        tag = f"{p['case']}_{p['bc']}"
        with open(os.path.join(a.outdir, f'{tag}_metrics.json'), 'w') as f:
            json.dump(m, f, indent=1)
        if not a.no_plots:
            plots(p, times, n, us, bb, m, a.outdir)
        rows.append((p, m))
        print(f"{tag:8s} u_max/V {m['umax_over_V']:6.2f}  z_j/H {m['zj_over_H']:5.2f}  "
              f"b1/B {m['b1_over_B']:7.1f}  z_inv/H {m['zinv_over_H']:5.2f}  "
              f"Fs_model/Fs {m['Fs_model_over_Fs']:5.2f}  (t = {times[-1]} s)")

    if len(rows) > 1:
        lines = ['| case | bc | u_max/V (paper) | z_j/H (paper) | b1/B (paper) | z_inv/H | Fs_model/Fs |',
                 '|---|---|---|---|---|---|---|']
        for p, m in sorted(rows, key=lambda r: (r[0]['case'], r[0]['bc'])):
            ref = REF[p['case']]['les']
            lines.append(
                f"| {p['case']} | {p['bc']} | {m['umax_over_V']:.1f} ({ref.get('umax_over_V', '-')}) "
                f"| {m['zj_over_H']:.2f} ({ref.get('zj_over_H', '-')}) "
                f"| {m['b1_over_B']:.0f} ({ref.get('b1_over_B', '-')}) "
                f"| {m['zinv_over_H']:.2f} | {m['Fs_model_over_Fs']:.2f} |")
        open(os.path.join(a.outdir, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
        print('\n'.join(lines))


if __name__ == '__main__':
    main()
