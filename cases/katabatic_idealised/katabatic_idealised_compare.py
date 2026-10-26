#!/usr/bin/env python3
"""
Compare the Cartesian IB run of katabatic_idealised with its rotated-frame
reference:

    python3 katabatic_idealised_compare.py --rundir runs/test [--tavg 19800] [--out runs/test/compare]

Reads ib/ (xz planes of u, w, thl; xy maps thl_fluxbot_ib; case_parameters.json) and
rotated/ (the statistics file: u, b, b_flux). Over the averaging window (the
last tau_osc by default) it gives:
  a  downslope wind u_s(n) at the stations (top, mid, station, foot) and in the
     rotated run, n the height above the surface along the normal
  b  theta - theta_a(z) at the same places (rotated: b theta_r / g)
  c  the jet maximum and its height along the slope
  d  the sensible heat flux along the slope, rho cp w'theta'_s with rho = 1.1
     kg/m3 in both (rotated: rho cp theta_r/g b_flux_s)
  e  the jet maximum at the weather station and mid-slope in time, and the rotated one
  f  the surface heat flux at the weather station in time, and the rotated one
plus metrics.json. u_s = u cos(beta) - w sin(beta) with beta the local angle
of the terrain (falling towards +x); the rotated frame's downslope is -x.
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

GRAV, CP = 9.81, 1005.
RHO = 1.1


def first(v):
    """The lowest finite value of a profile."""
    v = np.asarray(v)
    return float(v[np.isfinite(v)][0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rundir', required=True, help='the folder with ib/ and rotated/')
    ap.add_argument('--tavg', type=float, default=None, help='averaging window at the end [s] (default: last tau_osc)')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    di, dr = os.path.join(a.rundir, 'ib'), os.path.join(a.rundir, 'rotated')
    out = a.out or os.path.join(a.rundir, 'compare')
    os.makedirs(out, exist_ok=True)
    P = json.load(open(os.path.join(di, 'case_parameters.json')))
    it, jt, kt, dx, dz = P['itot'], P['jtot'], P['ktot'], P['dx'], P['dz']
    x = (np.arange(it) + 0.5) * dx
    z = (np.arange(kt) + 0.5) * dz
    zh = np.arange(kt + 1) * dz
    dem = np.fromfile(os.path.join(di, 'dem.0000000')).reshape(jt, it)[0]
    beta = np.arctan(np.clip(-np.gradient(dem, dx), 0., None))
    c0, gam = P['thl_a_coeffs']
    thl_a = c0 + gam * (z + P['datum_amsl'])

    fs = glob.glob(os.path.join(di, 'thl.xz.000.*'))
    times = sorted({int(f.rsplit('.', 1)[1]) for f in fs})
    if not times:
        raise SystemExit(f'no thl.xz crosses in {di}')
    tend = times[-1]
    if a.tavg is not None:
        win = a.tavg
    else:
        win = P['tau_osc'] if tend >= 1.5 * P['tau_osc'] else tend / 3.   # last tau, or the last third
    tw = [t for t in times if t >= tend - win]
    planes = sorted({f.split('.')[-2] for f in fs})

    def plane(name, loc, j, t):
        f = os.path.join(di, f'{name}.xz.{loc}.{j}.{t:07d}')
        return np.fromfile(f).reshape(kt, it) if os.path.exists(f) else None

    n_grid = np.arange(0., 400.01, dz / 2.)

    def slope_fields(t):
        """u_s(z, x) and theta'(z, x) at the cell centres, mean over the planes."""
        us, th = [], []
        for j in planes:
            u, w, tl = plane('u', '100', j, t), plane('w', '001', j, t), plane('thl', '000', j, t)
            if u is None or w is None or tl is None:
                continue
            uc = 0.5 * (u + np.roll(u, -1, axis=1))
            uc[:, -1] = u[:, -1]
            wc = 0.5 * (w + np.vstack([w[1:], w[-1:]]))          # w at the cell centres
            us.append(uc * np.cos(beta)[None] - wc * np.sin(beta)[None])
            th.append(tl - thl_a[:, None])
        return np.mean(us, axis=0), np.mean(th, axis=0)

    def to_n(f2, i0, half):
        """Profile against n over the columns i0-half .. i0+half."""
        prof = []
        for i in range(max(0, i0 - half), min(it, i0 + half + 1)):
            n = (z - dem[i]) * np.cos(beta[i])
            m = n > 0.
            prof.append(np.interp(n_grid, n[m], f2[m, i], left=np.nan, right=np.nan))
        return np.nanmean(prof, axis=0)

    st = P['stations']
    # older case_parameters.json files name the weather station after the site
    st = {('station' if k not in ('top', 'mid', 'foot') else k): v for k, v in st.items()}
    half = max(1, int(round(500. / dx)))
    i_st = {k: int(np.clip(round(v / dx - 0.5), 0, it - 1)) for k, v in st.items()}
    acc_u = {k: [] for k in st}
    acc_t = {k: [] for k in st}
    jet_x, jetn_x = [], []
    ts_jet = {k: [] for k in ('station', 'mid')}
    for t in times:
        us, th = slope_fields(t)
        for k in ('station', 'mid'):
            ts_jet[k].append(np.nanmax(to_n(us, i_st[k], half)))
        if t in tw:
            for k in st:
                acc_u[k].append(to_n(us, i_st[k], half))
                acc_t[k].append(to_n(th, i_st[k], half))
            jm, jn = np.full(it, np.nan), np.full(it, np.nan)
            for i in range(it):
                n = (z - dem[i]) * np.cos(beta[i])
                m = (n > 0.) & (n < 400.)
                if m.any():
                    k = int(np.argmax(us[m, i]))
                    jm[i], jn[i] = us[m, i][k], n[m][k]
            jet_x.append(jm)
            jetn_x.append(jn)
    with np.errstate(all='ignore'):
        import warnings
        warnings.simplefilter('ignore', RuntimeWarning)
        prof_u = {k: np.nanmean(v, axis=0) for k, v in acc_u.items()}
        prof_t = {k: np.nanmean(v, axis=0) for k, v in acc_t.items()}
    jet_x, jetn_x = np.nanmean(jet_x, axis=0), np.nanmean(jetn_x, axis=0)
    H_t, H_x = [], []
    for t in times:
        # the kinematic flux x the same rho cp as the rotated run (hfss_ib uses the
        # reference density, which is 1 in the Boussinesq set-up)
        f = os.path.join(di, f'thl_fluxbot_ib.xy.000.{t:07d}')
        if os.path.exists(f):
            H = RHO * CP * np.fromfile(f).reshape(jt, it).mean(axis=0)
            H_t.append(H[max(0, i_st['station'] - half):i_st['station'] + half + 1].mean())
            if t in tw:
                H_x.append(H)
    H_x = np.mean(H_x, axis=0) if H_x else np.full(it, np.nan)

    # rotated reference
    R = json.load(open(os.path.join(dr, 'case_parameters.json')))
    sf = sorted(glob.glob(os.path.join(dr, 'katabatic_rotated.default.*.nc')))
    rot = None
    if sf:
        d = nc.Dataset(sf[0])
        tr = np.array(d['time'][:])
        zr = np.array(d['z'][:])
        ur = R.get('downslope_sign', -1) * np.array(d['default']['u'][:])
        br = np.array(d['thermo']['b'][:])
        bfl = np.array(d['thermo']['b_flux'][:])
        mw = tr >= tr[-1] - win
        Hr = RHO * CP * R['theta_r'] / GRAV * bfl[:, 0]
        rot = dict(t=tr, z=zr, u=ur[mw].mean(axis=0), th=br[mw].mean(axis=0) * R['theta_r'] / GRAV,
                   jet_t=ur.max(axis=1), H_t=Hr, H=Hr[mw].mean())
        rot['jet'] = float(rot['u'].max())
        rot['jet_n'] = float(zr[np.argmax(rot['u'])])

    # figure
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5), constrained_layout=True)
    cols = {'top': '#1f3b73', 'mid': '#4a90c2', 'station': '#e08a2c', 'foot': '#b03a2e'}
    for k in st:
        ax[0, 0].plot(prof_u[k], n_grid, color=cols.get(k), label=f'IB {k} (x = {st[k]/1e3:.1f} km)')
        ax[0, 1].plot(prof_t[k], n_grid, color=cols.get(k))
    L = P['layer']
    nn = n_grid[n_grid > 0]
    if 'profile' in L:
        ax[0, 0].plot(np.interp(nn, L['profile']['n'], L['profile']['u'], right=0.), nn, 'k:',
                      label='inflow (rotated equilibrium)')
    else:
        ax[0, 0].plot(L['U_j'] * (nn / L['n_j']) * np.exp(1 - nn / L['n_j']), nn, 'k:', label='inflow (RACMO fit)')
    if 'profile' in L:
        ax[0, 1].plot(np.interp(nn, L['profile']['n'], L['profile']['dtheta'], left=L['profile']['dtheta'][0],
                                right=0.), nn, 'k:')
    else:
        ax[0, 1].plot(L['dtheta0'] * np.exp(-nn / L['h_theta']), nn, 'k:')
    if rot:
        ax[0, 0].plot(rot['u'], rot['z'], 'k-', lw=2, label=f'rotated frame ({R["alpha_deg"]:.2f} deg)')
        ax[0, 1].plot(rot['th'], rot['z'], 'k-', lw=2)
    for axx, lab in ((ax[0, 0], 'downslope wind u_s (m/s)'), (ax[0, 1], r'$\theta - \theta_a$ (K)')):
        axx.set_ylim(0, 300); axx.set_xlabel(lab); axx.set_ylabel('n, height above the surface (m)'); axx.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].set_title(f'a  wind profiles, mean {tend - win:.0f}-{tend:.0f} s')
    ax[0, 1].set_title('b  temperature deficit')
    a2 = ax[0, 2]
    a2.plot(x / 1e3, jet_x, 'b-', label='IB jet maximum')
    if rot:
        a2.axhline(rot['jet'], color='b', ls='--', label='rotated jet maximum')
    a2.set_ylabel('u_s max (m/s)'); a2.set_xlabel('x (km)'); a2.grid(alpha=0.3)
    a3 = a2.twinx()
    a3.plot(x / 1e3, jetn_x, 'g-', lw=0.8, label='IB jet height')
    if rot:
        a3.axhline(rot['jet_n'], color='g', ls='--', lw=0.8)
    a3.set_ylabel('height of the maximum (m)', color='g')
    a2.legend(fontsize=8, loc='upper left')
    a2.set_title('c  jet along the slope')
    for v in (P['x_plateau_end'], P['x_foot']):
        a2.axvline(v / 1e3, color='0.6', lw=0.8)
    a2.axvline(P['x_site'] / 1e3, color='#e08a2c', lw=0.8, ls=':')
    b0 = ax[1, 0]
    b0.plot(x / 1e3, H_x, 'r-', label='IB')
    if rot:
        b0.axhline(rot['H'], color='r', ls='--', label='rotated')
    b0.set_ylabel('H (W/m2)'); b0.set_xlabel('x (km)'); b0.grid(alpha=0.3); b0.legend(fontsize=8)
    b0.set_title('d  sensible heat flux along the slope')
    b0b = b0.twinx()
    b0b.fill_between(x / 1e3, P['datum_amsl'] + dem, P['datum_amsl'] + dem.min() - 50., color='0.85', zorder=-1)
    b0b.set_ylabel('terrain (m amsl)', color='0.5')
    b0.set_zorder(b0b.get_zorder() + 1)          # the flux in front of the terrain
    b0.patch.set_visible(False)
    b1 = ax[1, 1]
    tt = np.array(times) / 3600.
    b1.plot(tt, ts_jet['station'], color='#e08a2c', label='IB station')
    b1.plot(tt, ts_jet['mid'], color='#4a90c2', label='IB mid')
    if rot:
        b1.plot(rot['t'] / 3600., rot['jet_t'], 'k-', label='rotated')
    b1.set_xlabel('t (h)'); b1.set_ylabel('jet maximum (m/s)'); b1.grid(alpha=0.3); b1.legend(fontsize=8)
    b1.set_title('e  jet maximum in time')
    b2 = ax[1, 2]
    th_ = np.array(times[:len(H_t)]) / 3600.
    b2.plot(th_, H_t, color='#e08a2c', label='IB station')
    if rot:
        b2.plot(rot['t'] / 3600., rot['H_t'], 'k-', label='rotated')
    b2.set_xlabel('t (h)'); b2.set_ylabel('H (W/m2)'); b2.grid(alpha=0.3); b2.legend(fontsize=8)
    b2.set_title('f  surface heat flux in time')
    fig.suptitle(f'katabatic_idealised: Cartesian IB ({P["momentum"]}) against the rotated frame, '
                 f'tau_osc = {P["tau_osc"]:.0f} s, run to {tend} s', fontsize=11)
    fig.savefig(os.path.join(out, 'katabatic_compare.png'), dpi=90)

    met = dict(window=[tend - win, tend],
               ib={k: dict(jet=float(np.nanmax(prof_u[k])), jet_n=float(n_grid[np.nanargmax(prof_u[k])]),
                           dtheta_1=first(prof_t[k]), n_1=float(n_grid[np.isfinite(prof_t[k])][0])) for k in st},
               ib_H_station=float(np.nanmean(H_x[max(0, i_st['station'] - half):i_st['station'] + half + 1])),
               ib_H_slope_mean=float(np.nanmean(H_x[(x > P['x_plateau_end']) & (x < P['x_foot'])])))
    if rot:
        met['rotated'] = dict(jet=rot['jet'], jet_n=rot['jet_n'], H=float(rot['H']),
                              dtheta_1=float(rot['th'][0]), n_1=float(rot['z'][0]))
    json.dump(met, open(os.path.join(out, 'metrics.json'), 'w'), indent=1)
    print(f'window {tend - win:.0f}-{tend:.0f} s')
    for k, v in met['ib'].items():
        print(f'  IB {k:7s}: jet {v["jet"]:5.2f} m/s at {v["jet_n"]:5.0f} m, theta - theta_a at the first level '
              f'(n = {v["n_1"]:.0f} m) {v["dtheta_1"]:6.2f} K')
    print(f'  IB H at the station {met["ib_H_station"]:.1f} W/m2, slope mean {met["ib_H_slope_mean"]:.1f} W/m2')
    if rot:
        r = met['rotated']
        print(f'  rotated : jet {r["jet"]:5.2f} m/s at {r["jet_n"]:5.0f} m, theta - theta_a at n = {r["n_1"]:.0f} m '
              f'{r["dtheta_1"]:6.2f} K, '
              f'H {r["H"]:.1f} W/m2')
    print(f'figure and metrics in {out}')


if __name__ == '__main__':
    main()
