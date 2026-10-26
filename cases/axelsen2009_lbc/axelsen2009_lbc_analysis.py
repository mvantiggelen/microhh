#!/usr/bin/env python3
"""
Slope-averaged profiles of an axelsen2009_lbc run as functions of the height
above the local surface, the figures of Axelsen and van Dop (2009, Part 1;
= Axelsen 2010, ch. 3) for case P1, and the comparison with the observed
profiles at Pasterze site A1 (Smeets et al. 1998; the JD 212 mast and balloon
data as plotted by Axelsen).

    python3 axelsen2009_lbc_analysis.py --rundir runs/P1_lbc [--outdir figs]
            [--obs-json observations_P1.json] [--obs extra.csv]
            [--tavg 3600] [--dn 1] [--segment 0.25,0.75]

Slope average
-------------
Every xz plane, every output time and every column on the straight part of
the slope (the breaks + 1 cell excluded; --segment chooses a part of it,
default the middle half) gives one sample per air cell above the local
surface, at the slope-normal height n = h cos(beta), h = z - z_s(x), beta the
local terrain angle:

  along-slope (downslope) wind   u_s = u cos(beta) - w sin(beta)
  slope-normal wind              w_n = u sin(beta) + w cos(beta)
  buoyancy                       b = g (thl - thl_a(z_s + h)) / theta_a0

(u, w interpolated to the cell centre.) The samples are averaged in bins of
n (--dn). On the staircase the cell centres sit at a different h in every
column, so the bins resolve heights below dz; each sample is still a cell
mean. Fluxes: resolved u_s'w_n' and w_n'b' (departures from the time mean of
the same cell over the window: no dispersive flux of the staircase), plus the SFS part -K du_s/dn, -K_h db/dn when the run
has the evisc (and eviscs) xz crosses; the surface values are the IB wall
fluxes (ustar_ib, thl_fluxbot_ib) of the same columns. Time average: the last
tau_osc (as the paper) if the run is longer than 1.5 tau_osc, else the last
third, or --tavg seconds.

Figures (<outdir>/P1_lbc_*.png)
  fig3_3a      u_s and theta against height, 0-600 m, with the JD 212 sounding (paper Fig. 3.3a)
  fig3_4       UD/(VH), BD/(BH) and u/V at z/H = 0.06, 0.15, 1.01, 2.01 against t/tau_osc (Fig. 3.4)
  fig3_5       z_inv/H and b_s/B against t/tau_osc (Fig. 3.5)
  fig3_7       u/V, tau/V^2, b/B, F/(VB) against z/H <= 1.5 with mast, balloon and paper LES (Fig. 3.7)
  observed     the same in metres, m/s and deg C, log height, with the JD 212 mast and the
               Smeets et al. (1998) A1 mean profiles (Figs. 3, 4d)
  profiles     u_s/V and b/B against n/H up to 6 H
  timeheight   slope-averaged u_s and theta - theta_a against time and n
  alongslope   profiles on the upper, middle and lower third of the slope
and P1_lbc_metrics.json.
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GRAV = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
C_MOD, C_LES, C_OBS = 'tab:blue', '0.35', 'k'


def find(name):
    for d in (HERE, os.path.join(HERE, '..', 'axelsen2009')):
        f = os.path.join(d, name)
        if os.path.exists(f):
            return f
    return None


def load_obs_csv(path, case):
    obs = {}
    if not path:
        return obs
    with open(path) as f:
        rows = list(csv.DictReader(l for l in f if not l.lstrip().startswith('#')))
    for r in rows:
        if r['case'].strip() != case:
            continue
        o = obs.setdefault(r['source'].strip(), {'z': [], 'u': [], 'th': []})
        o['z'].append(float(r['z_m']))
        o['u'].append(float(r['u_ms']) if (r.get('u_ms') or '').strip() else np.nan)
        o['th'].append(float(r['theta_C']) if (r.get('theta_C') or '').strip() else np.nan)
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in obs.items()}


class Run:
    def __init__(self, rd):
        self.rd = rd
        p = self.p = json.load(open(os.path.join(rd, 'case_parameters.json')))
        self.it, self.jt, self.kt = p['itot'], p['jtot'], p['ktot']
        self.dx, self.dz = p['dx'], p['dz']
        self.x = (np.arange(self.it) + 0.5) * self.dx
        self.z = (np.arange(self.kt) + 0.5) * self.dz
        dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(self.jt, self.it)
        self.zs = dem.mean(axis=0)
        self.beta = np.arctan(-np.gradient(self.zs, self.dx))
        f = sorted(glob.glob(os.path.join(rd, 'thl.xz.000.*')))
        self.planes = sorted({s.split('.')[-2] for s in f})
        self.times = np.array(sorted({int(s.split('.')[-1]) for s in f}))
        self.thl_a = lambda zz: p['theta_a0'] + p['gamma'] * (zz - p['z_ref'])
        self.has_k = bool(glob.glob(os.path.join(rd, 'evisc.xz.*')))
        self.has_kh = bool(glob.glob(os.path.join(rd, 'eviscs.xz.*')))

    def plane(self, name, loc, j, t):
        f = os.path.join(self.rd, '%s.xz.%s.%s.%07d' % (name, loc, j, t))
        return np.fromfile(f).reshape(self.kt, self.it)

    def xy(self, name, t):
        f = os.path.join(self.rd, '%s.xy.000.%07d' % (name, t))
        return np.fromfile(f).reshape(self.jt, self.it) if os.path.exists(f) else None

    def columns(self, segment):
        p = self.p
        x0 = p['x_top'] + 0.5 * p['break_width'] + self.dx
        x1 = p['x_foot'] - 0.5 * p['break_width'] - self.dx
        a, b = x0 + segment[0] * (x1 - x0), x0 + segment[1] * (x1 - x0)
        return np.where((self.x >= a) & (self.x <= b))[0]

    def samples(self, t, cols):
        """Per air cell of the columns (all planes): n, u_s, w_n, b, and the
        SFS fluxes (nan without the evisc crosses)."""
        th0 = self.p['theta_a0']
        cb, sb = np.cos(self.beta[cols]), np.sin(self.beta[cols])
        out = {k: [] for k in ('n', 'us', 'wn', 'b', 'tsgs', 'fsgs', 'key')}
        nc_ = len(cols)
        for ip, j in enumerate(self.planes):
            u = self.plane('u', '100', j, t)
            w = self.plane('w', '001', j, t)
            th = self.plane('thl', '000', j, t)
            uc = 0.5 * (u + np.roll(u, -1, axis=1))
            uc[:, -1] = u[:, -1]
            wc = 0.5 * (w + np.vstack([w[1:], np.zeros((1, self.it))]))
            h = self.z[:, None] - self.zs[None, cols]
            air = h > 0.
            us = uc[:, cols] * cb - wc[:, cols] * sb
            wn = uc[:, cols] * sb + wc[:, cols] * cb
            b = GRAV / th0 * (th[:, cols] - self.thl_a(self.zs[None, cols] + h))
            ts = np.full(us.shape, np.nan)
            fs = np.full(us.shape, np.nan)
            if self.has_k:
                km = self.plane('evisc', '000', j, t)[:, cols]
                kh = self.plane('eviscs', '000', j, t)[:, cols] if self.has_kh else km
                # d/dn ~ cos(beta) d/dz for a flow that varies with n only;
                # the derivative from air cells only (central, else one-sided),
                # never across the terrain.
                def ddz(f):
                    d = np.full(f.shape, np.nan)
                    a_dn = np.zeros_like(air); a_dn[1:] = air[:-1]
                    a_up = np.zeros_like(air); a_up[:-1] = air[1:]
                    c = air & a_dn & a_up
                    d[1:-1][c[1:-1]] = ((f[2:] - f[:-2]) / (2. * self.dz))[c[1:-1]]
                    fw = air & ~a_dn & a_up
                    d[:-1][fw[:-1]] = ((f[1:] - f[:-1]) / self.dz)[fw[:-1]]
                    return d
                ts = -km * ddz(us) * cb
                fs = -kh * ddz(b) * cb
                lowest = np.argmax(air, axis=0)       # the wall model sets that cell's flux
                ts[lowest, np.arange(nc_)] = np.nan
                fs[lowest, np.arange(nc_)] = np.nan
            key = (ip * self.kt + np.arange(self.kt)[:, None]) * nc_ + np.arange(nc_)[None, :]
            for k, v in (('n', h * cb), ('us', us), ('wn', wn), ('b', b), ('tsgs', ts), ('fsgs', fs),
                         ('key', key)):
                out[k].append(v[air])
        return {k: np.concatenate(v) for k, v in out.items()}

    def surface(self, t, cols):
        """Wall values on the columns: tau_s = -u*^2, F_s = g/theta0 w'thl'_s, b_s."""
        th0 = self.p['theta_a0']
        us = self.xy('ustar_ib', t)
        fl = self.xy('thl_fluxbot_ib', t)
        ts = self.xy('thl_sbot_ib', t)
        r = {}
        r['tau'] = -np.mean(us[:, cols]**2) if us is not None else np.nan
        r['F'] = GRAV / th0 * np.mean(fl[:, cols]) if fl is not None else np.nan
        r['b'] = GRAV / th0 * np.mean(ts[:, cols] - self.thl_a(self.zs[None, cols])) if ts is not None else np.nan
        return r


def binned(n, v, edges):
    ok = np.isfinite(v)
    idx = np.digitize(n[ok], edges) - 1
    v = v[ok]
    m = (idx >= 0) & (idx < len(edges) - 1)
    cnt = np.bincount(idx[m], minlength=len(edges) - 1)
    s1 = np.bincount(idx[m], v[m], minlength=len(edges) - 1)
    s2 = np.bincount(idx[m], v[m]**2, minlength=len(edges) - 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = s1 / cnt
        sd = np.sqrt(np.maximum(s2 / cnt - mean**2, 0.))
    mean[cnt == 0] = np.nan
    return mean, sd, cnt


def flux_resolved(pooled, edges):
    """Resolved u_s'w_n' and w_n'b' over the averaging window: departures from
    the TIME mean of the same cell (plane, level, column), so the stationary
    spatial pattern of the staircase (a dispersive flux) is not counted."""
    key = np.concatenate([q['key'] for q in pooled])
    n = np.concatenate([q['n'] for q in pooled])
    out = []
    for f1, f2 in (('us', 'wn'), ('wn', 'b')):
        a = np.concatenate([q[f1] for q in pooled])
        b = np.concatenate([q[f2] for q in pooled])
        uk, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv)
        ma = np.bincount(inv, a) / cnt
        mb = np.bincount(inv, b) / cnt
        out.append(binned(n, (a - ma[inv]) * (b - mb[inv]), edges)[0])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--outdir', default=None, help='default: <rundir>/figs')
    ap.add_argument('--obs-json', default=None,
                    help='digitised observations (digitize_observations.py; default: observations_P1.json next to this script)')
    ap.add_argument('--obs', default=None, help='more observed profiles, CSV: case,source,z_m,u_ms,theta_C')
    ap.add_argument('--balloon', default=None,
                    help='Pasterze ADAS soundings (pasterze_balloon.py; default: pasterze_balloon_1994.json next to this script)')
    ap.add_argument('--tavg', type=float, default=None, help='averaging window at the end of the run [s]')
    ap.add_argument('--dn', type=float, default=None, help='bin width in n [m] (default min(1, dz/2))')
    ap.add_argument('--dn-flux', type=float, default=None,
                    help='bin width in n for the fluxes [m] (default max(1, dz/2): on the coarse grid 1 m bins are noisy)')
    ap.add_argument('--nmax', type=float, default=600., help='top of the profiles [m]')
    ap.add_argument('--integral-top', type=float, default=51.,
                    help='top of the UD, BD integrals [m] (paper: its 51 m domain)')
    ap.add_argument('--segment', default='0.25,0.75',
                    help='part of the straight slope, fractions top->foot (0,1 = all of it)')
    a = ap.parse_args()

    R = Run(a.rundir)
    p = R.p
    V, B, H, tau, th0 = p['V'], p['B'], p['H'], p['tau_osc'], p['theta_a0']
    case = p.get('case', 'P1')
    oj = a.obs_json or find(f'observations_{case}.json')
    O = json.load(open(oj)) if oj and os.path.exists(oj) else None
    ref = {}
    rf = find('reference_axelsen2009.json')
    if rf:
        ref = json.load(open(rf))['cases'].get(case, {})
    les_ref = ref.get('les', {})
    extra = load_obs_csv(a.obs, case)
    bf = a.balloon or find('pasterze_balloon_1994.json')
    BAL = json.load(open(bf)) if bf and os.path.exists(bf) else None
    p1s = None
    if BAL:
        p1s = next((q for q in BAL['soundings'] if q['doy'] == 212 and q['launch'] == '10:25'), None)
    if p1s:
        # Ascent only (as the paper); ambient theta fitted over 70-550 m as the paper (Table 3.1).
        up = np.array(p1s['ascent'])
        bz, bu, bth = (np.array(p1s[k])[up] for k in ('z_m', 'u_slope', 'theta_C'))
        mfit = (bz >= 70.) & (bz <= 550.)
        cfit = np.polyfit(bz[mfit], bth[mfit], 1)
        bb_obs = GRAV / th0 * (bth - np.polyval(cfit, bz))
        N_obs = np.sqrt(GRAV / (cfit[1] + 273.15) * cfit[0])
    outdir = a.outdir or os.path.join(a.rundir, 'figs')
    os.makedirs(outdir, exist_ok=True)
    tag = f'{case}_lbc'
    seg = tuple(float(s) for s in a.segment.split(','))
    cols = R.columns(seg)

    times = R.times[R.times > 0]
    if len(times) == 0:
        raise SystemExit('no xz crosses after t = 0')
    if a.tavg is not None:
        last = times >= times[-1] - a.tavg
        window = f'last {a.tavg:g} s'
    elif times[-1] >= 1.5 * tau:
        last = times >= times[-1] - tau
        window = 'last tau_osc'
    else:
        last = times >= times[-1] * 2. / 3.
        window = 'last third (run < 1.5 tau_osc: not the paper\'s steady state)'

    if a.dn is None:
        a.dn = min(1., 0.5 * R.dz)
    if a.dn_flux is None:
        a.dn_flux = max(1., 0.5 * R.dz)
    edges = np.arange(0., a.nmax + a.dn, a.dn)
    nc = 0.5 * (edges[1:] + edges[:-1])
    nt = len(times)
    U = np.full((nt, len(nc)), np.nan)
    Bt = np.full_like(U, np.nan)
    TS, FS = (np.full_like(U, np.nan) for _ in range(2))
    edges_f = np.arange(0., a.nmax + a.dn_flux, a.dn_flux)
    ncf = 0.5 * (edges_f[1:] + edges_f[:-1])
    TSf, FSf = [], []
    surf = {k: np.full(nt, np.nan) for k in ('tau', 'F', 'b')}
    pooled = []
    for it, t in enumerate(times):
        S = R.samples(t, cols)
        U[it] = binned(S['n'], S['us'], edges)[0]
        Bt[it] = binned(S['n'], S['b'], edges)[0]
        TS[it] = binned(S['n'], S['tsgs'], edges)[0]
        FS[it] = binned(S['n'], S['fsgs'], edges)[0]
        if last[it]:
            TSf.append(binned(S['n'], S['tsgs'], edges_f)[0])
            FSf.append(binned(S['n'], S['fsgs'], edges_f)[0])
        for k, v in R.surface(t, cols).items():
            surf[k][it] = v
        if last[it]:
            pooled.append(S)
    P = {k: np.concatenate([s[k] for s in pooled]) for k in pooled[0]}
    um, usd, cnt = binned(P['n'], P['us'], edges)
    bm, bsd, _ = binned(P['n'], P['b'], edges)
    good = cnt > 0
    tr, fr = flux_resolved(pooled, edges_f)
    with np.errstate(all='ignore'):
        tsg = np.nanmean(np.array(TSf), axis=0)
        fsg = np.nanmean(np.array(FSf), axis=0)
    # Bins holding only wall-model cells (no SFS estimate) are left out.
    tot_t = tr + (tsg if R.has_k else 0.)
    tot_f = fr + (fsg if R.has_k else 0.)
    tau_s, F_s, b_s = (float(np.nanmean(surf[k][last])) for k in ('tau', 'F', 'b'))

    # Metrics.
    k = int(np.nanargmax(np.where(good, um, -np.inf)))
    zj = nc[k]
    if 0 < k < len(nc) - 1 and np.all(good[k-1:k+2]):
        c = np.polyfit(nc[k-1:k+2], um[k-1:k+2], 2)
        if c[0] < 0:
            zj = -c[1] / (2. * c[0])
    first = int(np.argmax(good))
    pos = np.where(good & (bm >= 0.))[0]
    zinv = nc[pos[0]] if len(pos) else np.nan
    top = nc <= a.integral_top
    UD = _trapz(np.nan_to_num(U[:, top]), nc[top], axis=1)
    BD = _trapz(np.nan_to_num(Bt[:, top]), nc[top], axis=1)
    zinv_t = np.array([nc[np.where(np.isfinite(r) & (r >= 0.))[0][0]] if np.any(np.isfinite(r) & (r >= 0.))
                       else np.nan for r in Bt])
    m = dict(window=window, from_s=float(times[last][0]), to_s=float(times[-1]), segment=seg,
             x_range_m=[float(R.x[cols[0]]), float(R.x[cols[-1]])], n_planes=len(R.planes),
             umax_ms=float(um[k]), umax_over_V=float(um[k] / V), zj_m=float(zj), zj_over_H=float(zj / H),
             b_lowest_over_B=float(bm[first] / B), n_lowest_m=float(nc[first]),
             bs_over_B=b_s / B, tau_s_over_V2=tau_s / V**2, Fs_over_VB=F_s / (V * B),
             zinv_m=float(zinv), zinv_over_H=float(zinv / H), sfs_flux_included=R.has_k,
             UD_over_VH_end=float(UD[-1] / (V * H)), BD_over_BH_end=float(BD[-1] / (B * H)),
             n=nc[good].tolist(), u_s=um[good].tolist(), b=bm[good].tolist(),
             n_flux=ncf.tolist(), tau=[None if not np.isfinite(v) else float(v) for v in tot_t],
             F=[None if not np.isfinite(v) else float(v) for v in tot_f])
    json.dump(m, open(os.path.join(outdir, f'{tag}_metrics.json'), 'w'), indent=1)

    title = (f"MicroHH {tag} (dx {R.dx:g} m, dz {R.dz:g} m): slope x = {R.x[cols[0]]/1e3:.1f}-"
             f"{R.x[cols[-1]]/1e3:.1f} km, {len(R.planes)} xz plane(s), {window} "
             f"({times[last][0]:.0f}-{times[-1]:.0f} s)")
    lvl1 = 0.5 * R.dz
    f37 = O['axelsen']['fig3_7'] if O else None

    def obs37(ax, key, scale_mast=True):
        if not f37:
            return
        e = f37[key]
        les = np.array(e['les'])
        ax.plot(les[:, 0], les[:, 1], '-', color=C_LES, lw=1.3, label='Axelsen & van Dop (2009) LES')
        mm = e['mast']
        ax.errorbar([q['value'] for q in mm], [q['z_over_H'] for q in mm],
                    xerr=[q['sd'] or 0. for q in mm], fmt='s', ms=5, mfc='white', color=C_OBS,
                    capsize=0, lw=0.8, label='Mast, JD 212 (30-min mean +- sd)', zorder=6)
        if p1s is not None and key in ('u_over_V', 'b_over_B'):
            xv = bu / V if key == 'u_over_V' else bb_obs / B
            ax.plot(xv, bz / H, '+', color=C_OBS, ms=5, mew=0.8, label='Balloon, JD 212 10:25 (ADAS ascent)', zorder=5)
        elif 'balloon' in e:
            bb = np.array(e['balloon'])
            ax.plot(bb[:, 1], bb[:, 0], '+', color=C_OBS, ms=5, mew=0.8, label='Balloon, JD 212 (digitised)', zorder=5)

    # ---- Fig. 3.7 ------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(10, 9.5))
    y = nc / H
    g = good & (y <= 1.55)
    a0 = ax[0, 0]
    a0.fill_betweenx(y[g], ((um - usd) / V)[g], ((um + usd) / V)[g], color=C_MOD, alpha=0.15, lw=0)
    a0.plot((um / V)[g], y[g], '-', color=C_MOD, lw=2, label='MicroHH, slope average')
    obs37(a0, 'u_over_V')
    a0.set_xlabel('u / V'); a0.set_title('(a) Downslope velocity', fontsize=10)
    a1 = ax[0, 1]
    yf = ncf / H
    gt = np.isfinite(tot_t) & (yf <= 1.55)
    a1.plot((tot_t / V**2)[gt], yf[gt], 'o-', color=C_MOD, lw=2, ms=3,
            label='MicroHH, resolved + SFS' if R.has_k else 'MicroHH, RESOLVED only (no evisc crosses)')
    a1.plot(tau_s / V**2, 0., '>', color=C_MOD, ms=9, clip_on=False, label='MicroHH wall stress -u*^2')
    obs37(a1, 'tau_over_V2')
    a1.axvline(0, color='0.6', lw=0.6)
    a1.set_xlabel('tau / V^2'); a1.set_title('(b) Momentum flux', fontsize=10)
    a2 = ax[1, 0]
    a2.fill_betweenx(y[g], ((bm - bsd) / B)[g], ((bm + bsd) / B)[g], color=C_MOD, alpha=0.15, lw=0)
    a2.plot((bm / B)[g], y[g], '-', color=C_MOD, lw=2, label='MicroHH, slope average')
    if b_s / B >= -85.:
        a2.plot(b_s / B, 0., '>', color=C_MOD, ms=9, clip_on=False, label='MicroHH surface b_s')
    else:   # MOST surface value far off the paper's axis: say it instead of squeezing the panel
        a2.annotate(f'MicroHH surface b_s/B = {b_s / B:.0f}', xy=(-80, 0.), xytext=(-78, 0.12),
                    fontsize=7, color=C_MOD, arrowprops=dict(arrowstyle='->', color=C_MOD))
    a2.set_xlim(-80, 10)
    obs37(a2, 'b_over_B')
    a2.axvline(0, color='0.6', lw=0.6)
    a2.set_xlabel('b / B'); a2.set_title('(c) Buoyancy', fontsize=10)
    a3 = ax[1, 1]
    gf = np.isfinite(tot_f) & (yf <= 1.55)
    a3.plot((tot_f / (V * B))[gf], yf[gf], 'o-', color=C_MOD, lw=2, ms=3,
            label='MicroHH, resolved + SFS' if R.has_k else 'MicroHH, RESOLVED only (no evisc crosses)')
    a3.plot(F_s / (V * B), 0., '>', color=C_MOD, ms=9, clip_on=False, label='MicroHH wall flux')
    obs37(a3, 'F_over_VB')
    a3.set_xlabel('F / (VB)'); a3.set_title('(d) Buoyancy flux', fontsize=10)
    for axx in ax.flat:
        axx.set_ylim(0, 1.5)
        axx.set_ylabel('z / H')
        axx.axhline(lvl1 / H, color='0.6', ls=':', lw=0.8)
        axx.grid(alpha=0.25)
        axx.legend(fontsize=7, loc='upper right' if axx is not a2 else 'upper left')
    sec = ax[0, 1].secondary_yaxis('right', functions=(lambda v: v * H, lambda v: v / H))
    sec.set_ylabel('n (m)')
    fig.suptitle(f'{title}\nas Axelsen and van Dop (2009) Fig. 7 / Axelsen (2010) Fig. 3.7: Pasterze A1, '
                 f'JD 212; dotted: first model level over flat ground ({lvl1:g} m)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_fig3_7.png'), dpi=130)
    plt.close(fig)

    # ---- Fig. 3.3a: dimensional, 0-600 m ---------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw=dict(width_ratios=[1.4, 1]))
    th_mod = th0 - 273.15 + p['gamma'] * nc + bm * th0 / GRAV
    g6 = good & (nc <= 600.)
    for axx, zt in ((ax[0], 600.), (ax[1], 120.)):
        if p1s is not None:
            dn_ = ~np.array(p1s['ascent'])
            axx.plot(bu, bz, '.', color=C_OBS, ms=3, label='Balloon u_s, JD 212 10:25 ascent')
            axx.plot(bth, bz, '.', color='tab:red', ms=3, label='Balloon theta, ascent')
            axx.plot(np.array(p1s['u_slope'])[dn_], np.array(p1s['z_m'])[dn_], '.', color='0.7', ms=2,
                     label='descent (not used by the paper)')
            axx.plot(np.array(p1s['theta_C'])[dn_], np.array(p1s['z_m'])[dn_], '.', color='0.7', ms=2)
            zz = np.linspace(0., 600., 50)
            axx.plot(np.polyval(cfit, zz), zz, '-', color='tab:red', lw=0.8,
                     label=f'Balloon theta_a fit 70-550 m (N = {N_obs:.4f} 1/s)')
        elif O:
            s = np.array(O['axelsen']['fig3_3a']['z_m_u_ms'])
            axx.plot(s[:, 1], s[:, 0], '.', color=C_OBS, ms=3, label='Balloon u, JD 212 10:30 (digitised)')
            s = np.array(O['axelsen']['fig3_3a']['z_m_theta_C'])
            axx.plot(s[:, 1], s[:, 0], '.', color='tab:red', ms=3, label='Balloon theta (digitised)')
        axx.plot(um[g6], nc[g6], '-', color=C_MOD, lw=2, label='MicroHH u_s')
        axx.plot(th_mod[g6], nc[g6], '-', color='tab:orange', lw=2, label='MicroHH theta')
        axx.plot(th0 - 273.15 + p['gamma'] * nc[g6], nc[g6], ':', color='tab:orange', lw=1,
                 label='theta_a (model, = paper)')
        axx.set_ylim(0, zt)
        axx.grid(alpha=0.25)
        axx.set_xlabel('downslope wind (m/s), potential temperature (deg C)')
    ax[0].set_xlim(-5, 15); ax[1].set_xlim(-1, 14)
    ax[0].set_ylabel('height above the surface (m)')
    ax[0].legend(fontsize=7, loc='upper left')
    ax[1].set_title('lowest 120 m', fontsize=9)
    fig.suptitle(f'{title}\nas Axelsen (2010) Fig. 3.3a (theta: model theta_a0 = {th0 - 273.15:.2f} deg C '
                 f'at the surface, as the paper)', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_fig3_3a.png'), dpi=130)
    plt.close(fig)

    # ---- Fig. 3.4 --------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    tt = times / tau
    ax[0].plot(tt, UD / (V * H), '-', color=C_MOD, lw=2, label=f'MicroHH UD/(VH), n < {a.integral_top:g} m')
    ax[0].plot(tt, BD / (B * H), '--', color=C_MOD, lw=2, label='MicroHH BD/(BH)')
    if O:
        f = O['axelsen']['fig3_4']
        c = np.array(f['UD_over_VH']); ax[0].plot(c[:, 0], c[:, 1], '-', color=C_LES, label='paper LES UD/(VH)')
        c = np.array(f['BD_over_BH']); ax[0].plot(c[:, 0], c[:, 1], '--', color=C_LES, label='paper LES BD/(BH)')
    ax[0].axhline(0, color='0.6', lw=0.5)
    ax[0].set_ylabel('non-dimensional'); ax[0].set_title('(a) vertically integrated', fontsize=10)
    styles = {'0.06': '-', '0.15': '--', '1.01': '-.', '2.01': ':'}
    for zh, ls in styles.items():
        zz = float(zh) * H
        ser = np.array([np.interp(zz, nc[np.isfinite(r)], r[np.isfinite(r)]) if np.any(np.isfinite(r)) else np.nan
                        for r in U])
        ax[1].plot(tt, ser / V, ls, color=C_MOD, lw=1.8, label=f'MicroHH z/H = {zh} ({zz:.1f} m)')
        if O:
            c = np.array(O['axelsen']['fig3_4']['u_over_V'][zh])
            ax[1].plot(c[:, 0], c[:, 1], ls, color=C_LES, lw=1, label=f'paper LES {zh}')
    ax[1].set_ylabel('u / V'); ax[1].set_title('(b) downslope velocity at four heights', fontsize=10)
    for axx in ax:
        axx.set_xlabel('t / tau_osc'); axx.set_xlim(0, 3); axx.grid(alpha=0.25); axx.legend(fontsize=6.5, ncol=2)
    fig.suptitle(f'{title}\nas Axelsen (2010) Fig. 3.4', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_fig3_4.png'), dpi=130)
    plt.close(fig)

    # ---- Fig. 3.5 --------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(tt, zinv_t / H, '-', color=C_MOD, lw=2, label='MicroHH (slope average)')
    ax[1].plot(tt, surf['b'] / B, '-', color=C_MOD, lw=2, label='MicroHH thl_sbot_ib - thl_a(z_s)')
    if O:
        c = np.array(O['axelsen']['fig3_5']['zinv_over_H']); ax[0].plot(c[:, 0], c[:, 1], '-', color=C_LES, label='paper LES')
        c = np.array(O['axelsen']['fig3_5']['bs_over_B']); ax[1].plot(c[:, 0], c[:, 1], '-', color=C_LES, label='paper LES')
    ax[0].set_ylabel('z_inv / H'); ax[0].set_title('(a) height where b first becomes positive', fontsize=10)
    ax[1].set_ylabel('b_s / B'); ax[1].set_title('(b) surface buoyancy', fontsize=10)
    for axx in ax:
        axx.set_xlabel('t / tau_osc'); axx.set_xlim(0, 3); axx.grid(alpha=0.25); axx.legend(fontsize=8)
    ax[0].set_yscale('log')
    fig.suptitle(f'{title}\nas Axelsen (2010) Fig. 3.5 (z_inv on a log axis: the model value is far above the paper\'s)',
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_fig3_5.png'), dpi=130)
    plt.close(fig)

    # ---- Observed, dimensional, log height ----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    gl = good & (nc >= 0.2) & (nc <= 400.)
    ax[0].plot(um[gl], nc[gl], '-', color=C_MOD, lw=2, label='MicroHH, slope average (P1 forcing)')
    ax[1].plot(th_mod[gl], nc[gl], '-', color=C_MOD, lw=2, label='MicroHH')
    if f37:
        mm = f37['u_over_V']['mast']
        ax[0].errorbar([q['value'] * V for q in mm], [q['z_over_H'] * H for q in mm],
                       xerr=[(q['sd'] or 0.) * V for q in mm], fmt='s', mfc='white', color=C_OBS, ms=5,
                       lw=0.8, label='Mast JD 212 10:30 (P1)', zorder=6)
        les = np.array(f37['u_over_V']['les'])
        ax[0].plot(les[:, 0] * V, les[:, 1] * H, '-', color=C_LES, lw=1.2, label='paper LES P1')
        mm = f37['b_over_B']['mast']
        ax[1].errorbar([th0 - 273.15 + p['gamma'] * q['z_over_H'] * H + q['value'] * B * th0 / GRAV for q in mm],
                       [q['z_over_H'] * H for q in mm], xerr=[(q['sd'] or 0.) * B * th0 / GRAV for q in mm],
                       fmt='s', mfc='white', color=C_OBS, ms=5, lw=0.8,
                       label='Mast JD 212 (from b/B, theta_a as the paper)', zorder=6)
        les = np.array(f37['b_over_B']['les'])
        ax[1].plot(th0 - 273.15 + p['gamma'] * les[:, 1] * H + les[:, 0] * B * th0 / GRAV, les[:, 1] * H,
                   '-', color=C_LES, lw=1.2, label='paper LES P1')
    if O:
        s = O['smeets']
        c = np.array(s['fig4d']['z_m_U_ms']); ax[0].plot(c[:, 1], c[:, 0], 'o-', color='tab:green', mfc='none', ms=5,
                                                         lw=0.8, label='Smeets et al. (1998) Fig. 4d, A1 mast mean*')
        c = np.array(s['fig3']['z_m_speed_ms']); ax[0].plot(c[:, 1], c[:, 0], '-', color='tab:green', lw=0.8,
                                                            alpha=0.7, label='Smeets et al. (1998) Fig. 3, 12 soundings*')
        c = np.array(s['fig4d']['z_m_T_C']); ax[1].plot(c[:, 1], c[:, 0], 'o-', color='tab:green', mfc='none', ms=5,
                                                        lw=0.8, label='Smeets et al. (1998) Fig. 4d (T)*')
        c = np.array(s['fig3']['z_m_T_C']); ax[1].plot(c[:, 1], c[:, 0], '-', color='tab:green', lw=0.8,
                                                       alpha=0.7, label='Smeets et al. (1998) Fig. 3 (T)*')
    for src, o in extra.items():
        ax[0].plot(o['u'], o['z'], 'D', mfc='none', label=src)
        ax[1].plot(o['th'], o['z'], 'D', mfc='none', label=src)
    for axx in ax:
        axx.set_yscale('log'); axx.set_ylim(0.2, 400); axx.grid(alpha=0.25, which='both')
        axx.axhline(lvl1, color='0.6', ls=':', lw=0.8)
        axx.legend(fontsize=6.5, loc='upper left')
    ax[0].set_xlabel('wind speed (m/s)'); ax[1].set_xlabel('potential / air temperature (deg C)')
    ax[0].set_ylabel('height above the surface (m)')
    fig.suptitle(f'{title}\n* Smeets et al. (1998): means over katabatic runs with NO wind maximum below 13 m '
                 f'(large-scale forcing), not the P1 day; their T is air temperature', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_observed.png'), dpi=130)
    plt.close(fig)

    # ---- All PASTEX soundings of 31 Jul and 1 Aug 1994 ---------------------------------------
    if BAL:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
        cmap = plt.get_cmap('twilight_shifted')
        for q in BAL['soundings']:
            up = np.array(q['ascent'])
            zq = np.array(q['z_m'])[up]
            hh_, mm_ = (int(v) for v in q['launch'].split(':'))
            col = cmap(((hh_ + mm_ / 60.) % 24.) / 24.)
            lw = 2.2 if q is p1s else 0.8
            if q is p1s:
                col = 'tab:red'
            lab = f"{q['date'][5:]} {q['launch']}" + (' (P1)' if q is p1s else '')
            ax[0].plot(np.array(q['u_slope'])[up], zq, '-', color=col, lw=lw, label=lab)
            thq = np.array(q['theta_C'])[up]
            m_ = (zq >= 70.) & (zq <= 550.)
            if m_.sum() > 10:
                cq = np.polyfit(zq[m_], thq[m_], 1)
                ax[1].plot(thq - np.polyval(cq, zq), zq, '-', color=col, lw=lw)
        g6 = good & (nc <= 600.)
        ax[0].plot(um[g6], nc[g6], '-', color='k', lw=3, label='MicroHH (P1 forcing)')
        ax[1].plot((bm * th0 / GRAV)[g6], nc[g6], '-', color='k', lw=3)
        ax[0].set_xlabel(f"along-slope wind u_s (m/s; from {BAL['downslope_from_deg']:g} deg = downslope)")
        ax[1].set_xlabel('theta - theta_a (K; theta_a fitted over 70-550 m per sounding)')
        ax[0].set_ylabel('height above the surface (m)')
        for axx in ax:
            axx.set_ylim(0, 600); axx.grid(alpha=0.25); axx.axvline(0, color='0.6', lw=0.5)
        ax[0].legend(fontsize=6.5, ncol=2, title='launch (local time), ascent', title_fontsize=7)
        fig.suptitle(f'{title}\nPASTEX 1994, Pasterze A1: IMAU tethered-balloon (ADAS) soundings of 31 July '
                     f'(JD 212) and 1 August (JD 213)', fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f'{tag}_soundings.png'), dpi=130)
        plt.close(fig)

    # ---- Profiles to 6 H -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    g6 = good & (y <= 6.)
    ax[0].fill_betweenx(y[g6], ((um - usd) / V)[g6], ((um + usd) / V)[g6], color=C_MOD, alpha=0.15, lw=0)
    ax[0].plot((um / V)[g6], y[g6], '-', color=C_MOD, lw=2, label='MicroHH, slope average')
    ax[1].fill_betweenx(y[g6], ((bm - bsd) / B)[g6], ((bm + bsd) / B)[g6], color=C_MOD, alpha=0.15, lw=0)
    ax[1].plot((bm / B)[g6], y[g6], '-', color=C_MOD, lw=2)
    if f37:
        for axx, key in ((ax[0], 'u_over_V'), (ax[1], 'b_over_B')):
            les = np.array(f37[key]['les'])
            axx.plot(les[:, 0], les[:, 1], '-', color=C_LES, label='paper LES (to 1.5 H)')
    if 'zinv_over_H' in les_ref:
        ax[1].axhline(les_ref['zinv_over_H'], color=C_LES, ls=':', lw=1, label='paper LES: b = 0')
    for axx in ax:
        axx.axhline(lvl1 / H, color='0.6', ls=':', lw=0.8)
        axx.axvline(0, color='0.6', lw=0.5); axx.grid(alpha=0.25); axx.set_ylim(0, 6)
        axx.legend(fontsize=7)
    ax[0].set_xlabel('u_s / V'); ax[1].set_xlabel('b / B'); ax[0].set_ylabel('n / H')
    fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_profiles.png'), dpi=130)
    plt.close(fig)

    # ---- Time-height -----------------------------------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    hh = nc <= 150.
    lim = np.nanmax(np.abs(U[:, hh]))
    pc = ax[0].pcolormesh(times / 3600., nc[hh], U[:, hh].T, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
    fig.colorbar(pc, ax=ax[0], label='u_s (m/s)')
    D = Bt * th0 / GRAV
    lim = np.nanmax(np.abs(D[:, hh]))
    pc = ax[1].pcolormesh(times / 3600., nc[hh], D[:, hh].T, cmap='RdBu_r', vmin=-lim, vmax=lim, shading='auto')
    fig.colorbar(pc, ax=ax[1], label='thl - thl_a (K)')
    for axx in ax:
        axx.set_ylabel('n (m)')
    ax[1].set_xlabel('t (h)')
    ax[0].set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_timeheight.png'), dpi=130)
    plt.close(fig)

    # ---- Along the slope -------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
    for (f0, f1), col, lab in [((0., 1/3), 'tab:green', 'upper third'), ((1/3, 2/3), 'k', 'middle third'),
                               ((2/3, 1.), 'tab:purple', 'lower third')]:
        cc = R.columns((f0, f1))
        Q = [R.samples(t, cc) for t in times[last]]
        nn = np.concatenate([q['n'] for q in Q])
        u3, _, c3 = binned(nn, np.concatenate([q['us'] for q in Q]), edges)
        b3 = binned(nn, np.concatenate([q['b'] for q in Q]), edges)[0]
        gg = (c3 > 0) & (y <= 6.)
        lab = f'{lab} (x = {R.x[cc[0]]/1e3:.1f}-{R.x[cc[-1]]/1e3:.1f} km)'
        ax[0].plot(u3[gg] / V, y[gg], color=col, label=lab)
        ax[1].plot(b3[gg] / B, y[gg], color=col)
    for axx in ax:
        axx.grid(alpha=0.25); axx.axvline(0, color='0.6', lw=0.5); axx.set_ylim(0, 6)
    ax[0].legend(fontsize=7); ax[0].set_xlabel('u_s / V'); ax[1].set_xlabel('b / B'); ax[0].set_ylabel('n / H')
    fig.suptitle(f'{tag}: development along the slope, {window}', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f'{tag}_alongslope.png'), dpi=130)
    plt.close(fig)

    # ---- Printed summary -------------------------------------------------------------------
    print(title)
    mast_u = max(q['value'] for q in f37['u_over_V']['mast']) if f37 else np.nan
    print(f"  u_max = {um[k]:.2f} m/s = {um[k]/V:.1f} V at n = {zj:.1f} m = {zj/H:.2f} H   "
          f"(paper LES {les_ref.get('umax_over_V', '-')} V at {les_ref.get('zj_over_H', '-')} H; "
          f"mast max {mast_u:.1f} V at 0.15 H)")
    print(f"  b/B lowest bin (n = {nc[first]:.1f} m) {bm[first]/B:.1f}, surface b_s/B {b_s/B:.1f}  "
          f"(paper LES surface {les_ref.get('bs_over_B', '-')}; mast at 0.5 m "
          f"{f37['b_over_B']['mast'][0]['value']:.0f})" if f37 else '')
    print(f"  b = 0 at n = {zinv:.0f} m = {zinv/H:.2f} H (paper {les_ref.get('zinv_over_H', '-')})")
    print(f"  wall: tau_s/V^2 = {tau_s/V**2:.2f}, F_s/(VB) = {F_s/(V*B):.2f} (prescribed -1)")
    print(f"  UD/(VH) = {UD[-1]/(V*H):.1f}, BD/(BH) = {BD[-1]/(B*H):.1f} at t = {times[-1]} s "
          f"(n < {a.integral_top:g} m; paper LES about 11.5 and -8)")
    if p1s is not None:
        print(f"  JD 212 10:25 sounding: theta_a fit 70-550 m: {cfit[1]:.2f} C + {cfit[0]*1e3:.2f} K/km z "
              f"(N = {N_obs:.4f}; paper 11.5 C, N = 0.0121); downslope from {BAL['downslope_from_deg']:g} deg")
    print(f"  fluxes: {'resolved + SFS (evisc crosses)' if R.has_k else 'resolved only (add evisc, eviscs to crosslist)'}")
    print(f'  figures and metrics in {outdir}')


if __name__ == '__main__':
    main()
