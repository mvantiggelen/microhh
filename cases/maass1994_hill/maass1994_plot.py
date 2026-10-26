#!/usr/bin/env python3
"""
Compare a MicroHH run of the wavy-wall channel of Maass and Schumann (1994)
(maass1994_setup.py) with the LDV measurements of Hudson, Dykhno and
Hanratty (ERCOFTAC Classic Collection case 076; reference_hudson1996.json,
built by build_hudson_reference.py). Only the measurements and MicroHH are
plotted; the DNS of Maass and Schumann is not (its digitised data stay in
reference_maass1994.json for documentation).

    python3 maass1994_plot.py --rundir run_paper --outdir figs --tavg 60

What it reads from --rundir:
    maass1994.ini, dem.0000000, maass1994.default.0000000.nc (grid, top-wall
    stress), the 3-D dumps u/v/w/p.<iotime> (phase averages), and the IB
    cross-sections taux_form_ib / taux_skin_ib (drag time series).

Averaging: over y, over the waves (x mod lambda) and over every dump with
t >= --tavg. Normalisation as the experiment: velocities, second moments,
pressure and stresses with U_b, the bulk velocity of the LOWER channel half
(Hudson's U_b = 12 cm/s). x is measured from the crest; the LDV stations
sit 0.023 lambda (1.2 mm) downstream of their nominal x/lambda = 0.0 ... 0.9,
and MicroHH is sampled there.

Figures (PNG in --outdir):
    fig2_u_profiles.png      mean u at the ten LDV stations
    fig3_mean_flow.png       mean (u, w), Psi = 0, separation; LDV stations with reversed flow
    fig6_variances.png       u'u', w'w', v'v', u'w' at constant eta, averaged over the wave
    fig7a_surface_pressure.png  2 (p - <p>) / (rho U_b^2), <p> = wave mean
    drag_timeseries.png      IB form and skin drag, top-wall stress; experiment: u_tau^2
    near_wall_u.png          u at 0.02 H above the wave: separation and reattachment
    snapshot_xz.png          instantaneous u in an x-z plane
and summary.json with the scalar comparisons.
"""
import argparse
import configparser
import glob
import json
import os
import re
import sys

import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker

# Colours: MicroHH = categorical slot 1 (blue), the measurements = slot 2 (orange).
C_LES = '#2a78d6'
C_EXP = '#eb6834'
C_SKIN = '#1baf7a'
LAMBDA = 1.0
DELTA = 0.05
H = 1.0

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.6, 'lines.linewidth': 1.4,
                     'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
                     'legend.frameon': False, 'savefig.dpi': 150})


def read_ini(fn):
    cp = configparser.ConfigParser(inline_comment_prefixes=('#',), strict=False)
    cp.optionxform = str
    with open(fn) as f:
        cp.read_string(f.read())
    return cp


class Run:
    def __init__(self, rundir):
        self.dir = rundir
        ini = read_ini(os.path.join(rundir, 'maass1994.ini'))
        g = ini['grid']
        self.itot, self.jtot, self.ktot = int(g['itot']), int(g['jtot']), int(g['ktot'])
        self.xsize, self.ysize = float(g['xsize']), float(g['ysize'])
        self.iotimeprec = int(ini['time'].get('iotimeprec', '0'))
        self.dx = self.xsize / self.itot
        st = nc.Dataset(os.path.join(rundir, 'maass1994.default.0000000.nc'))
        self.z = st['z'][:].data
        self.zh = st['zh'][:].data
        self.stats = st
        self.dem = np.fromfile(os.path.join(rundir, 'dem.0000000')).reshape(self.jtot, self.itot)
        self.h_col = self.dem.mean(axis=0)                 # DEM per column (2-D wave)
        self.z_mean = self.h_col.mean()                    # the paper's z = 0
        self.ztop = self.zh[-1]
        self.Hc = self.ztop - self.z_mean                  # channel height (= H)
        nper = self.itot * LAMBDA / self.xsize
        if abs(nper - round(nper)) > 1e-9:
            raise ValueError('itot must hold a whole number of points per wavelength')
        self.nper = int(round(nper))
        self.nwave = self.itot // self.nper
        self.xc = (np.arange(self.nper) + 0.5) * self.dx   # cell centres in one wave
        self.hp = self.h_col[:self.nper] - self.z_mean     # paper h(x) at the centres
        # first air cell per column (staircase), as the IB defines it
        self.kf = np.array([np.argmax(self.z > hc) for hc in self.h_col[:self.nper]])

    def t_of(self, it):
        return int(it) * 10. ** self.iotimeprec

    def dump_times(self, tmin):
        its = []
        for fn in glob.glob(os.path.join(self.dir, 'p.[0-9]' * 1 + '*')):
            m = re.match(r'^p\.(\d{7})$', os.path.basename(fn))
            if not m:
                continue
            it = m.group(1)
            if all(os.path.exists(os.path.join(self.dir, '%s.%s' % (v, it))) for v in 'uvw'):
                if self.t_of(it) >= tmin - 1e-9:
                    its.append(it)
        return sorted(its)

    def load(self, var, it):
        return np.fromfile(os.path.join(self.dir, '%s.%s' % (var, it))).reshape(
            self.ktot, self.jtot, self.itot)

    def phase(self, a):
        """mean over y and the waves of a (k, j, i) field -> (k, nper)"""
        return a.reshape(self.ktot, self.jtot, self.nwave, self.nper).mean(axis=(1, 2))


def phase_averages(run, its):
    keys = ('u', 'v', 'w', 'p', 'uu', 'vv', 'ww', 'uw', 'uf')
    acc = {k: 0. for k in keys}
    for it in its:
        u = run.load('u', it); v = run.load('v', it); w = run.load('w', it); p = run.load('p', it)
        uc = 0.5 * (u + np.roll(u, -1, axis=2))
        vc = 0.5 * (v + np.roll(v, -1, axis=1))
        wt = np.concatenate([w, np.zeros((1,) + w.shape[1:])], axis=0)   # w = 0 at the top wall
        wc = 0.5 * (wt[:-1] + wt[1:])
        for k, a in (('u', uc), ('v', vc), ('w', wc), ('p', p), ('uu', uc * uc),
                     ('vv', vc * vc), ('ww', wc * wc), ('uw', uc * wc), ('uf', u)):
            acc[k] = acc[k] + run.phase(a)
    n = float(len(its))
    m = {k: acc[k] / n for k in keys}
    m['var_u'] = m['uu'] - m['u'] ** 2
    m['var_v'] = m['vv'] - m['v'] ** 2
    m['var_w'] = m['ww'] - m['w'] ** 2
    m['cov_uw'] = m['uw'] - m['u'] * m['w']
    return m


def air_mask(run):
    """(k, nper): True in air cells of the phase-averaged column"""
    return run.z[:, None] > run.h_col[None, :run.nper]


def bulk_velocities(run, m):
    """Um over the whole channel and over its lower half (paper p. 291)."""
    air = air_mask(run)
    dz = np.diff(run.zh)[:, None]
    u = np.where(air, m['u'], 0.)
    q_full = (u * dz).sum(axis=0).mean()
    zhalf = run.z_mean + 0.5 * run.Hc
    w_lo = np.clip((zhalf - run.zh[:-1]) / np.diff(run.zh), 0., 1.)[:, None]  # fraction below z_half
    q_lo = (u * dz * w_lo).sum(axis=0).mean()
    return q_full / run.Hc, q_lo / (0.5 * run.Hc)


def periodic_interp(xq, x, f):
    """linear interpolation in x on a periodic (period LAMBDA) grid, f(..., nx)"""
    xx = np.r_[x[-1] - LAMBDA, x, x[0] + LAMBDA]
    ff = np.concatenate([f[..., -1:], f, f[..., :1]], axis=-1)
    i = np.searchsorted(xx, xq % LAMBDA) - 1
    a = ((xq % LAMBDA) - xx[i]) / (xx[i + 1] - xx[i])
    return (1 - a) * ff[..., i] + a * ff[..., i + 1]


def exp_stations(exp):
    """Hudson's profiles: list of (x_true, z = h(x_true) + y [H], dict of arrays)."""
    out = []
    for key in sorted(exp['stations']):
        s = exp['stations'][key]
        xt = s['x_true']
        h = DELTA * np.cos(2 * np.pi * xt / LAMBDA)
        out.append((xt, h + np.array(s['y_over_H']), {k: np.array(s[k]) for k in ('U', 'W', 'uu', 'ww', 'uw')},
                    np.array(s['y_over_H'])))
    return out


def exp_near_wall(exp, d):
    """Hudson's mean U / U_b at the height d above the local surface, per station."""
    xs, us = [], []
    for xt, z, f, y in exp_stations(exp):
        xs.append(xt); us.append(np.interp(d, y, f['U']))
    return np.array(xs), np.array(us)


def fig2(run, m, umL, exp, out):
    """Mean u at the ten measurement stations: MicroHH sampled at the
    stations' true positions (1.2 mm downstream of the nominal ones)."""
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    air = air_mask(run)
    zp = run.z - run.z_mean
    xs_fine = np.linspace(0, 1, 201)
    ax.plot(xs_fine, DELTA * np.cos(2 * np.pi * xs_fine), '--', color='#999999', lw=0.8)
    for n, (xt, z, f, y) in enumerate(exp_stations(exp)):
        h = DELTA * np.cos(2 * np.pi * xt)
        col = periodic_interp(np.array([xt]), run.xc, np.where(air, m['u'], np.nan))[:, 0]
        ok = np.isfinite(col) & (zp > h) & (zp <= 0.5)
        lbl = (n == 0)
        ax.plot(xt + 0.1 * np.r_[0., col[ok] / umL], np.r_[h, zp[ok]], color=C_LES,
                label='MicroHH IB' if lbl else None)
        k = z <= 0.5
        ax.plot(xt + 0.1 * f['U'][k], z[k], 'o', mfc='none', mec=C_EXP, ms=3., mew=0.7,
                label='Hudson et al. LDV (ERCOFTAC case 076)' if lbl else None)
    ax.annotate('', xy=(0.1, -0.035), xytext=(0.0, -0.035),
                arrowprops=dict(arrowstyle='<->', lw=0.8))
    ax.text(0.05, -0.03, r'$U_b$', ha='center', va='bottom', fontsize=8)
    ax.set_xlim(-0.02, 1.15); ax.set_ylim(-0.055, 0.5)
    ax.set_xlabel(r'$x/\lambda$ from the crest  (profiles: $x/\lambda + 0.1\,\overline{u}/U_b$)')
    ax.set_ylabel(r'$z/H$')
    ax.set_title('Mean u at the ten LDV stations ($U_b$: lower-half bulk velocity)')
    ax.legend(loc='upper left', fontsize=7.5)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig2_u_profiles.png')); plt.close(fig)

def streamfunction(run, m):
    air = air_mask(run)
    dz = np.diff(run.zh)[:, None]
    u = np.where(air, m['u'], 0.)
    return np.cumsum(u * dz, axis=0) - 0.5 * u * dz


def at_height(run, f, d):
    """f (k, nper) at the height d above the smooth wave h(x) of each column,
    by linear interpolation in the air cells of that column (extrapolated
    from the two lowest air cells when d lies below the first air centre).
    The first air cell itself sits 0..1 dz above the wave on the staircase,
    so sampling it directly gives a zig-zag from column to column."""
    out = np.empty(run.nper)
    for i in range(run.nper):
        k0 = run.kf[i]
        zt = run.z_mean + run.hp[i] + d
        zz, ff = run.z[k0:], f[k0:, i]
        if zt <= zz[0]:
            out[i] = ff[0] + (ff[1] - ff[0]) * (zt - zz[0]) / (zz[1] - zz[0])
        else:
            out[i] = np.interp(zt, zz, ff)
    return out


def separated_region(run, m):
    """x where the mean streamline Psi = 0 (Psi from the wall) bounds a
    recirculation: min over the column of Psi < 0 (the paper's definition,
    p. 292). Returns the start and end of each such interval."""
    psi = streamfunction(run, m)
    air = air_mask(run)
    mn = np.array([psi[air[:, i], i].min() for i in range(run.nper)])
    return separation_points(run.xc, -mn)[::-1]


def near_wall(run, m):
    """phase-averaged u in the first air cell above the staircase floor"""
    return np.array([m['u'][run.kf[i], i] for i in range(run.nper)])


def separation_points(x, u):
    """x where the near-wall u turns negative (separation) and positive (reattachment)"""
    sep, rea = [], []
    n = len(x)
    for i in range(n):
        a, b = u[i], u[(i + 1) % n]
        xb = x[(i + 1) % n] + (LAMBDA if i == n - 1 else 0.)
        if a * b < 0:
            x0 = (x[i] + (xb - x[i]) * a / (a - b)) % LAMBDA
            (sep if a > 0 else rea).append(x0)
    return sep, rea


def fig3(run, m, umL, ref, out, sep, rea):
    air = air_mask(run)
    zp = run.z - run.z_mean
    u = np.where(air, m['u'], np.nan) / umL
    w = np.where(air, m['w'], np.nan) / umL
    psi = streamfunction(run, m)
    psi = np.where(air, psi, np.nan)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    k = zp <= 0.135
    pc = ax.pcolormesh(np.r_[0., (np.arange(run.nper) + 1) * run.dx], np.r_[run.zh[:-1][k] - run.z_mean,
                       run.zh[1:][k][-1] - run.z_mean], u[k], cmap='RdBu_r',
                       norm=matplotlib.colors.TwoSlopeNorm(vmin=-0.3, vcenter=0., vmax=1.2),
                       shading='flat')
    step = max(1, run.nper // 20)
    X, Z = np.meshgrid(run.xc, zp[k])
    ax.quiver(X[:, ::step], Z[:, ::step], u[k][:, ::step], w[k][:, ::step], scale=12, width=0.002,
              color='k')
    if np.isfinite(psi[k]).sum() > 4:
        try:
            ax.contour(run.xc, zp[k], psi[k], levels=[0.], colors=['#008300'], linewidths=1.2)
        except Exception:
            pass
    xs = np.linspace(0, 1, 201)
    # the model's terrain (the IB staircase: a column is rock below its DEM
    # height, cell by cell), and the paper's smooth wave as a dashed line
    zb = run.zh[run.kf] - run.z_mean
    xe = np.arange(run.nper + 1) * run.dx
    ax.stairs(zb, xe, baseline=-0.06, fill=True, color='#bbbbbb', zorder=3,
              label='model terrain (staircase)')
    ax.plot(xs, DELTA * np.cos(2 * np.pi * xs), '--', color='#555555', lw=0.9, zorder=4,
            label='paper wave $h(x)$')
    xe_, ue_ = exp_near_wall(ref, P_HEIGHT)
    for xv, uv in zip(xe_, ue_):
        ax.plot(xv, -0.05, marker='v', ms=6, zorder=5, color=C_EXP if uv < 0 else 'none',
                mec=C_EXP)
    ax.plot([], [], 'v', color=C_EXP, label='LDV: reversed flow %g H above the wall' % P_HEIGHT)
    for xv in sep:
        ax.axvline(xv, color=C_LES, lw=1, ls='--')
    for xv in rea:
        ax.axvline(xv, color=C_LES, lw=1, ls=':')
    ax.set_ylim(-0.055, 0.16); ax.set_xlim(0, 1)
    ax.set_xlabel(r'$x/\lambda$'); ax.set_ylabel(r'$z/H$')
    ax.set_title(r'Mean flow (colour $\overline{u}/U_{m,lower}$), green: $\Psi = 0$; '
                 'blue dashed/dotted: separation/reattachment', fontsize=8)
    fig.colorbar(pc, ax=ax, shrink=0.8)
    ax.legend(loc='upper right', fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig3_mean_flow.png')); plt.close(fig)


def eta_profiles(run, m, umL):
    """variances at constant eta (paper eq. 3), averaged over the wave"""
    eta = np.linspace(0.01, 0.99, 99)
    zp = run.z - run.z_mean
    out = {}
    for key in ('var_u', 'var_w', 'var_v', 'cov_uw'):
        prof = np.full((run.nper, eta.size), np.nan)
        for i in range(run.nper):
            h = DELTA * np.cos(2 * np.pi * run.xc[i] / LAMBDA)
            zq = h + eta * (H - h)
            kk = np.arange(run.kf[i], run.ktot)
            prof[i] = np.interp(zq, zp[kk], m[key][kk, i], left=np.nan, right=np.nan)
        out[key] = np.nanmean(prof, axis=0) / umL ** 2
    return eta, out


def exp_eta_profiles(exp, eta):
    """Hudson's second moments at constant eta = y / (H - h) (paper eq. 3,
    with y the height above the local surface), averaged over the ten
    stations (each needs the eta in its measured range)."""
    out = {}
    for key in ('uu', 'ww', 'uw'):
        prof = []
        for xt, z, f, y in exp_stations(exp):
            h = DELTA * np.cos(2 * np.pi * xt)
            e = y / (H - h)
            prof.append(np.interp(eta, e, f[key], left=np.nan, right=np.nan))
        out[key] = np.nanmean(np.array(prof), axis=0)
        out[key][np.sum(np.isfinite(prof), axis=0) < 10] = np.nan
    return out


def fig6(run, m, umL, exp, out):
    eta, pr = eta_profiles(run, m, umL)
    ee = np.linspace(0.005, 0.45, 90)
    ex = exp_eta_profiles(exp, ee)
    fig, axs = plt.subplots(2, 2, figsize=(6.5, 6.2))
    spec = [('var_u', 'uu', r"$\overline{u'u'}/U_b^2$", (0, 0.1)),
            ('var_w', 'ww', r"$\overline{w'w'}/U_b^2$", (0, 0.03)),
            ('var_v', None, r"$\overline{v'v'}/U_b^2$ (spanwise; not measured)", (0, 0.1)),
            ('cov_uw', 'uw', r"$\overline{u'w'}/U_b^2$", (-0.025, 0.006))]
    for ax, (key, rk, lab, xl) in zip(axs.flat, spec):
        ax.plot(pr[key], eta, color=C_LES, label='MicroHH IB (resolved)')
        if rk:
            ax.plot(ex[rk], ee, 'o', mfc='none', mec=C_EXP, ms=3., mew=0.7,
                    label='Hudson et al. LDV')
        ax.set_xlim(*xl); ax.set_ylim(0, 1)
        ax.set_xlabel(lab); ax.set_ylabel(r'$\eta/H$')
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axs[0, 0].legend(fontsize=7)
    fig.suptitle(r'Second moments at constant $\eta$, averaged over the wave; normalised with $U_b$ '
                 '(lower-half bulk velocity)', fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig6_variances.png')); plt.close(fig)
    return eta, pr



P_HEIGHT = 0.02     # H above the smooth wave for the "surface" pressure and near-wall u: above
#                     the first air cell centre everywhere on the paper grid (<= 1.5 dz = 0.019 H), so an
#                     interpolation, not an extrapolation from the staircase (that gave a zig-zag)


def surface_pressure(run, m):
    """p 0.02 H above the wave minus its mean over the wave."""
    ps = at_height(run, m['p'], P_HEIGHT)
    return ps - ps.mean()


def fig7a(run, m, umL, exp, out):
    """2 (p - <p>)/(rho U_b^2) along the wave, <p> = the wave mean of each
    curve (the LDV value nearest the crest is a sharp suction peak, P+ = -18,
    so a one-point reference would shift the whole curve by it). The LDV
    pressure (Hudson: from the measured velocity gradients) has the mean
    pressure gradient added back (error bar: its +-2 P+ uncertainty)."""
    e = exp['surface_pressure']
    xt = np.array(e['x_true'])
    cpe = np.array(e['Cp_corrected']); cpe = cpe - cpe.mean()
    ps = 2. * surface_pressure(run, m) / umL ** 2
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    x = np.r_[run.xc, run.xc[0] + LAMBDA]
    ax.plot(x, np.r_[ps, ps[0]], color=C_LES, marker='.', ms=3,
            label='MicroHH IB (%g H above the wave)' % P_HEIGHT)
    ax.errorbar(xt, cpe, yerr=e['Cp_corrected_err'], fmt='o', mfc='none', mec=C_EXP,
                ecolor=C_EXP, ms=4, lw=0.8, capsize=2, label='Hudson et al. (from LDV velocities)')
    ax.axhline(0, color='#999999', lw=0.6)
    ax.set_xlim(0, 1); ax.set_xlabel(r'$x/\lambda$ from the crest')
    ax.set_ylabel(r'$2(p - \langle p\rangle)/(\rho U_b^2)$')
    ax.set_title(r'Surface pressure; $\langle p\rangle$ = mean over the wave', fontsize=8.5)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'fig7a_surface_pressure.png')); plt.close(fig)
    return ps


def drag_series(run):
    out = {}
    for name in ('taux_form_ib', 'taux_skin_ib'):
        t, v = [], []
        for fn in sorted(glob.glob(os.path.join(run.dir, name + '.xy.000.*'))):
            it = fn.rsplit('.', 1)[1]
            t.append(run.t_of(it)); v.append(np.fromfile(fn).mean())
        out[name] = (np.array(t), np.array(v))
    st = run.stats
    out['top'] = (st['time'][:].data, st['default']['u_flux'][:, -1].data)
    out['umean'] = (st['time'][:].data, (st['default']['u'][:].data * np.diff(run.zh)).sum(axis=1) / run.Hc)
    return out


def fig_drag(run, ds, exp, out, tavg, umL):
    """Left: the stresses in time. Right: their means over t >= tavg; the
    experiment gives the total stress on the wavy wall only (u_tau = 1.29
    cm/s). All in units of rho U_b^2 (U_b = lower-half bulk velocity)."""
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.0, 3.3), gridspec_kw=dict(width_ratios=[1.6, 1]))
    tf, form = ds['taux_form_ib']; ts, skin = ds['taux_skin_ib']; tt, top = ds['top']
    sc = 1. / umL ** 2
    if form.size:
        ax.plot(tf, sc * form, color=C_LES, label='IB form drag (taux_form_ib)')
    if skin.size:
        ax.plot(ts, sc * skin, color=C_SKIN, label='IB skin friction, MOST (taux_skin_ib)')
    if form.size and skin.size and np.array_equal(tf, ts):
        ax.plot(tf, sc * (form + skin), color='k', lw=1.0, label='IB total (form + skin)')
    ax.plot(tt, sc * top, color=C_EXP, label='top wall (resolved)')
    etot = exp['tau_wavy_over_Ub2']
    ax.axhline(etot, color='k', ls='--', lw=0.8)
    ax.text(0.01, etot, ' experiment, wavy wall total', color='#555555', fontsize=6.5, va='bottom',
            transform=ax.get_yaxis_transform())
    ax.axvline(tavg, color='#999999', lw=0.8, ls=':')
    ax.set_ylim(0., 2.2 * etot)
    ax.set_xlabel(r'$t\,U_m/H$'); ax.set_ylabel(r'stress $/(\rho U_b^2)$')
    ax.set_title('Wall stresses, interval means (start-up spike clipped)', fontsize=8.5)
    ax.legend(fontsize=6.5, loc='upper right')

    def tmean(t, v):
        k = t >= tavg - 1e-9
        return float(v[k].mean()) if k.any() else np.nan
    les = [sc * tmean(tf, form), sc * tmean(ts, skin), sc * (tmean(tf, form) + tmean(ts, skin)),
           sc * tmean(tt, top)]
    lab = ['wavy wall\nform', 'wavy wall\nskin', 'wavy wall\ntotal', 'top wall']
    xb = np.arange(4)
    bx.bar(xb, les, 0.55, color=[C_LES, C_SKIN, '#444444', C_EXP], label='MicroHH IB')
    for x0, v in zip(xb, les):
        bx.text(x0, v, '%.4f' % v, ha='center', va='bottom', fontsize=6.5)
    bx.plot([1.65, 2.35], [etot, etot], color='k', ls='--', lw=1.2,
            label='experiment: $u_\\tau^2/U_b^2$ = %.4f' % etot)
    bx.set_xticks(xb); bx.set_xticklabels(lab, fontsize=7)
    bx.set_title('Mean over t >= %g' % tavg, fontsize=8.5)
    bx.legend(fontsize=6.5, loc='upper left')
    bx.set_ylim(0, 1.45 * max(np.nanmax(les), etot))
    bx.grid(axis='x', visible=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'drag_timeseries.png')); plt.close(fig)

def fig_nearwall(run, unw, umL, exp, out, sep, rea):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(run.xc, unw / umL, color=C_LES, marker='.', ms=3, label='MicroHH IB, %g H above the wave' % P_HEIGHT)
    xe, ue = exp_near_wall(exp, P_HEIGHT)
    ax.plot(xe, ue, 'o', mfc='none', mec=C_EXP, ms=4, label='Hudson et al. LDV, same height')
    ax.axhline(0, color='k', lw=0.6)
    for xv in sep:
        ax.axvline(xv, color=C_LES, ls='--', lw=0.8)
    for xv in rea:
        ax.axvline(xv, color=C_LES, ls=':', lw=0.8)
    ax.set_xlim(0, 1); ax.set_xlabel(r'$x/\lambda$ from the crest'); ax.set_ylabel(r'$\overline{u}/U_b$')
    ax.set_title('Near-wall mean u: separation and reattachment', fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'near_wall_u.png')); plt.close(fig)

def fig_snapshot(run, it, out):
    u = run.load('u', it)
    j = run.jtot // 2
    zp = run.z - run.z_mean
    air = run.z[:, None] > run.h_col[None, :]
    a = np.where(air, u[:, j, :], np.nan)
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    pc = ax.pcolormesh(np.arange(run.itot + 1) * run.dx, np.r_[run.zh - run.z_mean], a,
                       cmap='RdBu_r', vmin=-1.6, vmax=1.6, shading='flat')
    ax.set_xlabel(r'$x/H$'); ax.set_ylabel(r'$z/H$')
    ax.set_title('u at y = Ly/2, t = %g H/Um (instantaneous)' % run.t_of(it), fontsize=9)
    fig.colorbar(pc, ax=ax, pad=0.01)
    fig.tight_layout(); fig.savefig(os.path.join(out, 'snapshot_xz.png')); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rundir', default='.')
    ap.add_argument('--outdir', default='figs')
    ap.add_argument('--tavg', type=float, default=None,
                    help='average dumps with t >= tavg (default: second half of the run)')
    ap.add_argument('--exp', default=None, help='reference_hudson1996.json (default: next to this script '
                    'or in --rundir)')
    a = ap.parse_args()

    run = Run(a.rundir)
    here = os.path.dirname(os.path.abspath(__file__))
    expfn = a.exp or next((f for f in (os.path.join(a.rundir, 'reference_hudson1996.json'),
                                       os.path.join(here, 'reference_hudson1996.json')) if os.path.exists(f)),
                          None)
    if expfn is None:
        sys.exit('reference_hudson1996.json not found (build_hudson_reference.py writes it)')
    exp = json.load(open(expfn))
    os.makedirs(a.outdir, exist_ok=True)

    all_its = run.dump_times(0.)
    if not all_its:
        sys.exit('no dumps (u, v, w, p) found in %s' % a.rundir)
    tend = run.t_of(all_its[-1])
    tavg = a.tavg if a.tavg is not None else 0.5 * tend
    its = run.dump_times(tavg)
    if not its:
        sys.exit('no dumps with t >= %g (last dump at %g)' % (tavg, tend))
    print('averaging %d dumps, t = %g .. %g' % (len(its), run.t_of(its[0]), run.t_of(its[-1])))

    m = phase_averages(run, its)
    um, umL = bulk_velocities(run, m)
    print('bulk velocity: whole channel Um = %.4f, lower half U_b = %.4f (ratio %.3f); figures use U_b, '
          'as the experiment' % (um, umL, umL / um))

    unw = near_wall(run, m)
    sep, rea = separation_points(run.xc, unw)
    xe, ue = exp_near_wall(exp, P_HEIGHT)
    esep, erea = separation_points(xe, ue)
    fig2(run, m, umL, exp, a.outdir)
    fig3(run, m, umL, exp, a.outdir, sep, rea)
    eta, pr = fig6(run, m, umL, exp, a.outdir)
    ps = fig7a(run, m, umL, exp, a.outdir)
    ds = drag_series(run)
    fig_drag(run, ds, exp, a.outdir, tavg, umL)
    fig_nearwall(run, unw, umL, exp, a.outdir, sep, rea)
    fig_snapshot(run, its[-1], a.outdir)

    def tmean(key):
        t, v = ds[key]
        k = t >= tavg - 1e-9
        return float(v[k].mean()) if k.any() else float('nan')
    form, skin, top = tmean('taux_form_ib'), tmean('taux_skin_ib'), tmean('top')
    # The viscous wall stress a resolved (DNS-like) wall would have,
    # nu u1 / d1 with u1 the mean u in the first air cell above the
    # staircase floor at distance d1, and the wall distance of that cell in
    # wall units with the MOST u*. If d1+ < ~5 the cell is in the viscous
    # sublayer, where the log law (MOST) underestimates the stress.
    ini = read_ini(os.path.join(a.rundir, 'maass1994.ini'))
    nu = float(ini['fields']['visc'])
    d1 = run.z[run.kf] - run.zh[run.kf]
    tau_visc = float(np.mean(nu * unw / d1))
    ustar_skin = np.sqrt(max(skin, 1e-12))
    d1_plus = float(np.mean(d1) * ustar_skin / nu)
    tot = form + skin
    sp = exp['surface_pressure']
    cpe = np.array(sp['Cp_corrected']); cpe = cpe - cpe.mean()
    ee = np.linspace(0.005, 0.45, 90)
    ex = exp_eta_profiles(exp, ee)
    ib = eta < 0.45
    summ = {
        'rundir': os.path.abspath(a.rundir), 'tavg': tavg, 'tend': tend, 'n_dumps': len(its),
        'Um_whole': um, 'Ub_lower': umL, 'Ub_lower_over_Um': umL / um,
        'Re_H_run_lower_half': umL / nu, 'Re_H_experiment': exp['Re_H'],
        'separation_x_over_lambda': sep, 'reattachment_x_over_lambda': rea,
        'exp_separation_x_over_lambda': esep, 'exp_reattachment_x_over_lambda': erea,
        'surface_pressure_max': float(ps.max()), 'surface_pressure_max_x': float(run.xc[ps.argmax()]),
        'exp_surface_pressure_max': float(cpe.max()), 'exp_surface_pressure_max_x': float(sp['x_true'][cpe.argmax()]),
        'tau_form_wavy_over_Ub2': form / umL ** 2, 'tau_skin_wavy_over_Ub2': skin / umL ** 2,
        'tau_total_wavy_over_Ub2': tot / umL ** 2, 'form_fraction': form / tot if tot else float('nan'),
        'exp_tau_total_wavy_over_Ub2': exp['tau_wavy_over_Ub2'],
        'utau_wavy_over_Ub': float(np.sqrt(max(tot, 0.))) / umL, 'exp_utau_wavy_over_Ub': exp['utau_wavy_over_Ub'],
        'tau_top_over_Ub2': top / umL ** 2,
        'uu_max': float(np.nanmax(pr['var_u'][ib])), 'uu_max_eta': float(eta[ib][np.nanargmax(pr['var_u'][ib])]),
        'exp_uu_max': float(np.nanmax(ex['uu'])), 'exp_uu_max_eta': float(ee[np.nanargmax(ex['uu'])]),
        'uw_min': float(np.nanmin(pr['cov_uw'][ib])), 'exp_uw_min': float(np.nanmin(ex['uw'])),
        'tau_viscous_estimate_wavy_over_Ub2': tau_visc / umL ** 2,
        'first_cell_dn_plus_with_most_ustar': d1_plus,
    }
    with open(os.path.join(a.outdir, 'summary.json'), 'w') as f:
        json.dump(summ, f, indent=1)
    fmt = lambda v: ['%.2f' % x for x in v]
    print('reversed near-wall flow (%g H above the wave): separation %s, reattachment %s; '
          'experiment (LDV, 10 stations) %s .. %s' % (P_HEIGHT, fmt(sep), fmt(rea), fmt(esep), fmt(erea)))
    print('wavy wall: form %.4f + skin %.4f = %.4f U_b^2 (form %.0f %%), u*/U_b %.3f; experiment total %.4f '
          '(u*/U_b %.3f); top wall %.4f'
          % (summ['tau_form_wavy_over_Ub2'], summ['tau_skin_wavy_over_Ub2'], summ['tau_total_wavy_over_Ub2'],
             100 * summ['form_fraction'], summ['utau_wavy_over_Ub'], exp['tau_wavy_over_Ub2'],
             exp['utau_wavy_over_Ub'], summ['tau_top_over_Ub2']))
    print('viscous stress estimate at the floors nu*u1/d1 = %.4f U_b^2 (MOST skin %.4f); '
          'first air cell at d1+ = %.1f with the MOST u*'
          % (tau_visc / umL ** 2, skin / umL ** 2, d1_plus))
    print('surface pressure 2(p - <p>)/U_b^2: max %.3f at x/lambda %.2f; experiment %.3f at %.2f'
          % (summ['surface_pressure_max'], summ['surface_pressure_max_x'], cpe.max(),
             sp['x_true'][cpe.argmax()]))
    print('u\'u\' max (eta < 0.45) %.4f at eta %.2f; experiment %.4f at %.2f | u\'w\' min %.4f; experiment %.4f'
          % (summ['uu_max'], summ['uu_max_eta'], summ['exp_uu_max'], summ['exp_uu_max_eta'],
             summ['uw_min'], summ['exp_uw_min']))
    print('Reynolds number H U_b/nu: run %.0f, experiment %d' % (umL / nu, exp['Re_H']))
    print('figures in', a.outdir)


if __name__ == '__main__':
    main()
