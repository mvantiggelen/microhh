#!/usr/bin/env python3
"""
Compare MicroHH runs of the Andren et al. (1994) neutral Ekman layer with
the paper's reference data (reference_andren1994.json), and with each other.

    Andren, A., Brown, A. R., Graf, J., Mason, P. J., Moeng, C.-H.,
    Nieuwstadt, F. T. M. and Schumann, U. (1994), Q. J. R. Meteorol. Soc.
    120, 1457-1484, https://doi.org/10.1002/qj.49712052003

Usage:
    python3 andren1994_plot.py --rundir runs/paper --outdir runs/paper/figs
    python3 andren1994_plot.py --stats some.default.0000000.nc --tavg 70000 100000
    # overlay: the first run is the main one, --compare adds more (any surface)
    python3 andren1994_plot.py --rundir runs/flat_2i5 --compare runs/ib runs/flat_ref \\
        --labels "flat 2i5" "flat IB" "flat ref" --outdir runs/compare_figs

Reads the MicroHH statistics (<case>.default.<starttime>.nc; several files
from restarts are joined in time), averages the profiles over the paper's
window t = 7/f ... 10/f (or the last third of a shorter run), normalises them
as the paper does (z f/u*, u*^2, c* = (wc)0/u*) and writes:

    andren1994_phi.png         phi_M, phi_C (Fig. 4) and mean wind / hodograph
    andren1994_variances.png   u2, v2, w2 (Fig. 5) and c2 (Fig. 7)
    andren1994_fluxes.png      uw, vw (Fig. 6), wc (Fig. 8), K_M, K_C (Fig. 14)
    andren1994_timeseries.png  u*/U_G, integrated TKE (Fig. 2), C_u, C_v (Fig. 3)
    andren1994_surface.png     (with an IB run) the near-surface stress and wind
                               profiles in metres, and the surface-stress checks

and prints a table of bulk quantities against the paper (one column per run).

Surface stress and u*. u* = ((uw)0^2 + (vw)0^2)^(1/4) as in the paper, from
the mean surface stress over the window:
  - flat surface (swboundary=surface): (uw)0, (vw)0 = u_flux, v_flux at the
    surface half level (the MOST flux the surface model sets);
  - flat IB surface (--surface ib in andren1994_setup.py): the IB drag the
    wall model applied, (uw)0 = -(taux_skin_ib + taux_form_ib)/rho, from the
    cross-sections (interval means with [IB] sw_flux_accum), and likewise the
    scalar flux from s_fluxbot_ib. The statistics' u_flux at the IB floor face
    is the diffusion operator's flux with the ghost cell, which the wall model
    removes and replaces (MICROHH_PHYSICS.md 4.16), so it is NOT the surface
    stress; the plots and table use the IB value there instead.
Both are checked against the total momentum flux profile extrapolated
linearly to the surface from the first two half levels above it.

Heights are measured from the surface: for the IB run z - z_surface, with
z_surface from case_parameters.json (written by andren1994_setup.py), and
only the air levels are used.

MicroHH's smag2 has no SGS TKE, so MicroHH variances are RESOLVED only; the
paper's curves are resolved + SGS (its SGS part is drawn separately).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

KAPPA = 0.4
HERE = os.path.dirname(os.path.abspath(__file__))

# Paper line styles (as in the paper's legends), drawn in grey.
CODE_STYLE = {
    'AM':  dict(ls=(0, (6, 3)),            label='Andren/Moeng'),
    'MBb': dict(ls=(0, (1, 2)),            label='Mason/Brown, bsct'),
    'N':   dict(ls='-',                    label='Nieuwstadt'),
    'SG':  dict(ls=(0, (12, 4)),           label='Schumann/Graf'),
    'MBn': dict(ls=(0, (8, 3, 1.5, 3)),    label='Mason/Brown, nbsct'),
}
C_REF = '#6b6b6b'
# One colour per run (first = main run); second colour of a pair for <v>.
RUN_COLORS = ['#1f5fbf', '#c2410c', '#15803d', '#7e22ce', '#b45309', '#0e7490']
RUN_COLORS2 = ['#7fa6e0', '#f0a07a', '#7cc49a', '#c39ae6', '#e0b070', '#6fc0cf']
RUN_LS = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1))]


def read_stats(files):
    """Read and join (in time) the MicroHH default statistics files."""
    import netCDF4 as nc4
    out = {}
    for fn in sorted(files):
        with nc4.Dataset(fn) as d:
            g = d.groups['default']
            t = d.variables['time'][:].filled(np.nan)
            blk = {'time': t, 'z': d.variables['z'][:].filled(np.nan),
                   'zh': d.variables['zh'][:].filled(np.nan)}
            for v in g.variables:
                blk[v] = g.variables[v][:].filled(np.nan) if np.ma.isMaskedArray(g.variables[v][:]) \
                    else np.asarray(g.variables[v][:])
        if not out:
            out = blk
            continue
        keep = blk['time'] > out['time'][-1]          # drop overlap after a restart
        for v in blk:
            if v in ('z', 'zh') or v not in out:
                continue
            if np.ndim(blk[v]) >= 1 and np.shape(blk[v])[0] == blk['time'].size:
                out[v] = np.concatenate([out[v], blk[v][keep]], axis=0)
    return out


def find_stats(rundir, case):
    files = sorted(glob.glob(os.path.join(rundir, f'{case}.default.*.nc')))
    if not files:
        sys.exit(f'no {case}.default.*.nc in {rundir}')
    return files


def load_params(rundir):
    p = dict(fc=1e-4, ug=10., vg=0., sflux=1e-3, zsize=1500., tPr=0.7, t_avg=[7e4, 1e5],
             surface='flat', n_solid=0, z_surface=0.)
    pfile = os.path.join(rundir, 'case_parameters.json')
    if os.path.exists(pfile):
        p.update(json.load(open(pfile)))
    p.setdefault('zsize_air', p['zsize'] - p['z_surface'])
    return p


def read_xy(rundir, name, times, shape, level=None):
    """MicroHH xy cross-section files <name>.xy.000.<t> (IB surface fields) or
    <name>.xy.<loc>.<k:05d>.<t> (model level k; loc = 100 for u, 010 for v,
    000 for scalars), one 2-D field per time (NaN where a file is missing)."""
    loc = {'u': '100', 'v': '010', 'w': '001'}.get(name, '000')
    out = np.full((len(times),) + shape, np.nan)
    for n, t in enumerate(times):
        tag = f'{int(round(t)):07d}'
        fn = (os.path.join(rundir, f'{name}.xy.000.{tag}') if level is None
              else os.path.join(rundir, f'{name}.xy.{loc}.{level:05d}.{tag}'))
        if os.path.exists(fn):
            out[n] = np.fromfile(fn).reshape(shape)
    return out


def ib_surface(rundir, p, times):
    """Per statistics sample: the domain-mean IB drag (Pa, rho = 1 here, so
    kinematic m2 s-2 as well) and scalar flux, plus ustar_ib (mean)."""
    shape = (p['jtot'], p['itot'])
    f = {n: read_xy(rundir, n, times, shape) for n in
         ('taux_skin_ib', 'tauy_skin_ib', 'taux_form_ib', 'tauy_form_ib', 'ustar_ib', 's_fluxbot_ib')}
    m = {n: np.nanmean(v, axis=(1, 2)) for n, v in f.items()}
    if np.all(np.isnan(m['taux_skin_ib'])):
        sys.exit(f'{rundir}: an IB run, but no taux_skin_ib cross-sections (needs [cross] crosslist '
                 f'with taux_skin_ib, [IB] sw_flux_accum)')
    return m


def surface_flux(s, p, ib):
    """Kinematic surface momentum flux (uw)0, (vw)0 per sample, and its source."""
    if ib is None:
        k = p['n_solid']
        return s['u_flux'][:, k], s['v_flux'][:, k]
    # The drag on the terrain is +tau; the air receives -tau.
    return -(ib['taux_skin_ib'] + ib['taux_form_ib']), -(ib['tauy_skin_ib'] + ib['tauy_form_ib'])


def compute(s, p, t0, t1, ib=None):
    """Time means over [t0, t1] and the normalised profiles (air levels only,
    heights above the surface)."""
    t = s['time']
    sel = (t >= t0 - 1e-6) & (t <= t1 + 1e-6)
    if not sel.any():
        sys.exit(f'no statistics samples in [{t0:g}, {t1:g}] s (run covers {t[0]:g}-{t[-1]:g} s)')
    ns, zs = p['n_solid'], p['z_surface']
    m = lambda v: np.nanmean(s[v][sel], axis=0)[ns:]       # air levels / half levels from the surface up
    fc, sflux = p['fc'], p['sflux']
    uw0_t, vw0_t = surface_flux(s, p, ib)
    ust_t = (uw0_t**2 + vw0_t**2)**0.25
    uw0, vw0 = np.mean(uw0_t[sel]), np.mean(vw0_t[sel])
    ust = (uw0**2 + vw0**2)**0.25                         # u* of the mean surface stress
    if ib is not None:
        wc0 = np.mean(ib['s_fluxbot_ib'][sel])
    else:
        wc0 = np.mean(s['s_flux'][sel, ns])
    cst = sflux / ust
    z, zh = s['z'][ns:] - zs, s['zh'][ns:] - zs
    zh[0] = 0.                                             # the surface (IB: -ib_offset*dz from the face)
    dz = np.diff(s['zh'][ns:])
    r = dict(nsamples=int(sel.sum()), t0=t[sel][0], t1=t[sel][-1], ust=ust, cst=cst, uw0=uw0, vw0=vw0,
             wc0=wc0, z=z, zh=zh, surface=p['surface'])
    r['zn'], r['zhn'] = z * fc / ust, zh * fc / ust
    r['u'], r['v'], r['s'] = m('u'), m('v'), m('s')
    # Gradients at half levels (as MicroHH's u_grad), between the cell centres.
    ug, vg, sg = m('u_grad'), m('v_grad'), m('s_grad')
    r['phiM'] = KAPPA * zh / ust * np.sqrt(ug**2 + vg**2)
    r['phiC'] = KAPPA * zh / cst * sg
    u2 = ust**2
    r['u2'], r['v2'], r['w2'] = m('u_2') / u2, m('v_2') / u2, m('w_2') / u2
    r['c2'] = m('s_2') / cst**2
    uf, vf, sf = m('u_flux'), m('v_flux'), m('s_flux')
    usg, vsg, ssg = m('u_diff'), m('v_diff'), m('s_diff')
    # The statistics' flux at the surface half level, before the IB value
    # replaces it (for the IB: the operator's ghost-cell flux, not applied).
    r['stats_uw0'], r['stats_vw0'], r['stats_wc0'] = uf[0], vf[0], sf[0]
    if ib is not None:
        uf[0], vf[0], sf[0] = uw0, vw0, wc0
        usg[0], vsg[0], ssg[0] = uw0, vw0, wc0
    r['uw_kin'], r['vw_kin'] = uf, vf
    r['uw'], r['uw_res'], r['uw_sgs'] = uf / u2, m('u_w') / u2, usg / u2
    r['vw'], r['vw_res'], r['vw_sgs'] = vf / u2, m('v_w') / u2, vsg / u2
    r['wc'], r['wc_res'], r['wc_sgs'] = sf / (ust * cst), m('s_w') / (ust * cst), ssg / (ust * cst)
    r['KM'] = m('evisc')
    r['KC'] = r['KM'] / p.get('tPr', 0.7)
    # Surface stress against the total flux profile extrapolated linearly from
    # the first two half levels above the surface (zh[1], zh[2]).
    ex = lambda f: f[1] - (f[2] - f[1]) / (zh[2] - zh[1]) * zh[1]
    r['uw0_extrap'], r['vw0_extrap'], r['wc0_extrap'] = ex(uf), ex(vf), ex(sf)
    r['tau0_extrap_ratio'] = np.hypot(r['uw0_extrap'], r['vw0_extrap']) / np.hypot(uw0, vw0)
    r['wc0_extrap_ratio'] = r['wc0_extrap'] / wc0
    # Bulk quantities
    r['ust_over_UG'] = ust / p['ug']
    r['lid_n'] = p['zsize_air'] * fc / ust
    # Direction of the surface wind (= minus the surface stress), counter-
    # clockwise from the geostrophic wind: atan(vw0/uw0), cf. Fig. 6.
    r['stress_angle'] = np.degrees(np.arctan2(-vw0, -uw0))
    r['wind_angle_1'] = np.degrees(np.arctan2(r['v'][0], r['u'][0]))
    r['z1'] = z[0]
    r['U1'] = np.hypot(r['u'][0], r['v'][0])
    # MOST check with the mean first-level wind (neutral log law; MOST acts on
    # the instantaneous local wind, so this is only approximate).
    r['ust_most_1'] = KAPPA * r['U1'] / np.log(z[0] / p['z0'])
    r['U2'] = np.hypot(r['u'][1], r['v'][1])
    r['ust_most_2'] = KAPPA * r['U2'] / np.log(z[1] / p['z0'])
    if ib is not None:
        r['form_frac'] = (np.hypot(np.mean(ib['taux_form_ib'][sel]), np.mean(ib['tauy_form_ib'][sel]))
                          / np.hypot(uw0, vw0))
        r['ustar_ib_mean'] = np.nanmean(ib['ustar_ib'][sel])
    # h: lowest height where the total stress magnitude drops below 5% of its
    # surface value, divided by 0.95 (the convention of Kosovic and Curry 2000
    # and Beare et al. 2006, GABLS1). Not given in the paper.
    tau = np.sqrt(uf**2 + vf**2)
    k = np.where(tau[1:] < 0.05 * tau[0])[0]
    if k.size:
        k = k[0] + 1
        zc = zh[k - 1] + (0.05 * tau[0] - tau[k - 1]) / (tau[k] - tau[k - 1]) * (zh[k] - zh[k - 1])
        r['h'] = zc / 0.95
    else:
        r['h'] = np.nan
    r['h_n'] = r['h'] * fc / ust
    # Time series (each sample its own u*), air levels only
    a = slice(ns, None)
    tke = 0.5 * (s['u_2'][:, a] + s['v_2'][:, a] + 0.5 * (s['w_2'][:, a][:, :-1] + s['w_2'][:, a][:, 1:]))
    r['ts_tf'] = t * fc
    r['ts_ust'] = ust_t / p['ug']
    r['ts_tke'] = fc * (tke * dz).sum(axis=1) / ust_t**3
    r['ts_Cu'] = -fc / uw0_t * ((s['v'][:, a] - p['vg']) * dz).sum(axis=1)
    r['ts_Cv'] = fc / vw0_t * ((s['u'][:, a] - p['ug']) * dz).sum(axis=1)
    r['Cu'], r['Cv'] = np.mean(r['ts_Cu'][sel]), np.mean(r['ts_Cv'][sel])
    r['tke_int'] = np.mean(r['ts_tke'][sel])
    return r


def most_check(rundir, p, r, times):
    """IB run: the instantaneous ustar_ib of every column against the neutral
    log law kappa |U| / ln(d/z0) with the wind of that column at the wall
    model's sampling height. With [IB] sw_wall_sample the wall model samples
    wall_sample_dz*dz along the normal (1.5 dz = 56.2 m), the second air level
    (an xy cross-section of the setup); without it the first air cell
    (u_ib1, v_ib1). Returns the mean ratio and its spread over the window."""
    shape = (p['jtot'], p['itot'])
    sel = [t for t in times if r['t0'] - 1e-6 <= t <= r['t1'] + 1e-6]
    ust = read_xy(rundir, 'ustar_ib', sel, shape)
    sw = {k: v for k, v in (x.split('=', 1) for x in p.get('ib_set', []))}
    sample = sw.get('sw_wall_sample', 'true') == 'true'
    dz = p['dz']
    if sample:
        ks = p['n_solid'] + 1
        u = read_xy(rundir, 'u', sel, shape, level=ks)
        v = read_xy(rundir, 'v', sel, shape, level=ks)
        uc = 0.5 * (u + np.roll(u, -1, axis=2))         # u faces -> cell centres
        vc = 0.5 * (v + np.roll(v, -1, axis=1))
        d = (ks + 0.5) * dz - p['z_surface']
        where = f'image point = 2nd air level, d = {d:.2f} m'
    else:
        uc = read_xy(rundir, 'u_ib1', sel, shape)
        vc = read_xy(rundir, 'v_ib1', sel, shape)
        d = 0.5 * dz - p['ib_offset'] * dz
        where = f'first air cell, d = {d:.2f} m'
    ust_log = KAPPA * np.hypot(uc, vc) / np.log(d / p['z0'])
    ok = np.isfinite(ust) & np.isfinite(ust_log)
    if not ok.any():
        return None
    ratio = ust[ok].mean() / ust_log[ok].mean()
    rel = np.abs(ust[ok] - ust_log[ok]) / ust_log[ok]
    return dict(ratio=ratio, med_rel=np.median(rel), p95_rel=np.percentile(rel, 95), where=where,
                n=int(ok.sum()))


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------
def style_ax(ax, xlabel, ylabel=True, ymax=0.36):
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(r'$zf/u_*$')
    ax.set_ylim(0, ymax)
    ax.grid(True, color='#e3e3e3', lw=0.6)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def band(ax, item, label, alpha=0.35, hatch=None):
    e = item['envelope']
    z = np.array(e['z_f_over_ustar'])
    unc = item.get('uncertainty', 0.)
    lo, hi = np.array(e['min']) - unc, np.array(e['max']) + unc
    if hatch is None:
        ax.fill_betweenx(z, lo, hi, color='#b9b9b9', alpha=alpha, lw=0, label=label)
    else:
        ax.fill_betweenx(z, lo, hi, facecolor='none', edgecolor='#7a7a7a', hatch=hatch, lw=0.5, label=label)


def codes(ax, item, legend=True):
    for c, v in item['per_code'].items():
        st = CODE_STYLE[c]
        ax.plot(v['value'], v['z_f_over_ustar'], color=C_REF, lw=1.0, ls=st['ls'],
                label=st['label'] if legend else None)


def plot_all(runs, ref, p, outdir, title):
    """runs: list of (r, label). The paper data once, every run on top."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 7,
                         'legend.frameon': False})
    P = ref['profiles']
    files = []
    sty = lambda i: dict(color=RUN_COLORS[i % 6], ls=RUN_LS[i % 6])
    multi = len(runs) > 1
    lw = lambda i: 2. if i == 0 else 1.5

    # ---- 1. phi_M, phi_C, mean wind, hodograph --------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.0), constrained_layout=True)
    codes(ax[0], P['phi_M'])
    codes(ax[1], P['phi_C'], legend=False)
    for i, (r, lab) in enumerate(runs):
        ax[0].plot(r['phiM'][1:], r['zhn'][1:], marker='o', ms=3, lw=lw(i), label=lab, **sty(i))
        ax[1].plot(r['phiC'][1:], r['zhn'][1:], marker='o', ms=3, lw=lw(i), label=lab, **sty(i))
    style_ax(ax[0], r'$\Phi_M$', ymax=0.1); ax[0].set_xlim(0, 2.5)
    ax[0].set_title('(a) Fig. 4a: non-dimensional shear')
    ax[0].legend(loc='upper left')
    style_ax(ax[1], r'$\Phi_C$', ylabel=False, ymax=0.1); ax[1].set_xlim(0, 2.5)
    ax[1].set_title('(b) Fig. 4b: non-dimensional scalar gradient')
    A1 = np.array(ref['table_A1']['data'])
    r0 = runs[0][0]
    zA = A1[:, 0] * p['fc'] / r0['ust']
    ax[2].plot(A1[:, 1], zA, color=C_REF, lw=1, ls='--', label=r'$\langle u\rangle$ initial (Table A.1)')
    ax[2].plot(A1[:, 2], zA, color=C_REF, lw=1, ls=':', label=r'$\langle v\rangle$ initial (Table A.1)')
    ax[3].plot(A1[:, 1], A1[:, 2], color=C_REF, lw=1, ls='--', label='initial (Table A.1)')
    for i, (r, lab) in enumerate(runs):
        ax[2].plot(r['u'], r['zn'], color=RUN_COLORS[i % 6], ls=RUN_LS[i % 6], lw=lw(i),
                   label=r'$\langle u\rangle$ ' + lab)
        ax[2].plot(r['v'], r['zn'], color=RUN_COLORS2[i % 6], ls=RUN_LS[i % 6], lw=lw(i),
                   label=r'$\langle v\rangle$ ' + lab)
        ax[3].plot(r['u'], r['v'], lw=lw(i), label=lab, **sty(i))
        ax[3].plot(r['u'][0], r['v'][0], 'o', color=RUN_COLORS[i % 6], ms=5)
    style_ax(ax[2], r'wind (m s$^{-1}$)', ylabel=False)
    ax[2].set_title('(c) mean wind (final profiles not shown in paper)')
    ax[2].legend(loc='center right')
    ax[3].set_xlabel(r'$\langle u\rangle$ (m s$^{-1}$)'); ax[3].set_ylabel(r'$\langle v\rangle$ (m s$^{-1}$)')
    ax[3].grid(True, color='#e3e3e3', lw=0.6); ax[3].set_aspect('equal', adjustable='datalim')
    for s in ('top', 'right'):
        ax[3].spines[s].set_visible(False)
    ax[3].set_title('(d) hodograph (dot: first level)'); ax[3].legend(loc='upper right')
    fig.suptitle(title)
    files.append(os.path.join(outdir, 'andren1994_phi.png'))
    fig.savefig(files[-1], dpi=150); plt.close(fig)

    # ---- 2. variances --------------------------------------------------------
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.0), constrained_layout=True)
    for i, (key, name, fig_no) in enumerate((('u2', r"$\overline{u'^2}/u_*^2$", 'Fig. 5'),
                                             ('v2', r"$\overline{v'^2}/u_*^2$", 'Fig. 5'),
                                             ('w2', r"$\overline{w'^2}/u_*^2$", 'Fig. 5'),
                                             ('c2', r"$\overline{c'^2}/c_*^2$", 'Fig. 7'))):
        band(ax[i], P[key + '_total'], 'paper: total, all codes')
        band(ax[i], P[key + '_sgs'], 'paper: SGS part', alpha=1., hatch='////')
        for j, (r, lab) in enumerate(runs):
            zz = r['zhn'] if key == 'w2' else r['zn']
            ax[i].plot(r[key], zz, lw=lw(j), label=f'{lab} resolved', **sty(j))
        style_ax(ax[i], name, ylabel=(i == 0))
        ax[i].set_xlim(0, None)
        ax[i].set_title(f'({"abcd"[i]}) {fig_no}')
    ax[0].legend(loc='upper right')
    fig.suptitle(title + '  (MicroHH smag2 has no SGS TKE: resolved part only)')
    files.append(os.path.join(outdir, 'andren1994_variances.png'))
    fig.savefig(files[-1], dpi=150); plt.close(fig)

    # ---- 3. fluxes and eddy diffusivities -----------------------------------
    fig, ax = plt.subplots(1, 5, figsize=(15, 4.0), constrained_layout=True)
    for i, (key, name, fig_no, sgs) in enumerate((('uw', r"$\overline{uw}/u_*^2$", 'Fig. 6a', 'uw_sgs'),
                                                  ('vw', r"$\overline{vw}/u_*^2$", 'Fig. 6b', None),
                                                  ('wc', r"$\overline{wc}/(u_*c_*)$", 'Fig. 8', 'wc_sgs'))):
        band(ax[i], P[key + '_total'], 'paper: total, all codes')
        if sgs:
            band(ax[i], P[sgs], 'paper: SGS part', alpha=1., hatch='////')
        for j, (r, lab) in enumerate(runs):
            ax[i].plot(r[key], r['zhn'], lw=lw(j), label=f'{lab} total', **sty(j))
            if not multi:
                ax[i].plot(r[key + '_sgs'], r['zhn'], color=RUN_COLORS[j], lw=1.2, ls='--', label=f'{lab} SGS')
        style_ax(ax[i], name, ylabel=(i == 0))
        ax[i].set_title(f'({"abcde"[i]}) {fig_no}')
    ax[0].legend(loc='upper left')
    for i, (key, name, fig_no) in enumerate((('K_M', r'$K_M$ (m$^2$ s$^{-1}$)', 'Fig. 14a'),
                                             ('K_C', r'$K_C$ (m$^2$ s$^{-1}$)', 'Fig. 14b'))):
        a = ax[3 + i]
        band(a, P[key], 'paper: all codes')
        for j, (r, lab) in enumerate(runs):
            a.plot(r['KM' if key == 'K_M' else 'KC'], r['zn'], lw=lw(j), label=lab, **sty(j))
        style_ax(a, name, ylabel=False); a.set_xlim(0, None)
        a.set_title(f'({"de"[i]}) {fig_no} (grid dependent)')
    ax[3].legend(loc='upper right')
    fig.suptitle(title)
    files.append(os.path.join(outdir, 'andren1994_fluxes.png'))
    fig.savefig(files[-1], dpi=150); plt.close(fig)

    # ---- 4. time series ----------------------------------------------------
    B, T = ref['bulk'], ref['timeseries']
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.4), constrained_layout=True)
    win = (r0['t0'] * p['fc'], r0['t1'] * p['fc'])
    us = B['ustar_over_UG']
    vals = [us[c] for c in CODE_STYLE]
    ax[0].axhspan(min(vals), max(vals), color='#b9b9b9', alpha=0.5, lw=0, label='paper, all codes')
    it = B['integrated_tke']
    ax[1].axhspan(*it['range_all_codes_tf_gt_1'], color='#b9b9b9', alpha=0.5, lw=0, label='paper (resolved + SGS)')
    ax[1].axhline(it['value'], color=C_REF, lw=1)
    for a, key in ((ax[2], 'C_u'), (ax[3], 'C_v')):
        ts = T[key]
        tt, vv = np.array(ts['tf']), np.array(ts['value'])
        a.fill_between(tt, vv - ts['uncertainty'], vv + ts['uncertainty'], color='#b9b9b9', alpha=0.5, lw=0,
                       label='paper (coarse digitization)')
        a.axhline(1., color=C_REF, lw=0.8, ls=':')
    tmax = 10.
    for i, (r, lab) in enumerate(runs):
        tf = r['ts_tf']
        tmax = max(tmax, tf[-1])
        ax[0].plot(tf, r['ts_ust'], lw=1.5 if i == 0 else 1.2, label=lab, **sty(i))
        ax[1].plot(tf, r['ts_tke'], lw=1.5 if i == 0 else 1.2, label=f'{lab} resolved', **sty(i))
        ax[2].plot(tf, r['ts_Cu'], lw=1.5 if i == 0 else 1.2, label=lab, **sty(i))
        ax[3].plot(tf, r['ts_Cv'], lw=1.5 if i == 0 else 1.2, label=lab, **sty(i))
    ax[0].set_ylabel(r'$u_*/U_G$'); ax[0].set_title('(a) section 4(a)')
    ax[1].set_ylabel(r'$f\int q^2/2\,dz\,/u_*^3$'); ax[1].set_title('(b) Fig. 2')
    ax[2].set_ylabel(r'$C_u$'); ax[2].set_title('(c) Fig. 3a')
    ax[3].set_ylabel(r'$C_v$'); ax[3].set_title('(d) Fig. 3b')
    for a in ax:
        a.axvspan(*win, color=RUN_COLORS[0], alpha=0.07, lw=0)
        a.set_xlabel(r'$tf$'); a.set_xlim(0, tmax)
        a.grid(True, color='#e3e3e3', lw=0.6)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.legend(loc='upper right')
    fig.suptitle(title + '  (shaded: averaging window)')
    files.append(os.path.join(outdir, 'andren1994_timeseries.png'))
    fig.savefig(files[-1], dpi=150); plt.close(fig)

    # ---- 5. near-surface profiles in metres (for comparing surfaces) --------
    fig, ax = plt.subplots(1, 4, figsize=(13, 4.0), constrained_layout=True)
    zmax = 400.
    for i, (r, lab) in enumerate(runs):
        U = np.hypot(r['u'], r['v'])
        ax[0].plot(U, r['z'], marker='o', ms=3, lw=lw(i), label=lab, **sty(i))
        tau = np.hypot(r['uw_kin'], r['vw_kin'])
        ax[1].plot(tau, r['zh'], marker='o', ms=3, lw=lw(i), label=lab, **sty(i))
        ax[1].plot([np.hypot(r['uw0_extrap'], r['vw0_extrap'])], [0.], marker='x', ms=8, mew=1.5,
                   color=RUN_COLORS[i % 6], ls='none')
        ax[2].plot(r['phiM'][1:], r['zh'][1:], marker='o', ms=3, lw=lw(i), label=lab, **sty(i))
        ax[3].plot(r['KM'], r['z'], marker='o', ms=3, lw=lw(i), label=lab, **sty(i))
    zl = np.linspace(1., zmax, 200)
    U_log = runs[0][0]['ust'] / KAPPA * np.log(zl / p['z0'])
    ax[0].plot(U_log, zl, color=C_REF, lw=1, ls=':', label=r'log law with $u_*$ of ' + runs[0][1])
    ax[0].set_xlabel(r'$|\langle \mathbf{u}\rangle|$ (m s$^{-1}$)'); ax[0].set_title('(a) mean wind speed')
    ax[1].set_xlabel(r'$|\overline{u_h w}|$ total (m$^2$ s$^{-2}$)')
    ax[1].set_title('(b) stress; x: linear extrapolation\nfrom the 1st two half levels')
    ax[2].set_xlabel(r'$\Phi_M$'); ax[2].set_title(r'(c) $\Phi_M$ (each run its own $u_*$)')
    ax[2].axvline(1., color=C_REF, lw=0.8, ls=':')
    ax[3].set_xlabel(r'$K_M$ (m$^2$ s$^{-1}$)'); ax[3].set_title('(d) eddy viscosity')
    for j, a in enumerate(ax):
        a.set_ylim(0, zmax)
        a.set_ylabel('height above the surface (m)' if j == 0 else '')
        a.grid(True, color='#e3e3e3', lw=0.6)
        for s in ('top', 'right'):
            a.spines[s].set_visible(False)
        a.legend(loc='best')
    ax[1].set_xlim(0, None); ax[3].set_xlim(0, None)
    fig.suptitle(title + '  (near-surface, heights in m above the surface)')
    files.append(os.path.join(outdir, 'andren1994_surface.png'))
    fig.savefig(files[-1], dpi=150); plt.close(fig)
    return files


def bulk_table(runs, ref):
    B = ref['bulk']
    us = B['ustar_over_UG']
    vals = [us[c] for c in CODE_STYLE]
    sa = B['surface_stress_angle_deg']
    ts = ref['timeseries']
    win = lambda key: np.mean([v for t, v in zip(ts[key]['tf'], ts[key]['value']) if 7. <= t <= 10.])
    fmt = lambda f: [f(r) for r, _ in runs]
    rows = [
        ('u*/U_G', fmt(lambda r: f"{r['ust_over_UG']:.4f}"), f"{min(vals):.4f}-{max(vals):.4f}",
         'AM .0425, MBb .0448, MBn .0402, N .0402, SG .0425 (sec. 4a)'),
        ('u* (m/s)', fmt(lambda r: f"{r['ust']:.4f}"), f"{min(vals)*10:.3f}-{max(vals)*10:.3f}",
         'flat: MOST surface flux; IB: skin+form drag'),
        ('lid z_top f/u*', fmt(lambda r: f"{r['lid_n']:.3f}"), '~0.35', 'section 3'),
        ('h f/u* (5% stress/0.95)', fmt(lambda r: f"{r['h_n']:.3f}"), 'n/a', 'not given in the paper'),
        ('h (m)', fmt(lambda r: f"{r['h']:.0f}"), 'n/a', ''),
        ('surface stress angle (deg)', fmt(lambda r: f"{r['stress_angle']:.1f}"),
         f"{sa['range'][0]:.1f}-{sa['range'][1]:.1f}", 'atan(vw0/uw0), from Fig. 6 surface values'),
        ('wind angle 1st level (deg)', fmt(lambda r: f"{r['wind_angle_1']:.1f}"), 'n/a',
         'Table A.1 initial: 26.2'),
        ('z1 (m) / |U(z1)| (m/s)', fmt(lambda r: f"{r['z1']:.2f}/{r['U1']:.3f}"), 'n/a', 'first air level'),
        ('|U| 2nd level (m/s)', fmt(lambda r: f"{r['U2']:.3f}"), 'n/a', f'z1 + dz'),
        ('u* log law at z1 / u*', fmt(lambda r: f"{r['ust_most_1'] / r['ust']:.3f}"), 'n/a',
         'kappa |<U>(z1)|/ln(z1/z0), mean wind'),
        ('u* log law at z2 / u*', fmt(lambda r: f"{r['ust_most_2'] / r['ust']:.3f}"), 'n/a',
         'same with the 2nd level'),
        ('stress extrap./surface', fmt(lambda r: f"{r['tau0_extrap_ratio']:.3f}"), '1',
         'linear from 1st two half levels'),
        ('scalar flux extrap./surf.', fmt(lambda r: f"{r['wc0_extrap_ratio']:.3f}"), '1', ''),
        ('(wc)0 (kg m-2 s-1)', fmt(lambda r: f"{r['wc0']:.3e}"), '1.000e-03', 'IB: s_fluxbot_ib'),
        ('stats uw at surface/(uw)0', fmt(lambda r: f"{np.hypot(r['stats_uw0'], r['stats_vw0']) / np.hypot(r['uw0'], r['vw0']):.3f}"),
         'n/a', 'IB: ghost-cell operator flux (not applied)'),
        ('form drag / total', fmt(lambda r: f"{r['form_frac']:.4f}" if 'form_frac' in r else '-'), '0',
         'IB only: taux_form_ib (0 on flat)'),
        ('f int q2/2 dz / u*^3', fmt(lambda r: f"{r['tke_int']:.3f}"), '~0.7 (0.55-0.90)',
         'resolved only; paper res.+SGS'),
        ('C_u (window mean)', fmt(lambda r: f"{r['Cu']:.2f}"), f"{win('C_u'):.2f}", 'Fig. 3a, coarse'),
        ('C_v (window mean)', fmt(lambda r: f"{r['Cv']:.2f}"), f"{win('C_v'):.2f}", 'Fig. 3b, coarse'),
    ]
    head = ['quantity'] + [lab for _, lab in runs] + ['paper', 'note']
    body = [[q] + v + [pp, n] for q, v, pp, n in rows]
    w = [max(len(x[i]) for x in body + [head]) for i in range(len(head))]
    line = lambda x: '  '.join(s.ljust(w[i]) for i, s in enumerate(x))
    out = [line(head), '-' * (sum(w) + 2 * len(w))] + [line(x) for x in body]
    return '\n'.join(out)


def window(t, p, tavg):
    if tavg:
        return tavg
    if t[-1] >= 0.97 * p['t_avg'][1]:               # last sample may fall just short of 10/f
        return p['t_avg'][0], min(p['t_avg'][1], t[-1])
    t0, t1 = t[-1] - (t[-1] - t[0]) / 3., t[-1]
    print(f'NOTE: the run ends at t = {t[-1]:g} s < {p["t_avg"][1]:g} s (10/f): averaging over the last third, '
          f'{t0:g}-{t1:g} s, instead of the paper\'s 7/f-10/f. Not comparable with the paper.')
    return t0, t1


def analyse(rundir, case, stats, tavg):
    p = load_params(rundir)
    s = read_stats(stats or find_stats(rundir, case))
    ib = ib_surface(rundir, p, s['time']) if p['surface'] == 'ib' else None
    t0, t1 = window(s['time'], p, tavg)
    r = compute(s, p, t0, t1, ib)
    if ib is not None:
        r['most'] = most_check(rundir, p, r, s['time'])
    return p, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rundir', default='.', help='run directory (stats files, case_parameters.json)')
    ap.add_argument('--case', default='andren1994')
    ap.add_argument('--stats', nargs='*', help='statistics file(s) of --rundir; default: <rundir>/<case>.default.*.nc')
    ap.add_argument('--compare', nargs='*', default=[], help='more run directories to overlay')
    ap.add_argument('--labels', nargs='*', default=None, help='one label per run (--rundir first)')
    ap.add_argument('--ref', default=None, help='reference json (default: next to this script or in rundir)')
    ap.add_argument('--tavg', nargs=2, type=float, default=None, help='averaging window t0 t1 (s)')
    ap.add_argument('--outdir', default=None, help='figure directory (default <rundir>/figs)')
    ap.add_argument('--summary', default=None, help='also write the bulk numbers of every run to this json')
    a = ap.parse_args()

    reff = a.ref or next((f for f in (os.path.join(HERE, 'reference_andren1994.json'),
                                      os.path.join(a.rundir, 'reference_andren1994.json')) if os.path.exists(f)), None)
    if reff is None:
        sys.exit('reference_andren1994.json not found (use --ref)')
    ref = json.load(open(reff))

    dirs = [a.rundir] + a.compare
    labels = a.labels or [('MicroHH' if len(dirs) == 1 else os.path.basename(os.path.normpath(d))) for d in dirs]
    if len(labels) != len(dirs):
        sys.exit('--labels needs one label per run directory')
    runs, p0 = [], None
    for n, d in enumerate(dirs):
        p, r = analyse(d, a.case, a.stats if n == 0 else None, a.tavg)
        p0 = p0 or p
        runs.append((r, labels[n]))

    outdir = a.outdir or os.path.join(a.rundir, 'figs')
    os.makedirs(outdir, exist_ok=True)
    r0 = runs[0][0]
    if len(runs) == 1:
        title = (f"Andren et al. (1994) neutral Ekman layer: MicroHH {p0.get('itot', '?')}x{p0.get('jtot', '?')}x"
                 f"{p0.get('ktot', '?')} ({p0['surface']} surface), mean over t = {r0['t0']:g}-{r0['t1']:g} s "
                 f"({r0['nsamples']} samples), u* = {r0['ust']:.3f} m/s")
    else:
        title = (f"Andren et al. (1994) neutral Ekman layer: " + ', '.join(f"{lab} u* = {r['ust']:.3f}"
                 for r, lab in runs) + f" m/s; mean over t = {r0['t0']:g}-{r0['t1']:g} s")
    files = plot_all(runs, ref, p0, outdir, title)
    print(title)
    print(bulk_table(runs, ref))
    for r, lab in runs:
        m = r.get('most')
        if m:
            print(f'{lab}: MOST check, ustar_ib vs kappa|U|/ln(d/z0) per column and sample ({m["where"]}, '
                  f'{m["n"]} values): mean ratio {m["ratio"]:.4f}, median |rel. diff| {m["med_rel"]:.2e}, '
                  f'95th pct {m["p95_rel"]:.2e}; mean ustar_ib {r["ustar_ib_mean"]:.4f} m/s')
    print('figures:', ' '.join(files))
    if a.summary:
        keys = ('ust', 'ust_over_UG', 'uw0', 'vw0', 'wc0', 'stress_angle', 'wind_angle_1', 'z1', 'U1', 'U2',
                'ust_most_1', 'ust_most_2', 'tau0_extrap_ratio', 'wc0_extrap_ratio', 'uw0_extrap', 'vw0_extrap',
                'stats_uw0', 'stats_vw0', 'form_frac', 'ustar_ib_mean', 'lid_n', 'h', 'h_n', 'tke_int', 'Cu', 'Cv',
                't0', 't1', 'nsamples')
        out = {}
        for (r, lab), d in zip(runs, dirs):
            e = {k: float(r[k]) for k in keys if k in r}
            e['rundir'] = d
            e['surface'] = r['surface']
            if r.get('most'):
                e['most_check'] = {k: (float(v) if not isinstance(v, str) else v) for k, v in r['most'].items()}
            out[lab] = e
        with open(a.summary, 'w') as f:
            json.dump(out, f, indent=1)
        print('summary:', a.summary)


if __name__ == '__main__':
    main()
