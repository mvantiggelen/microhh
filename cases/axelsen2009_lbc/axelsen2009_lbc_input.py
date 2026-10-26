#!/usr/bin/env python3
"""
Axelsen and van Dop (2009) katabatic case P1 with OPEN lateral boundaries:
a plateau - slope - valley section along x, periodic in y, on a coarse grid
(dx = 100 m, dz = 10 m, 160 x 160 x 160 points by default; the resolution
of a regional LES over a large ice sheet).

Why not periodic
----------------
axelsen2009/ reproduces the paper's infinite slope with terrain-following
periodic lateral boundaries. P1 stayed unstable there (blow-up at
t = 3370 s). This case drops the infinite-slope idea: the slope
has a top and a foot, and the domain edges sit where the true state is
known, so that MicroHH's open BC (a Dirichlet value plus a relaxation zone,
Davies 1976) can be exact.

Layout along x (downslope = +x), default
----------------------------------------
    0 -  3.0 km   plateau, NOT cooled: the air stays the resting environment,
                  so the west LBC (u = v = 0, thl_a(z)) is exact (inflow side;
                  the west sponge covers 0 - 1 km)
    3.0 - 11.5 km the 5 deg slope, cooled with the paper's flux (drop 744 m;
                  the breaks smoothed over --break-width)
   11.5 - 16 km   valley, NOT cooled: the katabatic current spreads as a
                  gravity current / cold pool and only then reaches the east
                  sponge (15 - 16 km), which relaxes it to the resting
                  environment (outflow side)
y is periodic ([boundary_lateral] sw_periodic_y). The total mass
flux through the west/east LBCs is zero, so w_top = 0.

Same as axelsen2009 --case P1 --bc flux
---------------------------------------
alpha = 5 deg, Fs = -0.0012 m2/s3 as a kinematic heat flux per unit surface
area F = Fs theta_a0 / g = -0.0348 K m/s ([IB] sbcbot=flux, sbot_spatial
map), N = 0.0121 1/s (G = N^2 theta_a0/g = 4.25 K/km), theta_a0 = 284.65 K
at the height of the slope's midpoint, z0 = 4.4 mm, z0h yang2008, no
Coriolis, at rest; 1.5-order TKE closure (tke2, Pr = 1) with the paper's
initial SFS TKE (Beare et al. 2006 profile in the lowest 25 m above the local
surface, axelsen2009_lbc_init.py); random perturbations scaled with V and B;
run length 3 tau_osc = 19800 s. The recommended IB settings, as in
axelsen2009 (MOST wall model, sw_wall_slope, evisc cap_last; momentum by
--momentum: staircase, the MOST wall stress as a source term and the
default, or vr, the velocity reconstruction).

Usage
-----
    python3 axelsen2009_lbc_input.py --outdir runs/P1_lbc
    cd runs/P1_lbc
    ./microhh init axelsen2009_lbc
    python3 ../../axelsen2009_lbc_init.py --rundir .
    ./microhh run axelsen2009_lbc
(job.sh does these steps, on a laptop or a SLURM cluster.) Needs MicroHH
from the develop_openbc branch (for --momentum vr: a build with velocity
reconstruction).
"""
import argparse
import json
import os
import sys

import numpy as np
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
CASE = 'axelsen2009_lbc'
GRAV = 9.81
RD, KAPPA = 287.04, 287.04 / 1005.
T_MELT = 273.15
SGSTKE_MIN = 1e-7          # MicroHH Constants::sgstke_min [m2/s2]
NGC = 3                    # gd.igc = gd.jgc for swadvec=2i5

# Axelsen and van Dop (2009) Table 1, case P1 (as axelsen2009/case_parameters.json).
P1 = dict(alpha_deg=5.0, Fs=-0.0012, N=0.0121, z0=0.0044, theta_a0_K=284.65,
          tau_osc_table_s=6600.0)

# Heights above the local surface of the terrain-following crosses [m].
TF_HEIGHTS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300]
# Fine grids (dz < 5 m): whole metres (MicroHH names the planes <var>_tf<h>).
TF_HEIGHTS_FINE = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100]
# --output minimal (default since 22 Oct 2026): only what the assessment
# reads. axelsen2009_lbc_analysis.py (the test) needs the xz planes of u, w,
# thl and evisc (+ eviscs with tke2) and the xy maps ustar_ib,
# thl_fluxbot_ib, thl_sbot_ib; axelsen2009_lbc_check.py (quick look) the
# terrain-following u_tf and thl_tf at a few heights. On a parallel file
# system every cross file can cost ~2 s whatever its size: the full set (131
# files every 300 s, 365 with the tavg planes) once made a P1 run spend
# ~95 % of its 4.5 h writing.
# --output full gives the old set (yz planes, v, w_tf, sgstke, drag maps,
# 13 tf heights, tavg planes and towers) for the turbulence, drag-budget and
# vortex scripts.
TF_HEIGHTS_MIN = [5, 10, 20, 50]
TF_HEIGHTS_MIN_FINE = [1, 2, 5, 10, 20]

# Grid presets (--res); every option given on the command line overrides them.
#   coarse dx 100 m, dz 10 m, 160^3, 16 x 16 x 1.6 km (default)
#   fine  dz 1 m, dx 5 m (aspect 5; the 5 deg slope drops 0.44 dz per column),
#         512 x 128 x 300 = 2.56 x 0.64 x 0.30 km. With dz = 1 m the slope's
#         drop must fit below the buffer: a 1150 m slope drops 101 m, the
#         plateau sits at 106 m, 119 m (4.6 H) undamped above it, buffer
#         225-300 m with the paper's katabatic sigma 0.03 1/s. P1's jet
#         (z_j ~ 4 m) then spans ~4 cells, the layer (H = 26 m) 26 cells
#         (paper: dx 2.6 m, dz 0.4 m). Sponges 20 cells (100 m), tau 20 s: the
#         outflowing jet (~5 m/s) crosses them in ~20 s.
PRESETS = {
    'coarse': dict(itot=160, jtot=160, ktot=160, dx=100., dz=10., x_top=3000., slope_length=8500.,
                 break_width=500., buffer_start=1200., buf_sigma=0.01, npx=8, npy=16, n_sponge=10,
                 tau_sponge=60.),
    'fine': dict(itot=512, jtot=128, ktot=300, dx=5., dz=1., x_top=400., slope_length=1150.,
                 break_width=100., buffer_start=225., buf_sigma=0.03, npx=4, npy=32, n_sponge=20,
                 tau_sponge=20.),
}


# The IB momentum block of the .ini (--momentum). 'staircase' (default): the
# MOST wall stress as a source term; 'vr': the velocity reconstruction, which
# needs a MicroHH build with it.
MOMENTUM = {
    'staircase': '''# momentum, staircase: MOST wall stress as a source term, eddy viscosity
# rescaled to it, w on the floor faces along the sub-grid slope, momentum
# exchanged with the terrain only through MOST and pressure (needs swadvec=2i5)
sw_momentum_flux=true
sw_momentum_flux_vertical=true
sw_strain_most=true
sw_strain_most_vertical=false
strain_most_min=0.05
sw_impermeable=false
sw_wall_kinematic=true
sw_momentum_air_only=true
''',
    'vr': '''# momentum by velocity reconstruction on the true DEM surface:
# MOST profile along the normal at the first air points, 0 below the
# surface, the terrain in the pressure solve
sw_momentum_reconstruct=true
reconstruct_profile=most
reconstruct_dz=1.5
reconstruct_diag_every=300
# at least the MOST eddy viscosity on the faces between the R
# points and the free air (the stress the air loses)
reconstruct_stress=true
''',
}


def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--outdir', required=True)
    p.add_argument('--res', choices=sorted(PRESETS), default='coarse',
                   help='grid preset (coarse: dx 100 m, dz 10 m; fine: dx 5 m, dz 1 m); options below override it')
    p.add_argument('--itot', type=int, default=None)
    p.add_argument('--jtot', type=int, default=None)
    p.add_argument('--ktot', type=int, default=None)
    p.add_argument('--dx', type=float, default=None, help='dx = dy [m]')
    p.add_argument('--dz', type=float, default=None, help='uniform dz [m]')
    p.add_argument('--x-top', type=float, default=None,
                   help='x of the top of the slope (end of the plateau) [m]')
    p.add_argument('--slope-length', type=float, default=None,
                   help='horizontal length of the slope [m] (drop = this x tan alpha)')
    p.add_argument('--break-width', type=float, default=None,
                   help='width of the smoothed slope breaks at the top and the foot [m]')
    p.add_argument('--cool', default='slope', choices=['slope', 'plateau,slope', 'slope,valley', 'all'],
                   help='where the surface is cooled with the paper\'s flux (default: slope only, '
                        'so the west LBC is exact)')
    p.add_argument('--bc', choices=['flux', 'temp'], default='flux',
                   help='flux: the paper\'s surface buoyancy flux (default); temp: a MELTING surface '
                        '(prescribed temperature, MOST computes the flux)')
    p.add_argument('--melt', choices=['physical', 'uniform'], default='physical',
                   help='--bc temp: physical = T_s = 0 C, theta_s = 273.15 (p_bot/p(z_s))^(R/cp), so the '
                        'deficit grows downslope; uniform = theta_s = thl_a(z_s) + (273.15 - theta_a0), '
                        'a constant deficit (as axelsen2009 --bc temp)')
    p.add_argument('--t-surface', type=float, default=273.15,
                   help='--bc temp --melt physical: the surface temperature [K] (melting ice: 273.15)')
    p.add_argument('--melt-ramp', type=float, default=3600.,
                   help='--bc temp: the surface temperature goes linearly from thl_a(z_s) (no flux) to the '
                        'melting value over this many s ([IB] sw_timedep_sbot; 0 = from the start). A sudden '
                        '-11 K surface under a weak wind drives MOST to z/L = zL_max = 10 and the exchange '
                        'collapses (8 Oct test)')
    p.add_argument('--init-jet', type=float, default=None,
                   help='initial downslope jet maximum [m/s], Prandtl-shaped u_s = U e (n/n_j) exp(-n/n_j), '
                        'n_j = 0.15 H, on the cooled part (axelsen2009_lbc_init.py). Default 0 for '
                        '--bc flux, 2 m/s for --bc temp (MOST gives no flux at zero wind)')
    p.add_argument('--tavg-interval', type=float, default=None,
                   help='[IB] tavg_interval [s]: time-averaged planes and towers ('
                        '0 = off; default: 0 with --output minimal, 600 with full)')
    p.add_argument('--output', choices=('minimal', 'full'), default='minimal',
                   help='minimal: only what axelsen2009_lbc_analysis.py and _check.py read '
                        '(default); full: every plane of the turbulence/drag/vortex scripts')
    p.add_argument('--dump-every', type=float, default=0.,
                   help='[dump] u, v, w, thl (3-D) every this many s for vortex_criteria.py (0 = off; '
                        'fine grid: ~0.6 GB per dump)')
    p.add_argument('--z0m', type=float, default=None,
                   help='momentum roughness [m] of the surface (default: P1, 4.4 mm)')
    p.add_argument('--z0h', type=float, default=None,
                   help='heat roughness [m]; sets [IB] z0h_model = fixed (default: z0m/10 via ratio). '
                        'With --z0m and no --z0h, z0h stays at the P1 value 0.44 mm, so only the momentum '
                        'roughness changes')
    p.add_argument('--z0h-model', choices=['ratio', 'yang2008'], default='ratio',
                   help='[IB] z0h_model (default ratio: z0h = z0m/10; yang2008 diverges at small u*)')
    p.add_argument('--alpha', type=float, default=None, help='slope angle [deg] (P1: 5)')
    p.add_argument('--n-tau', type=float, default=3., help='run length in oscillation periods')
    p.add_argument('--sgs', choices=['tke2', 'smag2'], default='tke2')
    p.add_argument('--momentum', choices=('staircase', 'vr'), default='staircase',
                   help='IB momentum method: staircase (default; the MOST wall stress as a source term) '
                        'or vr (velocity reconstruction; needs a MicroHH build with it)')
    p.add_argument('--no-init-tke', action='store_true', help='tke2: no initial SFS TKE profile')
    p.add_argument('--n-sponge', type=int, default=None, help='west/east sponge width [cells]')
    p.add_argument('--tau-sponge', type=float, default=None, help='sponge relaxation time [s]')
    p.add_argument('--no-periodic-y', action='store_true',
                   help='open BCs on all four edges (no periodic y; south/north LBC files written)')
    p.add_argument('--buffer-start', type=float, default=None, help='[buffer] zstart [m]')
    p.add_argument('--buf-sigma', type=float, default=None,
                   help='[buffer] sigma [1/s] (~0.8 N; MicroHH default 0.00223)')
    p.add_argument('--solid-levels', type=int, default=5,
                   help='solid levels below the valley floor (the 2i5 stencil needs >= 4)')
    p.add_argument('--npx', type=int, default=None)
    p.add_argument('--npy', type=int, default=None)
    p.add_argument('--out-every', type=float, default=300., help='stats/cross interval [s]')
    p.add_argument('--endtime', type=float, default=None, help='override the run length [s]')
    a = p.parse_args()
    for k, v in PRESETS[a.res].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    if a.tavg_interval is None:
        a.tavg_interval = 0. if a.output == 'minimal' else 600.
    return a


def slope_weight(x, x0, x1, w):
    """1 on the slope [x0, x1], 0 off it, cosine tapers of width w centred on
    the breaks, so the horizontal integral (the drop) is tan(alpha) (x1 - x0)."""
    def ramp(d):            # 0 -> 1 across [-w/2, w/2]
        if w <= 0.:
            return (d >= 0.).astype(float)
        t = np.clip(d / w + 0.5, 0., 1.)
        return 0.5 - 0.5 * np.cos(np.pi * t)
    return ramp(x - x0) * ramp(x1 - x)


def main():
    a = parse()
    r = dict(P1)
    alpha = a.alpha if a.alpha is not None else r['alpha_deg']
    tana = np.tan(np.deg2rad(alpha))
    Fs, N, z0, th0 = r['Fs'], r['N'], r['z0'], r['theta_a0_K']
    z0h = z0 / 10.
    if a.z0m is not None or a.z0h is not None:
        z0h = a.z0h if a.z0h is not None else z0 / 10.
        z0 = a.z0m if a.z0m is not None else z0
        a.z0h_model = 'fixed'
    V = np.sqrt(abs(Fs) / N)
    B = abs(Fs) / V
    H = V / N
    gam = N**2 * th0 / GRAV
    F_thl = Fs * th0 / GRAV
    tau = r['tau_osc_table_s'] if a.alpha is None else 2. * np.pi / (N * np.sin(np.deg2rad(alpha)))

    itot, jtot, ktot, dx, dz = a.itot, a.jtot, a.ktot, a.dx, a.dz
    lx, ly, zsize = itot * dx, jtot * dx, ktot * dz
    npx, npy = a.npx, a.npy
    for n, d, what in [(itot, npx, 'itot % npx'), (itot, npy, 'itot % npy'),
                       (jtot, npx, 'jtot % npx'), (jtot, npy, 'jtot % npy'),
                       (ktot, npx, 'ktot % npx')]:
        if n % d:
            raise SystemExit(f'{what} != 0 ({n}, {d}): choose another --npx/--npy')
    if itot // npx < a.n_sponge + NGC + 1:
        raise SystemExit(f'imax = {itot // npx} too small for a {a.n_sponge}-cell sponge')

    os.makedirs(a.outdir, exist_ok=True)
    out = lambda f: os.path.join(a.outdir, f)

    # Terrain: z_s(x) = z_base + tan(alpha) * integral_x^lx of the slope weight.
    x0, x1 = a.x_top, a.x_top + a.slope_length
    if not (a.n_sponge * dx + a.break_width < x0 and x1 + a.break_width < lx - a.n_sponge * dx):
        raise SystemExit('the slope (with its breaks) must lie between the two sponges')
    z_base = (a.solid_levels + 0.25) * dz
    xf = np.linspace(0., lx, 200 * itot + 1)
    wf = slope_weight(xf, x0, x1, a.break_width)
    integral = np.concatenate(([0.], np.cumsum(0.5 * (wf[1:] + wf[:-1]) * np.diff(xf))))
    zs_f = z_base + tana * (integral[-1] - integral)
    x = (np.arange(itot) + 0.5) * dx
    y = (np.arange(jtot) + 0.5) * dx
    zs_x = np.interp(x, xf, zs_f)
    w_x = slope_weight(x, x0, x1, a.break_width)
    dem = np.broadcast_to(zs_x[None, :], (jtot, itot)).astype(np.float64)
    dem.tofile(out('dem.0000000'))
    drop = zs_x.max() - zs_x.min()
    z_top_terrain = zs_x.max()
    z_ref = z_base + 0.5 * tana * a.slope_length        # theta_a0 at the slope's midpoint
    free_min = min(200., 4. * H)
    if z_top_terrain + free_min > a.buffer_start or a.buffer_start >= zsize:
        raise SystemExit(f'the plateau ({z_top_terrain:.0f} m) needs >= {free_min:.0f} m of undamped air below '
                         f'--buffer-start ({a.buffer_start:.0f} m), inside zsize = {zsize:.0f} m')

    thl_a = lambda zz: th0 + gam * (zz - z_ref)

    # Surface cooling: the paper's flux where --cool says, on the slope weighted
    # with the local slope fraction (smooth across the breaks).
    cool = {'slope': w_x}
    plateau = (x < x0).astype(float) * (1. - w_x)
    valley = (x > x1).astype(float) * (1. - w_x)
    if a.cool in ('plateau,slope', 'all'):
        cool['plateau'] = plateau
    if a.cool in ('slope,valley', 'all'):
        cool['valley'] = valley
    frac = np.clip(sum(cool.values()), 0., 1.)
    z_c = (np.arange(ktot) + 0.5) * dz
    if a.bc == 'flux':
        thl_s = None
        sbot_x = F_thl * frac
    else:
        # A melting surface on the cooled part, the environment (no flux) elsewhere.
        thl_env = thl_a(zs_x)
        if a.melt == 'physical':
            # T_s fixed; theta relative to p_bot (= 1000 hPa at z = 0, [thermo] pbot):
            # hydrostatic p(z) of the environment, T = theta (p/p_bot)^(R/cp).
            zf = np.linspace(0., zs_x.max() + dz, 2000)
            lnp = np.zeros_like(zf)
            for k in range(1, len(zf)):
                zm = 0.5 * (zf[k] + zf[k-1])
                T = thl_a(zm) * np.exp(KAPPA * lnp[k-1])
                lnp[k] = lnp[k-1] - GRAV / (RD * T) * (zf[k] - zf[k-1])
            thl_melt = a.t_surface * np.exp(-KAPPA * np.interp(zs_x, zf, lnp))
        else:
            thl_melt = thl_env + (T_MELT - th0)
        thl_s = thl_env + frac * (thl_melt - thl_env)
        sbot_x = thl_s
    sbot = np.broadcast_to(sbot_x[None, :], (jtot, itot)).astype(np.float64)
    sbot.tofile(out('thl_sbot.0000000'))
    endtime = a.endtime if a.endtime is not None else float(np.ceil(a.n_tau * tau / 600.) * 600.)
    ramp = a.bc == 'temp' and a.melt_ramp > 0.
    sbot_load = a.out_every
    if ramp:
        # Time-dependent surface temperature: thl_sbot.<t> every sbot_load s
        # over the whole run (MicroHH interpolates linearly, no extrapolation).
        for t in np.arange(0., endtime + sbot_load + 0.5, sbot_load):
            r = min(t / a.melt_ramp, 1.)
            th_t = thl_env + r * (thl_s - thl_env)
            np.broadcast_to(th_t[None, :], (jtot, itot)).astype(np.float64).tofile(out('thl_sbot.%07d' % int(t)))
            np.zeros((jtot, itot)).tofile(out('qt_sbot.%07d' % int(t)))
    if not os.path.exists(out('qt_sbot.0000000')):
        np.zeros((jtot, itot)).tofile(out('qt_sbot.0000000'))

    # Initial profiles: the resting environment.
    z = (np.arange(ktot) + 0.5) * dz
    with nc.Dataset(out(f'{CASE}_input.nc'), 'w') as f:
        f.createDimension('z', ktot)
        f.createVariable('z', 'f8', ('z',))[:] = z
        g = f.createGroup('init')
        for name, val in [('thl', thl_a(z)), ('qt', 0*z), ('sgstke', 0*z + SGSTKE_MIN),
                          ('u', 0*z), ('v', 0*z), ('u_geo', 0*z), ('v_geo', 0*z)]:
            g.createVariable(name, 'f8', ('z',))[:] = val

    # LBCs: the same resting environment at every open edge (horizontally
    # uniform, so the periodic y halo of the west/east files is trivially right).
    slist = ['thl', 'qt'] + (['sgstke'] if a.sgs == 'tke2' else [])
    prof = {'u': 0*z, 'v': 0*z, 'thl': thl_a(z), 'qt': 0*z, 'sgstke': 0*z + SGSTKE_MIN}
    edges = ['west', 'east'] + ([] if not a.no_periodic_y else ['south', 'north'])
    for name in ['u', 'v'] + slist:
        pad_w = NGC + 1 if name == 'u' else NGC
        pad_s = NGC + 1 if name == 'v' else NGC
        shapes = {'west': (ktot, jtot + 2*NGC, pad_w + a.n_sponge),
                  'east': (ktot, jtot + 2*NGC, NGC + a.n_sponge),
                  'south': (ktot, pad_s + a.n_sponge, itot + 2*NGC),
                  'north': (ktot, NGC + a.n_sponge, itot + 2*NGC)}
        for edge in edges:
            arr = np.broadcast_to(prof[name][:, None, None], shapes[edge]).astype(np.float64)
            arr.tofile(out(f'lbc_{name}_{edge}.0000000'))

    init_tke = None
    if a.sgs == 'tke2' and not a.no_init_tke:
        init_tke = dict(kind='beare2006', e0=0.4, depth=25.)

    # Cross-sections: xz planes along the slope, yz planes on the
    # plateau, the slope and in the valley (cell centres).
    j_mid = jtot // 2
    # Four xz planes across y (the slope-averaged profiles of
    # axelsen2009_lbc_analysis.py average over all of them).
    n_xz = 2 if a.output == 'minimal' else 4
    j_xz = sorted(set(int(j) for j in np.round(np.linspace(0, jtot, n_xz + 1)[:-1] + jtot / (2 * n_xz))))
    yz_x = [0.5 * (a.n_sponge * dx + x0 - 0.5 * a.break_width), x0 + 0.25 * a.slope_length,
            x0 + 0.5 * a.slope_length, x0 + 0.75 * a.slope_length,
            0.5 * (x1 + 0.5 * a.break_width + lx - a.n_sponge * dx), lx - (a.n_sponge + 2) * dx]
    i_yz = sorted(set(int(np.clip(round(v / dx - 0.5), 0, itot - 1)) for v in yz_x))
    if a.output == 'minimal':
        i_yz = []
        crosslist = ['u', 'w', 'thl', 'u_tf', 'thl_tf', 'thlfluxbot_ib', 'thl_sbot_ib', 'ustar_ib']
    else:
        crosslist = ['u', 'v', 'w', 'thl', 'u_tf', 'v_tf', 'w_tf', 'thl_tf',
                     'thlfluxbot_ib', 'thl_sbot_ib', 'ustar_ib', 'obuk_ib', 'ch_ib',
                     'taux_form_ib', 'tauy_form_ib', 'taux_skin_ib', 'tauy_skin_ib']
        if a.sgs == 'tke2':
            crosslist.append('sgstke')
    # Eddy viscosity and diffusivity: the SFS momentum and buoyancy fluxes of
    # the slope-averaged flux profiles (axelsen2009_lbc_analysis.py).
    crosslist += ['evisc'] + (['eviscs'] if a.sgs == 'tke2' else [])
    if a.output == 'minimal':
        tf_all = TF_HEIGHTS_MIN_FINE if dz < 5. else TF_HEIGHTS_MIN
    else:
        tf_all = TF_HEIGHTS_FINE if dz < 5. else TF_HEIGHTS
    tf = [h for h in tf_all if h < a.buffer_start - z_top_terrain]

    if a.bc == 'flux':
        ib_sbc = ('# flux: sbot = kinematic heat flux per unit surface area; the\n'
                  '# map thl_sbot.0000000 holds F on the cooled part and 0 elsewhere.\n'
                  f'sbcbot=flux\nsbot[thl]={F_thl:.9g}')
    else:
        ib_sbc = ('# dirichlet: a MELTING surface; the map thl_sbot.0000000 holds theta_s\n'
                  f'# ({a.melt}: {thl_s[w_x > 0.99].max():.2f}-{thl_s[w_x > 0.99].min():.2f} K on the slope) on the\n'
                  '# cooled part and thl_a(z_s) (no flux) elsewhere; MOST computes the flux.\n'
                  f'sbcbot=dirichlet\nsbot[thl]={float(np.median(thl_s)):.6f}')
    init_jet = a.init_jet if a.init_jet is not None else (2. if a.bc == 'temp' else 0.)

    # Time-averaged output: terrain-following planes and
    # virtual towers on the slope. Interval: a multiple of the cross sampletime.
    tavg = a.tavg_interval
    if tavg > 0. and abs(tavg / a.out_every - round(tavg / a.out_every)) > 1e-9:
        raise SystemExit('--tavg-interval must be a multiple of --out-every')
    tavg_h = [h for h in (TF_HEIGHTS_FINE if dz < 5. else TF_HEIGHTS)
              if h < a.buffer_start - z_top_terrain and h >= 0.5 * dz]
    tower_x = [x0 + f * a.slope_length for f in (0.25, 0.5, 0.75)]
    tower_y = [0.5 * ly] * 3
    ib_tavg = ''
    if tavg > 0.:
        ib_tavg = ('# Time statistics on terrain-following planes: <q>_tavg<h>\n'
                   '# interval means, covariances, tke_res, tke_sgs; and the same as\n'
                   '# *_tower profiles in every [column] file.\n'
                   f'tavg_interval={tavg:g}\n'
                   'tavg_heights=' + ','.join(f'{h:g}' for h in tavg_h) + '\n'
                   '# Surface diagnostics in the column files.\n'
                   'columnlist=thl_fluxbot_ib,ustar_ib,obuk_ib,ch_ib,thl_sbot_ib,u_ib1,thl_ib1,'
                   'z_dem,k_dem,taux_skin_ib,taux_form_ib\n')
    if a.output == 'minimal' and tavg <= 0.:
        column_sec = '[column]\n# --output minimal: no towers\nswcolumn=0\n'
    else:
        column_sec = ('[column]\n# Towers at 1/4, 1/2, 3/4 of the slope (time-averaged\n'
                      '# *_tower profiles; the plain variables are instantaneous).\n'
                      f'swcolumn=1\nsampletime={tavg if tavg > 0. else a.out_every:g}\n'
                      'coordinates[x]=' + ','.join(f'{v:.2f}' for v in tower_x) + '\n'
                      'coordinates[y]=' + ','.join(f'{v:.2f}' for v in tower_y) + '\n')
    if a.dump_every > 0.:
        dump_sec = ('[dump]\n# 3-D fields for vortex_criteria.py (Q, lambda2, lambda_ci), written\n'
                    '# at every sampletime.\n'
                    f'swdump=1\nsampletime={a.dump_every:g}\nsw_dump_every_sample=true\n'
                    'dumplist=u,v,w,thl\n')
    else:
        dump_sec = '[dump]\nswdump=0\n'

    ini = f"""[master]
npx={npx}
npy={npy}

[grid]
itot={itot}
jtot={jtot}
ktot={ktot}
xsize={lx:g}
ysize={ly:g}
zsize={zsize:g}
swspatialorder=2

[pres]
# DCT in x (open), DFT in y (periodic).
sw_openbc=true

[advec]
swadvec=2i5
cflmax=1.3
fluxlimit_list=thl,qt

[limiter]
limitlist=qt

[diff]
swdiff={a.sgs}
dnmax=0.3
""" + ("""# Pr_SFS = 1, as the paper: K_h = (ch1 + ch2 l/Delta) K_m
ch1=1.
ch2=0.
""" if a.sgs == 'tke2' else "") + f"""
[thermo]
swthermo=moist
swbasestate=anelastic
pbot=100000.
swupdatebasestate=1

[force]
swlspres=0
swls=0
swwls=0

[boundary]
# The domain floor is buried under the terrain; the IB is the surface.
mbcbot=noslip
mbctop=freeslip
swboundary=surface
swconstantz0=1
z0m={z0:g}
z0h={z0h:g}
sbcbot=dirichlet
sbctop=neumann
sbot[thl]={thl_a(z_base):.6f}
stop[thl]={gam:.9g}
sbot[qt]=0.
stop[qt]=0.
""" + ("""sbot[sgstke]=0.
stop[sgstke]=0.
""" if a.sgs == 'tke2' else "") + f"""
[boundary_lateral]
# Open at west (inflow, plateau) and east (outflow, valley), periodic y.
# LBCs: the resting environment, lbc_*.0000000.
sw_openbc=true
sw_periodic_y={'false' if a.no_periodic_y else 'true'}
sw_openbc_uv=true
sw_openbc_w=false
sw_neumann_w=true
sw_wtop_2d=false
sw_timedep=false
loadfreq=3600
slist={','.join(slist)}
sw_sponge=true
n_sponge={a.n_sponge}
tau_sponge={a.tau_sponge:g}
w_diff=0.003

[IB]
sw_immersed_boundary=dem
n_idw_points=5
sw_dem_halo_replicate=true
sw_wall_model=most
z0m={z0:g}
z0h={z0h:g}
# ratio: z0h = z0m / z0h_ratio. yang2008 (z0h ~ 70 nu/u* ...) grows without bound as
# u* -> 0 and is capped at z0h_max = 0.1 m, 23x z0m, in the decoupled katabatic layer.
z0h_model={a.z0h_model}
z0h_ratio=10
z0_nu=1.5e-05
z0_yang_beta=7.2
z0h_min=1e-10
z0h_max=0.1
sw_scalar_flux=true
sw_wall_stability_vertical=false
sw_scalar_flux_vertical=false
{MOMENTUM[a.momentum]}# Mason's wall damping with the distance to the IB surface. Acts on smag2 and
# swanisotropic; tke2 ignores it (its wall cell is not IB-aware).
sw_mason_wall=true
evisc_solid=cap_last
dn_probe_every=20000
diag_z_mom=10
diag_z_scalar=2
sw_wall_slope=true
wall_dn_min=0.1
sw_blank_solid=true
sw_blank_solid_tend=true
sw_advec_wall=true
sw_advec_wall_conserve=true
sw_flux_accum=true
# wall model evaluated at a point wall_sample_dz*dz along the normal above
# the true surface
sw_wall_sample=true
wall_sample_dz=1.5
sw_tf_cross=true
tf_cross_heights={','.join(f'{h:g}' for h in tf)}
{ib_tavg}{ib_sbc}
sbot[qt]=0.
""" + ("""sbot[sgstke]=0.
""" if a.sgs == 'tke2' else "") + f"""sbot_spatial=thl,qt
{'sw_timedep_sbot=true' + chr(10) + f'sbot_loadtime={int(sbot_load)}' + chr(10) + f'# melting surface ramped in over {a.melt_ramp:g} s: thl_sbot.<t>' if ramp else 'sw_timedep_sbot=false'}

[fields]
visc=1.e-5
svisc=1.e-5
rndseed=2
rndamp[thl]={float('%.2g' % (0.1 * B * th0 / GRAV)):g}
rndamp[u]={float('%.2g' % (0.1 * V)):g}
rndamp[v]={float('%.2g' % (0.1 * V)):g}
rndamp[w]={float('%.2g' % (0.1 * V)):g}
rndz={z_top_terrain + 50.:.1f}
rndexp=2.

[buffer]
swbuffer=1
swupdate=0
zstart={a.buffer_start:g}
sigma={a.buf_sigma:g}
beta=2.

[time]
starttime=0.
endtime={endtime:g}
dt=1.
dtmax=10.
savetime=3600.
outputiter=50
adaptivestep=true
rkorder=3
iotimeprec=0

[stats]
swstats=1
sampletime={a.out_every:g}
masklist=ib

[cross]
swcross=1
sampletime={a.out_every:g}
xz={','.join(f'{(j + 0.5) * dx:g}' for j in j_xz)}
{('yz=' + ','.join(f'{(i + 0.5) * dx:g}' for i in i_yz) + chr(10)) if i_yz else ''}crosslist={','.join(crosslist)}

{column_sec}
{dump_sec}"""
    open(out(f'{CASE}.ini'), 'w').write(ini)

    pars = dict(case='P1', alpha_deg=alpha, Fs=Fs, N=N, z0=z0, z0h=z0h, z0h_model=a.z0h_model, theta_a0=th0,
                V=V, B=B, H=H, tau_osc=tau, gamma=gam, F_thl=F_thl,
                itot=itot, jtot=jtot, ktot=ktot, dx=dx, dz=dz, lx=lx, ly=ly, zsize=zsize,
                x_top=x0, x_foot=x1, slope_length=a.slope_length, break_width=a.break_width,
                cool=a.cool, z_base=z_base, z_ref=z_ref, drop=drop, z_top_terrain=z_top_terrain,
                n_sponge=a.n_sponge, tau_sponge=a.tau_sponge, periodic_y=not a.no_periodic_y,
                buffer_start=a.buffer_start, buf_sigma=a.buf_sigma, sgs=a.sgs,
                init_tke=init_tke, bc=a.bc, melt_ramp=a.melt_ramp if ramp else 0., melt=a.melt if a.bc == 'temp' else None,
                thl_s_slope=None if thl_s is None else [float(thl_s[w_x > 0.99].max()), float(thl_s[w_x > 0.99].min())],
                init_jet=dict(U=init_jet, n_j=0.15 * H) if init_jet > 0. else None,
                tavg_interval=tavg, tavg_heights=tavg_h if tavg > 0. else [], towers_x=tower_x,
                dump_every=a.dump_every, endtime=endtime, out_every=a.out_every, tf_heights=tf,
                xz_j=j_xz, yz_i=i_yz, npx=npx, npy=npy, momentum=a.momentum, output=a.output)
    json.dump(pars, open(out('case_parameters.json'), 'w'), indent=1)

    # The cross-section converter of axelsen2009/, if it is next door.
    for d in (HERE, os.path.join(HERE, '..', 'axelsen2009')):
        if os.path.exists(os.path.join(d, 'write_cross_to_nc.py')):
            sys.path.insert(0, d)
            from write_cross_to_nc import install
            install(a.outdir, CASE, window=600., quiet=True)
            break

    print(f'{CASE}: {itot} x {jtot} x {ktot}, dx {dx:g} m, dz {dz:g} m: '
          f'{lx/1e3:g} x {ly/1e3:g} x {zsize/1e3:g} km, npx x npy = {npx} x {npy}')
    print(f'  terrain: plateau {z_top_terrain:.1f} m (x < {x0/1e3:g} km), {alpha:g} deg slope to '
          f'x = {x1/1e3:g} km (drop {drop:.1f} m, breaks over {a.break_width:g} m), valley floor '
          f'{z_base:.2f} m')
    if a.bc == 'flux':
        print(f'  cooling ({a.cool}): F = {F_thl:.5f} K m/s; theta_a0 = {th0} K at z_ref = {z_ref:.1f} m, '
              f'G = {gam*1e3:.3f} K/km')
    else:
        dsl = thl_s - thl_a(zs_x)
        print(f'  melting surface ({a.cool}, {a.melt}): theta_s {thl_s[w_x > 0.99].max():.2f}-'
              f'{thl_s[w_x > 0.99].min():.2f} K on the slope, theta_s - thl_a(z_s) '
              f'{dsl[w_x > 0.99].max():.2f} to {dsl[w_x > 0.99].min():.2f} K; theta_a0 = {th0} K at '
              f'z_ref = {z_ref:.1f} m; initial jet {init_jet:g} m/s')
    print(f'  output: crosses every {a.out_every:g} s (instantaneous), '
          + (f'tavg planes + towers every {tavg:g} s at {len(tavg_h)} heights' if tavg > 0. else 'no tavg')
          + (f', 3-D dumps every {a.dump_every:g} s' if a.dump_every > 0. else ''))
    print(f'  V = {V:.4f} m/s, H = {H:.2f} m (= {H/dz:.1f} dz), B = {B:.5f} m/s2, '
          f'tau_osc = {tau:.0f} s, endtime {endtime:g} s')
    print(f'  open BC west/east (sponge {a.n_sponge} cells = {a.n_sponge*dx/1e3:g} km, tau {a.tau_sponge:g} s), '
          f'{"periodic y" if not a.no_periodic_y else "open south/north too"}')
    print(f'  buffer from {a.buffer_start:g} m (sigma {a.buf_sigma:g} 1/s): '
          f'{a.buffer_start - z_top_terrain:.0f} m undamped above the plateau')
    print(f'  wrote {a.outdir}')


if __name__ == '__main__':
    main()
