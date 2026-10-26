#!/usr/bin/env python3
"""
Fedorovich, Gibbs and Shapiro (2017, JAS 74, 2813-2834), case B13, on the
IB terrain: the slope is made of the immersed boundary on a gravity-aligned
grid, NOT a rotated (slope-following) frame.

B13: surface buoyancy b_s = 0.1 m s-2 for 3 h ("day"), 0.05 m s-2 after
("night"), Vg = 10 m s-1 southerly (along the slope contours, +y),
N = 0.01 s-1, alpha = 0.18 deg, terrain sloping DOWN towards the east (+x),
f = 1e-4 s-1, 15 h.

Translation to the gravity-aligned IB set-up
--------------------------------------------
- Environment: thl_e(z) = theta_r + G z, G = N^2 theta_r / g, with z the
  TRUE height (z is absolute, the terrain sits on top of a buried floor).
- Surface: Fedorovich's b is buoyancy relative to the environment at the
  same true height, so the IB wall temperature is
      thl_s(x, y, t) = thl_e(z_s(x, y)) + b_s(t) theta_r / g.
  b_s steps from day to night at t_sunset; the step is a linear ramp over
  one sbot_loadtime (default 60 s), via [IB] sw_timedep_sbot.
- Momentum: LES (smag2 + the MOST wall model of the IB), NOT the paper's
  constant-viscosity DNS. The wall stress needs a z0m.
- Lateral boundaries: open. Default --lbc tfperiodic (terrain-following
  periodic boundary of the develop_openbc branch): every edge
  is fed from the opposite edge at the same height above the surface, thl
  shifted with the environment (lbc_tf_gradient[thl] = G). --lbc nudge
  instead writes time-dependent lbc_* files from the mean profiles of a
  periodic precursor (MicroHH stats file), mapped to height above the
  surface, and switches sw_lbc_tf_periodic off.

Usage
-----
    python3 fedorovich2017_b13_input.py                    # default: dx=8, dz=4, Lx=4096 (4 steps)
    python3 fedorovich2017_b13_input.py --dx 4 --lx 1024   # the paper's grid (0 steps: see README)
    python3 fedorovich2017_b13_input.py --dx 16 --lx 1024 --ly 512 --endtime 3600
    python3 fedorovich2017_b13_input.py --alpha 0          # B11, flat twin
    python3 fedorovich2017_b13_input.py --lbc nudge --precursor prec.default.0000000.nc

Writes, in --outdir: fedorovich2017_b13.ini, fedorovich2017_b13_input.nc,
dem.0000000, thl_sbot.<iotime> (symlinks to two files), qt_sbot.0000000,
and case_parameters.json (read by the analysis script).
"""
import argparse
import json
import os
import sys

import numpy as np
import netCDF4 as nc

# The cross-section converter (cross_to_nc.sh, cross_to_nc.py, microhh_tools.py)
# is copied into every run folder by write_cross_to_nc.py, next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_cross_to_nc import install as install_cross_to_nc  # noqa: E402


HERE = os.path.dirname(os.path.abspath(__file__))

GRAV = 9.81          # MicroHH Constants::grav
CASE = 'fedorovich2017_b13'


def find_microhh_tools():
    """microhh_tools.py from $MICROHH_ROOT/python, or the copy next to this script."""
    candidates = []
    if 'MICROHH_ROOT' in os.environ:
        candidates.append(os.path.join(os.environ['MICROHH_ROOT'], 'python'))
    candidates += [os.path.join(HERE, '..', '..'), HERE]
    for c in candidates:
        if os.path.exists(os.path.join(c, 'microhh_tools.py')):
            sys.path.insert(0, c)
            import microhh_tools
            return microhh_tools
    raise SystemExit('microhh_tools.py not found: set MICROHH_ROOT to the MicroHH tree')


def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--dx', type=float, default=8.0, help='horizontal grid spacing [m] (paper: 4)')
    p.add_argument('--dz', type=float, default=4.0, help='vertical grid spacing [m] (paper: 4)')
    p.add_argument('--lx', type=float, default=4096., help='domain length, downslope (x) [m] (paper: 1024)')
    p.add_argument('--ly', type=float, default=1024., help='domain width, along the contours (y) [m]')
    p.add_argument('--lz', type=float, default=1536., help='depth ABOVE the highest terrain [m]')
    p.add_argument('--alpha', type=float, default=0.18, help='slope angle [deg]; 0 gives B11')
    p.add_argument('--vg', type=float, default=10., help='geostrophic wind, +y [m/s]')
    p.add_argument('--N', type=float, default=0.01, help='environmental Brunt-Vaisala frequency [1/s]')
    p.add_argument('--theta-r', type=float, default=300., help='reference potential temperature [K]')
    p.add_argument('--fc', type=float, default=1.e-4, help='Coriolis parameter [1/s]')
    p.add_argument('--bs-day', type=float, default=0.1, help='daytime surface buoyancy [m/s2]')
    p.add_argument('--bs-night', type=float, default=0.05, help='night-time surface buoyancy [m/s2]')
    p.add_argument('--t-sunset', type=float, default=10800., help='time of the forcing change [s]')
    p.add_argument('--endtime', type=float, default=54000., help='end time [s]')
    p.add_argument('--z0m', type=float, default=0.05, help='IB momentum roughness [m]')
    p.add_argument('--z0h', type=float, default=0.005, help='IB heat roughness [m]')
    p.add_argument('--wall-slope', action='store_true',
                   help='[IB] sw_wall_slope: MOST on the sub-grid DEM surface')
    p.add_argument('--kinematic', action='store_true',
                   help='[IB] sw_wall_kinematic: floor w = u.grad(eta), the sub-grid slope '
                        '(implies --wall-slope)')
    p.add_argument('--sgs', choices=['smag2', 'aniso', 'tke2'], default='smag2',
                   help='SGS closure (aniso and tke2 need the develop_openbc branch)')
    p.add_argument('--advec', choices=['2i5', '2i62'], default='2i5',
                   help='advection scheme (2i62 with sw_advec_wall needs the develop_openbc branch)')
    p.add_argument('--solid-levels', type=int, default=5,
                   help='levels of solid below the LOWEST terrain point (the 2i5 stencil needs >= 4)')
    p.add_argument('--sbot-loadtime', type=int, default=60, help='[IB] sbot_loadtime [s]')
    p.add_argument('--lbc', choices=['tfperiodic', 'nudge'], default='tfperiodic')
    p.add_argument('--precursor', default=None, help='stats .nc of a periodic run (--lbc nudge)')
    p.add_argument('--nudge-freq', type=int, default=600, help='LBC file interval for --lbc nudge [s]')
    p.add_argument('--n-sponge', type=int, default=5)
    p.add_argument('--npx', type=int, default=1)
    p.add_argument('--npy', type=int, default=1)
    p.add_argument('--cross-every', type=float, default=600., help='cross-section interval [s]')
    p.add_argument('--outdir', default='.')
    return p.parse_args()


def lbc_geometry(n, ngc, ns, dx, staggered, side):
    """Positions of the LBC buffer cells along the boundary-normal axis.

    The buffer of the low edge (west/south) is field index b = 0 .. nlbc-1,
    the high edge (east/north) is index itot+ngc-ns+b; one extra cell on the
    low edge for the staggered normal velocity (see read_lbc / set_lbc_gcs).
    """
    nlbc = ngc + ns + (1 if (staggered and side == 'low') else 0)
    b = np.arange(nlbc)
    i = b if side == 'low' else n + ngc - ns + b
    x = (i - ngc) * dx if staggered else (i - ngc + 0.5) * dx
    return nlbc, x


def write_nudge_lbcs(a, ini, zs_fun, z, zh, gam, times, prof):
    """Time-dependent lbc_<var>_<edge>.<iotime> from mean profiles of height AGL.

    prof(var, h, t) returns the precursor value at height h above ITS surface.
    thl is mapped as prof('thl') + G * z_s (the precursor's environment starts
    at its own surface).
    """
    igc = jgc = 3
    itot, jtot = ini['grid']['itot'], ini['grid']['jtot']
    dx, dy = a.dx, a.dx
    ns = a.n_sponge
    for t in times:
        it = int(round(t))
        for var in ['u', 'v', 'thl', 'qt']:
            for edge in ['west', 'east', 'south', 'north']:
                along_x = edge in ('west', 'east')
                side = 'low' if edge in ('west', 'south') else 'high'
                if along_x:
                    nlbc, xb = lbc_geometry(itot, igc, ns, dx, var == 'u', side)
                    yb = (np.arange(jtot + 2*jgc) - jgc + 0.5) * dy
                    X, Y = np.meshgrid(xb, yb)              # [j, i]
                else:
                    nlbc, yb = lbc_geometry(jtot, jgc, ns, dy, var == 'v', side)
                    xb = (np.arange(itot + 2*igc) - igc + 0.5) * dx
                    X, Y = np.meshgrid(xb, yb)              # [j, i]
                zs = zs_fun(X, Y)
                h = z[:, None, None] - zs[None, :, :]
                val = prof(var, np.maximum(h, 0.), t)
                if var == 'thl':
                    val = val + gam * zs[None, :, :]
                if var in ('u', 'v'):
                    val = np.where(h > 0, val, 0.)
                val.astype(np.float64).tofile(
                        os.path.join(a.outdir, f'lbc_{var}_{edge}.{it:07d}'))


def main():
    a = parse()
    mht = find_microhh_tools()
    os.makedirs(a.outdir, exist_ok=True)
    out = lambda f: os.path.join(a.outdir, f)

    itot = int(round(a.lx / a.dx))
    jtot = int(round(a.ly / a.dx))
    tana = np.tan(np.deg2rad(a.alpha))

    # Terrain: a plane falling towards +x. The lowest point sits
    # solid_levels*dz above the buried domain floor.
    # A quarter cell above a half level, so that no column's surface falls
    # exactly on a cell face (that made every MOST wall face non-finite in
    # the flat alpha = 0 twin).
    z_base = (a.solid_levels + 0.25) * a.dz
    drop = a.lx * tana
    zs_fun = lambda X, Y: z_base + tana * (a.lx - X) + 0.*Y
    ktot = int(np.ceil((a.lz + z_base + drop) / a.dz))
    zsize = ktot * a.dz

    x = (np.arange(itot) + 0.5) * a.dx
    y = (np.arange(jtot) + 0.5) * a.dx
    X, Y = np.meshgrid(x, y)
    dem = zs_fun(X, Y)
    dem.astype(np.float64).tofile(out('dem.0000000'))

    # Staircase steps: the number of distinct first-air levels along x.
    z_full = (np.arange(ktot) + 0.5) * a.dz
    k_dem = np.searchsorted(z_full, dem[0], side='right')
    n_steps = int(k_dem.max() - k_dem.min())

    gam = a.N**2 * a.theta_r / GRAV                 # K/m
    dth_day = a.bs_day * a.theta_r / GRAV
    dth_night = a.bs_night * a.theta_r / GRAV

    # IB surface temperature, day and night, as two files + symlinks.
    thl_day = a.theta_r + gam*dem + dth_day
    thl_night = a.theta_r + gam*dem + dth_night
    thl_day.astype(np.float64).tofile(out('thl_sbot_day.bin'))
    thl_night.astype(np.float64).tofile(out('thl_sbot_night.bin'))
    np.zeros_like(dem).tofile(out('qt_sbot.0000000'))
    # The t = 0 file must be a real file (also read by the IB at create).
    thl_day.astype(np.float64).tofile(out('thl_sbot.0000000'))

    n_links = 0
    t_last = a.endtime + 2*a.sbot_loadtime
    for t in range(a.sbot_loadtime, int(t_last) + 1, a.sbot_loadtime):
        target = 'thl_sbot_day.bin' if t <= a.t_sunset else 'thl_sbot_night.bin'
        link = out(f'thl_sbot.{t:07d}')
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
        n_links += 1

    # Initial state: resting environment with the geostrophic wind.
    z = (np.arange(ktot) + 0.5) * a.dz
    zh = np.arange(ktot) * a.dz
    with nc.Dataset(out(f'{CASE}_input.nc'), 'w') as f:
        f.createDimension('z', ktot)
        f.createVariable('z', 'f8', ('z',))[:] = z
        g = f.createGroup('init')
        for name, val in [('thl', a.theta_r + gam*z), ('qt', 0*z), ('sgstke', 0*z + 0.01),
                          ('u', 0*z), ('v', 0*z + a.vg),
                          ('u_geo', 0*z), ('v_geo', 0*z + a.vg)]:
            g.createVariable(name, 'f8', ('z',))[:] = val

    ini = mht.Read_namelist(os.path.join(HERE, f'{CASE}.ini.base'))
    ini['master']['npx'] = a.npx
    ini['master']['npy'] = a.npy
    ini['grid']['itot'] = itot
    ini['grid']['jtot'] = jtot
    ini['grid']['ktot'] = ktot
    ini['grid']['xsize'] = itot * a.dx
    ini['grid']['ysize'] = jtot * a.dx
    ini['grid']['zsize'] = zsize
    ini['force']['fc'] = a.fc
    ini['boundary']['stop[thl]'] = gam
    ini['boundary']['sbot[thl]'] = a.theta_r + gam*z_base + dth_day
    ini['IB']['sbot[thl]'] = a.theta_r + gam*z_base + dth_day
    ini['IB']['z0m'] = a.z0m
    ini['IB']['z0h'] = a.z0h
    ini['boundary']['z0m'] = a.z0m
    ini['boundary']['z0h'] = a.z0h
    ini['IB']['sbot_loadtime'] = a.sbot_loadtime
    ini['thermo']['thvref0'] = a.theta_r
    ini['boundary_lateral']['n_sponge'] = a.n_sponge
    ini['boundary_lateral']['lbc_tf_gradient[thl]'] = gam
    ini['buffer']['zstart'] = z_base + drop + 0.8*a.lz
    ini['time']['endtime'] = a.endtime

    if a.wall_slope or a.kinematic:
        ini['IB']['sw_wall_slope'] = True
    if a.kinematic:
        ini['IB']['sw_wall_kinematic'] = True
    if a.sgs == 'aniso':
        ini['diff']['swanisotropic'] = True
        ini['diff']['mlen0_h'] = a.dx
        ini['diff']['mlen0_v'] = a.dz
    elif a.sgs == 'tke2':
        ini['diff']['swdiff'] = 'tke2'
        ini['IB']['sbot[sgstke]'] = 0.
        ini['boundary']['sbot[sgstke]'] = 0.
        ini['boundary']['stop[sgstke]'] = 0.
        ini['boundary_lateral']['slist'] = 'thl,qt,sgstke'
        ini['boundary_lateral']['lbc_tf_gradient[sgstke]'] = 0.
    if a.advec == '2i62':
        # RK3 with 6th-order centred advection is stable to CFL ~1.09
        # (max modified wavenumber 1.59 vs RK3's 1.73 on the imaginary axis);
        # the 1.2 that suits 2i5 blew up a tke2 + 2i62 run in 650 s.
        ini['advec']['swadvec'] = '2i62'
        ini['advec']['cflmax'] = 1.0
        ini['advec']['fluxlimit_list'] = 'thl,qt,sgstke' if a.sgs == 'tke2' else 'thl,qt'
    ini['cross']['sampletime'] = a.cross_every
    ini['stats']['sampletime'] = a.cross_every
    ini['cross']['xz'] = ','.join(f'{v:g}' for v in (np.array([1, 3, 5, 7]) / 8 * jtot * a.dx))
    # Random temperature perturbations in the lowest 50 m above the terrain.
    ini['fields']['rndz'] = z_base + drop + 50.

    if a.lbc == 'nudge':
        if a.precursor is None:
            raise SystemExit('--lbc nudge needs --precursor <stats.nc of a periodic run>')
        with nc.Dataset(a.precursor) as p:
            tp = p.variables['time'][:]
            zp = p.variables['z'][:]
            grp = p.groups['default'] if 'default' in p.groups else p
            data = {v: grp.variables[v][:] for v in ['u', 'v', 'thl']}
            # Height above the precursor's surface: its floor (or IB base).
            zsp = float(p.getncattr('z_surface')) if 'z_surface' in p.ncattrs() else 0.
        def prof(var, h, t):
            if var == 'qt':
                return 0.*h
            it = np.clip(np.searchsorted(tp, t), 1, len(tp)-1)
            w = (t - tp[it-1]) / (tp[it] - tp[it-1])
            pr = (1-w)*data[var][it-1] + w*data[var][it]
            base = a.theta_r if var == 'thl' else 0.
            return np.interp(h, zp - zsp, pr - base) + base
        times = np.arange(0, a.endtime + a.nudge_freq + 1, a.nudge_freq)
        write_nudge_lbcs(a, ini, zs_fun, z, zh, gam, times, prof)
        ini['boundary_lateral']['sw_lbc_tf_periodic'] = False
        ini['boundary_lateral']['sw_timedep'] = True
        ini['boundary_lateral']['loadfreq'] = a.nudge_freq

    ini.save(out(f'{CASE}.ini'), allow_overwrite=True)
    # cross_to_nc.sh + cross_to_nc.py + microhh_tools.py (time means over 10 min)
    install_cross_to_nc(a.outdir, CASE, window=600.)

    pars = dict(vars(a))
    pars.update(dict(itot=itot, jtot=jtot, ktot=ktot, zsize=zsize, z_base=z_base,
                     drop=drop, gamma=gam, dth_day=dth_day, dth_night=dth_night,
                     n_steps=n_steps))
    with open(out('case_parameters.json'), 'w') as f:
        json.dump(pars, f, indent=1)

    print(f'{CASE}: {itot} x {jtot} x {ktot} = {itot*jtot*ktot/1e6:.1f} M cells, '
          f'dx = {a.dx:g} m, dz = {a.dz:g} m, alpha = {a.alpha:g} deg')
    print(f'  terrain drop over the domain {drop:.2f} m = {drop/a.dz:.2f} dz '
          f'-> the staircase has {n_steps} step(s)')
    if a.alpha != 0 and n_steps < 3 and not a.kinematic:
        print('  WARNING: fewer than 3 staircase steps. The IB terrain cannot carry the slope here:\n'
              '  the floors are flat, the near-surface flow cannot descend with the terrain, and\n'
              '  the N^2 u sin(alpha) and b sin(alpha) terms are largely missing. Use --kinematic\n'
              '  (w on the floor faces along the sub-grid slope), a longer domain (--lx) or a finer dz;\n'
              '  see README.md, "The slope is sub-grid".')
    elif a.alpha != 0 and a.kinematic:
        print(f'  --kinematic: the floor faces carry the sub-grid slope (w = u.grad(eta)); {n_steps} resolved step(s)')
    print(f'  environment dthl/dz = {gam*1e3:.4f} K/km, wall excess day/night '
          f'{dth_day:.3f} / {dth_night:.3f} K, step at t = {a.t_sunset:g} s')
    print(f'  lateral boundaries: {a.lbc}; {n_links} thl_sbot links every {a.sbot_loadtime} s')


if __name__ == '__main__':
    main()
