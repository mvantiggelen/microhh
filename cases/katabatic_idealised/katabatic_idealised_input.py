#!/usr/bin/env python3
"""
An idealised katabatic flow down a Greenland ice slope, as a test of the
immersed boundary (IB) in a realistic katabatic regime.

The slope, the air above it and the katabatic layer come from a regional
MicroHH simulation of the west Greenland ice-sheet margin on 24 Jul 2026,
driven by the regional climate model RACMO (greenland_20260724_inputs.json,
so no RACMO data is needed here). The case has two runs:

  rotated/  the reference: MicroHH's own rotated, periodic slope frame
            (swthermo=buoy), an infinite slope with MicroHH's surface model;
  ib/       the test: a finite slope as an IB on a gravity-aligned grid,
            with open boundaries in x.

TWO STAGES, RUN ONE AFTER THE OTHER (--inflow rotated, the default): the
inflow and the initial katabatic layer of the IB run are the equilibrium
profiles of the finished rotated run, so that both runs start from the same
balance and any drift down the IB slope is caused by the IB.

    python3 katabatic_idealised_input.py --outdir runs/test [options]
    #  stage 1 -> runs/test/rotated (MicroHH's rotated slope frame, the reference)
    #             runs/test/generator_args.json (the options, for stage 2)
    #  ... run rotated/ to its end ...
    python3 katabatic_idealised_input.py --outdir runs/test
    #  stage 2 -> runs/test/ib (the IB run, the test) and inputs.png, with
    #             the mean u_s(n) and theta'(n) = b theta_r/g of the rotated
    #             run over its last tau_osc (--rot-window) as the inflow
--stage auto (default) picks the stage from what is on disk; job.sh does
stage 2 itself when the rotated job ends (KAT_STAGE2=1, the default) and with
KAT_CHAIN=1 also submits ib/ (SLURM only). Options given at stage 2 override
those of stage 1; any that would change the rotated run (slope, theta_r,
surface deficit, dx, dz, z0m, z0h) are refused. --inflow racmo uses an
analytic fit to the RACMO profile instead and writes both directories at once.

Domain and resolution: --dx, --dz (both runs), --lx, --ly, --zsize (IB; the
defaults follow from the slope), --rot-lx, --rot-ly, --rot-lz (rotated). The
slope: --alpha, --slope-length (--terrain constant). Every size is checked:
the slope must end at least the sponge + 1 km before the east edge, the
terrain top must leave --min-free-air (500 m) below the buffer, the cell
counts must divide over the tasks; the slope must be 0-30 deg (a warning
above 15 deg). Radiation: --radiation lw|lwsw with --humidity racmo.

ib/
---
* Terrain (--terrain constant, the default): a uniform slope at the mean
  surface gradient of the ice in the regional simulation (1.93 deg on
  24 Jul 2026) over 16 km, breaks smoothed over 500 m, ending at the height
  of the ice margin (233 m amsl): the same slope as the rotated run.
  --terrain transect uses the real terrain along x through the weather
  station instead (from the 400-m and 100-m DEMs of the regional
  simulation), from ~1000 m down to the ice margin, mirrored so that the flow
  runs towards +x. Either way a flat plateau before the slope (the inflow,
  --plateau) and a flat valley after it (the outflow, --valley), uniform in y.
* Surface (Dirichlet thl_sbot), z0m = 2 cm. By default the IB run is kept as
  close as possible to the rotated reference:
    --surface deficit   theta_s = theta_a(z_s) + dtheta_s everywhere, dtheta_s
                        the melting deficit (T_s = 273.15 K) at the slope's
                        mid-height (the rotated run's b_s)
    --humidity off      qt = 0 (the rotated run's buoyancy is dry)
    --z0h-model fixed   z0h = --z0h (5e-4 m) in both runs
    --basestate boussinesq  constant reference thv0 = theta_r, as thermo buoy
  and the same initial noise in the lowest 100 m above the surface. The
  realistic alternatives for the IB run alone: --surface melt (T_s = 273.15 K
  everywhere), --humidity racmo (saturated surface), --z0h-model yang2008,
  --basestate anelastic.
* State: no large-scale pressure gradient, no Coriolis. The ambient theta and
  qt are RACMO's at the ice-sheet boundary at --racmo-time, above the
  katabatic layer (a linear fit of theta 200-1000 m above the surface,
  extrapolated downwards); the katabatic layer added to it is fitted to the
  same RACMO profile: u(n) = U_j (n/n_j) exp(1 - n/n_j) and
  theta'(n) = dtheta_0 exp(-n/h_theta), n the height above the local surface.
* Lateral boundaries: open in x (constant inflow at the west edge = ambient +
  katabatic layer above the plateau; the east edge, in the valley, gets the
  same profiles above the valley floor (--outflow jet: no net mass flux) or
  the ambient air at rest (--outflow rest)), periodic in y.
* The same initial katabatic layer (the inflow's) is laid over the whole
  surface by katabatic_idealised_init.py after "microhh init".
* IB settings: the recommended ones (staircase momentum method, MOST wall
  model, wall sampling, Mason wall damping, anisotropic smag2).
* Radiation: ready but off (--radiation off). The input .nc always holds the
  gas profiles and the RRTMGP background column; --radiation lw or lwsw
  switches it on (sw_ib_columns), and job.sh then links the coefficient files.
* Length: --n-tau periods of the katabatic oscillation 2 pi / (N sin alpha),
  alpha the mean slope of the ice and N of the ambient air.

rotated/
--------
MicroHH's rotated, periodic slope frame (swthermo=buoy, cf. cases/prandtlslope):
an infinite slope at the mean angle alpha (downslope = -x there) with the same N,
roughness, closure, grid spacing and the same initial katabatic layer; the
surface is a Dirichlet buoyancy b_s = g/theta_r (theta_melt - theta_a) at the
slope's mid-height. What still differs, by construction: the terrain is finite
(a plateau with the inflow before the slope and a valley with the outflow
after it, against an infinite periodic slope at the same angle), the surface
method (MicroHH's own surface vs the IB - the thing under test), and no
radiation (thermo buoy has none; radiation is off in the IB run by default too).

Compare with katabatic_idealised_compare.py --rundir runs/test.

Needs MicroHH from the develop_openbc branch.

References: Prandtl (1942); Axelsen and van Dop (2009),
https://doi.org/10.2478/s11600-009-0041-6; Yang et al. (2008),
https://doi.org/10.1175/2007JAMC1547.1 (z0h); Grisogono and Oerlemans
(2001), https://doi.org/10.1175/1520-0469(2001)058<3349:KFASFG>2.0.CO;2.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import netCDF4 as nc

HERE = os.path.dirname(os.path.abspath(__file__))
CASE_IB = 'katabatic_idealised'
CASE_ROT = 'katabatic_rotated'
GRAV = 9.81
RD, RV, CP = 287.04, 461.5, 1005.
KAPPA = RD / CP
EPS = RD / RV
P0 = 1.e5
T_MELT = 273.15
NGC = 3                     # ghost cells of swadvec=2i5
RADIATION_GASES = {'co2': 425e-6, 'ch4': 1930e-9, 'n2o': 338e-9, 'n2': 0.7808, 'o2': 0.2095}

MOMENTUM = {
    'staircase': '''# momentum, staircase (the default): MOST wall stress as a source term,
# eddy viscosity rescaled to it, w on the floor faces along the sub-grid
# slope, momentum exchanged with the terrain only through MOST and pressure
sw_momentum_flux=true
sw_momentum_flux_vertical=true
sw_strain_most=true
sw_strain_most_vertical=false
strain_most_min=0.05
sw_impermeable=false
sw_wall_kinematic=true
sw_momentum_air_only=true
''',
    'vr': '''# momentum by velocity reconstruction (needs a MicroHH build with it)
sw_momentum_reconstruct=true
reconstruct_profile=most
reconstruct_dz=1.5
reconstruct_diag_every=300
reconstruct_stress=true
''',
}


def parse():
    p = argparse.ArgumentParser(description='idealised katabatic flow down a Greenland ice slope')
    p.add_argument('--outdir', required=True)
    p.add_argument('--inputs', default=os.path.join(HERE, 'greenland_20260724_inputs.json'),
                   help='terrain and RACMO profiles (extract_inputs.py)')
    p.add_argument('--racmo-time', type=int, default=43200,
                   help='RACMO time [s after the case start, 03 UTC] of the inflow profiles (43200 = 15 UTC)')
    p.add_argument('--dx', type=float, default=100.)
    p.add_argument('--dz', type=float, default=10.)
    p.add_argument('--ly', type=float, default=3200., help='y width of the IB domain [m] (periodic)')
    p.add_argument('--jtot', type=int, default=None, help='cells in y; overrides --ly')
    p.add_argument('--lx', type=float, default=None,
                   help='x length of the IB domain [m]; the valley takes what is left after the slope '
                        '(checked: it must hold the sponge and 1 km); overrides --valley')
    p.add_argument('--zsize', type=float, default=None,
                   help='height of the IB domain [m] above its floor; checked against the terrain top, '
                        '--min-free-air and the buffer; overrides --free-air')
    p.add_argument('--min-free-air', type=float, default=500.,
                   help='at least this much air between the highest terrain and the buffer [m]')
    p.add_argument('--npx', type=int, default=16)
    p.add_argument('--npy', type=int, default=8)
    p.add_argument('--plateau', type=float, default=2500., help='flat inflow section incl. the sponge [m]')
    p.add_argument('--valley', type=float, default=4000., help='flat outflow section incl. the sponge, at least [m]')
    p.add_argument('--terrain', choices=('constant', 'transect'), default='constant',
                   help='constant (default): a uniform slope at --alpha over --slope-length, the same slope as '
                        'the rotated run; transect: the real terrain through the weather station')
    p.add_argument('--alpha', type=float, default=None,
                   help='--terrain constant: slope [deg]; default the mean surface gradient of the ice in the '
                        'regional simulation (1.93 deg for 24 Jul 2026)')
    p.add_argument('--slope-length', type=float, default=None,
                   help='--terrain constant: horizontal length of the slope [m]; default 16 km (the regional '
                        'simulation)')
    p.add_argument('--break-width', type=float, default=500.,
                   help='--terrain constant: the slope breaks are smoothed over this width [m]')
    p.add_argument('--x-top', type=float, default=None,
                   help='source x of the top of the slope [m, transect coordinates]; default: the last '
                        'cell of the 400-m DEM before its east edge')
    p.add_argument('--smooth', type=float, default=300., help='Gaussian smoothing of the transect [m] (0 = off)')
    p.add_argument('--solid-levels', type=int, default=5)
    p.add_argument('--free-air', type=float, default=1000., help='air above the plateau to the domain top [m]')
    p.add_argument('--buffer-depth', type=float, default=400.)
    p.add_argument('--buf-sigma', type=float, default=0.01)
    p.add_argument('--n-sponge', type=int, default=10)
    p.add_argument('--tau-sponge', type=float, default=60.)
    p.add_argument('--outflow', choices=('jet', 'rest'), default='jet')
    p.add_argument('--jet-speed', type=float, default=None, help='U_j [m/s] (default: RACMO fit)')
    p.add_argument('--jet-height', type=float, default=None, help='n_j [m] (default: RACMO fit)')
    p.add_argument('--dtheta0', type=float, default=None, help='theta deficit at the surface [K] (default: fit)')
    p.add_argument('--theta-depth', type=float, default=None, help='e-folding depth of the deficit [m]')
    p.add_argument('--z0m', type=float, default=0.02)
    # The IB run and the rotated reference are kept as alike as possible:
    # everything physical is the same in both by default, so that what
    # differs is the terrain (finite slope vs infinite mean slope) and the
    # IB method. The options below switch the IB run back to the more realistic
    # set-up the rotated frame cannot have.
    p.add_argument('--surface', choices=('deficit', 'melt'), default='deficit',
                   help='deficit (default): theta_s = theta_a(z_s) + dtheta_s, the melting deficit at the '
                        "slope's mid-height everywhere, as the rotated run's b_s; melt: T_s = 273.15 K")
    p.add_argument('--humidity', choices=('off', 'racmo'), default='off',
                   help='off (default): qt = 0, as the dry buoyancy of the rotated run; racmo: RACMO qt and '
                        'a saturated surface (needed with --radiation)')
    p.add_argument('--z0h-model', choices=('fixed', 'yang2008'), default='fixed',
                   help='fixed (default): z0h = --z0h in both runs; yang2008: the Yang et al. (2008) ice model (IB only)')
    p.add_argument('--z0h', type=float, default=5e-4,
                   help='z0h [m] of both runs with --z0h-model fixed (Yang 2008 at u* ~0.2 m/s gives ~5e-4)')
    p.add_argument('--basestate', choices=('boussinesq', 'anelastic'), default='boussinesq',
                   help='boussinesq (default): constant reference thv0 = theta_r and density 1, as thermo buoy')
    p.add_argument('--rnd-depth', type=float, default=100.,
                   help='initial noise in the lowest this many metres above the surface, both runs')
    p.add_argument('--momentum', choices=('staircase', 'vr'), default='staircase')
    p.add_argument('--n-tau', type=float, default=3.)
    p.add_argument('--endtime', type=float, default=None)
    p.add_argument('--out-every', type=float, default=600., help='stats/cross interval [s]')
    p.add_argument('--radiation', choices=('off', 'lw', 'lwsw'), default='off')
    p.add_argument('--albedo', type=float, default=0.55, help='ice albedo with --radiation lwsw')
    # rotated reference
    p.add_argument('--alpha-ref', type=float, default=None,
                   help='slope angle of the rotated run [deg]; default: the mean slope of the ice')
    p.add_argument('--rot-lx', type=float, default=6400., help='x length of the rotated domain [m]')
    p.add_argument('--rot-ly', type=float, default=3200., help='y width of the rotated domain [m]')
    p.add_argument('--rot-itot', type=int, default=None, help='cells in x of the rotated run; overrides --rot-lx')
    p.add_argument('--rot-jtot', type=int, default=None, help='cells in y of the rotated run; overrides --rot-ly')
    p.add_argument('--rot-lz', type=float, default=1000., help='depth of the rotated domain [m]')
    p.add_argument('--rot-npx', type=int, default=4)
    p.add_argument('--rot-npy', type=int, default=8)
    p.add_argument('--inflow', choices=('rotated', 'racmo'), default='rotated',
                   help='rotated (default): the IB inflow and initial layer are the '
                        'equilibrium profiles of the finished rotated run, so the two runs are generated and '
                        'run one after the other (see --stage); racmo: the analytic fit to RACMO, both '
                        'directories at once')
    p.add_argument('--stage', choices=('auto', 'rotated', 'ib', 'all'), default='auto',
                   help='auto (default): with --inflow rotated, write rotated/ when it has no output yet, '
                        'else ib/ from it; with --inflow racmo, both. rotated / ib: that directory only; '
                        'all: both (--inflow racmo only)')
    p.add_argument('--rot-window', type=float, default=None,
                   help='averaging window at the end of the rotated run for the inflow [s]; default the last '
                        'tau_osc of that run')
    p.add_argument('--only', choices=('ib', 'rotated'), default=None, help='old name of --stage ib|rotated')
    a = p.parse_args()
    if a.only and a.stage == 'auto':
        a.stage = a.only
    if a.radiation != 'off' and a.humidity == 'off':
        p.error('--radiation needs --humidity racmo (longwave cooling in dry air is meaningless)')
    if a.inflow == 'racmo' and a.stage == 'auto':
        a.stage = 'all'
    if a.inflow == 'rotated' and a.stage == 'all':
        p.error('--stage all needs --inflow racmo: with --inflow rotated the IB run is built from the '
                'finished rotated run')
    # the options given on the command line (the rest may come from the first stage)
    a.explicit = sorted({t[2:].split('=')[0].replace('-', '_') for t in sys.argv[1:] if t.startswith('--')})
    return a


# ---- thermodynamics -----------------------------------------------------------

def qsat_liq(T, p):
    es = 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))
    return EPS * es / (p - (1. - EPS) * es)


def hydrostatic_p(z_amsl, thl_of, qt_of, z0_amsl, p0):
    """p(z) [Pa] on z_amsl (ascending) by integrating dp/dz = -g p/(Rd T_v) from p0 at z0_amsl."""
    zf = np.linspace(min(z0_amsl, z_amsl.min()), z_amsl.max(), 6000)
    lnp = np.zeros_like(zf)
    i0 = np.argmin(abs(zf - z0_amsl))
    lnp[i0] = np.log(p0)
    for k in range(i0 + 1, zf.size):
        zm = 0.5 * (zf[k] + zf[k-1])
        T = thl_of(zm) * np.exp(KAPPA * (lnp[k-1] - np.log(P0)))
        lnp[k] = lnp[k-1] - GRAV / (RD * T * (1. + 0.608 * qt_of(zm))) * (zf[k] - zf[k-1])
    for k in range(i0 - 1, -1, -1):
        zm = 0.5 * (zf[k] + zf[k+1])
        T = thl_of(zm) * np.exp(KAPPA * (lnp[k+1] - np.log(P0)))
        lnp[k] = lnp[k+1] + GRAV / (RD * T * (1. + 0.608 * qt_of(zm))) * (zf[k+1] - zf[k])
    return np.exp(np.interp(z_amsl, zf, lnp))


def solar_zenith(when, lat, lon):
    """Solar zenith angle [rad] (NOAA approximation)."""
    doy = when.timetuple().tm_yday
    hr = when.hour + when.minute / 60.
    g = 2. * np.pi / 365. * (doy - 1 + (hr - 12.) / 24.)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    tst = hr * 60. + eqt + 4. * lon
    ha = np.deg2rad(tst / 4. - 180.)
    la = np.deg2rad(lat)
    return float(np.arccos(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)))


# ---- inputs -------------------------------------------------------------------

def transect(J, a):
    """z_amsl(x_src) along the row through the station: the 100-m DEM where it exists, the 400-m DEM elsewhere."""
    d0 = J['outer']
    x0, z0 = np.array(d0['x']), np.array(d0['z_amsl'])
    ice0 = np.array(d0['ice'])
    xs = np.arange(x0[0], x0[-1] + 1e-6, 25.)
    z = np.interp(xs, x0, z0)
    if 'inner' in J:
        x1, z1 = np.array(J['inner']['x']), np.array(J['inner']['z_amsl'])
        w = np.clip(np.minimum(xs - x1[0], x1[-1] - xs) / 400., 0., 1.)
        z = (1. - w) * z + w * np.interp(xs, x1, z1)
    if a.smooth > 0.:
        s = a.smooth / 25.
        k = np.arange(-int(4 * s), int(4 * s) + 1)
        ker = np.exp(-0.5 * (k / s)**2)
        ker /= ker.sum()
        zp = np.pad(z, k.size // 2, mode='edge')
        z = np.convolve(zp, ker, mode='valid')
    x_margin = float(x0[ice0 > 0].min())          # westernmost ice on the row
    if 'inner' in J:
        x1, i1 = np.array(J['inner']['x']), np.array(J['inner']['ice'])
        x_margin = float(x1[i1 > 0].min())
    return xs, z, x_margin, float(d0['x_site'])


def ambient_and_layer(J, a):
    """Ambient theta_a(z), qt_a(z) and the katabatic layer fitted to RACMO's east profile."""
    r = J['racmo_east']
    t = str(a.racmo_time)
    if t not in r['times']:
        raise SystemExit(f'--racmo-time {a.racmo_time}: available {list(r["times"])}')
    z = np.array(r['z_amsl'])
    zs = r['z_surface_amsl']
    pr = {k: np.array(v) for k, v in r['times'][t].items()}
    h = z - zs
    fit = (h >= 200.) & (h <= 1000.)
    gam, th_c = np.polyfit(z[fit], pr['thl'][fit], 1)
    if gam <= 0.:
        raise SystemExit(f'RACMO theta 200-1000 m above the surface is not stable (dtheta/dz = {gam:.4f} K/m)')
    thl_a = lambda zz: th_c + gam * np.asarray(zz, dtype=float)
    air = h > 0.
    q_above = pr['qt'][air & (h >= 200.)]
    z_above = z[air & (h >= 200.)]
    q200 = float(q_above[0])
    qt_a = lambda zz: np.where(np.asarray(zz) <= z_above[0], q200, np.interp(zz, z_above, q_above))
    # katabatic layer: the downslope wind (RACMO's -u: the slope falls to the west there)
    low = air & (h < 200.)
    us = -pr['u'][low]
    k = int(np.argmax(us))
    U_j = a.jet_speed if a.jet_speed is not None else float(us[k])
    n_j = a.jet_height if a.jet_height is not None else float(h[low][k])
    dth = pr['thl'][low] - thl_a(z[low])
    ok = dth < -0.2
    if ok.sum() >= 2:
        slope, icpt = np.polyfit(h[low][ok], np.log(-dth[ok]), 1)
        h_th_fit, dth0_fit = -1. / slope, -np.exp(icpt)
    else:
        h_th_fit, dth0_fit = 50., -5.
    dth0 = a.dtheta0 if a.dtheta0 is not None else float(dth0_fit)
    h_th = a.theta_depth if a.theta_depth is not None else float(h_th_fit)
    N = np.sqrt(GRAV / thl_a(zs) * gam)
    info = dict(racmo_time=a.racmo_time, z_surface_racmo=zs, gamma=float(gam), N=float(N),
                U_j=U_j, n_j=n_j, dtheta0=dth0, h_theta=h_th,
                racmo_profile=dict(h=h[air & (h < 1500.)].tolist(),
                                   thl=pr['thl'][air & (h < 1500.)].tolist(),
                                   qt=pr['qt'][air & (h < 1500.)].tolist(),
                                   u_downslope=(-pr['u'][air & (h < 1500.)]).tolist(),
                                   v=pr['v'][air & (h < 1500.)].tolist()))
    return thl_a, qt_a, info


def jet(n, U_j, n_j):
    n = np.asarray(n, dtype=float)
    return np.where(n > 0., U_j * (n / n_j) * np.exp(1. - n / n_j), 0.)


def deficit(n, dth0, h_th):
    n = np.asarray(n, dtype=float)
    return np.where(n > 0., dth0 * np.exp(-n / h_th), 0.)


def layer_u(n, info):
    """Downslope wind of the katabatic layer at n m above the surface: the rotated
    run's equilibrium profile (--inflow rotated) or the RACMO fit."""
    n = np.asarray(n, dtype=float)
    P = info.get('profile')
    if P is None:
        return jet(n, info['U_j'], info['n_j'])
    return profile_u(n, P)


def profile_u(n, P):
    """A tabulated profile, with the log law of the rotated run's z0m below its
    first level (MOST, neutral: the same shape MicroHH's surface assumes)."""
    n = np.asarray(n, dtype=float)
    n1, u1, z0 = P['n'][0], P['u'][0], P.get('z0m', 0.02)
    low = u1 * np.log(np.maximum(n, z0) / z0) / np.log(n1 / z0)
    return np.where(n <= 0., 0., np.where(n < n1, low, np.interp(n, P['n'], P['u'], right=0.)))


def layer_dtheta(n, info):
    """Theta deficit of the katabatic layer (the same two sources as layer_u)."""
    n = np.asarray(n, dtype=float)
    P = info.get('profile')
    if P is None:
        return deficit(n, info['dtheta0'], info['h_theta'])
    return np.where(n > 0., np.interp(n, P['n'], P['dtheta'], left=P['dtheta'][0], right=0.), 0.)


def rotated_profile(rot_dir, window=None):
    """The equilibrium katabatic layer of a finished rotated run: the mean of the
    downslope wind and of theta' = b theta_r / g over the last tau_osc (or
    --rot-window), as functions of the height n above the surface. The rotated
    buffer is tapered out (cosine over the 100 m below its start), so the
    profile is zero where the rotated run was damped."""
    import glob
    P = json.load(open(os.path.join(rot_dir, 'case_parameters.json')))
    files = sorted(glob.glob(os.path.join(rot_dir, f'{CASE_ROT}.default.*.nc')))
    if not files:
        raise SystemExit(f'{rot_dir}: no {CASE_ROT}.default.*.nc - run the rotated case first')
    t, u, v, b = [], [], [], []
    for fn in files:
        with nc.Dataset(fn) as f:
            z = np.array(f['z'][:], dtype=float)
            t.append(np.array(f['time'][:], dtype=float))
            u.append(np.array(f['default/u'][:], dtype=float))
            v.append(np.array(f['default/v'][:], dtype=float))
            b.append(np.array(f['thermo/b'][:], dtype=float))
    t, u, v, b = (np.concatenate(x) for x in (t, u, v, b))
    t, k = np.unique(t, return_index=True)               # restarts repeat times
    u, v, b = u[k], v[k], b[k]
    if t[-1] < P['endtime'] - 1.:
        if window is None:
            raise SystemExit(f'{rot_dir}: the rotated run stopped at t = {t[-1]:.0f} s of {P["endtime"]:.0f} s; '
                             'finish it, or give --rot-window to average what is there')
        print(f'WARNING: the rotated run stopped at t = {t[-1]:.0f} s of {P["endtime"]:.0f} s; '
              f'averaging its last {window:.0f} s (--rot-window)')
    win = min(window if window is not None else P['tau_osc'], t[-1] - t[0])
    w = t >= t[-1] - win
    if w.sum() < 3:
        raise SystemExit(f'{rot_dir}: fewer than 3 statistics times in the last {win:.0f} s')
    sgn = P.get('downslope_sign', -1)
    us = sgn * u[w].mean(0)
    dth = b[w].mean(0) * P['theta_r'] / GRAV
    # the buffer of the rotated run: taper to zero below its start
    ini = open(os.path.join(rot_dir, f'{CASE_ROT}.ini')).read()
    zstart = float(ini.split('zstart=')[1].split()[0])
    taper = np.clip((zstart - z) / 100., 0., 1.)
    taper = 0.5 - 0.5 * np.cos(np.pi * taper)
    us, dth = us * taper, dth * taper
    # how steady the jet is over the window (the oscillation amplitude)
    jet_t = (sgn * u[w]).max(axis=1)
    k = int(np.argmax(us))
    e = np.where(dth > dth[0] / np.e)[0]
    return dict(source=os.path.abspath(rot_dir), window=[float(t[-1] - win), float(t[-1])],
                n=z.tolist(), u=us.tolist(), dtheta=dth.tolist(), v_mean=float(np.abs(v[w].mean(0)).max()),
                U_j=float(us[k]), n_j=float(z[k]), dtheta0=float(dth[0]),
                h_theta=float(z[e[0]]) if len(e) else float(z[-1]),
                jet_range=[float(jet_t.min()), float(jet_t.max())], tau_osc=float(P['tau_osc']),
                alpha_deg=float(P['alpha_deg']), theta_r=float(P['theta_r']), dtheta_s=float(P['dtheta_s']),
                dx=float(P['dx']), dz=float(P['dz']), z0m=float(P['z0m']), z0h=float(P['z0h']))


def slope_weight(x, x0, x1, w):
    """1 on [x0, x1], 0 off it, cosine tapers of width w centred on the breaks, so
    that the horizontal integral (the drop / tan alpha) is x1 - x0."""
    def ramp(d):
        if w <= 0.:
            return (d >= 0.).astype(float)
        t = np.clip(d / w + 0.5, 0., 1.)
        return 0.5 - 0.5 * np.cos(np.pi * t)
    return ramp(x - x0) * ramp(x1 - x)


def lcm(a, b):
    return a * b // np.gcd(a, b)


# ---- the IB run -------------------------------------------------------------

def write_ib(a, J, thl_a, qt_a, info, d):
    os.makedirs(d, exist_ok=True)
    out = lambda f: os.path.join(d, f)
    dx, dz = a.dx, a.dz
    xs, zt, x_margin, x_site = transect(J, a)
    x_top = a.x_top if a.x_top is not None else float(np.array(J['outer']['x'])[-2])
    # the flow direction is -x in the source: model x = plateau + (x_top - x_src)
    sel = (xs >= x_margin) & (xs <= x_top)
    xm_s = a.plateau + (x_top - xs[sel])[::-1]
    zm_s = zt[sel][::-1]
    z_valley = float(zm_s[-1])
    z_plateau = float(zm_s[0])
    z_site_amsl = float(np.interp(x_site, xs, zt))
    if a.terrain == 'constant':
        # a uniform slope at alpha from the plateau to the valley (the valley at the
        # height of the ice margin on the row through the station), breaks smoothed over break_width
        S = J.get('inner_ice_slope', {})
        alpha_c = a.alpha if a.alpha is not None else S.get('mean_gradient_deg')
        L_c = a.slope_length if a.slope_length is not None else S.get('extent_m', 16000.)
        if alpha_c is None:
            raise SystemExit('--terrain constant: no inner_ice_slope in the inputs file; give --alpha')
        if not 0. < alpha_c <= 30.:
            raise SystemExit(f'--alpha {alpha_c:g}: the slope must be between 0 and 30 deg')
        if alpha_c > 15.:
            print(f'WARNING: alpha {alpha_c:g} deg: the IB wall model (MOST along the vertical) is untested '
                  'on slopes steeper than ~15 deg')
        if L_c <= 2. * a.break_width:
            raise SystemExit(f'--slope-length {L_c:g} m must be longer than twice --break-width ({a.break_width:g} m)')
        x0c, x1c = a.plateau + 0.5 * a.break_width, a.plateau + 0.5 * a.break_width + L_c
        xf = np.arange(a.plateau, x1c + 0.5 * a.break_width + 1e-6, 5.)
        wf = slope_weight(xf, x0c, x1c, a.break_width)
        integ = np.concatenate(([0.], np.cumsum(0.5 * (wf[1:] + wf[:-1]) * np.diff(xf))))
        xm_s = xf
        zm_s = z_valley + np.tan(np.deg2rad(alpha_c)) * (integ[-1] - integ)
        z_plateau = float(zm_s[0])
    elif a.alpha is not None or a.slope_length is not None:
        print('WARNING: --alpha and --slope-length only act with --terrain constant; ignored')
    # the domain: x from the slope (+ the valley, or --lx), y from --jtot or --ly,
    # z from the terrain top (+ --free-air, or --zsize); every size is checked so
    # that the slope, the sponges and the buffer cannot overlap
    step = lcm(a.npx, a.npy)
    valley_min = a.n_sponge * dx + 1000.                # the outflow sponge and 1 km of flat valley
    if a.lx is not None:
        itot = int(round(a.lx / dx))
        if abs(itot * dx - a.lx) > 1e-6 * a.lx:
            raise SystemExit(f'--lx {a.lx:g} is not a multiple of dx = {dx:g} m')
        if a.lx < xm_s[-1] + valley_min:
            raise SystemExit(f'--lx {a.lx:g} m: the slope ends at x = {xm_s[-1]:.0f} m and the valley needs '
                             f'{valley_min:.0f} m (the {a.n_sponge}-cell sponge and 1 km), so lx >= '
                             f'{np.ceil((xm_s[-1] + valley_min) / dx / step) * step * dx:.0f} m')
    else:
        itot = int(np.ceil((xm_s[-1] + max(a.valley, valley_min)) / dx / step) * step)
    if a.jtot is not None:
        jtot = a.jtot
    else:
        jtot = int(round(a.ly / dx))
        if abs(jtot * dx - a.ly) > 1e-6 * a.ly:
            raise SystemExit(f'--ly {a.ly:g} is not a multiple of dx = {dx:g} m')
    lx, ly = itot * dx, jtot * dx
    x = (np.arange(itot) + 0.5) * dx
    z_amsl_s = np.interp(x, xm_s, zm_s, left=z_plateau, right=z_valley)
    # model heights: valley floor at z_base
    z_base = (a.solid_levels + 0.25) * dz
    datum = z_valley - z_base                           # amsl of the domain floor (z = 0)
    zs = z_amsl_s - datum
    z_top_terrain = float(zs.max())
    z_need = z_top_terrain + a.min_free_air + a.buffer_depth
    if a.zsize is not None:
        ktot = int(round(a.zsize / dz))
        if abs(ktot * dz - a.zsize) > 1e-6 * a.zsize:
            raise SystemExit(f'--zsize {a.zsize:g} is not a multiple of dz = {dz:g} m')
        if a.zsize < z_need:
            raise SystemExit(f'--zsize {a.zsize:g} m: the terrain reaches {z_top_terrain:.0f} m above the domain '
                             f'floor, plus --min-free-air {a.min_free_air:g} m and the {a.buffer_depth:g}-m '
                             f'buffer: zsize >= {np.ceil(z_need / dz / a.npx) * a.npx * dz:.0f} m')
    else:
        ktot = int(np.ceil(max(z_top_terrain + a.free_air, z_need) / dz / a.npx) * a.npx)
    zsize = ktot * dz
    for n_, d_, what, opt in [(itot, a.npx, 'itot % npx', '--lx'), (itot, a.npy, 'itot % npy', '--lx'),
                              (jtot, a.npx, 'jtot % npx', '--jtot/--ly'), (jtot, a.npy, 'jtot % npy', '--jtot/--ly'),
                              (ktot, a.npx, 'ktot % npx', '--zsize')]:
        if n_ % d_:
            raise SystemExit(f'{what} != 0 ({n_} cells, {d_} tasks): change {opt} (nearest multiple: '
                             f'{int(np.ceil(n_ / d_) * d_)} cells), --npx or --npy')
    if itot // a.npx < a.n_sponge + NGC + 1:
        raise SystemExit(f'imax = {itot // a.npx} too small for a {a.n_sponge}-cell sponge and the ghost cells: '
                         f'use fewer tasks in x (--npx) or a longer domain (--lx, --valley)')
    if a.n_sponge * dx > a.plateau - dx:
        raise SystemExit('--plateau must be longer than the sponge')
    dem = np.broadcast_to(zs[None, :], (jtot, itot)).astype(np.float64)
    dem.tofile(out('dem.0000000'))

    z = (np.arange(ktot) + 0.5) * dz
    zh = np.arange(ktot + 1) * dz
    # pressure: hydrostatic from the floor pressure of the regional simulation through the ambient air
    p_of_amsl = lambda za: hydrostatic_p(np.atleast_1d(np.asarray(za, dtype=float)), thl_a, qt_a,
                                         J['floor_amsl'], J['pbot'])
    pbot = float(p_of_amsl(datum)[0])
    ps = p_of_amsl(z_amsl_s)
    thl_melt = T_MELT * (P0 / ps)**KAPPA
    # the melting deficit at the slope's mid-height: the rotated run's surface
    z_mid_amsl = 0.5 * (z_plateau + z_valley)
    theta_r = float(thl_a(z_mid_amsl))
    dth_s = float(np.interp(z_mid_amsl, z_amsl_s[::-1], thl_melt[::-1])) - theta_r
    if a.surface == 'melt':
        thl_s = thl_melt
    else:
        thl_s = thl_a(z_amsl_s) + dth_s
    qt_s = qsat_liq(T_MELT, ps) if a.humidity == 'racmo' else 0. * ps
    np.broadcast_to(thl_s[None, :], (jtot, itot)).astype(np.float64).tofile(out('thl_sbot.0000000'))
    np.broadcast_to(qt_s[None, :], (jtot, itot)).astype(np.float64).tofile(out('qt_sbot.0000000'))

    # slope statistics over the ice section (between the plateau and the valley)
    on_slope = (x > a.plateau) & (x < xm_s[-1])
    drop = z_plateau - z_valley
    if a.terrain == 'constant':
        alpha_mean = float(alpha_c)
        # the station where the slope has its height (24 Jul 2026 DEM), else mid-slope
        if z_valley < z_site_amsl < z_plateau:
            x_site_m = float(np.interp(-z_site_amsl, -zm_s, xm_s))
        else:
            x_site_m = 0.5 * (xm_s[0] + xm_s[-1])
    else:
        alpha_mean = np.degrees(np.arctan(drop / (xm_s[-1] - xm_s[0])))
        x_site_m = a.plateau + (x_top - x_site)
    # N with the reference theta_r of the buoyancy (the same as the rotated run's N2)
    N = float(np.sqrt(GRAV / theta_r * info['gamma']))
    tau = 2. * np.pi / (N * np.sin(np.deg2rad(alpha_mean)))
    endtime = a.endtime if a.endtime is not None else float(np.ceil(a.n_tau * tau / 3600.) * 3600.)

    # input .nc: ambient profiles (+ gases and the RRTMGP background column, radiation-ready)
    th_prof = thl_a(z + datum)
    q_prof = qt_a(z + datum) if a.humidity == 'racmo' else 0. * z
    with nc.Dataset(out(f'{CASE_IB}_input.nc'), 'w') as f:
        f.createDimension('z', ktot)
        f.createVariable('z', 'f8', ('z',))[:] = z
        g = f.createGroup('init')
        for name, val in [('thl', th_prof), ('qt', q_prof), ('u', 0*z), ('v', 0*z),
                          ('u_geo', 0*z), ('v_geo', 0*z)]:
            g.createVariable(name, 'f8', ('z',))[:] = val
        p_lay = p_of_amsl(z + datum)
        write_radiation_input(f, g, z, zsize, p_lay, float(p_of_amsl(zsize + datum)[0]), th_prof, q_prof)

    # LBCs: west = the inflow (ambient + katabatic layer over the plateau); east = the
    # same over the valley floor (--outflow jet) or the ambient air at rest.
    def column_profiles(z_surf, with_layer):
        n = z - z_surf
        th = th_prof + (layer_dtheta(n, info) if with_layer else 0.)
        u = layer_u(n, info) if with_layer else 0. * z
        return {'u': u, 'v': 0. * z, 'thl': th, 'qt': q_prof}
    prof_w = column_profiles(zs[0], True)
    prof_e = column_profiles(zs[-1], a.outflow == 'jet')
    for name in ('u', 'v', 'thl', 'qt'):
        pad_w = NGC + 1 if name == 'u' else NGC
        for edge, prof, ncol in (('west', prof_w, pad_w + a.n_sponge), ('east', prof_e, NGC + a.n_sponge)):
            arr = np.broadcast_to(prof[name][:, None, None], (ktot, jtot + 2 * NGC, ncol)).astype(np.float64)
            arr.tofile(out(f'lbc_{name}_{edge}.0000000'))
    m_w = np.sum(prof_w['u'] * dz)
    m_e = np.sum(prof_e['u'] * dz)

    # output
    j_xz = [jtot // 4, (3 * jtot) // 4]
    tf_h = [5, 10, 20, 50, 100]
    stations = {'top': a.plateau + 0.15 * (xm_s[-1] - xm_s[0]),
                'mid': a.plateau + 0.5 * (xm_s[-1] - xm_s[0]),
                'station': x_site_m,
                'foot': xm_s[-1] - 1000.}
    crosslist = ['u', 'w', 'thl', 'u_tf', 'thl_tf', 'thlfluxbot_ib', 'hfss_ib', 'hfls_ib',
                 'ustar_ib', 'obuk_ib', 'thl_sbot_ib']
    rad = a.radiation != 'off'
    if rad:
        crosslist += ['lw_dn_ib']

    lat, lon = J['site_latlon']
    t0 = datetime.strptime(J['datetime_utc_t0'], '%Y-%m-%d %H:%M:%S') if J.get('datetime_utc_t0') \
        else datetime(2026, 7, 24, 3)
    when = t0 + timedelta(seconds=a.racmo_time)
    sza = solar_zenith(when, lat, lon)
    if rad:
        rad_sec = f"""[radiation]
# RRTMGP from the IB surface. Gases and the background column
# above the LES top are in the input .nc; job.sh links the coefficient
# files (RRTMGP_DATA). The ice surface temperature is the IB's (273.15 K).
swradiation=rrtmgp
swlongwave=1
swshortwave={1 if a.radiation == 'lwsw' else 0}
dt_rad=60
t_sfc={T_MELT}
emis_sfc=0.99
swconstant_alb=1
sfc_alb_dir={a.albedo:g}
sfc_alb_dif={a.albedo:g}
# the sun fixed at {when:%Y-%m-%d %H:%M} UTC over the weather station
swfixedsza=1
sza={sza:.6f}
swclearskystats=0
sw_ib_columns=true

[micro]
# no microphysics; RRTMGP reads the droplet and ice crystal numbers for the cloud optics
swmicro=0
Nc0=1e+08
Ni0=1e+05
"""
    else:
        rad_sec = f"""[radiation]
# Radiation is off (--radiation off). Everything else is ready: the gases and
# the RRTMGP background column are in the input .nc. To switch it on,
# regenerate with --radiation lw (longwave only) or lwsw (with the sun fixed
# at {when:%H:%M} UTC, sza {np.degrees(sza):.1f} deg); job.sh then links the
# RRTMGP coefficient files (sw_ib_columns).
swradiation=0
"""

    if a.surface == 'melt':
        surf_comment = '# melting surface: theta_s = 273.15 K (p0/p(z_s))^(R/cp)'
    else:
        surf_comment = (f'# theta_s = theta_a(z_s) {dth_s:+.3f} K: the melting deficit at the slope\'s mid-height,\n'
                        '# the same as the rotated run\'s b_s (--surface deficit)')
    surf_comment += ('; qt_s = q_sat(273.15 K, p(z_s))' if a.humidity == 'racmo' else '; dry (qt = 0)')
    if a.z0h_model == 'yang2008':
        z0h_block = ('# melting ice: z0h = z0q from Yang et al. (2008)\nz0h_model=yang2008\n'
                     'z0_nu=1.5e-05\nz0_yang_beta=7.2\nz0h_min=1e-10\nz0h_max=0.1')
    else:
        z0h_block = '# z0h fixed, the same as the rotated run (--z0h-model fixed)\nz0h_model=fixed'
    if a.basestate == 'boussinesq':
        thermo_block = (f'# Boussinesq, as the rotated run (thermo buoy): reference thv0 = theta_r, the\n'
                        f"# ambient theta at the slope's mid-height; buoyancy g (thv - thv0)/thv0\n"
                        f'swbasestate=boussinesq\nthvref0={theta_r:.4f}\npbot={pbot:.3f}\nswupdatebasestate=0')
    else:
        thermo_block = f'swbasestate=anelastic\npbot={pbot:.3f}\nswupdatebasestate=1'
    ini = f"""# {CASE_IB}: idealised katabatic flow down a Greenland ice slope (24 Jul 2026),
# generated by katabatic_idealised_input.py on {datetime.now():%Y-%m-%dT%H:%M:%S}
# datum: z = 0 is {datum:.2f} m amsl; the valley floor {z_valley:.1f} m amsl, the plateau {z_plateau:.1f} m amsl
[master]
npx={a.npx}
npy={a.npy}

[grid]
itot={itot}
jtot={jtot}
ktot={ktot}
xsize={lx:g}
ysize={ly:g}
zsize={zsize:g}
swspatialorder=2
lat={lat:.6f}
lon={lon:.6f}

[pres]
# DCT in x (open), DFT in y (periodic)
sw_openbc=true

[advec]
swadvec=2i5
cflmax=1.2
fluxlimit_list=thl,qt

[limiter]
limitlist=qt

[diff]
# anisotropic Smagorinsky, lengths dx and dz (as the regional simulation)
swdiff=smag2
swanisotropic=true
mlen0_h={dx:g}
mlen0_v={dz:g}
dnmax=0.3

[thermo]
swthermo=moist
{thermo_block}

{rad_sec}
[force]
# no large-scale pressure gradient, no Coriolis (for now)
swlspres=0
swls=0
swwls=0

[boundary]
# the domain floor is buried under the terrain; the IB is the surface
mbcbot=noslip
mbctop=freeslip
swboundary=surface
swconstantz0=1
z0m={a.z0m:g}
z0h={a.z0h:g}
sbcbot=dirichlet
sbctop=neumann
sbot[thl]={th_prof[0]:.6f}
stop[thl]={info['gamma']:.9g}
sbot[qt]={q_prof[0]:.8g}
stop[qt]=0.

[boundary_lateral]
# open at west (inflow over the plateau) and east (outflow in the valley),
# periodic y; constant boundary values lbc_*.0000000
sw_openbc=true
sw_periodic_y=true
sw_openbc_uv=true
sw_openbc_w=false
sw_neumann_w=true
sw_wtop_2d=false
sw_timedep=false
loadfreq=3600
slist=thl,qt
sw_sponge=true
n_sponge={a.n_sponge}
tau_sponge={a.tau_sponge:g}
w_diff=0.003

[IB]
sw_immersed_boundary=dem
n_idw_points=5
sw_dem_halo_replicate=true
sw_wall_model=most
z0m={a.z0m:g}
z0h={a.z0h:g}
{z0h_block}
sw_scalar_flux=true
sw_wall_stability_vertical=false
sw_scalar_flux_vertical=false
{MOMENTUM[a.momentum]}sw_mason_wall=true
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
sw_wall_sample=true
wall_sample_dz=1.5
sw_tf_cross=true
tf_cross_heights={','.join(str(h) for h in tf_h)}
{surf_comment}
sbcbot=dirichlet
sbot[thl]={float(np.median(thl_s)):.6f}
sbot[qt]={float(np.median(qt_s)):.8g}
sbot_spatial=thl,qt
sw_timedep_sbot=false
columnlist=thl_fluxbot_ib,hfss_ib,hfls_ib,ustar_ib,obuk_ib,thl_sbot_ib,u_ib1,thl_ib1,z_dem,k_dem,taux_skin_ib,taux_form_ib

[fields]
visc=1.e-5
svisc=1.e-5
rndseed=2
# no noise here: katabatic_idealised_init.py adds it in the lowest
# {a.rnd_depth:g} m above the LOCAL surface, as MicroHH does above the floor in the
# rotated run (amplitude (1 - n/{a.rnd_depth:g})^2 x 0.1 m/s or 0.1 K)

[buffer]
swbuffer=1
swupdate=0
zstart={zsize - a.buffer_depth:g}
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
crosslist={','.join(crosslist)}

[column]
swcolumn=1
sampletime={a.out_every:g}
coordinates[x]={','.join(f'{v:.1f}' for v in stations.values())}
coordinates[y]={','.join(f'{0.5 * ly:.1f}' for _ in stations)}

[dump]
swdump=0
"""
    open(out(f'{CASE_IB}.ini'), 'w').write(ini)
    pars = dict(case=CASE_IB, frame='cartesian', itot=itot, jtot=jtot, ktot=ktot, dx=dx, dz=dz, lx=lx, ly=ly,
                zsize=zsize, datum_amsl=datum, z_valley_amsl=z_valley, z_plateau_amsl=z_plateau,
                x_plateau_end=a.plateau, x_foot=float(xm_s[-1]), x_site=x_site_m, x_top_src=x_top,
                x_margin_src=x_margin, alpha_mean_deg=float(alpha_mean), terrain=a.terrain, drop=float(drop), N=N, tau_osc=tau,
                endtime=endtime, gamma=info['gamma'], thl_a_coeffs=[float(thl_a(0.)), info['gamma']],
                layer=dict(U_j=info['U_j'], n_j=info['n_j'], dtheta0=info['dtheta0'], h_theta=info['h_theta'],
                           **({'profile': info['profile']} if 'profile' in info else {})),
                inflow=a.inflow, outflow=a.outflow, n_sponge=a.n_sponge, tau_sponge=a.tau_sponge, z0m=a.z0m,
                momentum=a.momentum, radiation=a.radiation, sza_deg=float(np.degrees(sza)),
                surface=a.surface, humidity=a.humidity, z0h_model=a.z0h_model, z0h=a.z0h,
                basestate=a.basestate, theta_r=theta_r, dtheta_s=dth_s, z_mid_amsl=z_mid_amsl,
                noise=dict(depth=a.rnd_depth, exp=2., u=0.1, v=0.1, w=0.1, thl=0.1, seed=2),
                stations=stations, xz_j=j_xz, tf_heights=tf_h, out_every=a.out_every,
                npx=a.npx, npy=a.npy, pbot=pbot, z_top_terrain=z_top_terrain,
                thl_s_range=[float(thl_s.min()), float(thl_s.max())],
                racmo=info)
    json.dump(pars, open(out('case_parameters.json'), 'w'), indent=1)
    np.savetxt(out('transect.txt'), np.c_[x, z_amsl_s, zs, thl_s, qt_s],
               header='x [m]  z_s amsl [m]  z_s model [m]  thl_s [K]  qt_s [kg/kg]', fmt='%.4f')
    print(f'ib/: {itot} x {jtot} x {ktot} = {itot*jtot*ktot/1e6:.2f} M cells, dx {dx:g} m, dz {dz:g} m, '
          f'{lx/1e3:g} x {ly/1e3:g} x {zsize/1e3:g} km, npx x npy = {a.npx} x {a.npy}')
    print(f'  terrain ({a.terrain}): plateau {z_plateau:.0f} m amsl (x < {a.plateau/1e3:g} km), ice slope to x = '
          f'{xm_s[-1]/1e3:.1f} km (drop {drop:.0f} m, mean {alpha_mean:.2f} deg, max '
          f'{np.degrees(np.arctan(np.abs(np.gradient(zs, dx))[on_slope].max())):.1f} deg), valley '
          f'{z_valley:.0f} m amsl; the station at x = {x_site_m/1e3:.1f} km')
    print(f'  surface ({a.surface}): theta_s {thl_s.min():.2f}-{thl_s.max():.2f} K, theta_s - theta_a '
          f'{(thl_s - thl_a(z_amsl_s)).min():.2f} to {(thl_s - thl_a(z_amsl_s)).max():.2f} K (melting at '
          f'mid-height: {dth_s:.2f} K); humidity {a.humidity}; z0h {a.z0h_model}'
          f'{" %g m" % a.z0h if a.z0h_model == "fixed" else ""}; {a.basestate}; pbot {pbot/100:.1f} hPa at '
          f'{datum:.1f} m amsl')
    if 'profile' in info:
        P = info['profile']
        print(f'  inflow: the rotated equilibrium ({P["window"][0]:.0f}-{P["window"][1]:.0f} s of {P["source"]}): '
              f'jet {P["U_j"]:.2f} m/s at {P["n_j"]:.0f} m (range over the window {P["jet_range"][0]:.2f}-'
              f'{P["jet_range"][1]:.2f}), deficit {P["dtheta0"]:.2f} K at the first level')
    else:
        print(f'  inflow: the RACMO fit, jet {info["U_j"]:.2f} m/s at {info["n_j"]:.0f} m, deficit '
              f'{info["dtheta0"]:.2f} K e-folding {info["h_theta"]:.0f} m')
    print(f'  inflow mass flux per width {m_w:.0f} m2/s, outflow target ({a.outflow}) {m_e:.0f} m2/s')
    print(f'  N = {N:.4f} 1/s, tau_osc = 2 pi/(N sin alpha) = {tau:.0f} s, endtime {endtime:g} s '
          f'({endtime/tau:.2f} tau)')
    print(f'  radiation: {a.radiation}; output every {a.out_every:g} s: {len(crosslist)} cross variables, '
          f'{len(j_xz)} xz planes, tf at {tf_h} m, {len(stations)} columns')
    return pars


def write_radiation_input(f, init, z, zsize, p_lay, p_top, thl_prof, qt_prof):
    """Gases on the LES levels and the RRTMGP background column above the top to
    70 km (as in the regional simulation): 6.5 K/km to 11 km, isothermal above,
    humidity decaying with a 2.5 km scale, ozone from the RCEMIP fit
    (Wing et al. 2018, https://doi.org/10.5194/gmd-11-793-2018)."""
    o3 = lambda p: 3.6478 * (p / 100.)**0.83209 * np.exp(-(p / 100.) / 11.3515) * 1e-6
    init.createVariable('h2o', 'f8', ('z',))[:] = qt_prof / (EPS - EPS * qt_prof)
    init.createVariable('o3', 'f8', ('z',))[:] = o3(p_lay)
    for gas, v in RADIATION_GASES.items():
        init.createVariable(gas, 'f8')[:] = v
    t_top = float(thl_prof[-1]) * (p_top / P0)**KAPPA
    q_top = float(qt_prof[-1])
    zh = np.append(np.arange(zsize, 70e3, 500.), 70e3)
    zf = np.linspace(zsize, 70e3, 40000)
    z_trop = max(11e3, zsize + 1e3)
    tf = np.where(zf < z_trop, t_top - 0.0065 * (zf - zsize), t_top - 0.0065 * (z_trop - zsize))
    lnp = np.log(p_top) - np.concatenate([[0.], np.cumsum(GRAV / (RD * tf[1:]) * np.diff(zf))])
    pf = np.exp(lnp)
    z_lay = 0.5 * (zh[1:] + zh[:-1])
    q_lay = np.maximum(3e-6, q_top * np.exp(-(z_lay - zsize) / 2500.))
    r = f.createGroup('radiation')
    r.createDimension('lay', z_lay.size)
    r.createDimension('lev', zh.size)
    for name, dim, val in (('z_lay', 'lay', z_lay), ('z_lev', 'lev', zh),
                           ('p_lay', 'lay', np.interp(z_lay, zf, pf)), ('p_lev', 'lev', np.interp(zh, zf, pf)),
                           ('t_lay', 'lay', np.interp(z_lay, zf, tf)), ('t_lev', 'lev', np.interp(zh, zf, tf)),
                           ('h2o', 'lay', q_lay / (EPS - EPS * q_lay)),
                           ('o3', 'lay', o3(np.interp(z_lay, zf, pf)))):
        r.createVariable(name, 'f8', (dim,))[:] = val
    for gas, v in RADIATION_GASES.items():
        r.createVariable(gas, 'f8')[:] = v


# ---- the rotated reference --------------------------------------------------

def write_rotated(a, info, pib, d):
    os.makedirs(d, exist_ok=True)
    out = lambda f: os.path.join(d, f)
    alpha = a.alpha_ref if a.alpha_ref is not None else pib['alpha_mean_deg']
    dx, dz = a.dx, a.dz
    itot = a.rot_itot if a.rot_itot is not None else int(round(a.rot_lx / dx))
    jtot = a.rot_jtot if a.rot_jtot is not None else int(round(a.rot_ly / dx))
    ktot = int(np.ceil(a.rot_lz / dz / a.rot_npx) * a.rot_npx)
    for n_, d_, what in [(itot, a.rot_npx, 'itot % npx'), (itot, a.rot_npy, 'itot % npy'),
                         (jtot, a.rot_npx, 'jtot % npx'), (jtot, a.rot_npy, 'jtot % npy'),
                         (ktot, a.rot_npx, 'ktot % npx')]:
        if n_ % d_:
            raise SystemExit(f'rotated: {what} != 0 ({n_}, {d_})')
    zsize = ktot * dz
    z = (np.arange(ktot) + 0.5) * dz
    # reference theta and the surface deficit at the slope's mid-height
    z_mid_amsl = pib['z_mid_amsl']
    a0, gam = pib['thl_a_coeffs']
    th_r = pib['theta_r']
    b_s = GRAV / th_r * pib['dtheta_s']
    N2 = GRAV / th_r * gam
    L = info
    with nc.Dataset(out(f'{CASE_ROT}_input.nc'), 'w') as f:
        f.createDimension('z', ktot)
        f.createVariable('z', 'f8', ('z',))[:] = z
        g = f.createGroup('init')
        g.createVariable('b', 'f8', ('z',))[:] = GRAV / th_r * deficit(z, L['dtheta0'], L['h_theta'])
        # downslope is -x in MicroHH's slope frame (thermo_buoy.cxx: u_t += b sin(alpha))
        g.createVariable('u', 'f8', ('z',))[:] = -jet(z, L['U_j'], L['n_j'])
        g.createVariable('v', 'f8', ('z',))[:] = 0. * z
    tau = 2. * np.pi / (np.sqrt(N2) * np.sin(np.deg2rad(alpha)))
    endtime = pib['endtime']
    ini = f"""# {CASE_ROT}: the rotated-frame reference of {CASE_IB} (infinite slope at
# alpha = {alpha:.3f} deg, N2 = {N2:.4e} 1/s2, melting-surface deficit at the slope's
# mid-height). Generated by katabatic_idealised_input.py on {datetime.now():%Y-%m-%dT%H:%M:%S}
[master]
npx={a.rot_npx}
npy={a.rot_npy}

[grid]
itot={itot}
jtot={jtot}
ktot={ktot}
xsize={itot * dx:g}
ysize={jtot * dx:g}
zsize={zsize:g}
swspatialorder=2

[advec]
swadvec=2i5
cflmax=1.2

[diff]
swdiff=smag2
swanisotropic=true
mlen0_h={dx:g}
mlen0_v={dz:g}
dnmax=0.3

[thermo]
swthermo=buoy
alpha={np.deg2rad(alpha):.10f}
N2={N2:.6e}

[force]
swlspres=0
swls=0
swwls=0

[boundary]
mbcbot=noslip
mbctop=freeslip
swboundary=surface
swconstantz0=1
z0m={a.z0m:g}
z0h={a.z0h:g}
# melting surface: b_s = g/theta_r (theta_melt - theta_a) at the slope's mid-height
sbcbot=dirichlet
sbctop=neumann
sbot[b]={b_s:.8g}
stop[b]=0.

[fields]
visc=1.e-5
svisc=1.e-5
rndseed=2
# the same noise as the IB run (katabatic_idealised_init.py): 0.1 m/s, 0.1 K
# in b units, (1 - z/{a.rnd_depth:g})^2 above the surface
rndamp[u]=0.1
rndamp[v]=0.1
rndamp[w]=0.1
rndamp[b]={GRAV / th_r * 0.1:.6g}
rndz={a.rnd_depth:g}
rndexp=2.

[buffer]
swbuffer=1
swupdate=0
zstart={zsize - a.buffer_depth:g}
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

[cross]
swcross=1
sampletime={a.out_every:g}
xz={0.5 * jtot * dx:g}
crosslist=u,w,b
"""
    open(out(f'{CASE_ROT}.ini'), 'w').write(ini)
    pars = dict(case=CASE_ROT, frame='rotated', downslope_sign=-1, alpha_deg=float(alpha), N2=N2, theta_r=th_r, b_s=b_s,
                dtheta_s=float(pib['dtheta_s']), z_mid_amsl=z_mid_amsl, itot=itot, jtot=jtot, ktot=ktot,
                dx=dx, dz=dz, zsize=zsize, z0m=a.z0m, z0h=a.z0h, tau_osc=tau, endtime=endtime,
                layer=dict(U_j=L['U_j'], n_j=L['n_j'], dtheta0=L['dtheta0'], h_theta=L['h_theta']),
                npx=a.rot_npx, npy=a.rot_npy, out_every=a.out_every)
    json.dump(pars, open(out('case_parameters.json'), 'w'), indent=1)
    print(f'rotated/: {itot} x {jtot} x {ktot}, alpha {alpha:.3f} deg, N {np.sqrt(N2):.4f} 1/s, '
          f'theta_r {th_r:.2f} K, surface deficit {pib["dtheta_s"]:.2f} K (b_s {b_s:.4f} m/s2), '
          f'z0h {a.z0h:g} m, tau_osc {tau:.0f} s, endtime {endtime:g} s')


def plot_inputs(a, J, thl_a, qt_a, info, pib, path):
    """inputs.png: the terrain and surface temperature along x, and the inflow
    profiles against RACMO's."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tr = np.loadtxt(os.path.join(a.outdir, 'ib', 'transect.txt'))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    ax[0].fill_between(tr[:, 0] / 1e3, tr[:, 1], tr[:, 1].min() - 50., color='0.8')
    ax[0].plot(tr[:, 0] / 1e3, tr[:, 1], 'k-')
    for v in (pib['x_plateau_end'], pib['x_foot']):
        ax[0].axvline(v / 1e3, color='0.5', lw=0.8)
    ax[0].axvline(pib['x_site'] / 1e3, color='#e08a2c', ls=':', label='station')
    ax[0].axvspan(0, pib['n_sponge'] * pib['dx'] / 1e3, color='m', alpha=0.15, label='sponges')
    ax[0].axvspan(pib['lx'] / 1e3 - pib['n_sponge'] * pib['dx'] / 1e3, pib['lx'] / 1e3, color='m', alpha=0.15)
    ax[0].set_xlabel('x (km), flow towards +x'); ax[0].set_ylabel('surface (m amsl)')
    ax[0].set_title(f'terrain: mean slope {pib["alpha_mean_deg"]:.2f} deg, drop {pib["drop"]:.0f} m')
    ax[0].legend(fontsize=8)
    a1 = ax[0].twinx()
    a1.plot(tr[:, 0] / 1e3, tr[:, 3], 'b-', lw=0.8)
    a1.set_ylabel(r'$\theta_s$ melting (K)', color='b')
    R = info['racmo_profile']
    h = np.array(R['h'])
    zs = info['z_surface_racmo']
    nn = np.linspace(0.5, 1000., 400)
    ax[1].plot(R['u_downslope'], h, 'ko', ms=3, label='RACMO -u (east boundary)')
    ax[1].plot(R['v'], h, 'o', color='0.6', ms=3, label='RACMO v (not used)')
    src = 'rotated equilibrium' if 'profile' in info else 'RACMO fit'
    if 'profile' in info:
        F = info['racmo_fit']
        ax[1].plot(jet(nn, F['U_j'], F['n_j']), nn, 'r:', lw=1, label='RACMO fit (not used)')
    ax[1].plot(layer_u(nn, info), nn, 'r-', label=f'inflow u ({src})')
    ax[1].set_ylim(0, 600); ax[1].set_xlabel('wind (m/s)'); ax[1].set_ylabel('height above the surface (m)')
    ax[1].set_title(f'inflow wind: {info["U_j"]:.1f} m/s at {info["n_j"]:.0f} m'); ax[1].legend(fontsize=8)
    ax[2].plot(R['thl'], h, 'ko', ms=3, label='RACMO thl')
    ax[2].plot(thl_a(zs + nn), nn, 'b--', label='ambient (fit 200-1000 m)')
    if 'profile' in info:
        F = info['racmo_fit']
        ax[2].plot(thl_a(zs + nn) + deficit(nn, F['dtheta0'], F['h_theta']), nn, 'r:', lw=1,
                   label='RACMO fit (not used)')
    ax[2].plot(thl_a(zs + nn) + layer_dtheta(nn, info), nn, 'r-', label=f'inflow ({src})')
    th_s = float(np.interp(0., tr[:, 0], tr[:, 3]))
    ax[2].plot([th_s], [0.], 'bs', label='melting surface')
    ax[2].set_ylim(0, 600); ax[2].set_xlabel(r'$\theta_l$ (K)')
    ax[2].set_title(f'inflow deficit {info["dtheta0"]:.1f} K at the first level'); ax[2].legend(fontsize=8)
    fig.suptitle(f'katabatic_idealised inputs: RACMO at the ice-sheet boundary, t = {a.racmo_time} s', fontsize=11)
    fig.savefig(path, dpi=90)
    plt.close(fig)


SHARED_CHECK = ('alpha_deg', 'theta_r', 'dtheta_s', 'dx', 'dz', 'z0m', 'z0h')
NOT_SAVED = ('outdir', 'stage', 'only', 'explicit', 'rot_window')


def build_ib(a, J, thl_a, qt_a, info, d, dry=False):
    """write_ib, or (dry) only its parameters: written to a scratch folder that is removed."""
    if not dry:
        return write_ib(a, J, thl_a, qt_a, info, d)
    import contextlib
    import io
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return write_ib(a, J, thl_a, qt_a, info, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def merge_saved_args(a):
    """Stage ib: take the options of the stage that wrote rotated/, except those given now."""
    f = os.path.join(a.outdir, 'generator_args.json')
    if not os.path.exists(f):
        print(f'WARNING: no {f}: using the command line only; it is checked against rotated/ below')
        return a
    saved = json.load(open(f))
    kept = []
    for k, v in saved.items():
        if k in NOT_SAVED or k in a.explicit:
            continue
        if getattr(a, k, None) != v:
            kept.append(k)
        setattr(a, k, v)
    # an option given now beats the saved one that it overrides
    for now, old in (('ly', 'jtot'), ('rot_ly', 'rot_jtot'), ('rot_lx', 'rot_itot')):
        if now in a.explicit and old not in a.explicit:
            setattr(a, old, None)
    over = [k for k in a.explicit if k not in NOT_SAVED and k in saved and saved[k] != getattr(a, k)]
    print(f'options of the first stage from {f}' + (f'; changed now: {", ".join(over)}' if over else ''))
    return a


def check_against_rotated(pib, prof, a):
    """The IB run must have the slope, theta_r, surface deficit, grid and roughness
    of the rotated run whose profile it takes."""
    want = dict(alpha_deg=a.alpha_ref if a.alpha_ref is not None else pib['alpha_mean_deg'],
                theta_r=pib['theta_r'], dtheta_s=pib['dtheta_s'], dx=pib['dx'], dz=pib['dz'],
                z0m=pib['z0m'], z0h=pib['z0h'])
    bad = [f'{k}: IB {want[k]:.6g}, rotated {prof[k]:.6g}' for k in SHARED_CHECK
           if abs(want[k] - prof[k]) > 1e-6 * max(1., abs(prof[k]))]
    if bad:
        raise SystemExit('the IB set-up does not match the rotated run it takes the inflow from:\n  '
                         + '\n  '.join(bad) + '\nRegenerate both (a new --outdir), or drop the options '
                         'that change the slope, grid, roughness or surface.')


def main():
    a = parse()
    rot_dir = os.path.join(a.outdir, 'rotated')
    stage = a.stage
    if stage == 'auto':
        import glob
        stage = 'ib' if glob.glob(os.path.join(rot_dir, f'{CASE_ROT}.default.*.nc')) else 'rotated'
    if a.inflow == 'rotated' and stage == 'ib':
        a = merge_saved_args(a)
    J = json.load(open(a.inputs))
    thl_a, qt_a, info = ambient_and_layer(J, a)
    print(f'RACMO east profile at t = {a.racmo_time} s: ambient dtheta/dz {info["gamma"]*1e3:.2f} K/km '
          f'(N {info["N"]:.4f} 1/s); katabatic layer U_j {info["U_j"]:.2f} m/s at n_j {info["n_j"]:.0f} m, '
          f'theta deficit {info["dtheta0"]:.2f} K e-folding {info["h_theta"]:.0f} m')
    ib_dir = os.path.join(a.outdir, 'ib')
    os.makedirs(a.outdir, exist_ok=True)
    if stage in ('rotated', 'all'):
        # the rotated run needs the IB geometry (slope, theta_r, surface deficit, endtime)
        pib = build_ib(a, J, thl_a, qt_a, info, ib_dir, dry=(stage == 'rotated'))
        if stage == 'all':
            plot_inputs(a, J, thl_a, qt_a, info, pib, os.path.join(a.outdir, 'inputs.png'))
        write_rotated(a, info, pib, rot_dir)
        json.dump({k: v for k, v in vars(a).items() if k not in NOT_SAVED},
                  open(os.path.join(a.outdir, 'generator_args.json'), 'w'), indent=1)
        if stage == 'rotated' and a.inflow == 'rotated':
            print(f"""
STAGE 1 of 2 done: {rot_dir} written; ib/ is NOT written yet.
  1. run the rotated case to its end (job.sh in {rot_dir}; with
     KAT_STAGE2=1, the default, the job writes ib/ itself when it finishes);
  2. otherwise: python3 {os.path.basename(__file__)} --outdir {a.outdir}
     (stage 2: ib/ with the rotated run's equilibrium as the inflow; the other
     options are read from {os.path.join(a.outdir, 'generator_args.json')});
  3. run ib/.""")
    elif stage == 'ib':
        if a.inflow == 'rotated':
            prof = rotated_profile(rot_dir, a.rot_window)
            info['racmo_fit'] = dict(U_j=info['U_j'], n_j=info['n_j'], dtheta0=info['dtheta0'],
                                     h_theta=info['h_theta'])
            info.update(profile={k: prof[k] for k in ('source', 'window', 'n', 'u', 'dtheta', 'U_j', 'n_j',
                                                      'dtheta0', 'jet_range', 'z0m')},
                        U_j=prof['U_j'], n_j=prof['n_j'], dtheta0=prof['dtheta0'], h_theta=prof['h_theta'])
            check_against_rotated(build_ib(a, J, thl_a, qt_a, info, ib_dir, dry=True), prof, a)
            print(f'rotated equilibrium over {prof["window"][0]:.0f}-{prof["window"][1]:.0f} s: jet '
                  f'{prof["U_j"]:.2f} m/s at {prof["n_j"]:.0f} m (between {prof["jet_range"][0]:.2f} and '
                  f'{prof["jet_range"][1]:.2f} over the window), deficit {prof["dtheta0"]:.2f} K at '
                  f'{prof["n"][0]:.0f} m, |v| <= {prof["v_mean"]:.2f} m/s')
        pib = build_ib(a, J, thl_a, qt_a, info, ib_dir)
        plot_inputs(a, J, thl_a, qt_a, info, pib, os.path.join(a.outdir, 'inputs.png'))
    print(f'wrote {a.outdir} (stage {stage}, inflow {a.inflow})')


if __name__ == '__main__':
    main()
