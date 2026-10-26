#!/usr/bin/env python3
"""
Axelsen and van Dop (2009, Acta Geophys. 57, 803-836), the slope-flow
cases A1, P1, P2, V1 and V2, on an immersed-boundary (IB) slope with the
recommended IB settings of the develop_openbc branch.

The paper runs a quasi-1-D steady slope flow in a ROTATED, periodic frame:
a uniform slope alpha, a stably stratified environment (N), a prescribed
surface buoyancy flux Fs, no Coriolis, no ambient wind, until 3 oscillation
periods tau_osc = 2 pi / (N sin alpha).

Translation to an IB slope on a gravity-aligned grid
---------------------------------------------------
- Terrain: an IB DEM falling at alpha towards +x (downslope = +x), on a
  gravity-aligned grid. --res paper (default): the paper's grid spacing and
  domain (katabatic 2.6 m x 0.4 m, 250 x 250 x 51 m; A1 13.3 m x 10 m,
  1700 x 850 x 600 m), plus the lateral sponges outside that interior.
  --res coarse: dx = 100 m, dz = 10 m, the resolution of a regional LES
  over a large ice sheet, 9.6 x 4.8 km (7.6 x 2.8 km inside the 10-cell
  sponges).
- "Periodic": a plane cannot be periodic in a gravity-aligned frame (the
  inflow edge is hundreds of metres higher). The terrain-following lateral
  boundary of the develop_openbc branch feeds every edge from
  the opposite edge at the same height above the surface, with thl shifted
  by the environmental gradient (lbc_tf_gradient[thl] = G): the IB form of
  the paper's periodic slope frame.
- Environment: thl_a(z) = theta_a0 + G (z - z_mid), G = N^2 theta_a0 / g,
  z the true height and z_mid the terrain height at the domain centre.
- Surface, two variants (--bc):
    flux  the paper's Fs as a kinematic heat flux per unit surface area,
          F = Fs theta_a0 / g, via [IB] sbcbot=flux.
    temp  a prescribed surface temperature, thl_s(x, y) = thl_a(z_s) + dth_s,
          so the deficit is uniform along the slope (what periodicity needs).
          Glacier cases: a melting surface at the domain centre,
          dth_s = 273.15 - theta_a0 (P1 -11.5 K, P2 -10.5, V1/V2 -7.5).
          A1 (no glacier): dth_s matched to Fs with the neutral bulk formula
          at the first level, dth_s = F / (C_hn U), U = 2.6 V (the paper's
          u_max). --dth overrides both.
- SGS: --res paper uses the paper's closure, 1.5-order TKE (tke2) with
  Pr_SFS = 1 ([diff] ch1 = 1, ch2 = 0), and its initial SFS TKE, set after
  "microhh init" by axelsen2009_init.py relative to the local surface:
  katabatic e(n) = [0.4 (1 - n/25)^3]^1/2 m2/s2 below n = 25 m, 0 above
  (paper section 3.1, after Beare et al. 2006); A1 e = 0.5 V^2
  exp(-20 n sin(alpha)/H) (section 2.3, after Schumann 1990).
  --res coarse uses smag2 (--sgs overrides both).
- Everything else as the recommended IB set-up: 2i5 + Koren on thl,qt,
  dnmax 0.3, cflmax 1.3, anelastic moist (dry), MOST wall model with
  sw_wall_slope, momentum by --momentum: staircase (default: the MOST wall
  stress as a source term) or vr (velocity reconstruction, needs a MicroHH
  build with it), evisc_solid=cap_last, advec wall, blank solid, z0h
  yang2008 for ice (ratio 10 for A1), open BCs with a 10-cell sponge.

Usage
-----
    python3 axelsen2009_input.py --case P1 --bc flux --outdir run_P1_flux
    python3 axelsen2009_input.py --case P1 --bc temp --outdir run_P1_temp
    python3 axelsen2009_input.py --all --outdir runs        # 10 runs, runs/<case>_<bc>
    python3 axelsen2009_input.py --case P1 --bc flux --res coarse --outdir run_P1_flux_coarse
then, in the run directory:
    ./microhh init axelsen2009
    python3 axelsen2009_init.py --rundir .   # initial SFS TKE (tke2), --init-jet
    ./microhh run axelsen2009
(job.sh does these three steps, on a laptop or a SLURM cluster.)

Needs MicroHH from the develop_openbc branch (for --momentum vr: a build with
velocity reconstruction). Writes, per run: axelsen2009.ini,
axelsen2009_input.nc, dem.0000000, thl_sbot.0000000, qt_sbot.0000000 and
case_parameters.json (read by axelsen2009_analysis.py).
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
GRAV = 9.81
KAPPA = 0.4
T_MELT = 273.15
SGSTKE_MIN = 1e-7          # MicroHH Constants::sgstke_min [m2/s2]
CASE = 'axelsen2009'
REF = json.load(open(os.path.join(HERE, 'reference_axelsen2009.json')))['cases']

# Heights above the local surface of the terrain-following crosses [m];
# MicroHH names the planes <var>_tf<h> with h rounded to whole metres, so
# sub-metre heights are not possible: the analysis takes the near-surface
# profiles from the xz crosses instead.
TF_HEIGHTS = {
    'coarse':    [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 120, 150,
                  200, 250, 300, 400, 500, 600, 800],
    'katabatic': [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50],
    'anabatic':  [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300,
                  350, 400, 450, 500, 550],
}

# Grid presets. 'paper': the paper's grid spacing, its domain as the
# INTERIOR (the lateral sponges come on top), its 1.5-order TKE closure.
# Katabatic: 250 x 250 x 51.2 m, 96 x 96 x 128 points, of which the top 25 %
# is the paper's sponge (section 3.1.1: "the sponge layer filling the top 25%
# of the domain"), i.e. 38.4 m undamped. Here (since 5 Oct 2026, after P1
# blew up at 5100 s from the free atmosphere above the cold layer) the same
# 38.4 m undamped above the highest terrain, with a buffer twice as deep
# (25.6 m: lz = 64, buf_frac = 0.4) and ~13x stronger (buf_sigma = 0.03 s-1
# ~ 2.5 N at the top, Klemp & Lilly 1978; was MicroHH's 0.00223 s-1, an
# e-folding time of 7.5 min against a buoyancy period of 8.7 min). The slope
# drop and the solid levels come on top: ktot 232. A1: 1700 x 850 x 600 m,
# 128 x 64 x 60 points. 'coarse': dx = 100 m, dz = 10 m and smag2, the set-up
# of a regional LES over a large ice sheet.
PRESETS = {
    ('paper', 'katabatic'): dict(dx=250./96, dz=0.4, lx=250., ly=250., lz=64.,
                                 n_sponge=5, sgs='tke2', out_every=300., buf_frac=0.4,
                                 buf_sigma=0.03),
    ('paper', 'anabatic'):  dict(dx=1700./128, dz=10., lx=1700., ly=850., lz=600.,
                                 n_sponge=5, sgs='tke2', out_every=300., buf_frac=0.2,
                                 buf_sigma=0.00223),
    ('coarse', 'katabatic'): dict(dx=100., dz=10., lx=7600., ly=2800., lz=1000.,
                                 n_sponge=10, sgs='smag2', out_every=300., buf_frac=0.25,
                                 buf_sigma=0.00223),
    ('coarse', 'anabatic'): dict(dx=100., dz=10., lx=7600., ly=2800., lz=1000.,
                                 n_sponge=10, sgs='smag2', out_every=300., buf_frac=0.25,
                                 buf_sigma=0.00223),
}


# Cost model, measured on one 128-core AMD EPYC node (128 tasks): P1 paper
# grid, run 2 (5.8 M cells, 45.5 k cells per
# task) 0.388 s per step, run 3 (20.6 M cells, 161 k per task) 1.31 s per
# step: c = 8.1-8.5 us per cell-step per task, the same for both. The time
# step was set by the diffusion number in the jet throughout (DNUM = dnmax =
# 0.3, CFL 0.3-0.7): the dn probe put the largest K in the air, K_max ~ 0.17
# m2/s at t = 2400 s (Delta = (dx dx dz)^1/3 = 1.39 m), dt 0.27 s.
C_CELL = 8.5e-6            # s per cell-step per MPI task
K_REF = 0.17               # m2/s, largest SGS K of the P1 paper-grid run
DELTA_REF = (250. / 96 * 250. / 96 * 0.4)**(1. / 3.)
OUTPUT_FAC = 1.05          # stats, crosses and restarts (~2 % measured) + start-up



# The staircase momentum keys of axelsen2009.ini.base; --momentum vr removes
# them (a velocity-reconstruction build does not use them).
STAIRCASE_KEYS = ('sw_momentum_flux', 'sw_momentum_flux_vertical', 'sw_strain_most',
                  'sw_strain_most_vertical', 'strain_most_min', 'sw_impermeable',
                  'sw_wall_kinematic', 'sw_momentum_air_only')

def cost_estimate(a, kind, V, alpha, itot, jtot, ktot, npx, npy, endtime, dnmax, cflmax):
    """Wall time of a run: steps x (c x cells per task).

    dt = min(dt_DN, dt_CFL). dt_DN = dnmax / (K_max (2/dx^2 + 1/dz^2)), the
    diffusion limit MicroHH applies (calc_dnmul), with K_max scaled from the
    measured P1 value by the inertial-range dependence K ~ eps^(1/3)
    Delta^(4/3). dt_CFL = cflmax / (U/dx + U sin(alpha)/dz) with the paper's
    jet speed U = 16 V (katabatic) or 2.6 V (anabatic)."""
    dz_min = a.dz
    delta = (a.dx * a.dx * dz_min)**(1. / 3.)
    kmax = K_REF * (delta / DELTA_REF)**(4. / 3.)
    dt_dn = dnmax / (kmax * (2. / a.dx**2 + 1. / dz_min**2))
    U = (16. if kind == 'katabatic' else 2.6) * V
    dt_cfl = cflmax / (U / a.dx + U * np.sin(np.deg2rad(alpha)) / dz_min)
    dt = min(dt_dn, dt_cfl, 5.)
    limit = 'DNUM' if dt_dn <= dt_cfl else 'CFL'
    ntask = npx * npy
    cells_task = itot * jtot * ktot / ntask
    s_step = C_CELL * cells_task
    steps = endtime / dt
    wall_h = steps * s_step * OUTPUT_FAC / 3600.
    nodes = int(np.ceil(ntask / 128.))
    measured = (a.res == 'paper' and kind == 'katabatic'
                and abs(a.dx - 250. / 96) < 1e-6 and abs(a.dz - 0.4) < 1e-9)
    return dict(how='measured P1 paper grid' if measured else 'scaled from P1, rough',
                dt_s=dt, dt_limit=limit, dt_dn_s=dt_dn, dt_cfl_s=dt_cfl, steps=steps,
                cells_per_task=cells_task, c_us=C_CELL * 1e6, s_per_step=s_step,
                wall_h=wall_h, nodes=nodes, core_h=ntask * wall_h,
                jobs=int(np.ceil(wall_h / max(a.job_hours - 0.25, 0.5))))


def find_microhh_tools():
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
    p.add_argument('--case', choices=sorted(REF), help='A1, P1, P2, V1 or V2')
    p.add_argument('--bc', choices=['flux', 'temp'], help='prescribed flux or temperature')
    p.add_argument('--all', action='store_true', help='every case with both --bc, in <outdir>/<case>_<bc>')
    p.add_argument('--res', choices=['paper', 'coarse'], default='paper',
                   help='grid/closure preset (default paper; see PRESETS); the options below override it')
    p.add_argument('--dx', type=float, default=None, help='horizontal grid spacing [m]')
    p.add_argument('--dz', type=float, default=None, help='vertical grid spacing [m]')
    p.add_argument('--lx', type=float, default=None,
                   help='INTERIOR length, downslope (x) [m]; the sponges are added on both sides')
    p.add_argument('--ly', type=float, default=None, help='INTERIOR width, cross-slope (y) [m]')
    p.add_argument('--lz', type=float, default=None, help='depth ABOVE the highest terrain [m]')
    p.add_argument('--grid-multiple', type=int, default=16,
                   help='itot and jtot are rounded up to a multiple of this, ktot to a multiple of 8 '
                        '(MPI decomposition)')
    p.add_argument('--fc', type=float, default=0., help='Coriolis parameter [1/s] (paper: 0; west Greenland: 1.34e-4)')
    p.add_argument('--alpha', type=float, default=None, help='override the slope angle [deg] (0: a flat twin)')
    p.add_argument('--n-tau', type=float, default=3., help='run length in oscillation periods (paper: 3)')
    p.add_argument('--dth', type=float, default=None, help='--bc temp: surface minus ambient thl [K]')
    p.add_argument('--init-jet', type=float, default=0.,
                   help='initial downslope jet maximum [m/s] (0: at rest, as the paper). '
                        'A Prandtl-shaped u_s = U e (n/n_j) exp(-n/n_j) along the slope, '
                        'n_j = max(z_j, first level). With --bc temp the run cannot start '
                        'from rest: MOST gives no flux at zero wind (see README)')
    p.add_argument('--z0h-model', default=None, help='[IB] z0h_model (default: yang2008 on ice, ratio for A1)')
    p.add_argument('--sgs', choices=['smag2', 'aniso', 'tke2'], default=None,
                   help='default: tke2 with Pr = 1 for --res paper, smag2 for --res coarse')
    p.add_argument('--no-init-tke', action='store_true',
                   help='tke2: start from the small uniform sgstke of the input profile, not the paper\'s')
    p.add_argument('--n-sponge', type=int, default=None,
                   help='lateral sponge width in cells (paper preset 5, coarse 10). The terrain-following '
                        'lateral boundary needs 2 n_sponge + 4 cells per MPI task: 10 costs 3.5x the wall '
                        'time of 5')
    p.add_argument('--tau-sponge', type=float, default=None,
                   help='lateral sponge relaxation time [s] (ini default 20); the cheap way to a '
                        'gentler sponge, unlike --n-sponge it costs nothing')
    p.add_argument('--no-lbc-zero-pgrad', action='store_true',
                   help='do not write [boundary_lateral] lbc_tf_zero_pgrad=true: the faces '
                        'then carry the band flux without the riser form drag, and the solve drives '
                        'the air above the jet with a uniform pressure gradient (0.38 m/s in 600 s, '
                        'blow-up at 2690 s in P1)')
    p.add_argument('--lbc-budget', type=int, default=20,
                   help='write lbc_tf_budget.txt every N steps (0 = off) [20]')
    p.add_argument('--stretch-above', type=float, default=None,
                   help='vertical grid: uniform dz up to this height ABOVE THE HIGHEST TERRAIN [m], '
                        'then dz grows by --stretch-ratio per level up to --dz-max (default: '
                        'uniform everywhere, as the paper). E.g. 25: ~25 %% fewer levels with lz 64, '
                        '~45 %% with lz 128, the same dt')
    p.add_argument('--stretch-ratio', type=float, default=1.04,
                   help='dz(k+1)/dz(k) above --stretch-above (default 1.04)')
    p.add_argument('--dz-max', type=float, default=None,
                   help='largest dz of the stretched grid [m] (default 4 dz)')
    p.add_argument('--dnmax', type=float, default=None,
                   help='[diff] dnmax, the diffusion-number cap that sets dt on the paper grid '
                        '(default 0.3, the recommended IB set-up; MicroHH default 0.4 = 1.33x larger dt; '
                        'RK3 limit ~0.63)')
    p.add_argument('--momentum', choices=('staircase', 'vr'), default='staircase',
                   help='IB momentum method: staircase (default; the MOST wall stress as a source term) '
                        'or vr (velocity reconstruction; needs a MicroHH build with it)')
    p.add_argument('--no-air-only', action='store_true',
                   help='staircase: [IB] sw_momentum_air_only=false (for older MicroHH versions)')
    p.add_argument('--reconstruct-dz', type=float, default=1.5,
                   help='[IB] reconstruct_dz: image point at max(c dz, d + dz) along the normal')
    p.add_argument('--no-reconstruct-stress', action='store_true',
                   help='[IB] reconstruct_stress=false: the SGS eddy viscosity above the R points '
                        'as it is (comparison only)')
    p.add_argument('--no-reconstruct', action='store_true',
                   help='[IB] sw_momentum_reconstruct=false: standard MicroHH ghost cells for the velocity '
                        '(comparison only)')
    p.add_argument('--no-flux-accum', action='store_true',
                   help='[IB] sw_flux_accum=false and no drag crosses (for older MicroHH versions)')
    p.add_argument('--evisc-cap', action='store_true',
                   help='[IB] evisc_solid=cap instead of cap_last (for older MicroHH versions)')
    p.add_argument('--job-hours', type=float, default=12.,
                   help='wall-time limit of one job [h] for the cost estimate (job.sh: 12)')
    p.add_argument('--solid-levels', type=int, default=5,
                   help='levels of solid below the LOWEST terrain point (the 2i5 stencil needs >= 4)')
    p.add_argument('--npx', type=int, default=None,
                   help='MPI tasks in x (default: the largest valid decomposition up to --max-tasks)')
    p.add_argument('--npy', type=int, default=None, help='MPI tasks in y (default: as --npx)')
    p.add_argument('--max-tasks', type=int, default=128, help='largest number of MPI tasks (128: one node)')
    p.add_argument('--no-fill-node', action='store_true',
                   help='do not widen the domain cross-slope (y) to use exactly --max-tasks '
                        'tasks; take the largest decomposition of the interior as it is')
    p.add_argument('--buf-frac', type=float, default=None,
                   help='top buffer depth as a fraction of lz (paper katabatic: 0.4 of lz = 64 m)')
    p.add_argument('--buf-sigma', type=float, default=None,
                   help='[buffer] sigma, the damping rate at the top [1/s] (paper katabatic: 0.03; '
                        'MicroHH default 0.00223)')
    p.add_argument('--tau-formula', action='store_true',
                   help='run length from tau_osc = 2 pi/(N sin alpha) instead of the paper\'s table value')
    p.add_argument('--out-every', type=float, default=None, help='cross/stats interval [s] (300)')
    p.add_argument('--outdir', default='.')
    a = p.parse_args()
    if not a.all and (a.case is None or a.bc is None):
        p.error('give --case and --bc, or --all')
    return a


def resolve(a, kind):
    """The preset for (--res, kind), overridden by any option given."""
    import copy
    a = copy.copy(a)
    for k, v in PRESETS[(a.res, kind)].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def make_run(a, mht, case, bc, outdir):
    r = REF[case]
    kind = 'katabatic' if r['glacier'] else 'anabatic'
    a = resolve(a, kind)
    os.makedirs(outdir, exist_ok=True)
    out = lambda f: os.path.join(outdir, f)

    alpha, Fs, N, z0, th0 = r['alpha_deg'], r['Fs'], r['N'], r['z0'], r['theta_a0_K']
    if a.alpha is not None:
        alpha = a.alpha
    tana = np.tan(np.deg2rad(alpha))
    V = np.sqrt(abs(Fs) / N)
    B = abs(Fs) / V
    H = V / N
    tau_formula = 2 * np.pi / (N * np.sin(np.deg2rad(max(alpha, 1e-3) if alpha > 0 else r['alpha_deg'])))
    # The paper's Table 1 tau_osc (P1: 6.6e3 s) is ~10 % longer than
    # 2 pi/(N sin alpha) with its own N and alpha (5958 s for P1); the runs
    # are "up to 3 tau_osc" with the table value, so that is the default.
    if a.tau_formula or a.alpha is not None or 'tau_osc_table_s' not in r:
        tau, tau_how = tau_formula, '2 pi/(N sin alpha)'
    else:
        tau, tau_how = float(r['tau_osc_table_s']), 'paper Table 1'
    gam = N**2 * th0 / GRAV
    F_thl = Fs * th0 / GRAV                       # K m/s, per unit surface area

    # The paper's domain is the interior; the sponges come on both sides.
    # Rounded up to a multiple of --grid-multiple for the MPI decomposition.
    gm = a.grid_multiple
    itot0 = int(np.ceil((a.lx / a.dx + 2*a.n_sponge) / gm - 1e-9)) * gm
    jtot0 = int(np.ceil((a.ly / a.dx + 2*a.n_sponge) / gm - 1e-9)) * gm
    # MPI decomposition: MicroHH's transposes need itot and jtot divisible
    # by npx and npy (grid.cxx; ktot by npx, done below); the terrain-following
    # lateral boundary needs the
    # source band of every edge on the edge rank: imax, jmax >= 2 n_sponge + 4.
    nmin = 2 * a.n_sponge + 4
    z_base = (a.solid_levels + 0.25) * a.dz

    def valid(it, jt, px, py):
        return (it % px == 0 and it % py == 0 and jt % px == 0 and jt % py == 0
                and it // px >= nmin and jt // py >= nmin)

    # Vertical grid. Default: uniform dz from the buried floor to lz above the
    # highest terrain (bit-identical to before). With --stretch-above S: the
    # same uniform dz up to S above the highest terrain, i.e. over the whole
    # slope, the jet and the cold pool, then dz_{k+1} = r dz_k up to dz_max.
    # The terrain, the IB stencils and the near-surface flow see only the
    # uniform part; the dt (set by DNUM in the jet) does not change.
    dz_max = a.dz_max if a.dz_max is not None else 4. * a.dz
    if a.stretch_above is not None:
        if a.stretch_above < 10. * a.dz:
            raise SystemExit(f'--stretch-above must be >= 10 dz = {10*a.dz:g} m (the IB stencils '
                             f'and the first levels above the highest terrain stay uniform)')
        if not (1. < a.stretch_ratio <= 1.1):
            raise SystemExit('--stretch-ratio must be in (1, 1.1]')
        if dz_max < a.dz:
            raise SystemExit('--dz-max must be >= dz')

    def uniform_z():
        return a.stretch_above is None or a.stretch_above >= a.lz

    def z_faces(it, cap, n_min=0):
        """Faces from the floor to at least lz above the highest terrain:
        uniform dz up to stretch_above above it, then dz_(k+1) = r dz_k up to cap."""
        top_terrain = z_base + it * a.dx * tana
        z_top = top_terrain + a.lz
        if uniform_z():
            n = max(int(np.ceil(z_top / a.dz - 1e-9)), n_min)
            return np.arange(n + 1) * a.dz
        n_u = int(np.ceil((top_terrain + a.stretch_above) / a.dz - 1e-9))
        f = list(np.arange(n_u + 1) * a.dz)
        d = a.dz
        while f[-1] < z_top - 1e-9 or len(f) - 1 < n_min:
            d = min(d * a.stretch_ratio, cap)
            f.append(f[-1] + d)
        return np.array(f)

    def ktot_of(it, px):
        kmul = int(np.lcm(8, px))
        return int(np.ceil((len(z_faces(it, dz_max)) - 1) / kmul - 1e-9)) * kmul

    def ktot_uniform(it, px):
        kmul = int(np.lcm(8, px))
        return int(np.ceil((a.lz + z_base + it * a.dx * tana) / a.dz / kmul - 1e-9)) * kmul

    def z_grid(it, px):
        """Full-level heights and zsize for ktot_of(it, px) levels. Stretched:
        the largest dz is lowered (bisection) until the levels needed to reach
        lz above the terrain are exactly ktot, so rounding ktot up to the MPI
        multiple does not deepen the domain (and the undamped layer)."""
        kt = ktot_of(it, px)
        if uniform_z():
            return (np.arange(kt) + 0.5) * a.dz, kt * a.dz
        lo, hi = a.dz, dz_max
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if len(z_faces(it, mid)) - 1 >= kt:
                lo = mid
            else:
                hi = mid
        f = z_faces(it, lo, n_min=kt)[:kt + 1]
        return 0.5 * (f[1:] + f[:-1]), f[-1]

    def best_decomp(it, jt, exact=None):
        c = [(px * py, -abs(px - py), px, py)
             for px in range(1, a.max_tasks + 1) for py in range(1, a.max_tasks + 1)
             if (px * py == exact if exact else px * py <= a.max_tasks) and valid(it, jt, px, py)]
        return max(c) if c else None

    itot, jtot = itot0, jtot0
    if a.npx is not None and a.npy is not None:
        npx, npy = a.npx, a.npy
        if not valid(itot, jtot, npx, npy):
            raise SystemExit(f'--npx {npx} --npy {npy} does not fit itot = {itot}, jtot = {jtot} '
                             f'(divisibility, and imax/jmax >= 2 n_sponge + 4 = {nmin})')
        fill_how = 'given'
    else:
        _, _, npx, npy = best_decomp(itot, jtot)
        fill_how = 'largest decomposition of the interior'
        if not a.no_fill_node and npx * npy < a.max_tasks:
            # Use the whole node: widen the domain until exactly max_tasks
            # tasks fit, with the fewest cells, if that keeps the cells per
            # task (the wall time) at most what the interior layout has. Widening across the slope (y)
            # costs no extra levels; along it (x) every cell adds dx tan(alpha)
            # of drop, so the search over both normally ends up in y.
            best = None
            for it in range(itot0, 4 * itot0 + 1, gm):
                for jt in range(jtot0, 4 * jtot0 + 1, gm):
                    d = best_decomp(it, jt, exact=a.max_tasks)
                    if d is None:
                        continue
                    cells = it * jt * ktot_of(it, d[2])
                    if best is None or cells < best[0]:
                        best = (cells, it, jt, d[2], d[3])
            # Only if it costs no wall time: the cells per task must not grow
            # (else the node is better left part-idle, e.g. the coarse preset).
            load0 = itot0 * jtot0 * ktot_of(itot0, npx) / (npx * npy)
            if best is not None and best[0] / a.max_tasks <= load0 * (1 + 1e-9):
                _, itot, jtot, npx, npy = best
                fill_how = (f'widened from {itot0} x {jtot0} so that {a.max_tasks} tasks fit '
                            f'(imax, jmax >= {nmin})')
    if itot > itot0:
        print(f'WARNING: the slope (x) was lengthened from {itot0} to {itot} cells to fit '
              f'{npx * npy} tasks with imax >= {nmin}: longer slope, more drop and levels. '
              f'A smaller --n-sponge, or --no-fill-node, keeps the paper length.')
    lx_tot = itot * a.dx
    ly_tot = jtot * a.dx
    lx_in = lx_tot - 2 * a.n_sponge * a.dx
    ly_in = ly_tot - 2 * a.n_sponge * a.dx

    # Terrain: a plane falling towards +x, the lowest point solid_levels*dz
    # above the buried floor, a quarter cell above a half level so no surface
    # falls exactly on a cell face.
    drop = lx_tot * tana
    zs_fun = lambda X, Y: z_base + tana * (lx_tot - X) + 0.*Y
    z_mid = z_base + 0.5 * drop
    ktot = ktot_of(itot, npx)
    z, zsize = z_grid(itot, npx)
    stretched = not np.allclose(np.diff(z), a.dz)

    x = (np.arange(itot) + 0.5) * a.dx
    y = (np.arange(jtot) + 0.5) * a.dx
    X, Y = np.meshgrid(x, y)
    dem = zs_fun(X, Y)
    dem.astype(np.float64).tofile(out('dem.0000000'))
    k_dem = np.searchsorted(z, dem[0], side='right')
    n_steps = int(k_dem.max() - k_dem.min())

    thl_a = lambda z: th0 + gam * (z - z_mid)

    # Surface condition.
    z1 = 0.5 * a.dz
    if bc == 'temp':
        if a.dth is not None:
            dth, how = a.dth, '--dth'
        elif r['glacier']:
            dth, how = T_MELT - th0, 'melting surface at the domain centre'
        else:
            z0h = z0 / 10.
            chn = KAPPA**2 / (np.log(z1 / z0) * np.log(z1 / z0h))
            dth = F_thl / (chn * 2.6 * V)
            how = f'neutral bulk at z1 = {z1:g} m, C_hn = {chn:.4f}, U = 2.6 V'
        sbot = thl_a(dem) + dth
    else:
        dth, how = None, 'prescribed flux'
        sbot = np.full_like(dem, F_thl)
    sbot.astype(np.float64).tofile(out('thl_sbot.0000000'))
    np.zeros_like(dem).tofile(out('qt_sbot.0000000'))

    # Initial state: the resting environment, optionally with a downslope
    # jet (--init-jet). The profile is 1-D in true height, so the jet is put
    # into the 3-D fields after init by the u.0000000/w.0000000 rewrite
    # below (heights above the local surface).
    with nc.Dataset(out(f'{CASE}_input.nc'), 'w') as f:
        f.createDimension('z', ktot)
        f.createVariable('z', 'f8', ('z',))[:] = z
        g = f.createGroup('init')
        # sgstke: MicroHH's floor (Constants::sgstke_min = 1e-7 m2/s2, i.e.
        # the paper's "zero"); axelsen2009_init.py puts the paper's
        # surface-relative profile in after init.
        for name, val in [('thl', thl_a(z)), ('qt', 0*z), ('sgstke', 0*z + SGSTKE_MIN),
                          ('u', 0*z), ('v', 0*z), ('u_geo', 0*z), ('v_geo', 0*z)]:
            g.createVariable(name, 'f8', ('z',))[:] = val
    if a.init_jet > 0.:
        nj = max(r['les'].get('zj_over_H', 0.15) * H, z1)
        jet = dict(U=a.init_jet, n_j=nj, alpha_deg=alpha)
        with open(out('init_jet.json'), 'w') as f:
            json.dump(jet, f)
    elif os.path.exists(out('init_jet.json')):
        os.remove(out('init_jet.json'))

    # Initial SFS TKE (tke2), applied by axelsen2009_init.py.
    init_tke = None
    if a.sgs == 'tke2' and not a.no_init_tke:
        if kind == 'katabatic':
            init_tke = dict(kind='beare2006', e0=0.4, depth=25.)
        else:
            init_tke = dict(kind='schumann1990', e0=0.5 * V**2, decay=20. * np.sin(np.deg2rad(alpha)) / H)

    ini = mht.Read_namelist(os.path.join(HERE, f'{CASE}.ini.base'))
    ini['master']['npx'] = npx
    ini['master']['npy'] = npy
    ini['grid']['itot'] = itot
    ini['grid']['jtot'] = jtot
    ini['grid']['ktot'] = ktot
    ini['grid']['xsize'] = lx_tot
    ini['grid']['ysize'] = ly_tot
    ini['grid']['zsize'] = zsize
    if a.fc != 0.:
        ini['force']['swlspres'] = 'geo'
        ini['force']['fc'] = a.fc
    ini['boundary']['z0m'] = z0
    ini['boundary']['z0h'] = z0 / 10.
    ini['boundary']['sbot[thl]'] = thl_a(z_base)
    ini['boundary']['stop[thl]'] = gam
    ini['boundary_lateral']['n_sponge'] = a.n_sponge
    if a.tau_sponge is not None:
        ini['boundary_lateral']['tau_sponge'] = a.tau_sponge
    ini['boundary_lateral']['lbc_tf_gradient[thl]'] = gam
    # Zero mean pressure gradient instead of a fixed face flux.
    if not a.no_lbc_zero_pgrad:
        ini['boundary_lateral']['lbc_tf_zero_pgrad'] = 'true'
    if a.lbc_budget > 0:
        ini['boundary_lateral']['lbc_tf_budget_every'] = a.lbc_budget
    ini['IB']['z0m'] = z0
    ini['IB']['z0h'] = z0 / 10.
    ini['IB']['z0h_model'] = a.z0h_model or ('yang2008' if r['glacier'] else 'ratio')
    tf_heights = [h for h in TF_HEIGHTS['coarse' if a.res == 'coarse' else kind] if h < a.lz]
    ini['IB']['tf_cross_heights'] = ','.join(f'{h:g}' for h in tf_heights)
    if bc == 'flux':
        ini['IB']['sbcbot'] = 'flux'
        ini['IB']['sbot[thl]'] = F_thl
    else:
        ini['IB']['sbcbot'] = 'dirichlet'
        ini['IB']['sbot[thl]'] = thl_a(z_mid) + dth
    ini['IB']['sbot[qt]'] = 0.
    # IB momentum (--momentum). The .ini.base holds the staircase block;
    # 'vr' replaces it by the velocity reconstruction.
    # sw_flux_accum: the IB surface fluxes and the drag as interval means, with
    # the drag split into form drag and MOST skin friction.
    if a.momentum == 'vr':
        for k in STAIRCASE_KEYS:
            ini['IB'].pop(k, None)
        ini['IB']['sw_momentum_reconstruct'] = 'false' if a.no_reconstruct else 'true'
        ini['IB']['reconstruct_profile'] = 'most'
        ini['IB']['reconstruct_dz'] = a.reconstruct_dz
        ini['IB']['reconstruct_diag_every'] = 300
        ini['IB']['reconstruct_stress'] = 'false' if a.no_reconstruct_stress else 'true'
    else:
        if a.no_reconstruct or a.no_reconstruct_stress:
            raise SystemExit('--no-reconstruct / --no-reconstruct-stress need --momentum vr')
        if a.no_air_only:
            ini['IB']['sw_momentum_air_only'] = 'false'
        elif ini['advec']['swadvec'] != '2i5':
            raise SystemExit('[IB] sw_momentum_air_only needs [advec] swadvec=2i5 (or --no-air-only)')
    if a.no_flux_accum:
        ini['IB']['sw_flux_accum'] = False
    if a.evisc_cap:
        ini['IB']['evisc_solid'] = 'cap'
    if a.dnmax is not None:
        if not (0. < a.dnmax <= 0.5):
            raise SystemExit('--dnmax must be in (0, 0.5] (RK3 explicit-diffusion limit ~0.63)')
        ini['diff']['dnmax'] = a.dnmax
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
        # Pr_SFS = 1, as the paper: K_h = (ch1 + ch2 l/Delta) K_m
        ini['diff']['ch1'] = 1.
        ini['diff']['ch2'] = 0.
    ini['buffer']['zstart'] = zsize - a.buf_frac * a.lz
    ini['buffer']['sigma'] = a.buf_sigma
    # Random perturbations near the surface (true height: the whole slope
    # up to min(50 m, lz/2) above its highest point); amplitudes scaled to
    # the paper's velocity and buoyancy scales.
    ini['fields']['rndz'] = z_base + drop + min(50., 0.5 * a.lz)
    ini['fields']['rndamp[thl]'] = float('%.2g' % (0.1 * B * th0 / GRAV))
    ini['fields']['rndamp[u]'] = float('%.2g' % (0.1 * V))
    ini['fields']['rndamp[v]'] = float('%.2g' % (0.1 * V))
    ini['fields']['rndamp[w]'] = float('%.2g' % (0.1 * V))
    endtime = int(np.ceil(a.n_tau * tau / 600.) * 600.)
    ini['time']['endtime'] = endtime
    ini['stats']['sampletime'] = a.out_every
    ini['cross']['sampletime'] = a.out_every
    # xz crosses: 4 planes inside the interior (between the y sponges), at
    # cell centres; the analysis builds the slope-normal profiles from them.
    j_lo, j_hi = a.n_sponge + 2, jtot - a.n_sponge - 3
    j_xz = np.unique(np.round(np.linspace(j_lo, j_hi, 4)).astype(int))
    ini['cross']['xz'] = ','.join(f'{(j + 0.5) * a.dx:.6g}' for j in j_xz)
    crosslist = ['u', 'w', 'thl', 'u_tf', 'v_tf', 'w_tf', 'thl_tf',
                 'thlfluxbot_ib', 'thl_sbot_ib', 'ustar_ib', 'obuk_ib', 'ch_ib']
    if not a.no_flux_accum:
        crosslist += ['taux_form_ib', 'tauy_form_ib', 'taux_skin_ib', 'tauy_skin_ib']
    ini['cross']['crosslist'] = ','.join(crosslist)
    ini.save(out(f'{CASE}.ini'), allow_overwrite=True)
    # cross_to_nc.sh + cross_to_nc.py + microhh_tools.py (time means over 10 min)
    install_cross_to_nc(outdir, CASE, window=600., quiet=True)

    cost = cost_estimate(a, kind, V, alpha, itot, jtot, ktot, npx, npy, endtime,
                         float(ini['diff']['dnmax']), float(ini['advec']['cflmax']))

    pars = dict(case=case, bc=bc, alpha_deg=alpha, Fs=Fs, N=N, z0=z0, theta_a0=th0,
                V=V, B=B, H=H, tau_osc=tau, gamma=gam, F_thl=F_thl, dth_s=dth,
                dth_how=how, itot=itot, jtot=jtot, ktot=ktot, dx=a.dx, dz=a.dz,
                lx=lx_tot, ly=ly_tot, lx_interior=lx_in, ly_interior=ly_in, lz=a.lz,
                buf_frac=a.buf_frac, buffer_depth=a.buf_frac * a.lz, buf_sigma=a.buf_sigma,
                tau_how=tau_how,
                tau_osc_formula=tau_formula, decomposition=fill_how,
                zsize=zsize, z_base=z_base, z_mid=z_mid, res=a.res,
                drop=drop, n_steps=n_steps, endtime=endtime, fc=a.fc,
                n_sponge=a.n_sponge, tau_sponge=a.tau_sponge, sgs=a.sgs, tf_heights=tf_heights,
                xz_j=[int(j) for j in j_xz], init_tke=init_tke,
                out_every=a.out_every, glacier=r['glacier'], init_jet=a.init_jet,
                npx=npx, npy=npy,
                stretch_above=a.stretch_above if stretched else None,
                stretch_ratio=a.stretch_ratio if stretched else None,
                dz_max=dz_max if stretched else None,
                momentum=a.momentum,
                momentum_reconstruct=(a.momentum == 'vr') and not a.no_reconstruct,
                reconstruct_dz=a.reconstruct_dz,
                reconstruct_stress=(a.momentum == 'vr') and not a.no_reconstruct_stress,
                momentum_air_only=(a.momentum == 'staircase') and not a.no_air_only,
                flux_accum=not a.no_flux_accum,
                dnmax=float(ini['diff']['dnmax']), cost_estimate=cost)
    if stretched:
        # Non-uniform levels: axelsen2009_init.py and _analysis.py read them here.
        pars['z_full'] = [float(v) for v in z]
    with open(out('case_parameters.json'), 'w') as f:
        json.dump(pars, f, indent=1)

    print(f'{case} {bc:4s} [{a.res}, {a.sgs}]: dx {a.dx:.3g} m, dz {a.dz:.3g} m, '
          f'{lx_tot:.0f} x {ly_tot:.0f} m ({lx_in:.0f} x {ly_in:.0f} m inside the sponges)')
    print(f'      MPI {npx} x {npy} = {npx*npy} tasks (subdomain {itot//npx} x {jtot//npy}; {fill_how})')
    print(f'      lz = {a.lz:g} m above the highest terrain, top buffer {a.buf_frac*a.lz:g} m '
          f'(zstart {zsize - a.buf_frac*a.lz:g} m), sigma {a.buf_sigma:g} 1/s '
          f'(N = {N:g} 1/s); {(1 - a.buf_frac)*a.lz:g} m undamped above the highest terrain')
    if stretched:
        dzs = np.diff(z)
        print(f'      stretched vertical grid: dz = {a.dz:g} m up to {a.stretch_above:g} m above the '
              f'highest terrain (z = {z_base + drop + a.stretch_above:.1f} m), then x{a.stretch_ratio:g} '
              f'per level to {dzs.max():.2f} m; zsize {zsize:.1f} m '
              f'({ktot_uniform(itot, npx)} levels if uniform)')
    print(f'      {itot} x {jtot} x {ktot} = {itot*jtot*ktot/1e6:.2f} M cells, '
          f'alpha {alpha:g} deg, {n_steps} steps, endtime {endtime} s (= {a.n_tau:g} tau_osc, '
          f'tau = {tau:.0f} s from {tau_how}; 2 pi/(N sin alpha) = {tau_formula:.0f} s)')
    print(f'      V = {V:.3f} m/s, B = {B*1e3:.2f}e-3 m/s2, H = {H:.1f} m '
          f'(dz/H = {a.dz/H:.2f}, first level at z/H = {z1/H:.2f}); G = {gam*1e3:.2f} K/km')
    if a.init_jet > 0.:
        print(f'      initial downslope jet {a.init_jet:g} m/s at {nj:.1f} m')
    if init_tke:
        print(f'      initial SFS TKE: {init_tke["kind"]} profile above the local surface')
    if a.init_jet > 0. or init_tke:
        print(f'      after "microhh init" run: python3 {os.path.join(HERE, "axelsen2009_init.py")} --rundir {outdir}')
    if bc == 'flux':
        print(f'      surface flux {F_thl:+.4f} K m/s (~{1.0*1005*F_thl:+.0f} W/m2 at rho = 1)')
    else:
        print(f'      surface thl - ambient = {dth:+.2f} K ({how})')
    zj = r['les'].get('zj_over_H', 0.15) * H
    if z1 > zj:
        print(f'      NOTE: the paper\'s jet (z_j = {zj:.1f} m) is BELOW the first level ({z1:g} m): '
              f'the coarse grid cannot resolve it; this tests the coarse set-up, it does not '
              f'reproduce the paper')
    print(f'      IB: momentum={a.momentum}, sw_flux_accum={not a.no_flux_accum} '
          f'evisc_solid={"cap" if a.evisc_cap else "cap_last"}')
    print(f'      LBC: lbc_tf_zero_pgrad={not a.no_lbc_zero_pgrad}, lbc_tf_budget_every={a.lbc_budget}')
    c = cost
    print(f'      COST ({c["how"]}): dt ~ {c["dt_s"]:.2f} s ({c["dt_limit"]}), '
          f'{c["steps"]/1e3:.0f} k steps x {c["s_per_step"]:.2f} s/step '
          f'({c["cells_per_task"]/1e3:.0f} k cells/task x {c["c_us"]:g} us) = '
          f'{c["wall_h"]:.1f} h wall on {c["nodes"]} node(s) of 128 cores, ~{c["core_h"]:,.0f} core-h; '
          f'{c["jobs"]} job(s) of {a.job_hours:g} h')
    if c['jobs'] > 1:
        print(f'      -> more than one {a.job_hours:g}-h job: chain restarts (job.sh with RESTART_TIME=T, README), '
              f'or sbatch --time with a margin if the queue allows it')
    if a.n_sponge > 5 and a.res == 'paper' and kind == 'katabatic':
        print(f'      NOTE: n_sponge {a.n_sponge} needs >= {nmin} x {nmin} columns per MPI task '
              f'(terrain-following boundary); n_sponge 5 needs 14 x 14: {(nmin/14.)**2:.1f}x the '
              f'columns per task. '
              f'For a gentler sponge, --tau-sponge costs nothing.')


def main():
    a = parse()
    mht = find_microhh_tools()
    if a.all:
        for case in sorted(REF):
            for bc in ('flux', 'temp'):
                make_run(a, mht, case, bc, os.path.join(a.outdir, f'{case}_{bc}'))
    else:
        make_run(a, mht, a.case, a.bc, a.outdir)


if __name__ == '__main__':
    main()
