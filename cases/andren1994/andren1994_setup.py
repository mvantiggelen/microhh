#!/usr/bin/env python3
"""
Generate the Andren et al. (1994) neutral Ekman-layer test case for MicroHH.

    Andren, A., Brown, A. R., Graf, J., Mason, P. J., Moeng, C.-H.,
    Nieuwstadt, F. T. M. and Schumann, U. (1994). Large-eddy simulation of
    a neutrally stratified boundary layer: A comparison of four computer
    codes. Q. J. R. Meteorol. Soc. 120, 1457-1484.
    https://doi.org/10.1002/qj.49712052003

Usage (generate a run directory):

    python3 andren1994_setup.py --outdir runs/paper                 # paper grid 40^3
    python3 andren1994_setup.py --res small --endtime 3600 --outdir runs/smoke
    python3 andren1994_setup.py --res fine  --outdir runs/fine      # 80^3, sensitivity

Flat immersed-boundary variant (README, section "The flat-IB variant"):

    python3 andren1994_setup.py --surface ib --outdir runs/ib       # IB floor, S5 [IB] block
    python3 andren1994_setup.py --surface ib --ib-set sw_wall_sample=false --outdir runs/ib_nosample
    python3 andren1994_setup.py --advec 2i5 --utrans 0 --outdir runs/flat_2i5   # its flat twin

With --surface ib the domain floor is buried: a constant DEM (dem.0000000)
IB_SOLID_LEVELS*dz + IB_OFFSET*dz above it is the surface, ktot and zsize grow
by IB_SOLID_LEVELS levels so that the air above the surface keeps the
paper's 1500 m (to IB_OFFSET*dz) and dz = 37.5 m. The [IB] block is the one
of the S5 production runs (IB_SWITCHES; IB momentum by --momentum: staircase,
the case026 method and the default since 20 Oct 2026, or vr, the velocity
reconstruction of patches 59-60),
with z0m = z0h = 0.1 m and the paper's scalar flux as a prescribed IB flux
(patch 35). Moist Boussinesq thermo with qt = 0 and uniform thl (the IB
wall model needs it), 2i5 advection (as the production runs) and
utrans = 0 (the IB ignores [grid] utrans, see README).

The run directory gets: andren1994.ini, andren1994_input.nc,
case_parameters.json, job_atos.sh and a copy of this script, the plot
script and the reference data (so the folder is self-contained).

Second mode, run by job_atos.sh between "microhh init" and "microhh run":

    python3 andren1994_setup.py --perturb [--rundir .]

adds the paper's initial random velocity perturbations (appendix: uniform
random numbers in [-0.5, 0.5], scaled to the TKE profile q^2/2 of Table A.1)
to u.0000000, v.0000000 and w.0000000.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

# The cross-section converter (cross_to_nc.sh, cross_to_nc.py, microhh_tools.py)
# is copied into every run folder by write_cross_to_nc.py, next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_cross_to_nc import install as install_cross_to_nc  # noqa: E402


CASE = 'andren1994'
THREF = 300.     # K, uniform potential temperature (neutral, buoyancy = 0)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# The paper's set-up (section 3, Tables 1, 2 and A.1)
# ---------------------------------------------------------------------------
PAPER = dict(
    xsize=4000., ysize=2000., zsize=1500.,  # m (section 3, Table 2)
    ug=10., vg=0.,                          # geostrophic wind, m/s
    fc=1.e-4,                               # vertical Coriolis parameter, 1/s
    z0=0.1,                                 # roughness length, m
    sflux=1.e-3,                            # passive scalar surface flux, kg m-2 s-1 (rho = 1)
    endtime=1.e5,                           # 10/f
    t_avg=(7.e4, 1.e5),                     # statistics over the last 3/f
)

# Table A.1: initial profiles at the 40 cell centres of the paper grid.
# z (m), <u> (m/s), <v> (m/s), q^2/2 (m2/s2)
TABLE_A1 = np.array([
    [18.75, 4.44, 2.18, 0.365], [56.25, 5.92, 2.67, 0.295],
    [93.75, 6.91, 2.83, 0.245], [131.25, 7.73, 2.84, 0.205],
    [168.75, 8.43, 2.75, 0.175], [206.25, 9.02, 2.57, 0.145],
    [243.75, 9.52, 2.34, 0.120], [281.25, 9.93, 2.06, 0.100],
    [318.75, 10.25, 1.75, 0.085], [356.25, 10.47, 1.44, 0.070],
    [393.75, 10.62, 1.12, 0.055], [431.25, 10.70, 0.82, 0.045],
    [468.75, 10.71, 0.55, 0.035], [506.25, 10.67, 0.31, 0.025],
    [543.75, 10.59, 0.12, 0.020], [581.25, 10.48, -0.02, 0.015],
    [618.75, 10.36, -0.11, 0.010], [656.25, 10.24, -0.16, 0.010],
    [693.75, 10.13, -0.17, 0.005], [731.25, 10.04, -0.15, 0.005],
    [768.75, 9.99, -0.11, 0.005], [806.25, 9.96, -0.06, 0.000],
    [843.75, 9.95, -0.02, 0.000], [881.25, 9.96, 0.01, 0.000],
    [918.75, 9.98, 0.02, 0.000], [956.25, 9.99, 0.02, 0.000],
    [993.75, 10.00, 0.02, 0.000], [1031.25, 9.99, 0.02, 0.000],
    [1068.75, 9.99, 0.02, 0.000], [1106.25, 9.99, 0.01, 0.000],
    [1143.75, 10.00, 0.01, 0.000], [1181.25, 10.00, 0.01, 0.000],
    [1218.75, 10.00, 0.01, 0.000], [1256.25, 10.00, 0.00, 0.000],
    [1293.75, 10.00, 0.00, 0.000], [1331.25, 10.00, 0.00, 0.000],
    [1368.75, 10.00, 0.00, 0.000], [1406.25, 10.00, 0.00, 0.000],
    [1443.75, 10.00, 0.00, 0.000], [1481.25, 10.00, 0.00, 0.000]])

# Grids: same domain, different resolution.
RES = {
    'paper': (40, 40, 40),   # dx = 100 m, dy = 50 m, dz = 37.5 m (Table 2)
    'small': (32, 32, 32),   # smoke test only
    'fine':  (80, 80, 80),   # 2x the paper resolution, sensitivity run
}


# ---------------------------------------------------------------------------
# Flat immersed-boundary variant (--surface ib)
# ---------------------------------------------------------------------------
# Solid levels under the IB surface: the 2i5 stencil reaches three cells into
# the terrain plus the ghost layer (racmo_to_openbc.py IB_MIN_SOLID_LEVELS).
IB_SOLID_LEVELS = 4
# Height of the surface above the half level IB_SOLID_LEVELS*dz, in units of
# dz. Not 0: a surface exactly on a cell face puts the image point of every
# scalar ghost cell exactly on the first air cell centre, and the IDW weight
# sqrt((dmax - d)/(dmax d)) (immersed_boundary.cxx precalculate_idw) is then
# inf/inf = NaN. A small offset keeps the first air level dz/2 - IB_OFFSET*dz
# above the surface, i.e. the same as the flat run to 0.1 %.
IB_OFFSET = 1.e-3

# The [IB] block of the S5 production runs (racmo_to_openbc.py IB_* constants
# and its [IB] writer), with the paper's z0 and scalar flux. The momentum keys
# come from MOMENTUM[--momentum] and are inserted where IB_SWITCHES has
# 'MOMENTUM': 'staircase' (default) is the case026 method and needs the
# staircase tree (patches/reapply_all.sh <tree> staircase); 'vr' is the
# velocity reconstruction and needs the vr tree. Each entry:
# (key, value, one-line comment). Override any of them with --ib-set key=value.
MOMENTUM = {
    'staircase': [
        ('sw_momentum_flux', 'true', 'momentum: MOST stress as a source term on u, v (patch 6)'),
        ('sw_momentum_flux_vertical', 'true', 'risers too (S5; forced off by sw_wall_slope, none on a flat DEM)'),
        ('sw_strain_most', 'true', 'evisc in floor cells rescaled so that K dU/dz = u*^2 (patch 7)'),
        ('sw_strain_most_vertical', 'false', ''),
        ('strain_most_min', '0.05', 'floor on that rescaling factor'),
        ('sw_impermeable', 'false', ''),
        ('sw_wall_kinematic', 'true', 'floor-face w = u.grad(eta), terrain out of the pressure solve (patch 32)'),
        ('sw_momentum_air_only', 'true', 'momentum exchange only through MOST and pressure (patches 49, 51; needs 2i5)'),
    ],
    'vr': [
        ('sw_momentum_reconstruct', 'true', 'momentum: MOST profile along the surface normal at the R points, '
         '0 below the surface (patch 59)'),
        ('reconstruct_profile', 'most', ''),
        ('reconstruct_dz', '1.5', 'image point at max(1.5 dz, d + dz) along the normal'),
        ('reconstruct_diag_every', '300', 'log slip, leak and IB force every 300 RK substeps'),
        ('reconstruct_stress', 'true', 'at least the MOST eddy viscosity above the R points (patch 60)'),
    ],
}
IB_SWITCHES = [
    ('sw_immersed_boundary', 'dem', 'DEM immersed boundary (dem.0000000)'),
    ('sw_dem_halo_replicate', 'true', 'DEM halo copied, not cyclic (no effect on a constant DEM)'),
    ('n_idw_points', '5', 'IDW points of the ghost-cell interpolation'),
    ('sw_wall_model', 'most', 'Monin-Obukhov wall model on the IB floor faces'),
    ('z0m', f"{PAPER['z0']:g}", 'the paper\'s roughness length'),
    ('z0h', f"{PAPER['z0']:g}", ''),
    ('sw_scalar_flux', 'true', 'scalars: SGS flux through the wall face replaced by the wall-model flux (patch 5)'),
    ('sw_wall_stability_vertical', 'false', ''),
    ('sw_scalar_flux_vertical', 'false', ''),
    ('sbcbot', 'flux', 'prescribed kinematic flux per unit surface area (patch 35)'),
    ('sbot[thl]', '0.', 'neutral'),
    ('sbot[qt]', '0.', ''),
    ('sbot[s]', f"{PAPER['sflux']:g}", 'the paper\'s passive-scalar flux'),
    'MOMENTUM',
    ('evisc_solid', 'cap_last', 'terrain evisc capped after the wall damping (patch 52)'),
    ('dn_probe_every', '20000', 'repeat the diffusion-limit probe'),
    ('diag_z_mom', '10', 'heights of the terrain-following surface diagnostics (m)'),
    ('diag_z_scalar', '2', ''),
    ('sw_wall_slope', 'true', 'MOST with the wall-normal distance to the sub-grid surface (patch 30)'),
    ('wall_dn_min', '0.1', 'lower bound of that distance, in dz'),
    ('sw_blank_solid', 'true', 'solid non-ghost cells reset every substep (patch 4)'),
    ('sw_blank_solid_tend', 'true', 'and their tendencies zeroed (patch 39)'),
    ('sw_advec_wall', 'true', 'air-only scalar advection next to the wall (patch 10)'),
    ('sw_advec_wall_conserve', 'true', 'exactly conservative across wall faces (patch 38)'),
    ('sw_mason_wall', 'true', 'Mason damping with the height above the local surface (patch 42)'),
    ('sw_wall_sample', 'true', 'wall model evaluated at an image point wall_sample_dz*dz off the surface (patch 45)'),
    ('wall_sample_dz', '1.5', ''),
    ('sw_flux_accum', 'true', 'IB fluxes and drag as interval means (patch 48, output only)'),
]
IB_CROSS = ['ustar_ib', 'obuk_ib', 'dn_ib', 'u_ib1', 'v_ib1', 'taux_skin_ib', 'tauy_skin_ib',
            'taux_form_ib', 'tauy_form_ib', 's_fluxbot_ib']


def ib_switches(overrides, momentum='staircase'):
    """IB_SWITCHES with the momentum block of `momentum` and --ib-set
    key=value overrides (a new key is appended)."""
    sw = []
    for x in IB_SWITCHES:
        sw += [list(y) for y in MOMENTUM[momentum]] if x == 'MOMENTUM' else [list(x)]
    for item in overrides or []:
        if '=' not in item:
            sys.exit(f'--ib-set {item}: expected key=value')
        key, val = item.split('=', 1)
        for x in sw:
            if x[0] == key:
                x[1], x[2] = val, (x[2] + ' ' if x[2] else '') + '[--ib-set]'
                break
        else:
            sw.append([key, val, '[--ib-set]'])
    return sw


def geometry(res, surface, ib_solid_levels=IB_SOLID_LEVELS, ib_offset=IB_OFFSET):
    """Grid and surface: the paper's 40 levels of 37.5 m above the surface, and
    for the IB variant ib_solid_levels more below it."""
    itot, jtot, ktot_air = RES[res]
    dz = PAPER['zsize'] / ktot_air
    ns = ib_solid_levels if surface == 'ib' else 0
    off = ib_offset if surface == 'ib' else 0.
    ktot = ktot_air + ns
    return dict(itot=itot, jtot=jtot, ktot=ktot, ktot_air=ktot_air, dz=dz, zsize=ktot * dz,
                n_solid=ns, ib_offset=off, z_surface=(ns + off) * dz)


def valid_decomp(itot, jtot, ktot, npx, npy):
    """MicroHH's checks in grid.cxx (Grid<TF>::init)."""
    return (itot % npx == 0 and itot % npy == 0 and jtot % npy == 0
            and (npy == 1 or jtot % npx == 0) and ktot % npx == 0)


def auto_decomp(itot, jtot, ktot, maxtasks=128, mincols=8):
    """Largest npx*npy <= maxtasks with at least mincols x mincols columns per task."""
    best = (1, 1)
    for npx in range(1, maxtasks + 1):
        for npy in range(1, maxtasks // npx + 1):
            if not valid_decomp(itot, jtot, ktot, npx, npy):
                continue
            if itot // npx < mincols or jtot // npy < mincols:
                continue
            n, nb = npx * npy, best[0] * best[1]
            # prefer more tasks, then the squarest sub-domain
            if n > nb or (n == nb and abs(itot / npx - jtot / npy) < abs(itot / best[0] - jtot / best[1])):
                best = (npx, npy)
    return best


def write_ini(fn, g, npx, npy, a):
    ib = g['n_solid'] > 0
    zc = (np.arange(g['ktot']) + 0.5) * g['dz']                     # model full levels
    zc = zc[zc > g['z_surface']]                                     # the air levels, for [cross] xy
    if ib:
        thermo = f"""\
# Neutral: moist Boussinesq with a uniform thl = {THREF:g} K and qt = 0 and zero
# fluxes of both, so the buoyancy is exactly zero (the reference thv comes
# is thvref0). The IB wall model needs the moist thermo
# class (dry has no thv base state, see the Maass hill case).
[thermo]
swthermo=moist
swbasestate=boussinesq
thvref0={THREF:g}
pbot=100000.
swupdatebasestate=0
"""
        boundary = f"""\
# The domain floor is buried {g['n_solid']} levels under the IB surface; its own
# surface model only sees terrain cells (blanked every substep).
[boundary]
mbcbot=noslip
mbctop=freeslip
sbcbot=flux
sbctop=neumann
sbot[s]=0.
stop[s]=0.
sbot[thl]=0.
stop[thl]=0.
sbot[qt]=0.
stop[qt]=0.
swboundary=surface
z0m={PAPER['z0']}
z0h={PAPER['z0']}
"""
        lines = ['# Flat immersed-boundary surface: a constant DEM at z = %.6g m, %d levels'
                 % (g['z_surface'], g['n_solid']),
                 '# + %g dz above the buried floor. The switches are the S5 production'
                 % g['ib_offset'],
                 '# block (racmo_to_openbc.py), with the paper\'s z0 and scalar flux.',
                 '[IB]']
        for key, val, com in ib_switches(a.ib_set, a.momentum):
            if com:
                lines.append(f'# {com}')
            lines.append(f'{key}={val}')
        ib_block = '\n'.join(lines) + '\n'
        masklist = 'masklist=ib\n'
        crosslist = 'u,v,w,s,' + ','.join(IB_CROSS)
    else:
        thermo = f"""\
# Neutral: a dry thermo with a uniform theta (= thref0) and zero theta flux,
# so the buoyancy is exactly zero. swthermo=0 cannot be used with this fork:
# Immersed_boundary::exec_wall_model fetches the base-state exner from the
# thermo every step, even with [IB] off, and Thermo_disabled throws
# "get_basestate_vector not implemented". The Boussinesq base state keeps
# rho = 1 kg m-3 at all heights, as in the paper (section 3).
[thermo]
swthermo=dry
swbasestate=boussinesq
thref0={THREF}
pbot=100000.
"""
        boundary = f"""\
[boundary]
mbcbot=noslip
mbctop=freeslip
sbcbot=flux
sbctop=neumann
sbot[s]={PAPER['sflux']}
stop[s]=0.
sbot[th]=0.
stop[th]=0.
swboundary=surface
z0m={PAPER['z0']}
z0h={PAPER['z0']}
"""
        ib_block = ''
        masklist = ''
        crosslist = 'u,v,w,s'
    ini = f"""\
# Andren et al. (1994) neutral Ekman layer, QJRMS 120, 1457-1484,
# https://doi.org/10.1002/qj.49712052003 . Generated by andren1994_setup.py
# (res={a.res}, surface={a.surface}, advec={a.advec}). Settings follow the
# paper's section 3 and Mason's code (Smagorinsky, Cs=0.17, Sc=0.7), see README.md.
[master]
npx={npx}
npy={npy}

[grid]
itot={g['itot']}
jtot={g['jtot']}
ktot={g['ktot']}

xsize={PAPER['xsize']}
ysize={PAPER['ysize']}
zsize={g['zsize']:.10g}

# Galilean transformation of Mason's code (paper section 2(b)); must be 0
# with the IB (the IB wall model and blanking ignore utrans).
utrans={a.utrans}
vtrans=0.

swspatialorder=2

[advec]
swadvec={a.advec}
cflmax={a.cflmax}

# With a thermo class on, smag2 takes its stability branch (N2 = 0 here), whose
# wall damping 1/l^n = 1/(cs D)^n + 1/(kappa (z+z0))^n has n = 2 as in Mason
# and Thomson (1992); the swthermo=0 branch would use n = 1.
[diff]
swdiff=smag2
cs={a.cs}
tPr={a.tpr}
dnmax=0.3

{thermo}
[force]
swlspres=geo
fc={PAPER['fc']}

{boundary}
{ib_block}
[fields]
visc=1.e-5
svisc=1.e-5
slist=s
# The paper's perturbations are added by "andren1994_setup.py --perturb"
# after "microhh init" and made divergence-free there (essential with a
# thermo scheme on, see README), so MicroHH's own random field is off.
rndseed={a.seed}
rndamp[u]=0.
rndamp[v]=0.
rndamp[w]=0.

[buffer]
swbuffer=0

[time]
endtime={a.endtime:g}
dt=2.
dtmax=20.
savetime={a.savetime}
outputiter=50
adaptivestep=true
starttime=0
rkorder=3

[stats]
swstats=1
sampletime={a.sampletime}
{masklist}
# Cross-sections for cross_to_nc.sh (written next to this ini): xy planes
# at the 2nd, 6th and 11th air level and one x-z slice through the middle
# of the domain; with the IB also the IB surface fields (ustar_ib, the skin
# and form drag, ...), which andren1994_plot.py uses for the surface stress.
[cross]
swcross=1
sampletime={a.sampletime}
xy={zc[1]:.6g},{zc[5]:.6g},{zc[10]:.6g}
xz={PAPER['ysize'] / 2.:.6g}
crosslist={crosslist}
"""
    with open(fn, 'w') as f:
        f.write(ini)


def write_dem(fn, g):
    """Constant DEM (height above the domain floor, m) on the itot x jtot grid."""
    np.full((g['jtot'], g['itot']), g['z_surface'], dtype=np.float64).tofile(fn)


def write_input_nc(fn, g):
    """Initial profiles (Table A.1) at the heights above the surface; zero
    wind inside the terrain. thl and qt for the moist IB variant, th for the
    dry flat one (MicroHH ignores the others)."""
    import netCDF4 as nc4
    ktot, dz = g['ktot'], g['dz']
    z = (np.arange(ktot) + 0.5) * dz
    za = z - g['z_surface']                       # height above the surface
    air = za > 0.
    zA, uA, vA = TABLE_A1[:, 0], TABLE_A1[:, 1], TABLE_A1[:, 2]
    u = np.where(air, np.interp(za, zA, uA), 0.)
    v = np.where(air, np.interp(za, zA, vA), 0.)
    with nc4.Dataset(fn, 'w') as nc:
        nc.createDimension('z', ktot)
        nc.createVariable('z', 'f8', ('z',))[:] = z
        gr = nc.createGroup('init')
        for name, val in (('u', u), ('v', v), ('s', np.zeros(ktot)), ('th', np.full(ktot, THREF)),
                          ('thl', np.full(ktot, THREF)), ('qt', np.zeros(ktot)),
                          ('u_geo', np.full(ktot, PAPER['ug'])),
                          ('v_geo', np.full(ktot, PAPER['vg']))):
            gr.createVariable(name, 'f8', ('z',))[:] = val


JOB = r"""#!/bin/bash
#
# Atos (ECMWF) job for one andren1994 run directory (Andren et al. 1994,
# neutral Ekman layer). Generated by andren1994_setup.py: res=@RES@,
# grid @ITOT@ x @JTOT@ x @KTOT@, MPI @NPX@ x @NPY@ = @NP@ tasks, endtime @ENDTIME@ s.
#
#   sbatch /path/to/run_dir/job_atos.sh      # from anywhere
#
# The script finds its own directory (scontrol Command=, else the submit
# directory, else the folder it sits in) and runs there, so no paths need
# editing. It does:
#   1. microhh init
#   2. andren1994_setup.py --perturb   (the paper's initial perturbations)
#   3. microhh run
#   4. andren1994_plot.py              (figures in ./figs, bulk table in the log)
# The binary: $MICROHH_BIN if set, else @BIN@ .
#
#SBATCH --job-name=andren1994_@RES@
#SBATCH --qos=@QOS@                  # nf: fractional (<= half a node, billed per core); np: whole nodes
#SBATCH --ntasks=@NP@
#SBATCH --hint=nomultithread          # one rank per physical core
#SBATCH --time=@WALL@                # estimate: @EST@
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

set -eu

# The SAME modules the build used; it must also give a python3 with numpy,
# netCDF4 and matplotlib.
source ${MICROHH_ENV:-$HOME/microhh_env.sh}

# --- run in the folder this script sits in ---------------------------------
RUN_DIR=""
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol > /dev/null; then
    CMD=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/^ *Command=/{print $2; exit}' | awk '{print $1}')
    if [ -n "$CMD" ]; then
        case "$CMD" in
            /*) ;;
            *)  CMD="${SLURM_SUBMIT_DIR:-$PWD}/$CMD" ;;
        esac
        RUN_DIR=$(dirname "$CMD")
    fi
fi
if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/andren1994.ini" ]; then
    RUN_DIR=${SLURM_SUBMIT_DIR:-}
fi
if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/andren1994.ini" ]; then
    RUN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi
if [ ! -f "$RUN_DIR/andren1994.ini" ]; then
    echo "ERROR: cannot find andren1994.ini (tried scontrol Command=, SLURM_SUBMIT_DIR, script dir)"
    exit 1
fi
cd "$RUN_DIR"
echo "Run directory: $RUN_DIR"

export OMPI_MCA_coll_hcoll_enable=0

npx=$(awk -F= '/^npx=/{print $2}' andren1994.ini)
npy=$(awk -F= '/^npy=/{print $2}' andren1994.ini)
NP=$((npx*npy))
if [ $NP -gt ${SLURM_NTASKS:-$NP} ]; then
    echo "ERROR: npx*npy = $NP in andren1994.ini, more than the ${SLURM_NTASKS} tasks of the job."
    exit 1
fi
echo "MPI: $npx x $npy = $NP tasks"
SRUN="srun -n $NP"

BIN=${MICROHH_BIN:-@BIN@}
[ -x "$BIN" ] || { echo "ERROR: no MicroHH binary at $BIN (set MICROHH_BIN or copy it to $RUN_DIR/microhh)"; exit 1; }

if [ -e grid.0000000 ]; then
    echo "ERROR: grid.0000000 exists - this directory has been initialised already."
    echo "       Regenerate it (andren1994_setup.py --outdir ...) for a clean start."
    exit 1
fi

$SRUN "$BIN" init andren1994
python3 andren1994_setup.py --perturb --rundir .
$SRUN "$BIN" run andren1994

python3 andren1994_plot.py --rundir . --outdir figs || \
    echo "plotting failed (the run itself finished)"
"""


def write_job(fn, a, itot, jtot, ktot, npx, npy):
    NP = npx * npy
    # Cost estimate, measured on Atos (qos nf, 18-19 Oct 2026, paper grid):
    # - both surfaces cost ~85 us per cell-step per core (flat: 25 tasks,
    #   0.21 s per step; IB: 20 tasks, 44 levels, 0.29 s per step). A 70 k-cell
    #   domain on 20-25 tasks is dominated by MPI and output, not by the IB;
    # - dt (CFL-limited, cflmax 1.2) is 11.3 s on the flat surface with
    #   utrans = 7.5, but 7.1 s with the IB, which needs utrans = 0: the CFL
    #   then sees the full ~10 m/s instead of u - 7.5 m/s.
    # So the IB run takes ~2.3x the flat run: ~70 min for 1e5 s. (The IB job
    # of 19 Oct was not slow: it reached 13 800 s in 10 min, then hung for
    # 21 min after a Lustre corruption of one cross-section file and hit the
    # 30-min limit; the flat job needed 29.5 of its 30 min.)
    ib = getattr(a, 'surface', 'flat') == 'ib'
    dt_est = a.dt_est if a.dt_est is not None else (11.3 if a.utrans else 7.1)
    dt = dt_est * (PAPER['xsize'] / itot / 100.)       # dt scales with dx (dy, dz alike)
    nsteps = a.endtime / dt
    core_s = itot * jtot * ktot * nsteps * 8.5e-5
    wall_s = core_s / NP
    wall_req = max(1800, int(1.5 * wall_s + 600))       # 50 % margin + 10 min (init, plotting)
    hh, mm = wall_req // 3600, (wall_req % 3600) // 60
    est = (f'{itot*jtot*ktot/1e6:.3g} M cells, dt ~{dt:.1f} s, {nsteps:.0f} steps, '
           f'{core_s/3600:.1f} core-h, ~{wall_s/60:.0f} min on {NP} tasks')
    qos = 'nf' if NP <= 64 else 'np'
    rep = {'@RES@': a.res, '@ITOT@': str(itot), '@JTOT@': str(jtot), '@KTOT@': str(ktot),
           '@NPX@': str(npx), '@NPY@': str(npy), '@NP@': str(NP), '@ENDTIME@': f'{a.endtime:g}',
           '@BIN@': a.microhh_bin, '@QOS@': qos, '@WALL@': f'{hh:02d}:{mm:02d}:00', '@EST@': est}
    txt = JOB
    for k, v in rep.items():
        txt = txt.replace(k, v)
    if qos == 'np':
        # np bills the whole node anyway: ask for all 128 tasks; srun uses npx*npy of them.
        txt = txt.replace('#SBATCH --ntasks=%d\n' % NP,
                          '#SBATCH --nodes=1\n#SBATCH --ntasks=128\n#SBATCH --ntasks-per-node=128\n')
    with open(fn, 'w') as f:
        f.write(txt)
    os.chmod(fn, 0o755)
    return est, qos


def divergence(u, v, w, dx, dy, dz):
    """Cell-centre divergence on MicroHH's staggered grid (periodic in x, y;
    u[i] on the west face, v[j] on the south face, w[k] on the bottom face,
    w = 0 at the surface and at the lid)."""
    wt = np.concatenate([w, np.zeros_like(w[:1])], axis=0)
    return ((np.roll(u, -1, axis=2) - u) / dx + (np.roll(v, -1, axis=1) - v) / dy
            + (wt[1:] - wt[:-1]) / dz)


def project(u, v, w, dx, dy, dz):
    """Remove the divergent part of (u, v, w): solve lap(phi) = div with the
    discrete (second-order, staggered) Laplacian - FFT in x and y, a
    tridiagonal solve with dphi/dz = 0 at the surface and lid in z - and
    subtract grad(phi). This is what MicroHH's own pressure solver does each
    step; doing it here makes the initial field divergence-free BEFORE the
    first step (see perturb)."""
    kt, jt, it = u.shape
    D = np.fft.rfft2(divergence(u, v, w, dx, dy, dz), axes=(1, 2))
    lx = (2. * np.cos(2. * np.pi * np.fft.rfftfreq(it)) - 2.) / dx**2
    ly = (2. * np.cos(2. * np.pi * np.fft.fftfreq(jt)) - 2.) / dy**2
    lh = ly[:, None] + lx[None, :]
    # Thomas algorithm for all horizontal modes at once.
    a = np.full(kt, 1. / dz**2); c = a.copy()
    b = lh[None, :, :] - 2. / dz**2 + np.zeros((kt, 1, 1))
    b[0] += 1. / dz**2; b[-1] += 1. / dz**2           # Neumann at both ends
    b[:, 0, 0] = 1.; D[:, 0, 0] = 0.                  # mean mode: phi = 0
    cp = np.zeros_like(b); dp = np.zeros_like(D)
    for k in range(kt):
        ak = 0. if k == 0 else a[k]
        ck = 0. if k == kt - 1 else c[k]
        mean_mode = np.zeros_like(b[k]); mean_mode[0, 0] = 1.
        ak_arr = ak * (1. - mean_mode)
        ck_arr = ck * (1. - mean_mode)
        den = b[k] - (ak_arr * cp[k - 1] if k > 0 else 0.)
        cp[k] = ck_arr / den
        dp[k] = (D[k] - (ak_arr * dp[k - 1] if k > 0 else 0.)) / den
    phi = np.zeros_like(D)
    phi[-1] = dp[-1]
    for k in range(kt - 2, -1, -1):
        phi[k] = dp[k] - cp[k] * phi[k + 1]
    phi = np.fft.irfft2(phi, s=(jt, it), axes=(1, 2))
    u = u - (phi - np.roll(phi, 1, axis=2)) / dx
    v = v - (phi - np.roll(phi, 1, axis=1)) / dy
    w = w.copy()
    w[1:] -= (phi[1:] - phi[:-1]) / dz
    w[0] = 0.
    return u, v, w


def perturb(rundir):
    """The paper's appendix: uniform random numbers in [-0.5, 0.5] added to
    all velocity components, scaled to give the TKE q^2/2 of Table A.1.
    For r ~ U(-0.5, 0.5), var(r) = 1/12, so a perturbation a*r on each of the
    three components gives TKE = 3 a^2 / 24 = a^2 / 8, i.e. a = sqrt(8 q^2/2).
    The horizontal mean of each perturbation is removed per level.

    The perturbation is then made divergence-free (project). This is
    essential here: MicroHH evaluates the advection of the first step with
    the field as loaded and only then removes the divergence, so a
    non-solenoidal perturbation of ~1 m/s gives a flux-form advection term
    th * div(u) of tens of K per second on the uniform theta = 300 K. That
    grid-scale theta noise (up to +-4 K) then drives a spurious convective
    burst (w'^2 ~ 1.5 m2/s2 up to the lid within 5 min) and a 3.6 K cooling
    of the surface layer. The paper's codes had no buoyancy, so for them the
    first pressure solve did the same as this projection. The projection
    removes the divergent part (about a third) of the perturbation energy, as
    the first pressure solve in the paper's codes did."""
    p = json.load(open(os.path.join(rundir, 'case_parameters.json')))
    it, jt, kt = p['itot'], p['jtot'], p['ktot']
    dx, dy, dz = p['xsize'] / it, p['ysize'] / jt, p['zsize'] / kt
    # IB variant: perturb and project only the air above the flat surface;
    # the floor face (w at k = ns) stays 0, the terrain stays at rest.
    ns = p.get('n_solid', 0)
    zs = p.get('z_surface', 0.)
    ka = kt - ns
    z = (np.arange(ns, kt) + 0.5) * dz - zs
    zh = np.arange(ns, kt) * dz - zs
    e = lambda zz: np.interp(zz, TABLE_A1[:, 0], TABLE_A1[:, 3], left=TABLE_A1[0, 3], right=0.)
    rng = np.random.default_rng(p['seed'])
    fld, add = {}, {}
    for name, zz in (('u', z), ('v', z), ('w', zh)):
        f = os.path.join(rundir, f'{name}.0000000')
        if not os.path.exists(f):
            raise SystemExit(f'{f} not found: run "microhh init" first')
        fld[name] = np.fromfile(f).reshape(kt, jt, it)
        amp = np.sqrt(8. * e(zz))
        r = rng.uniform(-0.5, 0.5, size=(ka, jt, it))
        r -= r.mean(axis=(1, 2), keepdims=True)
        add[name] = amp[:, None, None] * r
    add['w'][0] = 0.                                  # w = 0 at the surface
    e0 = 0.5 * sum((add[n]**2).mean() for n in add)
    du, dv, dw = project(add['u'], add['v'], add['w'], dx, dy, dz)
    e1 = 0.5 * ((du**2).mean() + (dv**2).mean() + (dw**2).mean())
    air = lambda f: f[ns:]
    div0 = np.abs(divergence(air(fld['u']), air(fld['v']), air(fld['w']), dx, dy, dz)).max()
    for name, d in (('u', du), ('v', dv), ('w', dw)):
        fld[name][ns:] += d
        fld[name].tofile(os.path.join(rundir, f'{name}.0000000'))
        print(f'{name}.0000000: perturbation added, rms at first air level {d[1 if name == "w" else 0].std():.3f} m/s')
    div1 = np.abs(divergence(air(fld['u']), air(fld['v']), air(fld['w']), dx, dy, dz)).max()
    print(f'projection kept {e1 / e0:.2f} of the perturbation energy; max |div| {div0:.1e} (before) '
          f'-> {div1:.1e} 1/s (after)' + (f', in the {ka} air levels above the IB surface' if ns else ''))
    if div1 > 1e-10:
        raise SystemExit('ERROR: the perturbed field is not divergence-free')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--outdir', default='run_andren1994', help='run directory to create')
    ap.add_argument('--res', choices=RES.keys(), default='paper', help='grid (domain is always the paper\'s)')
    ap.add_argument('--npx', type=int, default=None, help='MPI tasks in x (default: auto for one Atos node)')
    ap.add_argument('--npy', type=int, default=None, help='MPI tasks in y (default: auto)')
    ap.add_argument('--maxtasks', type=int, default=128, help='upper limit for the auto decomposition')
    ap.add_argument('--endtime', type=float, default=PAPER['endtime'], help='end time in s (paper: 10/f = 1e5)')
    ap.add_argument('--sampletime', type=int, default=300, help='statistics interval in s')
    ap.add_argument('--savetime', type=int, default=10000, help='restart file interval in s')
    ap.add_argument('--cs', type=float, default=0.17, help='Smagorinsky constant (Mason: 0.17)')
    ap.add_argument('--tpr', type=float, default=0.7, help='turbulent Schmidt number of s (Mason: 0.70)')
    ap.add_argument('--utrans', type=float, default=None,
                    help='Galilean translation velocity in x (default: Mason\'s 7.5 for the flat surface, '
                         '0 for --surface ib, which requires 0)')
    ap.add_argument('--surface', choices=('flat', 'ib'), default='flat',
                    help='flat: MicroHH\'s own surface model (default, the earlier runs); ib: a flat '
                         'immersed-boundary surface with the S5 [IB] switches')
    ap.add_argument('--advec', choices=('2', '2i5'), default=None,
                    help='[advec] swadvec (default: 2 for flat, 2i5 for ib, as the production runs)')
    ap.add_argument('--ib-solid-levels', type=int, default=IB_SOLID_LEVELS,
                    help='--surface ib: solid levels under the surface')
    ap.add_argument('--ib-offset', type=float, default=IB_OFFSET,
                    help='--surface ib: surface height above the half level, in dz (0 = on a face, '
                         'gives NaN IDW weights)')
    ap.add_argument('--momentum', choices=('staircase', 'vr'), default='staircase',
                    help='IB momentum method (--surface ib): staircase (default; case026, the staircase '
                         'tree) or vr (velocity reconstruction, patches 59-60, the vr tree)')
    ap.add_argument('--ib-set', action='append', metavar='KEY=VALUE',
                    help='--surface ib: override or add an [IB] key (repeatable), e.g. sw_wall_sample=false')
    ap.add_argument('--cflmax', type=float, default=1.2)
    ap.add_argument('--seed', type=int, default=2, help='seed of the initial perturbations')
    ap.add_argument('--dt-est', type=float, default=None,
                    help='expected mean dt (s) on the paper grid, only for the wall-time estimate '
                         '(default: measured 11.3 s for the flat surface with utrans 7.5, 7.1 s with utrans 0 / the IB; scaled with dx for other grids)')
    ap.add_argument('--microhh-bin', default='./microhh', help='default binary path written into job_atos.sh')
    ap.add_argument('--perturb', action='store_true', help='mode 2: add the initial perturbations in --rundir')
    ap.add_argument('--rundir', default='.', help='run directory for --perturb')
    a = ap.parse_args()

    if a.perturb:
        perturb(a.rundir)
        return

    if a.utrans is None:
        a.utrans = 0. if a.surface == 'ib' else 7.5
    if a.advec is None:
        a.advec = '2i5' if a.surface == 'ib' else '2'
    if a.surface == 'ib':
        sw = {k: v for k, v, _ in ib_switches(a.ib_set, a.momentum)}
        if a.utrans != 0.:
            sys.exit('--surface ib needs --utrans 0: the IB (wall model, ghost cells, blanking) works in the '
                     'model frame and does not add [grid] utrans, so the terrain would move at -utrans')
        gone = [k for k in ('sw_momentum_flux', 'sw_momentum_flux_vertical', 'sw_strain_most',
                            'sw_strain_most_vertical', 'strain_most_min', 'sw_strain_mlen',
                            'sw_wall_kinematic', 'sw_momentum_air_only', 'sw_impermeable',
                            'sw_wall_slantwise', 'sw_wall_momentum_relay') if k in sw]
        if gone and a.momentum == 'vr':
            sys.exit(f'--ib-set {", ".join(gone)}: removed by patch 59, not in the vr tree '
                     f'(MicroHH ignores them); use --momentum staircase')
        vr_keys = [k for k in sw if k.startswith(('sw_momentum_reconstruct', 'reconstruct_'))]
        if vr_keys and a.momentum == 'staircase':
            sys.exit(f'{", ".join(vr_keys)}: velocity-reconstruction keys, not in the staircase '
                     f'tree; use --momentum vr')
        if not 0. <= a.ib_offset < 1.:
            sys.exit('--ib-offset must be in [0, 1)')
        if a.ib_offset == 0.:
            print('WARNING: --ib-offset 0 puts the surface on a cell face: the IDW weights of the scalar '
                  'ghost cells are NaN (image point on a cell centre). Only for demonstrating that.')
    elif a.ib_set:
        sys.exit('--ib-set needs --surface ib')
    g = geometry(a.res, a.surface, a.ib_solid_levels, a.ib_offset)
    itot, jtot, ktot = g['itot'], g['jtot'], g['ktot']
    if a.npx is None and a.npy is None:
        npx, npy = auto_decomp(itot, jtot, ktot, a.maxtasks)
    else:
        npx, npy = a.npx or 1, a.npy or 1
    if not valid_decomp(itot, jtot, ktot, npx, npy):
        sys.exit(f'npx={npx}, npy={npy} is not valid for {itot}x{jtot}x{ktot}: MicroHH needs itot and '
                 f'jtot divisible by npx and npy, and ktot by npx')
    if npx * npy > 128:
        print(f'WARNING: {npx*npy} tasks is more than one Atos node (128 cores)')

    os.makedirs(a.outdir, exist_ok=True)
    if os.path.exists(os.path.join(a.outdir, 'grid.0000000')):
        sys.exit(f'{a.outdir} has been initialised already (grid.0000000): use a new --outdir')
    write_ini(os.path.join(a.outdir, f'{CASE}.ini'), g, npx, npy, a)
    write_input_nc(os.path.join(a.outdir, f'{CASE}_input.nc'), g)
    if a.surface == 'ib':
        write_dem(os.path.join(a.outdir, 'dem.0000000'), g)
    params = dict(case=CASE, res=a.res, itot=itot, jtot=jtot, ktot=ktot, npx=npx, npy=npy,
                  dx=PAPER['xsize'] / itot, dy=PAPER['ysize'] / jtot, dz=g['dz'],
                  cs=a.cs, tPr=a.tpr, utrans=a.utrans, seed=a.seed, endtime=a.endtime,
                  t_avg=list(PAPER['t_avg']), **{k: v for k, v in PAPER.items() if k not in ('endtime', 't_avg')})
    # zsize is the MODEL domain height; zsize_air (= the paper's 1500 m, to
    # ib_offset*dz) the air column above the surface at z_surface.
    params.update(zsize=g['zsize'], zsize_air=g['zsize'] - g['z_surface'], surface=a.surface,
                  advec=a.advec, n_solid=g['n_solid'], ib_offset=g['ib_offset'], z_surface=g['z_surface'],
                  ib_set=a.ib_set or [], momentum=a.momentum if a.surface == 'ib' else None)
    with open(os.path.join(a.outdir, 'case_parameters.json'), 'w') as f:
        json.dump(params, f, indent=1)
    est, qos = write_job(os.path.join(a.outdir, 'job_atos.sh'), a, itot, jtot, ktot, npx, npy)
    # cross_to_nc.sh + cross_to_nc.py + microhh_tools.py (time means over 1 h)
    install_cross_to_nc(a.outdir, CASE, window=3600.)
    for fn in ('andren1994_setup.py', 'andren1994_plot.py', 'reference_andren1994.json'):
        src = os.path.join(HERE, fn)
        if os.path.exists(src) and os.path.abspath(os.path.dirname(src)) != os.path.abspath(a.outdir):
            shutil.copy(src, a.outdir)

    print(f'{a.outdir}: res={a.res} {itot}x{jtot}x{ktot} (dx={params["dx"]:g}, dy={params["dy"]:g}, '
          f'dz={params["dz"]:g} m), MPI {npx}x{npy}={npx*npy} (qos {qos}), endtime {a.endtime:g} s, '
          f'surface={a.surface}, advec={a.advec}, utrans={a.utrans:g}')
    if a.surface == 'ib':
        print(f'IB surface at z = {g["z_surface"]:g} m ({g["n_solid"]} solid levels + {g["ib_offset"]:g} dz), '
              f'first air level {0.5 * g["dz"] - g["ib_offset"] * g["dz"]:g} m above it, air column '
              f'{params["zsize_air"]:g} m' + (f'; --ib-set {" ".join(a.ib_set)}' if a.ib_set else ''))
    print(f'estimate: {est}')
    if a.endtime < PAPER['t_avg'][1]:
        print(f'NOTE: endtime < {PAPER["t_avg"][1]:g} s: the paper\'s averaging window (7/f-10/f) is not '
              f'reached; andren1994_plot.py then averages over the last third of the run.')


if __name__ == '__main__':
    main()
