#!/usr/bin/env python3
"""
Maass and Schumann (1994): turbulent channel flow over a wavy lower wall,
as a test case for the immersed-boundary (IB) MicroHH.

    Maass, C. and Schumann, U. (1994). Numerical simulation of turbulent flow
    over a wavy boundary. In: Voke, P. R., Kleiser, L. and Chollet, J.-P.
    (eds.), Direct and Large-Eddy Simulation I, Kluwer, 287-297.
    https://doi.org/10.1007/978-94-011-1000-6_25

The paper's flow (its eqs. 1-4, Table I, Fig. 1):
    - plane channel, height H, flat no-slip top at z = H, lower wall
      h(x) = delta cos(2 pi x / lambda), delta = 0.05 H, lambda = H
      (case of Hudson 1993, water channel, H = 50.8 mm)
    - Re = Um H / nu = 4780, Um = bulk velocity, held constant by a mean
      pressure gradient that is re-computed every step
    - periodic in x and y, Lx x Ly x H = 4 x 2 x 1 (four waves)
    - DNS, case 4: 160 x 80 x 64, terrain-following, stretched vertically
    - results at t = 60 H/Um, averaged over y and over the four waves

Here the lower wall is a DEM for the IB (staircase), the wall stress comes
from the IB wall model (MOST with a smooth-wall roughness length, neutral)
and the resolved pressure (form drag), and the top wall is MicroHH's
no-slip top boundary. Units: H = 1 m, Um = 1 m/s, nu = 1/Re m2/s, so one
time unit H/Um is one second and rho = 1 (neutral Boussinesq): every stress
in Pa is a kinematic stress in Um^2.

Usage:
    python3 maass1994_setup.py --grid small --outdir run_small --endtime 20
    python3 maass1994_setup.py --grid paper --outdir run_paper            # Atos
    python3 maass1994_setup.py --grid paper --variant ghost --outdir run_paper_ghost

Variants (IB momentum; 20 Oct 2026):
    staircase (default) the case026 method: MOST stress as a source term
           (patch 6), strain-MOST (7), kinematic floor-face w (32), momentum
           exchanged with the terrain only through MOST and pressure (49, 51).
           Needs the staircase tree (patches/reapply_all.sh <tree> staircase).
    The others need the vr tree (patches/reapply_all.sh <tree> vr):
    vr     velocity reconstruction on the true wave surface: the
           first air points get the MOST profile along the surface normal,
           points below the surface 0, the terrain in the pressure solve
           (sw_momentum_reconstruct, patches/README.md section 59)
    ghost  the upstream Tseng & Ferziger ghost cells for the velocity (no
           wall model on momentum), for comparison

Writes in --outdir: maass1994.ini, maass1994_input.nc, dem.0000000,
job_atos.sh, cross_to_nc.sh (+ cross_to_nc.py, microhh_tools.py), and copies maass1994_plot.py, reference_hudson1996.json (the LDV data the plots use) and reference_maass1994.json
(when found next to this script) so that the directory is self-contained.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import netCDF4 as nc

# The cross-section converter (cross_to_nc.sh, cross_to_nc.py, microhh_tools.py)
# is copied into every run folder by write_cross_to_nc.py, next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from write_cross_to_nc import install as install_cross_to_nc  # noqa: E402


# ---------------------------------------------------------------------------
# The paper's parameters (Maass and Schumann 1994, Table I and text).
# ---------------------------------------------------------------------------
H = 1.0              # channel height (mean surface to top wall)
UM = 1.0             # bulk velocity over the whole channel
RE = 4780.           # Um H / nu
DELTA = 0.05 * H     # wave amplitude
LAMBDA = 1.0 * H     # wavelength
NU = UM * H / RE     # kinematic viscosity
KAPPA = 0.4          # von Karman constant used by MicroHH
B_SMOOTH = 5.0       # smooth-wall log-law constant: u+ = ln(z+)/kappa + B
UTAU_SKIN_REF = 0.070  # paper Table I, case 4, flat wall (no form drag)
TH0 = 300.           # uniform potential temperature (neutral, no buoyancy)

# Grids. dx = dy everywhere; dz0 is the uniform spacing of the terrain band.
GRIDS = {
    # local smoke test: 2 waves wide in y is not needed, 1 H is enough
    'small': dict(xsize=4., ysize=1., itot=64, jtot=16, dz0=0.025, ktot=36,
                  npx=1, npy=1, cs=0.15, dump_dt=1.),
    # the paper's resolution in a box of one wavelength x 0.5 H: a cheap
    # local check of the numerics (stability, transition) at that grid
    'paper_box': dict(xsize=1., ysize=0.5, itot=40, jtot=20, dz0=0.0125,
                      ktot=64, npx=1, npy=1, cs=0.1, dump_dt=1.),
    # the paper's horizontal grid (case 4: 160 x 80, dx = dy = 0.025 H);
    # vertically 8 cells over the wave height 2 delta
    'paper': dict(xsize=4., ysize=2., itot=160, jtot=80, dz0=0.0125, ktot=64,
                  npx=8, npy=16, cs=0.1, dump_dt=1.),
    # twice the paper's horizontal resolution
    'fine':  dict(xsize=4., ysize=2., itot=320, jtot=160, dz0=0.00625, ktot=128,
                  npx=8, npy=16, cs=0.1, dump_dt=2.),
}


def smooth_wall_z0(utau, nu=NU):
    """z0 of a hydrodynamically smooth wall: ln(z/z0)/kappa = ln(z+)/kappa + B."""
    return nu / utau * np.exp(-KAPPA * B_SMOOTH)


def vertical_grid(dz0, z_uniform_top, z_top, ktot, rmax=1.08):
    """
    Half levels: uniform dz0 from 0 to z_uniform_top (rounded up to a whole
    number of cells), then a stretched part to the no-slip top wall: spacing
    grows from dz0 by at most rmax per cell to a plateau A and shrinks again
    to dz0 at the top (the paper refines near both walls, its eq. 4).
    The plateau A is found by bisection so that exactly ktot cells fill z_top.
    """
    n_lo = int(np.ceil(z_uniform_top / dz0 - 1e-9))
    n_up = ktot - n_lo
    L_up = z_top - n_lo * dz0
    if n_up < 4 or L_up <= n_up * dz0 * 0.999:
        raise ValueError('ktot=%d too large or too small for this grid '
                         '(uniform part %d cells, %.4f left for %d cells)'
                         % (ktot, n_lo, L_up, n_up))
    i = np.arange(n_up)

    def spacing(A):
        ramp = np.minimum(dz0 * rmax ** (i + 1), dz0 * rmax ** (n_up - i))
        return np.minimum(A, np.maximum(ramp, dz0))

    lo, hi = dz0, 100. * L_up
    if spacing(hi).sum() < L_up:
        raise ValueError('ktot=%d too small: the stretched part cannot reach '
                         'the top with growth %.2f per cell' % (ktot, rmax))
    for _ in range(200):
        A = 0.5 * (lo + hi)
        if spacing(A).sum() > L_up:
            hi = A
        else:
            lo = A
    s_up = spacing(0.5 * (lo + hi))
    s_up *= L_up / s_up.sum()
    dz = np.r_[np.full(n_lo, dz0), s_up]
    zh = np.r_[0., np.cumsum(dz)]
    z = 0.5 * (zh[1:] + zh[:-1])
    return z, zh, dz, n_lo


def check_decomposition(itot, jtot, ktot, npx, npy):
    bad = []
    for n, name in ((itot, 'itot'), (jtot, 'jtot')):
        for p, pn in ((npx, 'npx'), (npy, 'npy')):
            if n % p:
                bad.append('%s=%d not divisible by %s=%d' % (name, n, pn, p))
    if ktot % npx:
        bad.append('ktot=%d not divisible by npx=%d' % (ktot, npx))
    if bad:
        raise ValueError('MPI decomposition: ' + '; '.join(bad))


def write_ini(fn, p):
    """The .ini, written out in full so that every choice is visible here."""
    vr = p['variant'] in ('vr', 'vrlin')
    lin = p['variant'] == 'vrlin'
    stair = p['variant'] == 'staircase'
    ib_mom = (
        "# momentum, variant 'staircase' (case026): MOST stress as a source term\n"
        "# (patch 6), strain-MOST (7), kinematic floor-face w (32), momentum\n"
        "# exchanged with the terrain only through MOST and pressure (49, 51)\n"
        "sw_momentum_flux=true\n"
        "sw_momentum_flux_vertical=true\n"
        "sw_strain_most=true\n"
        "sw_strain_most_vertical=false\n"
        "strain_most_min=0.05\n"
        "sw_impermeable=false\n"
        "sw_wall_kinematic=true\n"
        "sw_momentum_air_only=true\n") if stair else ((
        "# momentum: velocity reconstruction (patch 59). R points (air next to the\n"
        "# true bilinear wave surface) get the MOST profile along the normal,\n"
        "# image point at max(1.5 dz, d + dz); S points (below it) are 0; one\n"
        "# pressure solve over the whole domain, terrain included.\n"
        "sw_momentum_reconstruct=true\n"
        + ("# variant 'vrlin': the first air cell is at d+ ~ 1-2, in the viscous\n"
           "# sublayer, where u+ = d+ and the rough-wall log law does not hold:\n"
           "# a linear (resolved no-slip) profile, and no MOST eddy viscosity\n"
           "reconstruct_profile=linear\n" if lin else "reconstruct_profile=most\n")
        + "reconstruct_dz=1.5\n"
        "reconstruct_diag_every=300\n"
        + ("reconstruct_stress=false\n" if lin else
           "# patch 60: at least the MOST eddy viscosity on the faces between the R\n"
           "# points and the free air (the stress the air loses)\n"
           "reconstruct_stress=true\n")) if vr else (
        "# variant 'ghost': upstream ghost cells for u, v, w (no wall model on\n"
        "# momentum; patch 59 removed the MOST stress source of patch 6)\n"
        "sw_momentum_reconstruct=false\n"))
    txt = f"""# Maass and Schumann (1994) wavy-wall channel on the IB MicroHH.
# Generated by maass1994_setup.py --grid {p['grid']} --variant {p['variant']}
# Units: H = 1 m, Um = 1 m/s, nu = 1/{p['re']:.0f} m2/s, rho = 1 (Boussinesq, neutral).

[master]
npx={p['npx']}
npy={p['npy']}

[grid]
itot={p['itot']}
jtot={p['jtot']}
ktot={p['ktot']}
xsize={p['xsize']}
ysize={p['ysize']}
zsize={p['zsize']:.10g}
swspatialorder=2

[advec]
# 2i5 as in the production runs
swadvec=2i5
cflmax=1.2

[diff]
# Smagorinsky plus the molecular viscosity [fields] visc (Re = 4780).
# cs = 0.1 as in channel-flow LES (Deardorff 1970); MicroHH's default 0.23
# is tuned for the atmospheric boundary layer and laminarises this
# low-Re flow (the top wall has no Mason damping).
swdiff=smag2
cs={p['cs']:g}
dnmax=0.4

[thermo]
# Neutral and constant density (rho = 1): moist Boussinesq with a uniform
# thl and qt = 0, so there is no buoyancy. The IB wall model needs the
# moist thermo class (dry has no thv base state); thl is the same in the
# air, on the wall and at the top, so MOST stays neutral (z/L = 0).
swthermo=moist
swbasestate=boussinesq
thvref0={TH0:g}
pbot=100000.
swupdatebasestate=0

[force]
# The paper keeps the bulk velocity Um constant with a mean pressure
# gradient re-computed every step. MicroHH's uflux does the same for the
# domain-mean u, which includes the terrain cells (u = 0 there): with the
# volume flux Um*H per unit width through every y-z plane,
# uflux = Um*H/zsize.
swlspres=uflux
uflux={p['uflux']:.10g}
swls=0
swwls=0

[boundary]
# The domain floor is buried under the terrain; the IB is the lower wall.
# The top is the paper's flat no-slip wall, resolved: swboundary=default
# sets the wall flux to -visc*grad (molecular stress from the resolved
# gradient, first level at dz/2). With swboundary=surface a Dirichlet top
# carries NO flux in LES (boundary.cxx: set_flux_grad only for 'default'),
# i.e. the top would be stress-free. z0m only enters the Mason damping.
swboundary=default
mbcbot=noslip
mbctop=noslip
z0m={p['z0m']:.6g}
sbcbot=dirichlet
sbctop=dirichlet
sbot[thl]={TH0:g}
stop[thl]={TH0:g}
sbot[qt]=0.
stop[qt]=0.
sbot[s]=1.
stop[s]=0.

[IB]
sw_immersed_boundary=dem
n_idw_points=5
sw_wall_model=most
# smooth wall: z0m = (nu/u*_ref) exp(-kappa B), B = {B_SMOOTH}, u*_ref = {p['utau_ref']}
z0m={p['z0m']:.6g}
z0h={p['z0m']:.6g}
sw_wall_stability_vertical=false
# passive tracer s: s = 1 on the wavy wall, 0 at the top (wall exchange)
sw_scalar_flux=true
# the risers exchange nothing with sw_wall_slope (MOST on the true surface)
sw_scalar_flux_vertical=false
sbcbot=dirichlet
sbot[thl]={TH0:g}
sbot[qt]=0.
sbot[s]=1.
# MOST along the local slope with the wall-normal distance (patch 30)
sw_wall_slope=true
wall_dn_min=0.1
# patch 52: rock evisc capped after the Mason and MOST wall damping
# (evisc_solid=cap for a binary without patch 52)
evisc_solid={p['evisc_solid']}
sw_mason_wall=true
sw_blank_solid=true
sw_blank_solid_tend=true
sw_advec_wall=true
sw_advec_wall_conserve=true
{ib_mom}# patch 48: IB drag and fluxes as interval means
sw_flux_accum=true
# wall model at an image point wall_sample_dz*dz along the normal off the
# true surface, as in the production runs (patch 45, racmo_to_openbc
# IB_WALL_SAMPLE)
sw_wall_sample=true
wall_sample_dz=1.5
# diagnostic heights (default 10 m and 2 m, far too high here)
diag_z_mom={0.05*H:.4g}
diag_z_scalar={0.02*H:.4g}

[fields]
slist=s
visc={p['nu']:.10g}
svisc={p['nu']:.10g}
rndseed=2
rndamp[u]={0.2*UM:.4g}
rndamp[v]={0.2*UM:.4g}
rndamp[w]={0.2*UM:.4g}
rndz={0.9*p['zsize']:.6g}
rndexp=1.

[buffer]
swbuffer=0

[time]
starttime=0.
endtime={p['endtime']:g}
dt=0.002
dtmax=0.05
savetime={p['savetime']:g}
outputiter=50
adaptivestep=true
rkorder=3
iotimeprec=-2

[stats]
swstats=1
sampletime={p['stats_dt']:g}
masklist=ib

[cross]
swcross=1
sampletime={p['cross_dt']:g}
xz={p['ysize']/2.:.6g}
crosslist=u,w,p,s,taux_form_ib,tauy_form_ib,taux_skin_ib,tauy_skin_ib,ustar_ib

[dump]
# 3-D fields for the phase averages (Figs. 2, 3, 6, 7 of the paper)
swdump=1
sampletime={p['dump_dt']:g}
sw_dump_every_sample=true
dumplist=u,v,w,p
"""
    with open(fn, 'w') as f:
        f.write(txt)


JOB = r"""#!/bin/bash
#
# Atos (ECMWF) job for one maass1994 run directory (written by
# maass1994_setup.py). Submit from anywhere:
#     sbatch /path/to/run_dir/job_atos.sh
# The job finds its own directory; no paths to edit.
#
#SBATCH --job-name=maass1994_@GRID@
#SBATCH --qos=np
#SBATCH --nodes=1
#SBATCH --ntasks=@NP@
#SBATCH --ntasks-per-node=@NP@
#SBATCH --hint=nomultithread
#SBATCH --time=@WALL@
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

set -eu

# The directory of this script, also under Slurm (which runs a copy of it):
# 1. the Command= of the job, 2. the submit directory, 3. dirname $0.
HERE=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    CMD=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | awk -F= '/^ *Command=/{print $2; exit}')
    CMD=${CMD%% *}
    [ -n "$CMD" ] && [ -f "$CMD" ] && HERE=$(cd "$(dirname "$CMD")" && pwd)
fi
if [ -z "$HERE" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/maass1994.ini" ]; then
    HERE=$SLURM_SUBMIT_DIR
fi
[ -z "$HERE" ] && HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
echo "run directory: $HERE"

# The modules the binary was built with (adapt the path once in your env).
source ${MICROHH_ENV:-$HOME/microhh_env.sh}
export OMPI_MCA_coll_hcoll_enable=0

# The binary: ./microhh, else $MICROHH_BIN.
if [ ! -x ./microhh ]; then
    if [ -n "${MICROHH_BIN:-}" ] && [ -x "$MICROHH_BIN" ]; then
        ln -sf "$MICROHH_BIN" ./microhh
    else
        echo "ERROR: no ./microhh here and MICROHH_BIN not set"; exit 1
    fi
fi

npx=$(awk -F= '/^npx=/{print $2}' maass1994.ini)
npy=$(awk -F= '/^npy=/{print $2}' maass1994.ini)
NP=$((npx*npy))
echo "MPI: $npx x $npy = $NP tasks"

if [ -e grid.0000000 ]; then
    echo "ERROR: grid.0000000 exists: this directory has been initialised already."
    echo "       Regenerate it with maass1994_setup.py for a clean start."
    exit 1
fi

srun -n $NP ./microhh init maass1994
srun -n $NP ./microhh run maass1994

python3 maass1994_plot.py --rundir . --outdir figs --tavg @TAVG@ || \
    echo "plotting failed (the run itself finished)"
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--grid', choices=GRIDS.keys(), default='paper')
    ap.add_argument('--variant', choices=('staircase', 'vr', 'vrlin', 'ghost'),
                    default='staircase',
                    help="staircase (default): the case026 IB momentum method, the staircase tree; "
                         "vr: velocity reconstruction (patch 59); vrlin: the same with a linear "
                         "(viscous-sublayer) profile and reconstruct_stress=false; ghost: upstream "
                         "ghost cells for the velocity (vr, vrlin, ghost: the vr tree)")
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--endtime', type=float, default=None,
                    help='end time in H/Um (default: 20 small, 100 otherwise)')
    ap.add_argument('--npx', type=int, default=None)
    ap.add_argument('--npy', type=int, default=None)
    ap.add_argument('--ktot', type=int, default=None,
                    help='override the number of levels')
    ap.add_argument('--re', type=float, default=RE,
                    help='Reynolds number Um H/nu, Um = whole-channel bulk velocity (default %g, '
                         'Maass & Schumann 1994). The LDV data (ERCOFTAC case 076) are at '
                         'H U_b/nu = 6760 with U_b the LOWER-half bulk velocity; with U_b/Um = 0.92 '
                         'that is --re 7300' % RE)
    ap.add_argument('--z0m', type=float, default=None,
                    help='override the IB roughness length (units of H)')
    ap.add_argument('--utau-ref', type=float, default=UTAU_SKIN_REF,
                    help='friction velocity (Um) for the smooth-wall z0m')
    ap.add_argument('--cs', type=float, default=None,
                    help='Smagorinsky constant (default 0.1; 0.15 on the small '
                         'grid, where 0.1 is unstable at the staircase steps)')
    ap.add_argument('--evisc-cap', action='store_true',
                    help='[IB] evisc_solid=cap instead of cap_last (binary without patch 52)')
    ap.add_argument('--dump-dt', type=float, default=None,
                    help='interval of the 3-D dumps (H/Um)')
    ap.add_argument('--tavg', type=float, default=None,
                    help='start of the averaging window used by the job\'s plot step')
    a = ap.parse_args()

    g = dict(GRIDS[a.grid])
    if a.npx: g['npx'] = a.npx
    if a.npy: g['npy'] = a.npy
    if a.ktot: g['ktot'] = a.ktot
    small = a.grid == 'small'
    endtime = a.endtime if a.endtime is not None else (20. if small else 120.)

    dx = g['xsize'] / g['itot']
    dz0 = g['dz0']
    # Terrain: trough four cells above the buried domain floor, so that
    # sw_advec_wall covers every wall face (it leaves faces within three
    # levels of the domain floor to advec_2i5's own near-wall treatment).
    z_trough = 4. * dz0
    z_mean = z_trough + DELTA          # the paper's z = 0
    z_crest = z_mean + DELTA
    z_top = z_mean + H                 # the flat no-slip top wall
    z, zh, dz, n_lo = vertical_grid(dz0, z_crest + 0.15 * H, z_top, g['ktot'])
    check_decomposition(g['itot'], g['jtot'], g['ktot'], g['npx'], g['npy'])

    nu = UM * H / a.re
    z0m = a.z0m if a.z0m is not None else smooth_wall_z0(a.utau_ref, nu)
    if not z0m < 0.25 * dz0:
        raise ValueError('z0m = %g is not well below the first-cell height %g'
                         % (z0m, 0.5 * dz0))

    os.makedirs(a.outdir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    # DEM: one height per column at the cell centres, crest at x = 0
    # (the paper's Fig. 2 has its crest at x/lambda = 0).
    x = (np.arange(g['itot']) + 0.5) * dx
    dem1 = z_mean + DELTA * np.cos(2. * np.pi * x / LAMBDA)
    dem = np.tile(dem1, (g['jtot'], 1))
    dem.astype(np.float64).tofile(os.path.join(a.outdir, 'dem.0000000'))

    # Initial profile: a turbulent-like 1/7 power-law profile in the air
    # between the mean surface and the top wall with bulk velocity Um
    # (the paper starts from a laminar profile plus noise; the details do not
    # matter for the statistics at late times, its p. 290). Zero in the terrain.
    zeta = np.clip((z - z_mean) / H, 0., 1.)
    u0 = (8. / 7.) * UM * (1. - np.abs(2. * zeta - 1.)) ** (1. / 7.)
    u0[z < z_mean] = 0.
    s0 = np.clip(1. - zeta, 0., 1.)

    with nc.Dataset(os.path.join(a.outdir, 'maass1994_input.nc'), 'w') as f:
        f.createDimension('z', g['ktot'])
        f.createVariable('z', 'f8', ('z',))[:] = z
        grp = f.createGroup('init')
        for name, val in (('u', u0), ('v', 0. * z), ('s', s0),
                          ('thl', 0. * z + TH0), ('qt', 0. * z)):
            grp.createVariable(name, 'f8', ('z',))[:] = val

    p = dict(g)
    p.update(re=a.re, nu=nu, evisc_solid='cap' if a.evisc_cap else 'cap_last', grid=a.grid, variant=a.variant, cs=a.cs if a.cs else g['cs'], zsize=zh[-1], endtime=endtime,
             uflux=UM * H / zh[-1], z0m=z0m, utau_ref=a.utau_ref,
             savetime=max(endtime / 2., 1.) if small else 20.,
             stats_dt=0.5 if small else 0.5,
             cross_dt=0.5 if small else 0.5,
             dump_dt=a.dump_dt if a.dump_dt else g['dump_dt'])
    write_ini(os.path.join(a.outdir, 'maass1994.ini'), p)

    # the paper reports t = 60 H/Um; average from there (small: 2nd half)
    tavg = a.tavg if a.tavg is not None else (endtime / 2. if small else 60.)
    ncell = g['itot'] * g['jtot'] * g['ktot']
    # rough wall time: 5 us per cell-step per core. dt: CFL-limited at
    # cflmax = 1.2 with |u|max ~ 1.6-2 Um, i.e. dt ~ 0.6 dx/Um (paper_box
    # check: dt = 0.014..0.024 at dx = 0.025)
    dt_est = 0.6 * dx / UM
    nsteps = endtime / dt_est
    np_ = g['npx'] * g['npy']
    wall_h = 5e-6 * ncell * nsteps / np_ / 3600.
    wall = '%02d:00:00' % min(48, int(wall_h * 2 + 1))
    job = (JOB.replace('@GRID@', a.grid).replace('@NP@', str(np_))
              .replace('@WALL@', wall).replace('@TAVG@', '%g' % tavg))
    with open(os.path.join(a.outdir, 'job_atos.sh'), 'w') as f:
        f.write(job)
    os.chmod(os.path.join(a.outdir, 'job_atos.sh'), 0o755)

    # cross_to_nc.sh + cross_to_nc.py + microhh_tools.py; no time-mean
    # files by default (WINDOW=10 at submit time for means over 10 H/Um)
    install_cross_to_nc(a.outdir, 'maass1994', window=0.)

    for fn in ('maass1994_plot.py', 'reference_hudson1996.json', 'reference_maass1994.json'):
        src = os.path.join(here, fn)
        if os.path.exists(src) and os.path.abspath(a.outdir) != here:
            shutil.copy(src, a.outdir)

    print('grid %s, variant %s: %d x %d x %d = %.2f M cells, %d tasks'
          % (a.grid, a.variant, g['itot'], g['jtot'], g['ktot'], ncell / 1e6, np_))
    print('  dx = dy = %.4g H, dz = %.4g H over the terrain band (%d levels to '
          'z = %.3f), %.4g..%.4g above; zsize = %.4f H'
          % (dx, dz0, n_lo, zh[n_lo], dz[n_lo:].min(), dz.max(), zh[-1]))
    print('  wave: trough %.4f, mean %.4f, crest %.4f, top wall %.4f; '
          '%.1f cells over 2 delta' % (z_trough, z_mean, z_crest, z_top, 2 * DELTA / dz0))
    print('  nu = %.4g, z0m = %.3g H (dn+ of the first cell at u* = %.3f: %.1f), '
          'uflux = %.5f' % (nu, z0m, a.utau_ref, 0.5 * dz0 * a.utau_ref / nu,
                            UM * H / zh[-1]))
    print('  max adjacent spacing ratio %.3f' % (dz[1:] / dz[:-1]).max())
    print('  estimate: dt ~ %.3g, %d steps to t = %g, ~%.2f h wall on %d cores'
          % (dt_est, nsteps, endtime, wall_h, np_))


if __name__ == '__main__':
    sys.exit(main())
