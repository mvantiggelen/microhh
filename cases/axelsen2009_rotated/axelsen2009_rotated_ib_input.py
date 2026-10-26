#!/usr/bin/env python3
"""
Does the IB wall model reproduce MicroHH's own surface? A test in the
ROTATED slope frame, where the IB surface is flat (no staircase steps).

    python3 axelsen2009_rotated_ib_input.py --outdir runs/P1 [--momentum staircase|vr]
            [--itot 64 --dx 4 --dz 1 --lz 128] [--k-ib 4] [--endtime 3600] [--npx 1 --npy 2]

Writes two run directories with the same Axelsen and van Dop (2009) case
P1 in the rotated, periodic frame of [thermo] swthermo=buoy (alpha = 5 deg,
N2 = N^2; the frame of cases/prandtlslope and drycblslope), the same grid
spacing, closure, noise and surface buoyancy flux Fs:

  native/  the surface is the domain floor: [boundary] swboundary=surface,
           MOST with z0m, z0h, sbot[b] = Fs (flux)
  ib/      the domain floor is buried under a FLAT IB DEM at z = k_ib dz
           (a cell face, so the first air cell centre is dz/2 above the
           wall, as in native/); the IB MOST wall model with every switch of
           the axelsen2009_lbc runs (sw_wall_slope, sw_mason_wall,
           sw_wall_sample, evisc_solid = cap_last, sw_advec_wall,
           sw_blank_solid, sw_flux_accum), [IB] sbot[b] = Fs, and the
           momentum method of --momentum: staircase (default: the MOST
           wall stress as a source term; sw_momentum_flux, sw_strain_most,
           sw_wall_kinematic, sw_momentum_air_only) or vr (velocity
           reconstruction; needs a MicroHH build with it)

ib/ level k_ib + k is native/ level k. Needs MicroHH from the develop_openbc
branch (the IB with [thermo] swthermo=buoy). Compare with
axelsen2009_rotated_ib_compare.py.
"""
import argparse
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P1 = dict(alpha_deg=5.0, Fs=-0.0012, N=0.0121, z0=0.0044)


def momentum_block(a):
    """The IB momentum keys of --momentum (staircase or vr, velocity reconstruction)."""
    if a.momentum == 'vr':
        return f"""# velocity reconstruction (MOST profile along the normal at the
# first air points, 0 below the surface, terrain in the pressure solve)
sw_momentum_reconstruct={a.momentum_reconstruct}
reconstruct_profile=most
reconstruct_dz={a.reconstruct_dz:g}
reconstruct_diag_every=300
# at least the MOST eddy viscosity on the faces between the R
# points and the free air (the stress the air loses)
reconstruct_stress={a.reconstruct_stress}
"""
    return f"""# momentum, staircase: MOST wall stress as a source term, eddy viscosity
# rescaled to it, w on the floor faces along the sub-grid slope, momentum
# exchanged with the terrain only through MOST and pressure
sw_momentum_flux=true
sw_momentum_flux_vertical=true
sw_strain_most={a.strain_most}
sw_strain_most_vertical=false
strain_most_min=0.05
sw_impermeable=false
sw_wall_kinematic={a.wall_kinematic}
sw_momentum_air_only={a.momentum_air_only}
"""


def write_case(d, a, ib):
    os.makedirs(d, exist_ok=True)
    kib = a.k_ib if ib else 0
    ktot = int(round(a.lz / a.dz)) + kib
    z = (np.arange(ktot) + 0.5) * a.dz
    zsize = ktot * a.dz
    alpha = np.deg2rad(P1['alpha_deg'])
    N2 = P1['N']**2
    import netCDF4 as nc
    with nc.Dataset(os.path.join(d, 'axelsen2009_rotated_input.nc'), 'w') as f:
        f.createDimension('z', ktot)
        f.createVariable('z', 'f8', ('z',))[:] = z
        g = f.createGroup('init')
        g.createVariable('b', 'f8', ('z',))[:] = 0.
        # Beare et al. (2006)-like initial SFS TKE, 0.4 (1 - h/25)^3 above the surface.
        h = z - kib * a.dz
        e = np.where((h > 0) & (h < 25.), 0.4 * np.clip(1. - h / 25., 0, 1)**3, 0.)
        if a.sgs == 'tke2':
            g.createVariable('sgstke', 'f8', ('z',))[:] = np.maximum(e, 1e-5)
    jtot = a.jtot or a.itot
    if ib:
        # The DEM sits a_off dz ABOVE the cell face: exactly on the face, the
        # image point of every ghost cell falls on the first air cell centre
        # and the IB's inverse-distance weight 1/0 makes it NaN
        # (precalculate_idw clamps only the boundary point's distance).
        np.full((jtot, a.itot), (kib + a.dem_offset) * a.dz).astype(np.float64).tofile(os.path.join(d, 'dem.0000000'))
    rndz = (kib * a.dz + 20.)
    zbuf = kib * a.dz + 0.75 * a.lz
    sec_boundary = f"""[boundary]
mbcbot=noslip
mbctop=freeslip
swboundary=surface
swconstantz0=1
z0m={P1['z0']:g}
z0h={P1['z0'] / 10.:g}
sbcbot=flux
sbctop=neumann
sbot[b]={0. if ib else P1['Fs']:g}
stop[b]=0.
""" + ("sbot[sgstke]=0.\nstop[sgstke]=0.\n" if a.sgs == 'tke2' else "") + f""""""
    sec_ib = f"""
[IB]
sw_immersed_boundary=dem
n_idw_points=5
sw_dem_halo_replicate=true
sw_wall_model=most
z0m={P1['z0']:g}
z0h={P1['z0'] / 10.:g}
z0h_model=ratio
z0h_ratio=10
z0h_min=1e-10
z0h_max=0.1
sw_scalar_flux=true
sw_wall_stability_vertical=false
sw_scalar_flux_vertical=false
{momentum_block(a)}# Mason's wall damping with the distance to the IB surface. Acts on smag2 and
# swanisotropic; tke2 ignores it (its wall cell is not IB-aware).
sw_mason_wall=true
evisc_solid=cap_last
sw_wall_slope={a.wall_slope}
wall_dn_min=0.1
sw_blank_solid=true
sw_blank_solid_tend=true
sw_advec_wall={a.advec_wall}
sw_advec_wall_conserve=true
sw_flux_accum=true
# wall model evaluated at a point wall_sample_dz*dz along the normal above
# the true surface; --wall-sample false samples at the
# first air level, as the native surface does
sw_wall_sample={a.wall_sample}
wall_sample_dz=1.5
# |b_s - b_1| cap of the diagnosed surface value (the default 30 is meant for K)
flux_ds_max=0.3
sbcbot=flux
sbot[b]={P1['Fs']:g}
""" + ("sbot[sgstke]=0.\n" if a.sgs == 'tke2' else "") + f"""""" if ib else ""
    ini = f"""# Axelsen and van Dop (2009) P1, rotated periodic frame, {'flat IB' if ib else 'native surface'}
[master]
npx={a.npx}
npy={a.npy}

[grid]
itot={a.itot}
jtot={jtot}
ktot={ktot}
xsize={a.itot * a.dx:g}
ysize={jtot * a.dx:g}
zsize={zsize:g}
swspatialorder=2

[advec]
swadvec=2i5
cflmax=1.0

[diff]
""" + ("swdiff=tke2\ndnmax=0.3\nch1=1.\nch2=0.\n" if a.sgs == 'tke2' else "swdiff=smag2\ndnmax=0.3\ncs=0.17\ntPr=1.\n") + f"""
[thermo]
swthermo=buoy
alpha={alpha:.10f}
N2={N2:.6e}

[force]
swlspres=0
swls=0
swwls=0

{sec_boundary}{sec_ib}
[fields]
visc=1.e-5
svisc=1.e-5
rndseed=2
rndamp[u]=0.031
rndamp[v]=0.031
rndamp[w]=0.031
rndz={rndz:g}
rndexp=2.

[buffer]
swbuffer=1
swupdate=0
zstart={zbuf:g}
sigma=0.03
beta=2.

[time]
starttime=0.
endtime={a.endtime:g}
dt=0.5
dtmax=5.
savetime={a.endtime:g}
outputiter=50
adaptivestep=true
rkorder=3
iotimeprec=0

[stats]
swstats=1
sampletime={a.sampletime:g}

[cross]
swcross=1
sampletime={a.sampletime:g}
xy={(kib + 0.5) * a.dz:g},{(kib + 4.5) * a.dz:g}
crosslist=u,b{',ustar_ib,obuk_ib' if ib else ',ustar,obuk'}
"""
    open(os.path.join(d, 'axelsen2009_rotated.ini'), 'w').write(ini)
    pars = dict(P1, case='P1', frame='rotated', ib=ib, k_ib=kib, itot=a.itot, jtot=jtot, ktot=ktot, dx=a.dx,
                dz=a.dz, lz=a.lz, sgs=a.sgs, endtime=a.endtime, npx=a.npx, npy=a.npy,
                momentum=a.momentum if ib else None,
                switches=(dict(wall_slope=a.wall_slope, advec_wall=a.advec_wall, wall_sample=a.wall_sample)
                          | (dict(momentum_reconstruct=a.momentum_reconstruct,
                                  reconstruct_stress=a.reconstruct_stress, reconstruct_dz=a.reconstruct_dz)
                             if a.momentum == 'vr' else
                             dict(strain_most=a.strain_most, wall_kinematic=a.wall_kinematic,
                                  momentum_air_only=a.momentum_air_only))) if ib else None)
    json.dump(pars, open(os.path.join(d, 'case_parameters.json'), 'w'), indent=1)
    print(f'wrote {d}: {a.itot} x {jtot} x {ktot}, dx {a.dx:g} m, dz {a.dz:g} m, '
          f"{'flat IB at z = %g m' % (kib * a.dz) if ib else 'native surface'}, endtime {a.endtime:g} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--itot', type=int, default=64)
    ap.add_argument('--jtot', type=int, default=None)
    ap.add_argument('--dx', type=float, default=4.)
    ap.add_argument('--dz', type=float, default=1.)
    ap.add_argument('--lz', type=float, default=128., help='depth of the air [m]')
    ap.add_argument('--k-ib', type=int, default=4, help='solid levels under the flat IB')
    ap.add_argument('--dem-offset', type=float, default=0.02,
                    help='flat DEM height above the cell face, in dz (0 gives NaN ghost cells, see write_case)')
    ap.add_argument('--endtime', type=float, default=3600.)
    ap.add_argument('--sampletime', type=float, default=60.)
    ap.add_argument('--npx', type=int, default=1)
    ap.add_argument('--npy', type=int, default=2)
    ap.add_argument('--momentum', choices=('staircase', 'vr'), default='staircase',
                    help='IB momentum method: staircase (default; the MOST wall stress as a source term) '
                         'or vr (velocity reconstruction; needs a MicroHH build with it)')
    # on/off switches; the momentum ones only for their --momentum
    for s in ('wall_slope', 'advec_wall', 'wall_sample',
              'strain_most', 'wall_kinematic', 'momentum_air_only',      # staircase
              'momentum_reconstruct', 'reconstruct_stress'):             # vr
        ap.add_argument('--' + s.replace('_', '-'), default='true', choices=['true', 'false'])
    ap.add_argument('--reconstruct-dz', type=float, default=1.5,
                    help='[IB] reconstruct_dz: image point at max(c dz, d + dz) along the normal')
    ap.add_argument('--sgs', choices=['tke2', 'smag2'], default='tke2',
                    help='SFS closure ([diff] swdiff; smag2 with cs 0.17 and Pr_t 1, as tke2 with ch1 = 1)')
    ap.add_argument('--only', choices=['native', 'ib'], default=None)
    a = ap.parse_args()
    if a.momentum == 'vr' and a.momentum_reconstruct == 'true' and a.advec_wall != 'true':
        ap.error('--momentum-reconstruct true needs --advec-wall true (MicroHH refuses it otherwise)')
    for ib in (False, True):
        name = 'ib' if ib else 'native'
        if a.only and a.only != name:
            continue
        write_case(os.path.join(a.outdir, name), a, ib)


if __name__ == '__main__':
    main()
