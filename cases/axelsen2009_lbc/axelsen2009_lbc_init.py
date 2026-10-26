#!/usr/bin/env python3
"""
Initial SFS TKE (and the optional initial jet) of an axelsen2009_lbc run, applied to sgstke.0000000 after
"microhh init" and before "microhh run":

    python3 axelsen2009_lbc_init.py --rundir runs/P1_lbc

As axelsen2009_init.py (Axelsen and van Dop 2009, section 3.1, after Beare
et al. 2006): e(n) = [0.4 (1 - n/25)^3]^(1/2) m2/s2 below n = 25 m, MicroHH's
floor 1e-7 m2/s2 above, n the slope-normal height above the LOCAL surface,
n = h cos(beta) with h the vertical height above dem.0000000 and beta the
local terrain angle (so on the plateau and in the valley n = h). Cells in
the terrain keep the floor. Running it twice gives the same result.
"""
import argparse
import json
import os
import numpy as np

E_MIN = 1e-7

ap = argparse.ArgumentParser()
ap.add_argument('--rundir', default='.')
a = ap.parse_args()
rd = a.rundir
p = json.load(open(os.path.join(rd, 'case_parameters.json')))
it, jt, kt, dx, dz = p['itot'], p['jtot'], p['ktot'], p['dx'], p['dz']
tke = p.get('init_tke')
jet = p.get('init_jet')
if not tke and not jet:
    raise SystemExit('case_parameters.json: no init_tke and no init_jet: nothing to do')

dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jt, it)
dzdx = np.gradient(dem, dx, axis=1)
dzdy = np.gradient(dem, dx, axis=0)
cosb = 1. / np.sqrt(1. + dzdx**2 + dzdy**2)
z = (np.arange(kt) + 0.5) * dz
zh = np.arange(kt) * dz
n = (z[:, None, None] - dem[None, :, :]) * cosb[None, :, :]


def check(f, size):
    if not os.path.exists(f):
        raise SystemExit(f'{f} not found: run "microhh init" first')
    if os.path.getsize(f) != size * 8:
        raise SystemExit(f'{f} has {os.path.getsize(f)} bytes, expected {size * 8}')


if tke:
    e0, depth = tke['e0'], tke['depth']
    e = np.where((n > 0.) & (n < depth), np.sqrt(e0 * np.clip(1. - n / depth, 0., 1.)**3), E_MIN)
    e = np.maximum(e, E_MIN)
    f = os.path.join(rd, 'sgstke.0000000')
    check(f, e.size)
    e.astype(np.float64).tofile(f)
    print(f'sgstke.0000000: Beare et al. (2006) profile, e0 = {e0} m2/s2 over {depth} m above the '
          f'local surface; max {e.max():.3f} m2/s2, {np.count_nonzero(e > E_MIN)} cells above the floor')

if jet:
    # Prandtl-shaped downslope jet u_s = U e (n/n_j) exp(-n/n_j) on the cooled
    # part of the slope (weight: the local slope fraction, |dz_s/dx| / tan(alpha)),
    # u = u_s cos(beta), w = -u_s sin(beta) (terrain falling towards +x). It
    # is ADDED to the resting state of init (keeping its random perturbations);
    # the first pressure solve removes the divergent part. Running this
    # script twice adds the jet twice: rerun "microhh init" first.
    U, nj = jet['U'], jet['n_j']
    slope = -dzdx                                     # tan(beta) > 0 on the slope
    wgt = np.clip(slope / np.tan(np.deg2rad(p['alpha_deg'])), 0., 1.)
    beta = np.arctan(np.clip(slope, 0., None))

    def us_of(nn):
        return np.where(nn > 0., U * np.e * (nn / nj) * np.exp(-nn / nj), 0.)

    # u at the west faces: surface and slope from the two neighbouring columns
    dem_f = 0.5 * (dem + np.roll(dem, 1, axis=1))
    w_f = 0.5 * (wgt + np.roll(wgt, 1, axis=1))
    b_f = 0.5 * (beta + np.roll(beta, 1, axis=1))
    nu = (z[:, None, None] - dem_f[None]) * np.cos(b_f)[None]
    fu = os.path.join(rd, 'u.0000000'); check(fu, kt * jt * it)
    u = np.fromfile(fu).reshape(kt, jt, it)
    u += (us_of(nu) * w_f[None] * np.cos(b_f)[None])
    u.tofile(fu)
    nw = (zh[:, None, None] - dem[None]) * np.cos(beta)[None]
    fw = os.path.join(rd, 'w.0000000'); check(fw, kt * jt * it)
    w = np.fromfile(fw).reshape(kt, jt, it)
    w -= (us_of(nw) * wgt[None] * np.sin(beta)[None])
    w[0] = 0.
    w.tofile(fw)
    print(f'u.0000000, w.0000000: initial jet U = {U} m/s, n_j = {nj:.2f} m on the slope '
          f'(max u {u.max():.2f} m/s)')
