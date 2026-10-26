#!/usr/bin/env python3
"""
Lay the initial katabatic layer over the whole terrain of a katabatic_idealised
ib/ run, after "microhh init" and before "microhh run":

    python3 katabatic_idealised_init.py --rundir runs/test/ib

With n the height above the local surface along the normal and beta the local
terrain angle (the terrain falls towards +x):
    u_s(n)    = U_j (n/n_j) exp(1 - n/n_j),  u += u_s cos(beta),  w -= u_s sin(beta)
    theta'(n) = dtheta_0 exp(-n/h_theta),     thl += theta'
(the parameters are those of the inflow, case_parameters.json `layer`; with
--inflow rotated, the default, the tabulated equilibrium profile of the
rotated run, `layer.profile`, replaces the two formulas), and
the initial noise in the lowest `noise.depth` m above the local surface, the
same as the rotated run gets from MicroHH above its floor. The
fields are ADDED to what "microhh init" wrote (the ambient state; the .ini
asks MicroHH for no noise); the first pressure solve removes any divergence.
Running it twice adds the layer twice: rerun "microhh init" first.
"""
import argparse
import json
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--rundir', default='.')
a = ap.parse_args()
rd = a.rundir
p = json.load(open(os.path.join(rd, 'case_parameters.json')))
it, jt, kt, dx, dz = p['itot'], p['jtot'], p['ktot'], p['dx'], p['dz']
L = p['layer']
dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jt, it)
z = (np.arange(kt) + 0.5) * dz
zh = np.arange(kt) * dz


PR = L.get('profile')        # the rotated run's equilibrium (--inflow rotated), else the RACMO fit


def us_of(n):
    if PR is not None:
        # the log law of the rotated run's z0m below its first level, as the generator
        n1, u1, z0 = PR['n'][0], PR['u'][0], PR.get('z0m', 0.02)
        low = u1 * np.log(np.maximum(n, z0) / z0) / np.log(n1 / z0)
        return np.where(n <= 0., 0., np.where(n < n1, low, np.interp(n, PR['n'], PR['u'], right=0.)))
    return np.where(n > 0., L['U_j'] * (n / L['n_j']) * np.exp(1. - n / L['n_j']), 0.)


def th_of(n):
    if PR is not None:
        return np.where(n > 0., np.interp(n, PR['n'], PR['dtheta'], left=PR['dtheta'][0], right=0.), 0.)
    return np.where(n > 0., L['dtheta0'] * np.exp(-n / L['h_theta']), 0.)


def load(name):
    f = os.path.join(rd, f'{name}.0000000')
    if not os.path.exists(f):
        raise SystemExit(f'{f} not found: run "microhh init" first')
    v = np.fromfile(f)
    if v.size != kt * jt * it:
        raise SystemExit(f'{f}: {v.size} values, expected {kt * jt * it}')
    return f, v.reshape(kt, jt, it)


beta = np.arctan(np.clip(-np.gradient(dem, dx, axis=1), 0., None))       # > 0 where it falls to +x
# u at the west faces (x = i dx): the surface and angle of the two neighbouring columns
dem_f = 0.5 * (dem + np.roll(dem, 1, axis=1))
dem_f[:, 0] = dem[:, 0]
b_f = 0.5 * (beta + np.roll(beta, 1, axis=1))
b_f[:, 0] = beta[:, 0]
fu, u = load('u')
nu = (z[:, None, None] - dem_f[None]) * np.cos(b_f)[None]
u += us_of(nu) * np.cos(b_f)[None]
u.tofile(fu)
fw, w = load('w')
nw = (zh[:, None, None] - dem[None]) * np.cos(beta)[None]
w -= us_of(nw) * np.sin(beta)[None]
w[0] = 0.
w.tofile(fw)
ft, th = load('thl')
nt = (z[:, None, None] - dem[None]) * np.cos(beta)[None]
th += th_of(nt)

# Initial noise in the lowest `depth` m above the LOCAL surface, as MicroHH's
# [fields] rndamp/rndz/rndexp put it above the floor of the rotated run:
# amplitude (1 - n/depth)^exp x rndamp x U(-0.5, 0.5).
nz = p.get('noise')
if nz:
    rng = np.random.default_rng(nz.get('seed', 2))

    def noise(n, amp):
        fac = np.where((n > 0.) & (n < nz['depth']), (1. - np.clip(n, 0., None) / nz['depth'])**nz['exp'], 0.)
        return fac * amp * (rng.random(n.shape) - 0.5)
    u += noise(nu, nz['u'])
    fv, v = load('v')
    v += noise(nt, nz['v'])
    v.tofile(fv)
    w += noise(nw, nz['w'])
    w[0] = 0.
    th += noise(nt, nz['thl'])
    u.tofile(fu)
    w.tofile(fw)
th.tofile(ft)
print(f'u, v, w, thl .0000000: katabatic layer ({"rotated equilibrium" if PR is not None else "RACMO fit"}) '
      f'U_j {L["U_j"]:.2f} m/s at n_j {L["n_j"]:.0f} m, '
      f'theta deficit {L["dtheta0"]:.2f} K over {L["h_theta"]:.0f} m above the local surface '
      f'(max u {u.max():.2f} m/s, min w {w.min():.3f} m/s)')
