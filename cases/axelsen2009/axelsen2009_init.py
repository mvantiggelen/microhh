#!/usr/bin/env python3
"""
Initial state of an axelsen2009 run, applied to the fields that
"microhh init" wrote, before "microhh run":

    python3 axelsen2009_init.py --rundir run_P1_flux

1. Initial SFS TKE (swdiff=tke2, case_parameters.json 'init_tke'), as a
   function of the slope-normal height n = h cos(alpha) above the LOCAL
   surface (h the vertical height above it), into sgstke.0000000:
     katabatic (P1, P2, V1, V2; paper section 3.1, after Beare et al. 2006)
         e(n) = [0.4 (1 - n/25)^3]^(1/2) m2/s2 for n < 25 m, 0 above
     anabatic (A1; paper section 2.3, after Schumann 1990)
         e(n) = 0.5 V^2 exp(-20 n sin(alpha) / H)
   (Axelsen and van Dop 2009, Part I, section 3.1: "e'(z) =
   [0.4(1 - z/25)^3]^1/2 m2 s-2 in the lowest 25 m of the vertical domain
   and zero above", z slope-normal in their rotated frame.) "Zero" is
   MicroHH's floor Constants::sgstke_min = 1e-7 m2/s2, which tke2 enforces
   anyway; cells inside the terrain keep that input value too.

2. The optional downslope jet of --init-jet (init_jet.json), into
   u.0000000 and w.0000000: u_s(n) = U e (n/n_j) exp(-n/n_j), split into
   u = u_s cos(alpha) and w = -u_s sin(alpha) (the terrain falls towards
   +x). The first pressure solve removes what is not divergence-free.

3. --from-run DIR --from-time T: a warm start. The 3-D fields u, v, w,
   thl, qt (and sgstke) of another run of the same grid at time T replace
   the ones init wrote; e.g. a --bc temp run started from the developed
   katabatic flow of the --bc flux run at t = tau_osc, which avoids the
   "no wind, no MOST flux" start (README). The TKE profile and the jet are
   then not applied.

Running it twice adds the jet twice: rerun "microhh init" first.
"""
import argparse
import json
import os
import numpy as np

E_MIN = 1e-7          # MicroHH Constants::sgstke_min [m2/s2]

ap = argparse.ArgumentParser()
ap.add_argument('--rundir', default='.')
ap.add_argument('--from-run', default=None, help='warm start: run directory to take the fields from')
ap.add_argument('--from-time', type=int, default=None, help='warm start: restart time [s] in --from-run')
a = ap.parse_args()
rd = a.rundir
p = json.load(open(os.path.join(rd, 'case_parameters.json')))
it, jt, kt, dx, dz = p['itot'], p['jtot'], p['ktot'], p['dx'], p['dz']
al = np.deg2rad(p['alpha_deg'])
tana, cosa = np.tan(al), np.cos(al)


def zs(x):
    return p['z_base'] + tana * (p['lx'] - x)


if 'z_full' in p:
    # Stretched vertical grid (--stretch-above): the levels the generator
    # wrote; the half levels as MicroHH makes them (grid.cxx: midpoints,
    # zh = 0 at the floor).
    z = np.array(p['z_full'])
    zh = np.concatenate(([0.], 0.5 * (z[1:] + z[:-1])))
else:
    z = (np.arange(kt) + 0.5) * dz
    zh = np.arange(kt) * dz
xh = np.arange(it) * dx                 # u faces
x = (np.arange(it) + 0.5) * dx          # cell centres

if a.from_run:
    import shutil
    if a.from_time is None:
        raise SystemExit('--from-run needs --from-time')
    q = json.load(open(os.path.join(a.from_run, 'case_parameters.json')))
    for key in ('itot', 'jtot', 'ktot', 'dx', 'dz', 'alpha_deg', 'z_base'):
        if abs(float(q[key]) - float(p[key])) > 1e-9 * max(1., abs(float(p[key]))):
            raise SystemExit(f'--from-run has a different {key}: {q[key]} vs {p[key]}')
    for name in ('u', 'v', 'w', 'thl', 'qt', 'sgstke'):
        src = os.path.join(a.from_run, '%s.%07d' % (name, a.from_time))
        dst = os.path.join(rd, f'{name}.0000000')
        if os.path.exists(dst):
            if not os.path.exists(src):
                raise SystemExit(f'{src} not found')
            shutil.copyfile(src, dst)
            print(f'{name}.0000000 <- {src}')
    raise SystemExit(0)

done = False
tke = p.get('init_tke')
if tke:
    f = os.path.join(rd, 'sgstke.0000000')
    if not os.path.exists(f):
        raise SystemExit(f'{f} not found: run "microhh init" first (and swdiff must be tke2)')
    fld = np.fromfile(f).reshape(kt, jt, it)
    h = z[:, None] - zs(x)[None, :]                 # [k, i]
    n = np.maximum(h, 0.) * cosa
    if tke['kind'] == 'beare2006':
        e = np.where(n < tke['depth'],
                     np.sqrt(tke['e0'] * np.maximum(1. - n / tke['depth'], 0.)**3), 0.)
    else:
        e = tke['e0'] * np.exp(-tke['decay'] * n)
    e = np.maximum(e, E_MIN)
    fld[:] = np.where(h[:, None, :] > 0, e[:, None, :], fld)
    fld.tofile(f)
    print(f'sgstke.0000000: {tke["kind"]} profile, max {e.max():.3f} m2/s2 at the surface')
    done = True

fj = os.path.join(rd, 'init_jet.json')
if os.path.exists(fj):
    jet = json.load(open(fj))
    U, nj = jet['U'], jet['n_j']

    def us(n):
        n = np.maximum(n, 0.)
        return U * np.e * (n / nj) * np.exp(-n / nj)

    for name, xx, zz, fac in (('u', xh, z, cosa), ('w', x, zh, -np.sin(al))):
        f = os.path.join(rd, f'{name}.0000000')
        fld = np.fromfile(f).reshape(kt, jt, it)
        h = zz[:, None] - zs(xx)[None, :]
        add = np.where(h > 0, fac * us(h * cosa), 0.)
        fld += add[:, None, :]
        fld.tofile(f)
        print(f'{name}.0000000: jet added, max {np.abs(add).max():.3f} m/s')
    done = True

if not done:
    print('nothing to do: no init_tke (swdiff is not tke2) and no --init-jet')
