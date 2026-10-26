#!/usr/bin/env python3
"""
Vortex identification in MicroHH 3-D fields: Q, lambda2 and the swirling
strength lambda_ci, for isosurfaces of turbulent eddies and coherent
vortices (written for the IB slope cases, works for any MicroHH run).

    python3 vortex_criteria.py --rundir runs/P1_lbc_fine --time 7200
            [--xrange 400,1550] [--zmax-above 60] [--vtk] [--png]

Input: the 3-D fields u, v, w (and thl) of one time, u.<t>, v.<t>, w.<t>:
the [dump] files (axelsen2009_lbc_input.py --dump-every, dumplist u,v,w,thl)
or the restart files written every [time] savetime. They are on the
staggered grid (u at the x faces, v at the y faces, w at the half levels)
and are interpolated to the cell centres; derivatives are 2nd-order
central differences (periodic in y). Cells in the terrain (z <= dem) are
NaN.

Criteria, from the velocity gradient tensor A = grad u = S + Omega:
  Q         = 1/2 (|Omega|^2 - |S|^2) > 0: rotation dominates strain
              (Hunt, Wray and Moin 1988).
  lambda2   = the middle eigenvalue of S^2 + Omega^2; a vortex core is
              lambda2 < 0 (Jeong and Hussain 1995).
  lambda_ci = the imaginary part of the complex eigenvalue pair of A, the
              local swirling strength (> 0 in a vortex; 2 pi / lambda_ci is
              the time of one revolution; Zhou et al. 1999; Chakraborty et
              al. 2005).
  omega     = the vorticity vector (curl u), and omega_y, the cross-slope
              component that the slope-flow shear produces.
Isosurface thresholds are usually a fraction of a characteristic value:
Q = 0.1-1 x (U/L)^2, lambda2 = -(0.1-1) x (U/L)^2, lambda_ci at 10-20 % of
its maximum (Chakraborty et al. 2005); the defaults below are percentiles
of the air cells in the box.

Output (next to the run, in <rundir>/vortex/):
  vortex_<t>.nc       Q, lambda2, lambda_ci, omega_x/y/z, u_c, v_c, w_c, thl_c, height_above_surface
  vortex_<t>.vtk      (--vtk) the same as a VTK structured-points file for ParaView
                      (isosurface of Q or lambda_ci, coloured by u_c or the height)
  vortex_<t>.png      (--png) slices and a pyvista isosurface render (if pyvista is installed)
"""
import argparse
import json
import os

import numpy as np
import netCDF4 as nc


def read3d(rd, name, t, shape):
    f = os.path.join(rd, '%s.%07d' % (name, t))
    if not os.path.exists(f):
        raise SystemExit(f'{f} not found: dump it ([dump] dumplist) or use a savetime (restart) time')
    return np.fromfile(f).reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--rundir', required=True)
    ap.add_argument('--time', type=int, required=True)
    ap.add_argument('--xrange', default=None, help='x0,x1 [m] of the box (default: the straight slope)')
    ap.add_argument('--zmax-above', type=float, default=None,
                    help='box top above the highest terrain in the box [m] (default 2.5 H)')
    ap.add_argument('--vtk', action='store_true', help='write a VTK file for ParaView')
    ap.add_argument('--png', action='store_true', help='slices + isosurface render')
    ap.add_argument('--hmin', type=float, default=None,
                    help='leave out the cells below this height above the surface from the percentiles and the '
                         'isosurface [m] (default 2 dz: the staircase edges carry strong spurious Q)')
    ap.add_argument('--iso', choices=['Q', 'lambda2', 'lambda_ci'], default='Q')
    ap.add_argument('--iso-value', type=float, default=None,
                    help='isosurface value (default: the 99th percentile of Q / lambda_ci, 1st of lambda2)')
    a = ap.parse_args()
    rd = a.rundir
    p = json.load(open(os.path.join(rd, 'case_parameters.json')))
    it, jt, kt, dx, dz = p['itot'], p['jtot'], p['ktot'], p['dx'], p['dz']
    shape = (kt, jt, it)
    dem = np.fromfile(os.path.join(rd, 'dem.0000000')).reshape(jt, it)
    x = (np.arange(it) + 0.5) * dx
    y = (np.arange(jt) + 0.5) * dx
    z = (np.arange(kt) + 0.5) * dz
    if a.xrange:
        x0, x1 = (float(v) for v in a.xrange.split(','))
    else:
        x0 = p['x_top'] + 0.5 * p['break_width']
        x1 = p['x_foot'] - 0.5 * p['break_width']
    i0, i1 = max(int(x0 / dx), 1), min(int(np.ceil(x1 / dx)), it - 1)
    ztop = dem[:, i0:i1].max() + (a.zmax_above if a.zmax_above else 2.5 * p['H'])
    k1 = min(int(np.ceil(ztop / dz)) + 1, kt - 1)

    u = read3d(rd, 'u', a.time, shape)
    v = read3d(rd, 'v', a.time, shape)
    w = read3d(rd, 'w', a.time, shape)
    thl = read3d(rd, 'thl', a.time, shape) if os.path.exists(os.path.join(rd, 'thl.%07d' % a.time)) else None

    # Cell centres (one extra cell on each side in x for the gradients).
    sl = np.s_[:k1 + 1, :, i0 - 1:i1 + 1]
    uc = 0.5 * (u + np.roll(u, -1, axis=2))[sl]
    vc = 0.5 * (v + np.roll(v, -1, axis=1))[sl]            # periodic y
    wp = np.concatenate([w, np.zeros((1, jt, it))], axis=0)
    wc = 0.5 * (wp[:-1] + wp[1:])[sl]
    del u, v, w

    def grad(f):
        """d/dx, d/dy, d/dz of a centred field (central; periodic in y)."""
        fx = (f[:, :, 2:] - f[:, :, :-2]) / (2. * dx)
        fy = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1))[:, :, 1:-1] / (2. * dx)
        fz = np.gradient(f, dz, axis=0)[:, :, 1:-1]
        return fx, fy, fz

    A = np.empty(uc[:, :, 1:-1].shape + (3, 3))
    for r, f in enumerate((uc, vc, wc)):
        gx, gy, gz = grad(f)
        A[..., r, 0], A[..., r, 1], A[..., r, 2] = gx, gy, gz
    S = 0.5 * (A + np.swapaxes(A, -1, -2))
    Om = 0.5 * (A - np.swapaxes(A, -1, -2))
    Q = 0.5 * (np.sum(Om**2, axis=(-1, -2)) - np.sum(S**2, axis=(-1, -2)))
    # lambda2, in z slabs (memory).
    lam2 = np.empty(Q.shape)
    lci = np.empty(Q.shape)
    for k in range(Q.shape[0]):
        M = S[k] @ S[k] + Om[k] @ Om[k]
        lam2[k] = np.linalg.eigvalsh(M)[..., 1]
        ev = np.linalg.eigvals(A[k])
        lci[k] = np.abs(ev.imag).max(axis=-1)
    omx = A[..., 2, 1] - A[..., 1, 2]
    omy = A[..., 0, 2] - A[..., 2, 0]
    omz = A[..., 1, 0] - A[..., 0, 1]
    del A, S, Om

    xs, zsl = x[i0:i1], z[:k1 + 1]
    hgt = zsl[:, None, None] - dem[None, :, i0:i1]
    solid = hgt <= 0.
    fields = dict(Q=Q, lambda2=lam2, lambda_ci=lci, omega_x=omx, omega_y=omy, omega_z=omz,
                  u_c=uc[:, :, 1:-1], v_c=vc[:, :, 1:-1], w_c=wc[:, :, 1:-1], height_above_surface=hgt)
    if thl is not None:
        fields['thl_c'] = thl[:k1 + 1, :, i0:i1]
    for k_ in fields:
        if k_ != 'height_above_surface':
            fields[k_] = np.where(solid, np.nan, fields[k_])

    od = os.path.join(rd, 'vortex')
    os.makedirs(od, exist_ok=True)
    fn = os.path.join(od, 'vortex_%07d.nc' % a.time)
    with nc.Dataset(fn, 'w') as f:
        f.description = ('Vortex criteria from MicroHH 3-D fields at cell centres: Q (Hunt et al. 1988), lambda2 '
                         '(Jeong and Hussain 1995), lambda_ci swirling strength (Zhou et al. 1999). NaN in the terrain.')
        f.time = a.time
        for d, vals in (('z', zsl), ('y', y), ('x', xs)):
            f.createDimension(d, len(vals))
            f.createVariable(d, 'f4', (d,))[:] = vals
        units = dict(Q='s-2', lambda2='s-2', lambda_ci='s-1', omega_x='s-1', omega_y='s-1', omega_z='s-1',
                     u_c='m s-1', v_c='m s-1', w_c='m s-1', thl_c='K', height_above_surface='m')
        for k_, v_ in fields.items():
            var = f.createVariable(k_, 'f4', ('z', 'y', 'x'), zlib=True)
            var[:] = v_.astype(np.float32)
            var.units = units[k_]
    hmin = a.hmin if a.hmin is not None else 2. * dz
    air = ~solid & (hgt > hmin)
    pct = {k_: np.nanpercentile(fields[k_][air], [1, 50, 99]) for k_ in ('Q', 'lambda2', 'lambda_ci')}
    print(f'wrote {fn}: box x {xs[0]:.0f}-{xs[-1]:.0f} m, z <= {zsl[-1]:.0f} m, {air.sum()} air cells '
          f'more than {hmin:g} m above the surface')
    for k_, v_ in pct.items():
        print(f'  {k_:9s} 1/50/99 %: {v_[0]:.3g} / {v_[1]:.3g} / {v_[2]:.3g}')
    iso = a.iso_value
    if iso is None:
        iso = pct[a.iso][0] if a.iso == 'lambda2' else pct[a.iso][2]

    if a.vtk:
        fv = os.path.join(od, 'vortex_%07d.vtk' % a.time)
        nz, ny, nx = Q.shape
        with open(fv, 'wb') as f:
            f.write(b'# vtk DataFile Version 3.0\nMicroHH vortex criteria\nBINARY\nDATASET STRUCTURED_POINTS\n')
            f.write(f'DIMENSIONS {nx} {ny} {nz}\nORIGIN {xs[0]} {y[0]} {zsl[0]}\nSPACING {dx} {dx} {dz}\n'
                    f'POINT_DATA {nx * ny * nz}\n'.encode())
            for k_, v_ in fields.items():
                f.write(f'SCALARS {k_} float 1\nLOOKUP_TABLE default\n'.encode())
                f.write(np.nan_to_num(v_, nan=0.).astype('>f4').tobytes())
        print(f'wrote {fv} (ParaView: Contour on {a.iso} = {iso:.3g}, colour by u_c or height_above_surface)')

    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        jm = jt // 2
        hmax = 2.5 * p['H']
        fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
        for axx, k_, cm in ((ax[0], 'u_c', 'viridis'), (ax[1], 'Q', 'RdBu_r'), (ax[2], 'lambda_ci', 'magma')):
            fld = fields[k_][:, jm, :]
            if k_ == 'Q':
                lim = np.nanpercentile(np.abs(fld), 99)
                pc = axx.pcolormesh(xs, zsl, fld, cmap=cm, vmin=-lim, vmax=lim, shading='auto')
            else:
                pc = axx.pcolormesh(xs, zsl, fld, cmap=cm, shading='auto')
            axx.fill_between(xs, 0, dem[jm, i0:i1], color='0.6')
            axx.set_ylim(dem[jm, i0:i1].min() - 2., dem[jm, i0:i1].max() + hmax)
            fig.colorbar(pc, ax=axx, label=f"{k_} ({units[k_]})")
            axx.set_ylabel('z (m)')
        ax[-1].set_xlabel('x (m)')
        ax[0].set_title(f'xz slice at y = {y[jm]:.0f} m, t = {a.time} s', fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(od, 'vortex_slices_%07d.png' % a.time), dpi=120)
        plt.close(fig)
        try:
            import pyvista as pv
            pv.OFF_SCREEN = True
            grid = pv.ImageData(dimensions=(len(xs), len(y), len(zsl)), spacing=(dx, dx, dz),
                                origin=(xs[0], y[0], zsl[0]))
            fiso = np.where(hgt > hmin, fields[a.iso], 0.)
            grid.point_data[a.iso] = np.nan_to_num(fiso, nan=0.).transpose(2, 1, 0).ravel(order='F')
            grid.point_data['u_c'] = np.nan_to_num(fields['u_c'], nan=0.).transpose(2, 1, 0).ravel(order='F')
            surf = grid.contour([iso], scalars=a.iso)
            pl = pv.Plotter(off_screen=True, window_size=(1600, 900))
            ter = pv.StructuredGrid(*np.meshgrid(xs, y, indexing='xy'), dem[:, i0:i1])
            pl.add_mesh(ter, color='lightgrey')
            if surf.n_points:
                pl.add_mesh(surf, scalars='u_c', cmap='viridis', clim=np.nanpercentile(fields['u_c'][air], [2, 98]),
                            scalar_bar_args=dict(title='u (m/s)'))
            pl.view_vector((-0.35, -1., 0.55), viewup=(0., 0., 1.))   # from the south, above, looking downslope-ish
            pl.camera.zoom(1.3)
            pl.set_scale(zscale=4.)
            pl.add_text(f'{a.iso} = {iso:.3g} (z x4), t = {a.time} s', font_size=10)
            pl.screenshot(os.path.join(od, 'vortex_iso_%07d.png' % a.time))
            pl.close()
            print(f'wrote {od}/vortex_slices_{a.time:07d}.png, vortex_iso_{a.time:07d}.png')
        except Exception as e:   # pyvista or an OpenGL context missing
            print(f'isosurface render skipped ({e}); slices written')


if __name__ == '__main__':
    main()
