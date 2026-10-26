#
#  MicroHH
#  Copyright (c) 2011-2024 Chiel van Heerwaarden
#  Copyright (c) 2011-2024 Thijs Heus
#  Copyright (c) 2014-2024 Bart van Stratum
#
#  This file is part of MicroHH
#
#  MicroHH is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  MicroHH is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
#
# ---------------------------------------------------------------------------
# MODIFIED COPY of microhh/python/cross_to_nc.py, kept beside the case rather
# than in the MicroHH tree. Three changes, all of them about files that do not
# exist under the name the crosslist gives:
#
#   1. resolve_stem() maps a crosslist entry to the stem actually written.
#      `thlfluxbot_ib` in the .ini becomes `thl_fluxbot_ib.xy.000.*` on disk
#      (immersed_boundary.cxx:744-752 vs :979), which is what produced
#      "Failed to create thlfluxbot_ib.yz.nc".
#   2. Modes with no files are SKIPPED with a message instead of raising.
#      Surface planes are written by cross.cross_plane and exist as `xy` only,
#      so asking for xz/yz is not an error.
#   3. --mask-ib adds a second variable `<var>_ib` to each file: the same
#      cross-section with cells inside the immersed boundary set to NaN, so
#      the raw and masked fields sit side by side. MicroHH's [stats] masks
#      apply to statistics only; cross-sections are written as raw field
#      planes with solid cells included.
#   4. microhh_tools is located via --microhh-python / $MICROHH_PYTHON / the
#      usual relative paths, so this can live next to the case.
#   5. TERRAIN-FOLLOWING PLANES. [IB] sw_tf_cross writes <var>_tf<h> - <var>
#      at h metres above the LOCAL surface, one plane per
#      [IB] tf_cross_heights entry (develop_openbc branch). The
#      crosslist names them once, as `thl_tf`; the files carry the height.
#      expand_tf_variables() turns the one into the other, they are never
#      given a masked `_ib` copy (a terrain-following plane is air in every
#      column), and wdir_tf<h> is rebuilt from the mean VECTOR in the hourly
#      pass rather than block-averaged.
#
#   6. SURFACE PLANES WITHOUT xy= HEIGHTS. With no [cross] xy= line the
#      upstream script converts no xy files at all, so the IB surface planes
#      (always written as xy) were silently skipped in cases that only ask
#      for xz slices (maass1994_hill, axelsen2009). Now xy is added when
#      xy files exist on disk.
#
# MULTIPLE xz/yz SLICES ([cross] xz=8000,13650) need no change: the sliced
# coordinate is written as a real dimension carrying one entry per slice, in
# the order MicroHH numbered them, and each slice gets its own IB mask from
# its own DEM row or column. The conversion reports the positions it found.
#
# Everything else is upstream's. Still GPLv3, as above.
# ---------------------------------------------------------------------------
#

import os
import sys

# microhh_tools lives in microhh/python. This script is meant to sit next to
# the case, not inside the MicroHH tree, so find it: --microhh-python, then
# $MICROHH_PYTHON, then the usual relative locations.
def _find_microhh_tools():
    cand = []
    for i, a in enumerate(sys.argv):
        if a == '--microhh-python' and i + 1 < len(sys.argv):
            cand.append(sys.argv[i + 1])
        elif a.startswith('--microhh-python='):
            cand.append(a.split('=', 1)[1])
    if os.environ.get('MICROHH_PYTHON'):
        cand.append(os.environ['MICROHH_PYTHON'])
    here = os.path.dirname(os.path.abspath(__file__))
    cand += [here,
             os.path.join(here, '..', '..', '..', 'python'),
             os.path.join(here, '..', '..', 'python'),
             os.path.join(here, '..', 'python')]
    for c in cand:
        if os.path.isfile(os.path.join(c, 'microhh_tools.py')):
            return os.path.abspath(c)
    raise SystemExit(
        'cannot find microhh_tools.py. Pass --microhh-python '
        '<path-to-microhh/python>, or set $MICROHH_PYTHON.')

sys.path.insert(0, _find_microhh_tools())
# strip our own option so argparse below does not see it
sys.argv = [a for i, a in enumerate(sys.argv)
            if not (a == '--microhh-python'
                    or a.startswith('--microhh-python=')
                    or (i > 0 and sys.argv[i - 1] == '--microhh-python'))]

import microhh_tools as mht  # noqa: E402
import argparse
import collections
import glob
import re
import numpy as np
import warnings
from netCDF4 import Dataset
from multiprocessing import Pool, set_start_method
import platform

if platform.system() == 'Darwin':
    set_start_method('fork')

def load_dem(itot, jtot, path='dem.0000000'):
    """
    The immersed-boundary DEM as MicroHH read it: itot*jtot heights above the
    LES floor. Returns None if there is no IB in this run.
    """
    if not os.path.isfile(path):
        return None
    n = os.path.getsize(path)
    dt = 'f8' if n == itot * jtot * 8 else ('f4' if n == itot * jtot * 4
                                            else None)
    if dt is None:
        print('{} is {} B, which is not {}x{} in either precision - '
              'not masking'.format(path, n, itot, jtot))
        return None
    return np.fromfile(path, dtype=dt).reshape(jtot, itot).astype('f8')


# Terrain-following xy planes written by [IB] sw_tf_cross: <var>_tf<h> is
# <var> interpolated to h metres above the LOCAL surface in every column
# (develop_openbc branch). Two things follow from that and are handled
# below:
#
#   1. The crosslist says `thl_tf`, but the FILES are thl_tf10, thl_tf50, ...
#      - one per [IB] tf_cross_heights entry. expand_tf_variables() turns the
#        one crosslist name into the names actually on disk.
#   2. The plane is never inside the terrain, so it must NOT get the masked
#      `_ib` copy that an ordinary fixed-height xy plane gets. Masking it with
#      `dem > 0` (the rule for a flat-floor surface field) would blank every
#      column that has any terrain at all, which here is all of them.
TF_PLANE = re.compile(r'^(?P<var>.+)_tf(?P<h>\d+)$')


def tf_height(stem):
    """Height above the local surface in metres, or None if not a tf plane."""
    m = TF_PLANE.match(stem)
    return int(m.group('h')) if m else None


def expand_tf_variables(variables):
    """
    Replace every bare `<var>_tf` by the `<var>_tf<h>` names found on disk.

    An explicit `<var>_tf100` is left alone, so either spelling works in
    [cross] crosslist. A bare name with no files is reported rather than
    silently dropped - that is what "I asked for the planes and got nothing"
    looks like, and it usually means the run had sw_tf_cross off or the
    binary does not have the terrain-following planes (develop_openbc).
    """
    out = []
    for v in variables:
        if not v.endswith('_tf'):
            out.append(v)
            continue
        found = set()
        for f in glob.glob('{}[0-9]*.*'.format(v)):
            stem = os.path.basename(f).split('.')[0]
            if tf_height(stem) is not None and stem.startswith(v):
                found.add(stem)
        if not found:
            print('{}: no terrain-following planes on disk. Was '
                  '[IB] sw_tf_cross on, and is the binary built from the '
                  'develop_openbc branch?'.format(v))
            continue
        names = sorted(found, key=tf_height)
        print('{} -> {}'.format(v, ', '.join(names)))
        out.extend(names)
    return out


def is_ib_surface(stem):
    """
    An immersed-boundary surface field: `<x>_ib`, or its floor/riser split
    `<x>_ib_floor` / `<x>_ib_wall` (z0m_ib_floor, ustar_ib_floor, ...), or the
    first-air-cell fields `<x>_ib1` (thl_ib1, u_ib1, ...). These
    sit ON the terrain and are valid in every column.
    """
    return stem.endswith(('_ib', '_ib_floor', '_ib_wall', '_ib1'))


def solid_mask(dem, zc, mode, index, at_surface, variable):
    """
    True where the cell is INSIDE the terrain.

    MicroHH marks a cell solid when its centre is at or below the DEM
    (immersed_boundary.cxx:121, `if (z[k] <= zdem)`), which is the test
    reproduced here. `zc` is the vertical coordinate the cross uses - z or zh
    depending on the halflevel flag.

    Surface planes need opposite treatment to each other: thl_bot, ustar and
    friends belong to the FLAT floor, which under terrain is inside the solid,
    so they are solid wherever the DEM rises above it. The immersed-boundary
    fluxes (*_ib) belong to the terrain surface and are valid exactly there.
    """
    if at_surface:
        # Valid everywhere: the IB surface fields sit ON the terrain, and a
        # terrain-following plane sits a fixed height ABOVE it.
        if is_ib_surface(variable) or tf_height(variable) is not None:
            return np.zeros(dem.shape, dtype=bool)
        return dem > 0.0
    if mode == 'xy':
        return zc[index] <= dem
    if mode == 'xz':
        return zc[:, None] <= dem[index, :][None, :]
    if mode == 'yz':
        return zc[:, None] <= dem[:, index][None, :]
    raise ValueError(mode)


def resolve_stem(variable):
    """
    Map a [cross] crosslist entry to the stem MicroHH actually writes.

    They are not always the same. The immersed-boundary fluxes are listed as
    `thlfluxbot_ib` - the parser strips the literal "fluxbot_ib" and what
    remains must be a scalar name (immersed_boundary.cxx:744-752) - but the
    file is written as `thl_fluxbot_ib.xy.000.*`, with an underscore
    (:979). The unmodified cross_to_nc.py looks for the crosslist name, finds
    nothing, and reports "Failed to create thlfluxbot_ib.yz.nc".
    """
    if glob.glob('{}.[xy][yz].*'.format(variable)):
        return variable

    # Every <scalar><suffix> entry in the crosslist is written as
    # <scalar>_<suffix> on disk. Add new ones here when the IB grows a
    # diagnostic - the symptom of forgetting is "Skipping <name>_xy: no such
    # cross-sections" for a variable whose files are sitting right there.
    for suffix in ('fluxbot_ib', 'dsurf_ib'):
        if variable.endswith(suffix):
            alt = variable[:-len(suffix)] + '_' + suffix
            if glob.glob('{}.[xy][yz].*'.format(alt)):
                return alt
    return variable


def _to_centre(a, axis):
    """
    Move a staggered field onto the cell centres along `axis`.

    MicroHH writes u at xh[0..itot-1] and v at yh[0..jtot-1]: itot values, with
    the closing face (xh[itot]) not saved. So the centre value is the average of
    the two bounding faces everywhere except the last cell, where the closing
    face is missing and the face value is used as-is.

    With cyclic boundaries the honest thing there would be to wrap. These runs
    have OPEN boundaries, where wrapping is wrong by the full domain contrast,
    so the last row/column is left half a cell off instead - an error of one
    half grid cell in one line of pixels, rather than a spurious seam.
    """
    b = 0.5 * (a + np.roll(a, -1, axis=axis))
    idx = [slice(None)] * a.ndim
    idx[axis] = -1
    b[tuple(idx)] = a[tuple(idx)]
    return b


def _wind_dir(u, v):
    """
    Meteorological wind direction: the compass bearing the wind comes FROM,
    degrees clockwise from north. A northerly is 0/360, an easterly 90.
    """
    return np.mod(270.0 - np.degrees(np.arctan2(v, u)), 360.0)


WIND_PAIRS = [
    # (u stem, v stem, output stem, variable suffix)
    #
    # `u`/`v` are the raw prognostic components, stored STAGGERED at xh and yh,
    # so they are brought to the cell centre first. The `_ib` surface
    # diagnostics are already co-located there - MicroHH interpolates before
    # it applies the MOST ratio - so they are used as they
    # are. _load_centred() tells the two apart from the file's own dimensions
    # rather than from the name.
    ('u',        'v',        'wind',        ''),
    ('u_10m_ib', 'v_10m_ib', 'wind_10m_ib', '_10m_ib'),
    ('u_ib1',    'v_ib1',    'wind_ib1',    '_ib1'),
]


def _load_centred(ds, name):
    """
    Read a horizontal wind component and put it on the CELL CENTRES.

    Staggering is read off the variable's own dimensions: a trailing `xh` means
    u at the x faces, a `yh` in the second-to-last slot means v at the y faces.
    Anything else is taken as already centred, which is what the terrain-
    following `_ib` diagnostics are.
    """
    a = np.array(ds[name][:], dtype='f8')
    dims = ds[name].dimensions
    if dims[-1] == 'xh':
        a = _to_centre(a, -1)
    if len(dims) > 1 and dims[-2] == 'yh':
        a = _to_centre(a, -2)
    return a


def _centre_dims(dims):
    return tuple('x' if d == 'xh' else 'y' if d == 'yh' else d for d in dims)


def derive_wind_pair(mode, u_stem, v_stem, out_stem, suffix):
    """
    <u>.<mode>.nc + <v>.<mode>.nc -> <out>.<mode>.nc, holding U and wdir.

    XY PLANES ONLY for the STAGGERED pair, and the reason is staggering rather
    than laziness. On an xz plane at index j, u is stored at xh on that plane
    but v is stored at yh - half a cell off the plane itself - and the
    neighbouring plane that would be needed to interpolate it back is not
    written. Computing a speed from the two as-stored would mix a value from
    one place with a value from half a cell away and report it as a wind speed.

    The `_ib` pairs are surface fields and only exist as xy in any case.
    """
    fu = '{}.{}.nc'.format(u_stem, mode)
    fv = '{}.{}.nc'.format(v_stem, mode)
    if not (os.path.isfile(fu) and os.path.isfile(fv)):
        return False

    out = '{}.{}.nc'.format(out_stem, mode)
    with Dataset(fu) as du, Dataset(fv) as dv, Dataset(out, 'w') as dst:
        dst.setncattr('note',
                      'U{0} = sqrt(u^2+v^2) on the cell centres; wdir{0} is '
                      'the meteorological direction the wind comes FROM, '
                      'degrees clockwise from north.'.format(suffix))

        dims_out = _centre_dims(du[u_stem].dimensions)
        for d in dims_out:
            n = (len(du.dimensions[d]) if d in du.dimensions
                 else len(dv.dimensions[d]))
            dst.createDimension(d, n)
            src = du if d in du.variables else (dv if d in dv.variables else None)
            if src is not None:
                cv = dst.createVariable(d, 'f8', (d,))
                cv[:] = src[d][:]
                for at in src[d].ncattrs():
                    cv.setncattr(at, src[d].getncattr(at))

        for ib in ('', '_ib'):
            nu, nv = u_stem + ib, v_stem + ib
            if nu not in du.variables or nv not in dv.variables:
                continue
            u = _load_centred(du, nu)
            v = _load_centred(dv, nv)

            vs = dst.createVariable('U' + suffix + ib, 'f4', dims_out,
                                    zlib=compression)
            vs.units = 'm s-1'
            vs.long_name = 'horizontal wind speed'
            vs[:] = np.sqrt(u * u + v * v)

            vd = dst.createVariable('wdir' + suffix + ib, 'f4', dims_out,
                                    zlib=compression)
            vd.units = 'degrees'
            vd.long_name = 'wind direction (from), clockwise from north'
            vd[:] = _wind_dir(u, v)

    print('wind: wrote {}'.format(out))
    return True


def derive_wind(mode):
    """Every pair in WIND_PAIRS that has files on disk for this mode."""
    if mode != 'xy':
        print('wind: skipping {} - u and v cannot be co-located on that '
              'plane (see derive_wind_pair)'.format(mode))
        return
    made = 0
    for u_stem, v_stem, out_stem, suffix in WIND_PAIRS:
        made += derive_wind_pair(mode, u_stem, v_stem, out_stem, suffix)
    if made == 0:
        print('wind: no u/v pair found for {}'.format(mode))


def write_hourly(path, window):
    """
    Block-average every time-dependent field in `path` over `window` seconds.

    The output time is the CENTRE of each block, with time_bnds carrying the
    edges, so a plot against `time` lands where the average actually applies.
    A block is written only if it holds at least one sample; a partial block at
    the end of the run is averaged over what it has and its bounds say so.
    """
    out = path[:-3] + '.hourly.nc'
    with Dataset(path) as src:
        if 'time' not in src.dimensions:
            return
        t = np.array(src['time'][:], dtype='f8')
        if t.size == 0:
            return

        binid = np.floor(t / window).astype(int)
        bins = np.unique(binid)

        with Dataset(out, 'w') as dst:
            dst.setncattr('note',
                          'block means of {} over {:g} s. time is the block '
                          'centre; time_bnds are its edges.'
                          .format(os.path.basename(path), window))
            for name, d in src.dimensions.items():
                dst.createDimension(name,
                                    len(bins) if name == 'time' else len(d))
            dst.createDimension('nv', 2)

            tv = dst.createVariable('time', 'f8', ('time',))
            tv.units = 's'
            tv.long_name = 'centre of the averaging block'
            tv.bounds = 'time_bnds'
            tv[:] = (bins + 0.5) * window
            tb = dst.createVariable('time_bnds', 'f8', ('time', 'nv'))
            tb[:, 0] = bins * window
            tb[:, 1] = (bins + 1) * window

            nsv = dst.createVariable('nsamples', 'i4', ('time',))
            nsv.long_name = 'samples contributing to each block'
            nsv[:] = [int((binid == b).sum()) for b in bins]

            for name, var in src.variables.items():
                if name in ('time', 'time_bnds', 'nsamples'):
                    continue
                if 'time' not in var.dimensions:
                    v_ = dst.createVariable(name, var.dtype, var.dimensions)
                    v_[:] = var[:]
                    for a in var.ncattrs():
                        v_.setncattr(a, var.getncattr(a))
                    continue
                if name.startswith('wdir'):
                    # circular: rebuilt from the mean vector afterwards
                    continue

                v_ = dst.createVariable(name, 'f4', var.dimensions,
                                        zlib=compression)
                for a in var.ncattrs():
                    v_.setncattr(a, var.getncattr(a))
                data = np.array(var[:], dtype='f8')
                for i, b in enumerate(bins):
                    sel = (binid == b)
                    # An all-NaN column is normal, not an error: with
                    # --mask-ib the `_ib` copies are NaN wherever the level
                    # sits inside the terrain, and at z = 9 m that is most of
                    # the domain. numpy warns about it once per slice.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        v_[i] = np.nanmean(data[sel], axis=0)

    print('hourly: wrote {}'.format(out))


def rebuild_hourly_tf_wind():
    """
    The same repair for the terrain-following planes, which WIND_PAIRS cannot
    cover because their names carry a height: wdir_tf10, wdir_tf100, ...

    u_tf<h> and v_tf<h> live in their own files (one variable per file), so
    the mean vector is assembled across three of them. wspd_vec is added
    alongside, and wspd_vec / wspd_tf<h> is the steadiness at that height.
    """
    for f in sorted(glob.glob('wdir_tf[0-9]*.xy.hourly.nc')):
        stem = os.path.basename(f).split('.')[0]
        h = tf_height(stem)
        if h is None:
            continue
        fu = 'u_tf{}.xy.hourly.nc'.format(h)
        fv = 'v_tf{}.xy.hourly.nc'.format(h)
        if not (os.path.isfile(fu) and os.path.isfile(fv)):
            print('hourly: {} left as a block mean - u_tf{}/v_tf{} are not '
                  'there to rebuild it from. Treat it with suspicion: a mean '
                  'of 350 and 10 degrees is 180.'.format(f, h, h))
            continue
        with Dataset(f, 'a') as dst, Dataset(fu) as du, Dataset(fv) as dv:
            u = np.asarray(du['u_tf{}'.format(h)][:], dtype='f8')
            v = np.asarray(dv['v_tf{}'.format(h)][:], dtype='f8')
            # write_hourly deliberately leaves a wdir* variable OUT of the
            # hourly file rather than block-averaging it, so this creates it.
            if stem not in dst.variables:
                dims = du['u_tf{}'.format(h)].dimensions
                vw = dst.createVariable(stem, 'f4', dims, zlib=compression)
                vw.units = 'degrees'
            dst[stem][:] = _wind_dir(u, v)
            dst[stem].long_name = (
                'direction of the block-mean wind VECTOR (from), clockwise '
                'from north, at {} m above the local surface'.format(h))
            if 'wspd_vec' not in dst.variables:
                vv = dst.createVariable('wspd_vec', 'f4',
                                        dst[stem].dimensions,
                                        zlib=compression)
                vv.units = 'm s-1'
                vv.long_name = ('magnitude of the block-mean wind VECTOR; '
                                'divide by wspd_tf{} for the steadiness'
                                .format(h))
                vv[:] = np.sqrt(u * u + v * v)
        print('hourly: rebuilt {} from the mean vector'.format(f))


def rebuild_hourly_wind(mode):
    """
    Put wdir back into <out>.<mode>.hourly.nc, from the hourly MEAN VECTOR.

    Averaging a direction as if it were a scalar is wrong in a way that is easy
    to miss: 350 deg and 10 deg average to 180, the exact opposite of the
    truth. The mean wind direction is the direction of the mean vector, so that
    is what is written - and U_vec, its magnitude, comes with it. U_vec/U is
    the steadiness: 1 for a wind that held its direction all hour, near 0 for
    one that boxed the compass.
    """
    for u_stem, v_stem, out_stem, suffix in WIND_PAIRS:
        f  = '{}.{}.hourly.nc'.format(out_stem, mode)
        fu = '{}.{}.hourly.nc'.format(u_stem, mode)
        fv = '{}.{}.hourly.nc'.format(v_stem, mode)
        if not all(os.path.isfile(x) for x in (f, fu, fv)):
            continue

        with Dataset(f, 'a') as dst, Dataset(fu) as du, Dataset(fv) as dv:
            for ib in ('', '_ib'):
                nu, nv = u_stem + ib, v_stem + ib
                if nu not in du.variables or nv not in dv.variables:
                    continue
                u = _load_centred(du, nu)
                v = _load_centred(dv, nv)
                dims = _centre_dims(du[nu].dimensions)

                name = 'U_vec' + suffix + ib
                if name not in dst.variables:
                    vv = dst.createVariable(name, 'f4', dims, zlib=compression)
                    vv.units = 'm s-1'
                    vv.long_name = ('magnitude of the block-mean wind VECTOR; '
                                    'U_vec/U is the steadiness')
                    vv[:] = np.sqrt(u * u + v * v)

                name = 'wdir' + suffix + ib
                if name not in dst.variables:
                    vd = dst.createVariable(name, 'f4', dims, zlib=compression)
                    vd.units = 'degrees'
                    vd.long_name = ('direction of the block-mean wind vector '
                                    '(from), clockwise from north')
                    vd[:] = _wind_dir(u, v)

        print('hourly: rebuilt wdir in {} from the mean vector'.format(f))


def convert_to_nc(variables):
    dem = load_dem(itot, jtot, dem_path) if mask_ib else None

    # Loop over the different variables and crosssections
    for variable in variables:
        stem = resolve_stem(variable)
        if stem != variable:
            print('{}: files are written as {}'.format(variable, stem))

        for mode in modes:
            # Skip modes this variable does not have. Surface planes
            # (thl_bot, ustar, the IB fluxes ...) are written by
            # cross.cross_plane and exist as `xy` only, so asking for xz/yz
            # is not an error - there is simply nothing there.
            if not glob.glob('{}.{}.*'.format(stem, mode)):
                print('Skipping {}_{}: no such cross-sections'
                      .format(variable, mode))
                continue

            print('Processing {}_{}'.format(variable, mode))

            try:
                otime = int(round(starttime / 10**iotimeprec))

                if os.path.isfile("{0}.xy.000.{1:07d}".format(stem, otime)):
                    if mode != 'xy':
                        continue
                    at_surface = True
                else:
                    at_surface = False

                filename = "{0}.{1}.nc".format(stem, mode)
                halflevel = '000'
                if not at_surface:
                    if indexes is None:
                        indexes_local,halflevel = mht.get_cross_indices(stem, mode)
                    else:
                        indexes_local = indexes

                        files = glob.glob("{0:}.{1}.*.{2:05d}.{3:07d}".format(
                                stem, mode, indexes_local[0], otime))
                        if len(files) == 0:
                            raise Exception('Cannot find any cross-section')
                        halflevel = files[0].split('.')[-3]

                dim = collections.OrderedDict()
                dim['time'] = []
                dim['z'] = range(ktot)
                dim['y'] = range(jtot)
                dim['x'] = range(itot)

                if at_surface:
                    dim.pop('z')
                    n = itot * jtot
                    indexes_local = [-1]
                elif mode == 'xy':
                    dim.update({'z': []})
                    n = itot * jtot
                elif mode == 'xz':
                    dim.update({'y': []})
                    n = itot * ktot
                elif mode == 'yz':
                    dim.update({'x': []})
                    n = ktot * jtot

                if halflevel[0] == '1':
                    dim['xh'] = dim.pop('x')
                if halflevel[1] == '1':
                    dim['yh'] = dim.pop('y')
                if halflevel[2] == '1':
                    dim['zh'] = dim.pop('z')
                ncfile = mht.Create_ncfile(
                    grid, filename, stem, dim, precision, compression)

                h_tf = tf_height(stem)
                if h_tf is not None:
                    base = TF_PLANE.match(stem).group('var')
                    ncfile.var.long_name = (
                        '{} at {} m above the LOCAL surface '
                        '(terrain-following; dem(i,j) + {} m, interpolated '
                        'in z)'.format(base, h_tf, h_tf))
                    ncfile.var.height_above_surface = float(h_tf)
                    ncfile.var.vertical_reference = 'local IB surface (dem)'

                if not at_surface and len(indexes_local) > 1:
                    sliced = ('y' if mode == 'xz' else 'x')
                    sliced = sliced + ('h' if (
                        (mode == 'xz' and halflevel[1] == '1') or
                        (mode == 'yz' and halflevel[0] == '1')) else '')
                    print('{}.{}: {} parallel slices at {} = {}'.format(
                        stem, mode, len(indexes_local), sliced,
                        ', '.join('{:.0f}'.format(grid.dim[sliced][i])
                                  for i in indexes_local)))

                # A second variable in the SAME file holding the masked copy,
                # so raw and masked sit side by side and can be compared
                # without opening two datasets.
                var_ib = None
                zc_all = None
                # A surface plane that already belongs to the immersed
                # boundary (thl_fluxbot_ib and friends) is valid everywhere
                # it is written, so a masked copy would be identical: skip it
                # rather than write `thl_fluxbot_ib_ib`.
                if dem is not None and at_surface and (
                        is_ib_surface(stem) or tf_height(stem) is not None):
                    print('{}: terrain-following, valid in every column - no '
                          'masked copy'.format(stem))
                elif dem is not None:
                    var_ib = ncfile.ncfile.createVariable(
                        stem + '_ib', ncfile.var.dtype,
                        ncfile.var.dimensions, zlib=compression)
                    var_ib.long_name = (
                        'as ' + stem + ', with cells inside the immersed '
                        'boundary set to NaN')
                    # NOTE the truncation. Read_grid gives zh with ktot+1
                    # entries - it is the half-level grid, and it has one more
                    # level than z. But a cross-section plane carries ktot
                    # values whatever the staggering: `n` above is ktot*itot
                    # or ktot*jtot, and the vertical dimension built here is
                    # range(ktot), merely RENAMED to 'zh'. Using all ktot+1
                    # heights builds a mask one row too tall and the reshape
                    # fails with "cannot reshape array of size 4096 into shape
                    # (65,64)" - which is exactly what w.yz did.
                    zc_all = (grid.dim['zh'][:ktot] if halflevel[2] == '1'
                              else grid.dim['z'])
                
                for key, val in dim.items():
                    if key == 'time':
                        continue
                    elif val == []:
                        ncfile.dimvar[key][:] = grid.dim[key][indexes_local]

                stop = False
                for t, time in enumerate(np.arange(starttime, endtime + sampletime, sampletime)):
                    for k in range(len(indexes_local)):
                        index = indexes_local[k]
                        otime = int(
                            round(
                                (time) / 10**iotimeprec))
                        if at_surface:
                            f_in = "{0}.{1}.{2}.{3:07d}".format(
                                stem, mode, halflevel, otime)
                        else:
                            f_in = "{0:}.{1}.{2}.{3:05d}.{4:07d}".format(
                                stem, mode, halflevel, index, otime)
                        try:
                            fin = mht.Read_binary(grid, f_in)
                        except Exception as ex:
                            print (ex)
                            stop = True
                            break

                        #print(
                        #    "Processing %8s, time=%7i, index=%4i" %
                        #    (variable, otime, index))

                        ncfile.dimvar['time'][t] = time

                        data = fin.read(n)

                        if at_surface:
                            ncfile.var[t, :, :] = data
                        elif mode == 'xy':
                            ncfile.var[t, k, :, :] = data
                        elif mode == 'xz':
                            ncfile.var[t, :, k, :] = data
                        elif mode == 'yz':
                            ncfile.var[t, :, :, k] = data

                        if var_ib is not None:
                            # mht.Read_binary.read() returns a FLAT array;
                            # netCDF4 reshapes on assignment, numpy does not.
                            m = solid_mask(dem, zc_all, mode, index,
                                           at_surface, stem)
                            md = np.where(m, np.nan,
                                          np.asarray(data).reshape(m.shape))
                            if at_surface:
                                var_ib[t, :, :] = md
                            elif mode == 'xy':
                                var_ib[t, k, :, :] = md
                            elif mode == 'xz':
                                var_ib[t, :, k, :] = md
                            elif mode == 'yz':
                                var_ib[t, :, :, k] = md

                        fin.close()

                    if stop:
                        break

                ncfile.close()

            except Exception as ex:
                print(ex)
                print("Failed to create %s" % filename)



# Parse command line and namelist options
cross_modes = ['xy', 'xz', 'yz']
parser = argparse.ArgumentParser(
    description='Convert MicroHH binary cross-sections to netCDF4 files.')
parser.add_argument(
    '-m',
    '--modes',
    nargs='*',
    help='mode of the cross section',
    choices=cross_modes)
parser.add_argument('-f', '--filename', help='ini file name')
parser.add_argument('-d', '--directory', help='directory')
parser.add_argument('-v', '--vars', nargs='*', help='variable names')
parser.add_argument('-x', '--index', nargs='*', help='indices', type=int)
parser.add_argument('-t0', '--starttime', help='first time step to be parsed')
parser.add_argument('-t1', '--endtime', help='last time step to be parsed')
parser.add_argument(
    '-tstep',
    '--sampletime',
    help='time interval to be parsed')
parser.add_argument(
    '-p',
    '--precision',
    help='precision',
    choices=[
        'single',
         'double'])
parser.add_argument(
    '-n',
    '--nprocs',
    help='Number of processes',
    type=int,
    default=1)
parser.add_argument(
    '-c',
    '--nocompression',
    help='do not compress the netcdf file',
    action='store_true')

parser.add_argument(
    '--mask-ib',
    dest='mask_ib',
    action='store_true',
    help='also write <var>_ib, a copy with cells inside the immersed '
         'boundary set to NaN')
parser.add_argument(
    '--dem',
    help='DEM file for --mask-ib (default: dem.0000000)',
    default='dem.0000000')
parser.add_argument(
    '--wind',
    action='store_true',
    help='derive wind SPEED and DIRECTION for every u/v pair on disk: the '
         'prognostic u,v into wind.xy.nc, and the terrain-following '
         'u_10m_ib,v_10m_ib and u_ib1,v_ib1 into wind_10m_ib.xy.nc and '
         'wind_ib1.xy.nc. xy planes only - see derive_wind_pair()')
parser.add_argument(
    '--hourly',
    action='store_true',
    help='also write <stem>.<mode>.hourly.nc, every field block-averaged over '
         '--hourly-window seconds')
parser.add_argument(
    '--hourly-window',
    dest='hourly_window',
    type=float,
    default=3600.0,
    help='averaging window in seconds (default 3600)')
parser.add_argument(
    '-o',
    '--order',
    help='order',
    choices=[
        2, 4], type = int)

args = parser.parse_args()

if args.directory is not None:
    os.chdir(args.directory)

modes = args.modes
indexes = args.index
mask_ib = args.mask_ib
dem_path = args.dem

nl = mht.Read_namelist(args.filename)
itot = nl['grid']['itot']
jtot = nl['grid']['jtot']
ktot = nl['grid']['ktot']

starttime = float(
    args.starttime) if args.starttime is not None else nl['time']['starttime']
endtime = float(
    args.endtime) if args.endtime is not None else nl['time']['endtime']
sampletime = float(
    args.sampletime) if args.sampletime is not None else nl['cross']['sampletime']

if args.modes is None:
    modes = list(nl['cross'].keys() & cross_modes)

    # Check if there are paths in the cross-list
    if 'xy' not in modes:
        for v in nl['cross']['crosslist']:
            if 'path' in v:
                modes.append('xy')
                break
    # Surface planes (the IB's *_ib fields, taux_form_ib, ...) are written
    # as xy even when [cross] has no xy= heights: convert them too.
    if 'xy' not in modes and glob.glob('*.xy.*[0-9]'):
        modes.append('xy')
else:
    modes = args.modes

if 'iotimeprec' in nl['time']:
    iotimeprec = nl['time']['iotimeprec']
else:
    iotimeprec = 0.

variables = args.vars if args.vars is not None else nl['cross']['crosslist']

# In case variables is a single string, convert to list.
variables = [ variables ] if not isinstance(variables, list) else variables

# `thl_tf` in the crosslist is one plane per [IB] tf_cross_heights entry on
# disk. Expand before the work is chunked over the pool, so the heights are
# spread over the processes like any other variable.
variables = expand_tf_variables(variables)

precision = args.precision
nprocs = args.nprocs if args.nprocs is not None else len(variables)
compression = not(args.nocompression)
do_wind = args.wind
do_hourly = args.hourly
hourly_window = args.hourly_window
try:
    order = args.order if args.order is not None else nl['grid']['swspatialorder']
except KeyError:
    order = 2

# End option parsing
grid = mht.Read_grid(itot, jtot, ktot, order = order)

chunks = [variables[i::nprocs] for i in range(nprocs)]

pool = Pool(processes=nprocs)

pool.imap_unordered(convert_to_nc, chunks)

pool.close()
pool.join()

# ---------------------------------------------------------------------------
#   Post-passes: derived wind, then hourly means.
#
# Both work on the netCDFs the pool has just written rather than on the
# binaries, so they cost one extra read and are independent of the staggering
# bookkeeping above.
# ---------------------------------------------------------------------------
if do_wind:
    for mode in modes:
        derive_wind(mode)

if do_hourly:
    targets = sorted(set(glob.glob('*.xy.nc') + glob.glob('*.xz.nc')
                         + glob.glob('*.yz.nc')))
    for f in targets:
        if f.endswith('.hourly.nc'):
            continue
        write_hourly(f, hourly_window)
    # wdir cannot be block-averaged like a scalar: 350 deg and 10 deg average
    # to 180, which is the opposite direction. Rebuild it from the hourly MEAN
    # VECTOR instead, which is what a mean wind direction means.
    for mode in modes:
        rebuild_hourly_wind(mode)
    rebuild_hourly_tf_wind()
