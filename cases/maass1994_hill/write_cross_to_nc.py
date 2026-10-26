#!/usr/bin/env python3
"""
Put the cross-section converter into a MicroHH run folder.

    python3 write_cross_to_nc.py --rundir run_dir --case maass1994 [--window 10]

or, from a case generator,

    from write_cross_to_nc import install
    install(rundir, 'maass1994', window=10.)

It writes, in the run folder:
    cross_to_nc.sh   job that converts the cross-section binaries to
                     netCDF (one task with one CPU per crosslist entry).
                     It runs on a laptop and on a SLURM cluster:
                         bash cross_to_nc.sh
                         sbatch cross_to_nc.sh     (from the run folder)
    cross_to_nc.py   an extended copy of MicroHH's converter (IB mask with
                     --mask-ib, odd IB file stems, terrain-following planes,
                     several xz/yz slices; see its header)
    microhh_tools.py the MicroHH module it imports
    write_cross_to_nc.py  this script, so the folder can redo it

Both .py files are taken from the folder of this script, else from one or
two folders up.

Options in the job (environment variables at submit time):
    T0, T1   first and last time to convert [s] (default: all)
    CASE     another folder to convert instead of the job's own
    WINDOW   averaging window [s] for <stem>.<mode>.hourly.nc; 0 = none
             (the default is the --window given here)
"""
import argparse
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
# write_cross_to_nc.py itself too: a generator copied into the run folder
# (andren1994_setup.py --perturb) imports it from there.
SOURCES = ('cross_to_nc.py', 'microhh_tools.py', 'write_cross_to_nc.py')


def _find(name):
    for d in (HERE, os.path.join(HERE, '..', '..'), os.path.join(HERE, '..')):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return os.path.abspath(p)
    raise SystemExit(f'write_cross_to_nc: {name} not found next to {__file__} '
                     f'or one or two folders up')


def _crosslist(ini_path):
    """crosslist entries of the [cross] section, [] when cross is off."""
    sec, on, lst = None, False, []
    with open(ini_path) as f:
        for line in f:
            s = line.split('#', 1)[0].strip()
            if s.startswith('[') and s.endswith(']'):
                sec = s[1:-1]
                continue
            if sec != 'cross' or '=' not in s:
                continue
            k, v = (x.strip() for x in s.split('=', 1))
            if k == 'swcross':
                on = v.lower() in ('1', 'true')
            elif k == 'crosslist':
                lst = [x.strip() for x in v.split(',') if x.strip()]
    return lst if on else []


JOB = r"""#!/bin/bash
#SBATCH --job-name=@NAME@_cross2nc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=@NCPU@         # crosslist has @NCROSS@ entries
#SBATCH --mem=16G
#SBATCH --time=@WALL@
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
#
# Cross-section binaries -> netCDF for this run folder (written by
# write_cross_to_nc.py). On a laptop:   bash cross_to_nc.sh
# On a SLURM cluster, from the run folder: sbatch cross_to_nc.sh
# Options (environment variables): T0, T1 (s, the times to convert), CASE
# (another folder), WINDOW (s, time-mean files <stem>.<mode>.hourly.nc; 0 = none),
# MICROHH_ENV (a script to source first, e.g. module loads).
#
# cross_to_nc.py uses shared-memory multiprocessing, not MPI: one task with
# at most one CPU per crosslist entry.

set -euo pipefail

# This script's folder, also under Slurm (which runs a copy of it):
# 1. the Command= of the job, 2. the submit directory, 3. dirname $0.
HERE=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    CMD=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | awk -F= '/^ *Command=/{print $2; exit}')
    CMD=${CMD%% *}
    [ -n "$CMD" ] && [ -f "$CMD" ] && HERE=$(cd "$(dirname "$CMD")" && pwd)
fi
if [ -z "$HERE" ] && [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/@CASE@.ini" ]; then
    HERE=$SLURM_SUBMIT_DIR
fi
[ -z "$HERE" ] && HERE=$(cd "$(dirname "$0")" && pwd)
cd "${CASE:-$HERE}"
echo "converting cross-sections in $(pwd)"

# Optional environment (module loads); python3 needs numpy and netCDF4.
if [ -n "${MICROHH_ENV:-}" ]; then source "$MICROHH_ENV"; fi

for f in cross_to_nc.py microhh_tools.py; do
    if [ ! -f "$f" ] && [ -f "$HERE/$f" ]; then cp "$HERE/$f" .; fi
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not in $(pwd) (write_cross_to_nc.py copies it there)"; exit 1
    fi
done

NPROCS=${SLURM_CPUS_PER_TASK:-@NCPU@}
ARGS=(-f @CASE@.ini -n "$NPROCS" -t0 "${T0:-0}")
if [ -n "${T1:-}" ]; then ARGS+=(-t1 "$T1"); fi
@MASK@
@WIND@
WINDOW=${WINDOW:-@WINDOW@}
if [ "$WINDOW" != "0" ]; then ARGS+=(--hourly --hourly-window "$WINDOW"); fi

python3 cross_to_nc.py "${ARGS[@]}"
echo "done: $(ls -1 *.nc 2>/dev/null | wc -l) netCDF files in $(pwd)"
"""


def install(rundir, case, window=0., mask_ib=None, wind=None, wall='2:00:00', quiet=False):
    """Write cross_to_nc.sh and copy the converter into rundir.

    window: default averaging window [s] for the time-mean files (0: none).
    mask_ib/wind: None = decide from the ini ([IB] present / u and v in
    the crosslist)."""
    ini = os.path.join(rundir, f'{case}.ini')
    if not os.path.isfile(ini):
        raise SystemExit(f'write_cross_to_nc: {ini} does not exist')
    cl = _crosslist(ini)
    if not cl:
        if not quiet:
            print(f'  cross_to_nc: [cross] is off in {case}.ini, nothing installed')
        return False
    if mask_ib is None:
        txt = open(ini).read()
        mask_ib = '[IB]' in txt and 'sw_immersed_boundary=dem' in txt.replace(' ', '')
    if wind is None:
        wind = 'u' in cl and 'v' in cl
    ncpu = max(1, min(len(cl), 32))
    job = (JOB.replace('@NAME@', case).replace('@CASE@', case)
              .replace('@NCPU@', str(ncpu)).replace('@NCROSS@', str(len(cl)))
              .replace('@WALL@', wall).replace('@WINDOW@', '%g' % window)
              .replace('@MASK@', 'ARGS+=(--mask-ib)   # also <var>_ib: the field with the terrain set to NaN'
                       if mask_ib else '# no immersed boundary: no --mask-ib')
              .replace('@WIND@', 'ARGS+=(--wind)      # wind speed and direction from u, v (xy planes)'
                       if wind else '# no u and v pair in the crosslist: no --wind'))
    fn = os.path.join(rundir, 'cross_to_nc.sh')
    with open(fn, 'w') as f:
        f.write(job)
    os.chmod(fn, 0o755)
    for name in SOURCES:
        src = _find(name)
        dst = os.path.join(rundir, name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
    if not quiet:
        print(f'  cross_to_nc.sh + cross_to_nc.py + microhh_tools.py in {rundir} '
              f'({len(cl)} cross variables, mask_ib={mask_ib}, wind={wind}, window={window:g} s)')
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rundir', default='.')
    ap.add_argument('--case', required=True, help='case name: <case>.ini in --rundir')
    ap.add_argument('--window', type=float, default=0.,
                    help='default averaging window [s] for time-mean files (0: none)')
    ap.add_argument('--wall', default='2:00:00', help='Slurm time limit')
    a = ap.parse_args()
    install(a.rundir, a.case, window=a.window, wall=a.wall)


if __name__ == '__main__':
    main()
