#!/bin/bash
#
# Job for one axelsen2009_lbc run directory. It runs on a laptop or workstation
# as well as on a SLURM cluster such as Snellius:
#
#   cd runs/P1_lbc
#   bash ../../job.sh          # laptop / workstation (uses mpirun)
#   sbatch ../../job.sh        # SLURM cluster (check the #SBATCH lines first)
#
# The run directory must hold what axelsen2009_lbc_input.py wrote and a copy of
# (or a link to) the microhh binary. The script does the three steps of a run:
#   1. microhh init           (grid, base state, the input profiles)
#   2. axelsen2009_lbc_init.py (initial SFS TKE above the local surface)
#   3. microhh run
# and the quick look (check_<time>.png) and the slope-averaged profiles
# (figs/, axelsen2009_lbc_analysis.py; OBS=<csv> adds observed profiles).
#
# Restart (a run longer than one job, or one that was cut off): in the same
# run directory, with the restart files of time T (u.%07d etc., written
# every [time] savetime = 3600 s),
#   RESTART_TIME=7200 sbatch --export=ALL ../../job.sh     (or: RESTART_TIME=7200 bash ../../job.sh)
# skips steps 1-2, sets [time] starttime = T in axelsen2009_lbc.ini and continues.
#
# Settings you can pass as environment variables:
#   MICROHH_ENV  a script to source first, e.g. the module loads of your build
#   MPIRUN       the MPI launcher (default: "srun -n" in a SLURM job, else "mpirun -np")
#   CASE_DIR     the folder with the case scripts (default: two levels up)
#
#SBATCH --job-name=axelsen2009_lbc
#SBATCH --nodes=1
#SBATCH --ntasks=128                # at least npx*npy of axelsen2009_lbc.ini (both presets use 128)
#SBATCH --time=12:00:00             # --res coarse: < 1 h on 128 cores; --res fine: 8-14 h (use RESTART_TIME to continue)
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
##SBATCH --partition=...            # add the partition / account your cluster needs

set -eu

# Optional: the modules of the build. Python needs numpy, netCDF4 and matplotlib.
if [ -n "${MICROHH_ENV:-}" ]; then source "$MICROHH_ENV"; fi

RUN_DIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$RUN_DIR"
# Where the case scripts are (SLURM runs a copy of this script, so $0 does
# not tell): CASE_DIR if set, else two levels up (runs/<case>_<bc>).
CASE_DIR=${CASE_DIR:-$(cd "$RUN_DIR/../.." && pwd)}
if [ ! -f "$CASE_DIR/axelsen2009_lbc_init.py" ]; then
    echo "ERROR: axelsen2009_lbc_init.py not in $CASE_DIR - set CASE_DIR=/path/to/cases/axelsen2009_lbc"
    exit 1
fi

# The decomposition the generator chose ([master] npx x npy).
npx=$(awk -F= '/^npx=/{print $2}' axelsen2009_lbc.ini)
npy=$(awk -F= '/^npy=/{print $2}' axelsen2009_lbc.ini)
NP=$((npx*npy))
if [ $NP -gt ${SLURM_NTASKS:-$NP} ]; then
    echo "ERROR: npx*npy = $NP in axelsen2009_lbc.ini, more than the ${SLURM_NTASKS} tasks of the job."
    exit 1
fi
if [ -z "${MPIRUN:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then MPIRUN="srun -n"; else MPIRUN="mpirun -np"; fi
fi
RUN="$MPIRUN $NP"
echo "MPI: $npx x $npy = $NP tasks ($RUN)"

[ -x ./microhh ] || { echo "ERROR: no ./microhh in $RUN_DIR (copy or link the binary)"; exit 1; }

if [ -n "${RESTART_TIME:-}" ]; then
    T=$(printf '%07d' "$RESTART_TIME")
    for f in u v w thl qt time; do
        [ -e "$f.$T" ] || { echo "ERROR: restart file $f.$T not found in $RUN_DIR"; exit 1; }
    done
    # MicroHH does not overwrite output: move everything written after T by
    # the previous job (crosses, restarts, stats files) out of the way, or
    # the run stops at the first save ("FAILED", Timeloop::save exception).
    BAK="after_${T}_$(date +%Y%m%d%H%M%S)"
    mkdir -p "$BAK"
    for f in *; do
        [ -f "$f" ] || continue
        t=${f##*.}; [ "$t" = nc ] && { t=${f%.nc}; t=${t##*.}; }
        if [[ "$t" =~ ^[0-9]{7}$ ]] && [ $((10#$t)) -gt $((10#$T)) ]; then mv "$f" "$BAK/"; fi
        if [[ "$f" =~ \.$T\.nc$ ]]; then mv "$f" "$BAK/"; fi
    done
    echo "Restart: output after t = $RESTART_TIME s (and stats files of t = $RESTART_TIME s) moved to $BAK/"
    sed -i "s/^starttime=.*/starttime=$RESTART_TIME/" axelsen2009_lbc.ini
    echo "Restart from t = $RESTART_TIME s ([time] starttime set in axelsen2009_lbc.ini)"
    $RUN ./microhh run axelsen2009_lbc
    python3 "$CASE_DIR/axelsen2009_lbc_analysis.py" --rundir . ${OBS:+--obs "$OBS"} || echo "analysis failed"
    python3 "$CASE_DIR/axelsen2009_lbc_turbulence.py" --rundir . || echo "turbulence diagnostics failed"
    python3 "$CASE_DIR/axelsen2009_lbc_check.py" --rundir . || \
        echo "check failed (the run itself finished)"
    exit 0
fi

# A fresh start only: init refuses to overwrite its own files.
if [ -e grid.0000000 ]; then
    echo "ERROR: grid.0000000 exists - this directory has been initialised already."
    echo "       Make a new one (axelsen2009_lbc_input.py --outdir ...) for a clean start."
    exit 1
fi

$RUN ./microhh init axelsen2009_lbc

# Initial SFS TKE (tke2); skipped for smag2.
python3 "$CASE_DIR/axelsen2009_lbc_init.py" --rundir . || echo "no initial TKE applied"

$RUN ./microhh run axelsen2009_lbc

python3 "$CASE_DIR/axelsen2009_lbc_analysis.py" --rundir . ${OBS:+--obs "$OBS"} || echo "analysis failed"
python3 "$CASE_DIR/axelsen2009_lbc_turbulence.py" --rundir . || echo "turbulence diagnostics failed"
python3 "$CASE_DIR/axelsen2009_lbc_check.py" --rundir . || \
    echo "check failed (the run itself finished)"
