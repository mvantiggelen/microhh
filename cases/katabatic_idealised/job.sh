#!/bin/bash
#
# Job for one katabatic_idealised run directory, ib/ or rotated/. With
# --inflow rotated (the default) the two run ONE AFTER THE OTHER: the IB
# inflow is the equilibrium of the finished rotated run. It runs on a laptop
# or workstation as well as on a SLURM cluster such as Snellius:
#
#   python3 katabatic_idealised_input.py --outdir runs/test       # stage 1: rotated/ only
#   cd runs/test/rotated && ln -s /path/to/microhh .
#   bash ../../../job.sh            # laptop / workstation (uses mpirun), or
#   sbatch ../../../job.sh          # SLURM cluster (check the #SBATCH lines first)
#       -> at its end the job writes ../ib (stage 2, KAT_STAGE2=1, the default)
#          and links the binary there; on SLURM, KAT_CHAIN=1 also submits ib/:
#          sbatch --export=ALL,KAT_CHAIN=1 ../../../job.sh
#   cd ../ib && bash ../../../job.sh                          # (or sbatch) if not chained
#
# Copy or link the microhh binary as ./microhh in each directory (MicroHH
# from the develop_openbc branch).
# ib/: init, the initial katabatic layer (katabatic_idealised_init.py), run,
# and the comparison with rotated/ if that has finished. rotated/: init, run.
#
# Restart from the restart files of time T (written every 3600 s):
#   RESTART_TIME=7200 sbatch --export=ALL ../../../job.sh     (or: RESTART_TIME=7200 bash ../../../job.sh)
#
# Settings you can pass as environment variables:
#   MICROHH_ENV  a script to source first, e.g. the module loads of your build
#   MPIRUN       the MPI launcher (default: "srun -n" in a SLURM job, else "mpirun -np")
#   CASE_DIR     the folder with the case scripts (default: three levels up)
#   RRTMGP_DATA  the RRTMGP coefficient files (only with --radiation lw/lwsw)
#
#SBATCH --job-name=katabatic_idealised
#SBATCH --nodes=1
#SBATCH --ntasks=128                # at least npx*npy of the .ini (ib/ 16 x 8, rotated/ 4 x 8)
#SBATCH --time=08:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
##SBATCH --partition=...            # add the partition / account your cluster needs

set -eu
if [ -n "${MICROHH_ENV:-}" ]; then source "$MICROHH_ENV"; fi
RUN_DIR=${SLURM_SUBMIT_DIR:-$PWD}
cd "$RUN_DIR"
CASE_DIR=${CASE_DIR:-$(cd "$RUN_DIR/../../.." && pwd)}
if [ ! -f "$CASE_DIR/katabatic_idealised_init.py" ]; then
    echo "ERROR: katabatic_idealised_init.py not in $CASE_DIR - set CASE_DIR=/path/to/cases/katabatic_idealised"
    exit 1
fi

if [ -f katabatic_idealised.ini ]; then CASE=katabatic_idealised; else CASE=katabatic_rotated; fi
[ -f $CASE.ini ] || { echo "ERROR: no katabatic_idealised.ini or katabatic_rotated.ini in $RUN_DIR"; exit 1; }
npx=$(awk -F= '/^npx=/{print $2}' $CASE.ini)
npy=$(awk -F= '/^npy=/{print $2}' $CASE.ini)
NP=$((npx*npy))
if [ -z "${MPIRUN:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then MPIRUN="srun -n"; else MPIRUN="mpirun -np"; fi
fi
RUN="$MPIRUN $NP"
echo "$CASE: MPI $npx x $npy = $NP tasks ($RUN)"
[ -x ./microhh ] || { echo "ERROR: no ./microhh in $RUN_DIR (copy or link the binary)"; exit 1; }

# RRTMGP coefficient files, only with --radiation lw/lwsw
if grep -q '^swradiation=rrtmgp' $CASE.ini; then
    RRTMGP_DATA=${RRTMGP_DATA:-$HOME/microhh/rte-rrtmgp-cpp/rrtmgp-data}
    for pair in rrtmgp-gas-lw-g128.nc:coefficients_lw.nc rrtmgp-gas-sw-g112.nc:coefficients_sw.nc \
                rrtmgp-clouds-lw.nc:cloud_coefficients_lw.nc rrtmgp-clouds-sw.nc:cloud_coefficients_sw.nc; do
        src=${pair%%:*}; dst=${pair##*:}
        [ -f "$RRTMGP_DATA/$src" ] || { echo "ERROR: no $RRTMGP_DATA/$src (set RRTMGP_DATA)"; exit 1; }
        ln -sf "$RRTMGP_DATA/$src" "$dst"
    done
fi

stage2() {
    # rotated/ finished: write ../ib from its equilibrium (--inflow rotated only)
    [ $CASE = katabatic_rotated ] || return 0
    [ "${KAT_STAGE2:-1}" = 1 ] || return 0
    grep -q '"inflow": "rotated"' ../generator_args.json 2>/dev/null || return 0
    if [ -e ../ib/grid.0000000 ]; then
        echo "stage 2: ../ib is already initialised - not rewritten"
        return 0
    fi
    python3 "$CASE_DIR/katabatic_idealised_input.py" --outdir .. --stage ib || { echo "stage 2 failed"; return 0; }
    [ -e ../ib/microhh ] || ln -s "$(readlink -f ./microhh)" ../ib/microhh
    if [ "${KAT_CHAIN:-0}" = 1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
        (cd ../ib && sbatch --export=ALL,CASE_DIR="$CASE_DIR" "$CASE_DIR/job.sh")
    else
        echo "stage 2: ../ib written; run it: cd ../ib && bash $CASE_DIR/job.sh (or sbatch)"
    fi
}

compare() {
    if [ $CASE = katabatic_idealised ] && ls ../rotated/katabatic_rotated.default.*.nc >/dev/null 2>&1; then
        python3 "$CASE_DIR/katabatic_idealised_compare.py" --rundir .. || echo "comparison failed"
    fi
}

if [ -n "${RESTART_TIME:-}" ]; then
    T=$(printf '%07d' "$RESTART_TIME")
    [ -e "u.$T" ] || { echo "ERROR: restart file u.$T not found"; exit 1; }
    BAK="after_${T}_$(date +%Y%m%d%H%M%S)"
    mkdir -p "$BAK"
    for f in *; do
        [ -f "$f" ] || continue
        t=${f##*.}; [ "$t" = nc ] && { t=${f%.nc}; t=${t##*.}; }
        if [[ "$t" =~ ^[0-9]{7}$ ]] && [ $((10#$t)) -gt $((10#$T)) ]; then mv "$f" "$BAK/"; fi
        if [[ "$f" =~ \.$T\.nc$ ]]; then mv "$f" "$BAK/"; fi
    done
    sed -i "s/^starttime=.*/starttime=$RESTART_TIME/" $CASE.ini
    echo "restart from t = $RESTART_TIME s (later output moved to $BAK/)"
    $RUN ./microhh run $CASE
    stage2
    compare
    exit 0
fi

if [ -e grid.0000000 ]; then
    echo "ERROR: grid.0000000 exists - make a new directory for a clean start."
    exit 1
fi
$RUN ./microhh init $CASE
if [ $CASE = katabatic_idealised ]; then
    python3 "$CASE_DIR/katabatic_idealised_init.py" --rundir .
fi
$RUN ./microhh run $CASE
stage2
compare
