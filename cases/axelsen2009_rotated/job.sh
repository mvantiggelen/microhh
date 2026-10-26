#!/bin/bash
#
# Job for the two run directories written by axelsen2009_rotated_ib_input.py
# (native/ and ib/). Run it in each directory, on a laptop or workstation or
# on a SLURM cluster such as Snellius:
#
#   cd runs/P1/native && bash ../../../job.sh      # laptop / workstation (uses mpirun)
#   cd runs/P1/ib     && sbatch ../../../job.sh    # SLURM cluster (check the #SBATCH lines first)
#
# Copy or link the binary as ./microhh in each directory. ib/ needs MicroHH
# from the develop_openbc branch; native/ runs on any MicroHH build.
#
# Settings you can pass as environment variables:
#   MICROHH_ENV  a script to source first, e.g. the module loads of your build
#   MPIRUN       the MPI launcher (default: "srun -n" in a SLURM job, else "mpirun -np")
#
#SBATCH --job-name=axelsen_rotated
#SBATCH --ntasks=2                  # at least npx*npy of axelsen2009_rotated.ini (default 1 x 2)
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out
##SBATCH --partition=...            # add the partition / account your cluster needs

set -eu
if [ -n "${MICROHH_ENV:-}" ]; then source "$MICROHH_ENV"; fi
cd "${SLURM_SUBMIT_DIR:-$PWD}"

npx=$(awk -F= '/^npx=/{print $2}' axelsen2009_rotated.ini)
npy=$(awk -F= '/^npy=/{print $2}' axelsen2009_rotated.ini)
NP=$((npx*npy))
if [ -z "${MPIRUN:-}" ]; then
    if [ -n "${SLURM_JOB_ID:-}" ]; then MPIRUN="srun -n"; else MPIRUN="mpirun -np"; fi
fi
RUN="$MPIRUN $NP"
echo "MPI: $npx x $npy = $NP tasks ($RUN)"
[ -x ./microhh ] || { echo "ERROR: no ./microhh here (copy or link the binary)"; exit 1; }

$RUN ./microhh init axelsen2009_rotated
$RUN ./microhh run axelsen2009_rotated
echo "done: compare with python3 ../../../axelsen2009_rotated_ib_compare.py --native ../native --ib ../ib"
