#!/bin/bash
#
# Atos (ECMWF) job for test T1: the rotated-frame twins written by
# axelsen2009_rotated_ib_input.py (native/ and ib/). Run in each directory:
#
#   cd runs/T1/native && sbatch ../../../job_atos.sh
#   cd runs/T1/ib     && sbatch ../../../job_atos.sh
#
# Needs a MicroHH tree from patches/reapply_all.sh (mode staircase for the
# default --momentum staircase, vr for --momentum vr; patch 55 for ib/);
# native/ runs on any build. Copy or link the binary as ./microhh here.
#
#SBATCH --job-name=axelsen_rot
#SBATCH --qos=np
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --ntasks-per-node=128
#SBATCH --hint=nomultithread
#SBATCH --time=06:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.out

set -eu
source ${MICROHH_ENV:-$HOME/microhh_env.sh}
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export OMPI_MCA_coll_hcoll_enable=0

npx=$(awk -F= '/^npx=/{print $2}' axelsen2009_rotated.ini)
npy=$(awk -F= '/^npy=/{print $2}' axelsen2009_rotated.ini)
NP=$((npx*npy))
echo "MPI: $npx x $npy = $NP tasks"
[ -x ./microhh ] || { echo "ERROR: no ./microhh here (copy or link the binary)"; exit 1; }

srun -n $NP ./microhh init axelsen2009_rotated
srun -n $NP ./microhh run axelsen2009_rotated
echo "done: compare with python3 ../../../axelsen2009_rotated_ib_compare.py --native ../native --ib ../ib"
