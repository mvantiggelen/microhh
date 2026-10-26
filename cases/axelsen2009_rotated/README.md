# axelsen2009_rotated: Axelsen P1 in the rotated slope frame, native surface against a flat IB

Does the immersed-boundary (IB) wall model reproduce MicroHH's own surface
model when the IB surface is perfectly flat (no staircase steps)? The test
runs case P1 of Axelsen & van Dop (2009) twice in MicroHH's rotated,
periodic slope frame (`[thermo] swthermo=buoy`, α = 5°, the surface buoyancy
flux Fs of the paper), with the same grid (64 × 64, dx 4 m, dz 1 m), closure
and noise:

- `native/`: the domain floor is the surface (MicroHH's surface model, MOST with z0m, z0h);
- `ib/`: the floor is buried under a flat IB DEM `--k-ib` levels up, with the
  recommended IB settings (as in `../axelsen2009_lbc`) and the IB momentum
  method of `--momentum` (see `../README.md`).

Level k_ib + k of `ib/` is level k of `native/`, so the two runs can be
compared level by level. `ib/` needs MicroHH from the develop_openbc branch.

| file | what |
|---|---|
| `axelsen2009_rotated_ib_input.py` | writes `native/` and `ib/` (`.ini`, input `.nc`, DEM, `case_parameters.json`) |
| `axelsen2009_rotated_ib_compare.py` | profiles and jet time series of the two runs against each other and the paper |
| `job.sh` | init and run, in each of `native/` and `ib/`; on a laptop or a SLURM cluster |

## Run

The default (3600 s on 2 MPI tasks, `--npx 1 --npy 2`) is small enough for a laptop.

    python3 axelsen2009_rotated_ib_input.py --outdir runs/P1      # --momentum staircase|vr, --sgs tke2|smag2
    cd runs/P1/native && cp /path/to/microhh . && bash ../../../job.sh
    cd ../ib          && cp /path/to/microhh . && bash ../../../job.sh
    # on a SLURM cluster: sbatch ../../../job.sh instead of bash
    # or by hand: ./microhh init axelsen2009_rotated && ./microhh run axelsen2009_rotated

Switches of `ib/` (true/false): `--wall-slope`, `--advec-wall`, `--wall-sample`;
for staircase: `--strain-most`, `--wall-kinematic`, `--momentum-air-only`;
for vr: `--momentum-reconstruct`, `--reconstruct-stress`, `--reconstruct-dz`.

## Process the output

    python3 axelsen2009_rotated_ib_compare.py --native runs/P1/native --ib runs/P1/ib \
            --tavg 600 --out runs/P1/rotated_compare.png
    # optional: --lbc-metrics ../axelsen2009_lbc/runs/P1_lbc/figs/P1_lbc_metrics.json

It prints the jet maximum, its height, the first-level buoyancy and the
wall momentum flux of both runs. The figure shows u_s/V, b/B, the momentum
flux and the TKE against n/H, and the jet against t/τ_osc. The paper's
profiles come from `../axelsen2009_lbc/observations_P1.json`, so keep the
folders side by side.

## References

- Axelsen, S. L. and van Dop, H. (2009), Large-eddy simulation of katabatic winds. Part 1, *Acta Geophysica* 57, 803–836. https://doi.org/10.2478/s11600-009-0041-6
- Fedorovich, E. and Shapiro, A. (2009), Structure of numerically simulated katabatic and anabatic flows along steep slopes, *Acta Geophysica* 57, 981–1010. https://doi.org/10.2478/s11600-009-0027-4
