# axelsen2009_rotated: Axelsen P1 in the rotated slope frame (test T1)

Does the IB wall model reproduce MicroHH's own surface when there is no
staircase? Case P1 of Axelsen & van Dop (2009) in MicroHH's rotated, periodic
slope frame (`[thermo] swthermo=buoy`, α = 5°, N² = N², the surface buoyancy
flux Fs), run twice with the same grid (64 × 64, dx 4 m, dz 1 m), closure and
noise:

- `native/`: the domain floor is the surface (MOST with z0m, z0h);
- `ib/`: the floor is buried under a flat IB DEM `--k-ib` levels up, with
  the [IB] switches of `../axelsen2009_lbc` and the IB momentum method of
  `--momentum` (staircase, default; or vr, see `../README.md`).

`ib/` level k_ib + k is `native/` level k. Needs patch 55 (IB with thermo buoy).

| file | what |
|---|---|
| `axelsen2009_rotated_ib_input.py` | writes `native/` and `ib/` (`.ini`, input `.nc`, DEM, `case_parameters.json`) |
| `axelsen2009_rotated_ib_compare.py` | profiles and jet time series of the twins against each other and the paper |
| `job_atos.sh` | Atos job, run in each of `native/` and `ib/` |

## Run

    python3 axelsen2009_rotated_ib_input.py --outdir runs/T1       # --momentum staircase|vr, --sgs tke2|smag2
    cd runs/T1/native && cp /path/to/microhh . && sbatch ../../../job_atos.sh
    cd ../ib          && cp /path/to/microhh . && sbatch ../../../job_atos.sh
    # locally: ./microhh init axelsen2009_rotated && ./microhh run axelsen2009_rotated

Switches of `ib/` (true/false): `--wall-slope`, `--advec-wall`, `--wall-sample`;
staircase: `--strain-most`, `--wall-kinematic`, `--momentum-air-only`;
vr: `--momentum-reconstruct`, `--reconstruct-stress`, `--reconstruct-dz`.

## Process the output

    python3 axelsen2009_rotated_ib_compare.py --native runs/T1/native --ib runs/T1/ib \
            --tavg 600 --out runs/T1/T1_compare.png
    # optional: --lbc-metrics ../axelsen2009_lbc/runs/P1_lbc/figs/P1_lbc_metrics.json

Prints the jet maximum, its height, the first-level buoyancy and the wall
momentum flux of both runs; the figure has u_s/V, b/B, the momentum flux and
TKE against n/H, and the jet against t/τ_osc. Uses
`../axelsen2009_lbc/observations_P1.json` for the paper's profiles.

## References

- Axelsen, S. L. and van Dop, H. (2009), Large-eddy simulation of katabatic winds. Part 1, *Acta Geophysica* 57, 803–836. https://doi.org/10.2478/s11600-009-0041-6
- Fedorovich, E. and Shapiro, A. (2009), Structure of numerically simulated katabatic and anabatic flows along steep slopes, *Acta Geophysica* 57, 981–1010. https://doi.org/10.2478/s11600-009-0027-4
