# maass1994_hill: channel flow over a wavy wall (Maass & Schumann 1994)

Neutral turbulent flow (Re = U_m H/ν = 4780) through a plane channel whose
lower wall is a sine wave, h(x) = δ cos(2πx/λ) with 2δ/λ = 0.1 and λ = H. The
domain is 4 × 2 × 1 H (four waves), periodic in x and y, with a no-slip lid.
The wave is an immersed boundary (IB), given as a DEM. The flow separates
behind each crest, so the case tests the form drag, the skin friction and the
separation of the IB against the DNS of Maass & Schumann (1994) and the LDV
measurements of Hudson et al. (1996). Units: H = 1 m, U_m = 1 m/s.

| file | what |
|---|---|
| `maass1994_setup.py` | writes a run directory (`.ini`, DEM, input `.nc`, `job.sh`, copies of the plot script and reference data) |
| `maass1994_plot.py` | wall stress, pressure and velocity profiles along the wave against the DNS and LDV data (`figs/`, `summary.json`) |
| `reference_maass1994.json`, `reference_hudson1996.json` | the DNS and LDV data |
| `build_hudson_reference.py`, `hudson_case076_raw.json` | how `reference_hudson1996.json` was made (ERCOFTAC case 076) |
| `digitise/*.py` | how the paper figures were digitised |
| `cross_to_nc.py`, `write_cross_to_nc.py`, `microhh_tools.py` | conversion of the cross-sections to netCDF |

`--variant` sets how the IB applies the wall stress to the velocity:
`staircase` (default, works with the develop_openbc branch), or `vr`, `vrlin`
and `ghost`, which need a MicroHH build with velocity reconstruction. See
`../README.md`.

## Run

On a laptop (small grid, about 15 s):

    python3 maass1994_setup.py --grid small --outdir run_small --endtime 20
    cd run_small && cp /path/to/microhh . && ./microhh init maass1994 && ./microhh run maass1994

The paper grid has 0.82 M cells: about 1 h on one core to t = 83, or about
12 min on a 128-core node to t = 120. The generated `job.sh` runs init, run and
the plots, on a laptop with `bash job.sh` or on a SLURM cluster such as
Snellius with `sbatch job.sh`:

    python3 maass1994_setup.py --grid paper --outdir run_paper
    cp /path/to/microhh run_paper/ && cd run_paper && sbatch job.sh

Options: `--grid small|paper|fine`, `--variant`, `--endtime`, `--dump-dt`, `--npx/--npy`.

## Process the output

    python3 maass1994_plot.py --rundir run_small --outdir run_small/figs --tavg 10
    python3 write_cross_to_nc.py --rundir run_small --case maass1994 --window 0

## References

- Maass, C. and Schumann, U. (1994), Numerical simulation of turbulent flow over a wavy boundary, in *Direct and Large-Eddy Simulation I*, Kluwer, 287–297. https://doi.org/10.1007/978-94-011-1000-6_25
- Hudson, J. D., Dykhno, L. and Hanratty, T. J. (1996), Turbulence production in flow over a wavy wall, *Exp. Fluids* 20, 257–265. https://doi.org/10.1007/BF00192670
- Cherukat, P., Na, Y., Hanratty, T. J. and McLaughlin, J. B. (1998), Direct numerical simulation of a fully developed turbulent flow over a wavy wall, *Theor. Comput. Fluid Dyn.* 11, 109–134. https://doi.org/10.1007/s001620050083
- Tseng, Y.-H. and Ferziger, J. H. (2003), A ghost-cell immersed boundary method for flow in complex geometry, *J. Comput. Phys.* 192, 593–623. https://doi.org/10.1016/j.jcp.2003.07.024
