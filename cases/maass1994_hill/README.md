# maass1994_hill: channel flow over a wavy wall (Maass & Schumann 1994)

Neutral turbulent flow (Re = U_m H/ν = 4780) through a plane channel whose lower wall
is a sinusoid, h(x) = δ cos(2πx/λ) with 2δ/λ = 0.1, λ = H: four waves in a
4 × 2 × 1 H domain, periodic in x and y, no-slip lid. The wave is the IB DEM.
The flow separates behind each crest, so the case tests the IB's form drag,
skin friction and separation against the DNS of Maass & Schumann (1994) and
the LDV measurements of Hudson et al. (1996). Units: H = 1 m, Um = 1 m/s.

| file | what |
|---|---|
| `maass1994_setup.py` | writes a run directory (`.ini`, DEM, input `.nc`, `job_atos.sh`, copies of the plot script and reference data) |
| `maass1994_plot.py` | wall stress, pressure and velocity profiles along the wave against DNS and LDV (`figs/`, `summary.json`) |
| `reference_maass1994.json`, `reference_hudson1996.json` | the DNS and LDV data |
| `build_hudson_reference.py`, `hudson_case076_raw.json` | how `reference_hudson1996.json` was made (ERCOFTAC case 076) |
| `digitise/*.py` | how the paper figures were digitised |
| `cross_to_nc.py`, `write_cross_to_nc.py`, `microhh_tools.py` | cross-section conversion |

IB momentum, `--variant`: `staircase` (default; case026: MOST stress source,
strain-MOST, kinematic floor, air-only exchange; the staircase tree), or for
the vr tree `vr` (velocity reconstruction), `vrlin` (the same with a linear,
viscous-sublayer profile) or `ghost` (upstream ghost cells). See `../README.md`.

## Run

    python3 maass1994_setup.py --grid small --outdir run_small --endtime 20      # ~15 s
    cd run_small && microhh init maass1994 && microhh run maass1994

Paper grid (0.82 M cells; ~1 h serial to t = 83, 12 min on one Atos node to t = 120):

    python3 maass1994_setup.py --grid paper --outdir $SCRATCH/maass1994/paper
    cp /path/to/microhh $SCRATCH/maass1994/paper/ && sbatch $SCRATCH/maass1994/paper/job_atos.sh

Options: `--grid small|paper|fine`, `--variant`, `--endtime`, `--dump-dt`, `--npx/--npy`.

## Process the output

    python3 maass1994_plot.py --rundir run_small --outdir run_small/figs --tavg 10
    python3 write_cross_to_nc.py --rundir run_small --case maass1994 --window 0

## References

- Maass, C. and Schumann, U. (1994), Numerical simulation of turbulent flow over a wavy boundary, in *Direct and Large-Eddy Simulation I*, Kluwer, 287–297. https://doi.org/10.1007/978-94-011-1000-6_25
- Hudson, J. D., Dykhno, L. and Hanratty, T. J. (1996), Turbulence production in flow over a wavy wall, *Exp. Fluids* 20, 257–265. https://doi.org/10.1007/BF00192670
- Cherukat, P., Na, Y., Hanratty, T. J. and McLaughlin, J. B. (1998), Direct numerical simulation of a fully developed turbulent flow over a wavy wall, *Theor. Comput. Fluid Dyn.* 11, 109–134. https://doi.org/10.1007/s001620050083
