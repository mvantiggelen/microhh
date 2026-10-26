# Overview of cases

## Dynamical core validation
* `conservation`: Random noise advection for checking mass, momentum, and energy conservation.
* `taylorgreen`: Convergence test for spatial discretizations in Taylor-Green vortex.

## Atmospheric LES
* `andren1994`: LES of neutral boundary layer (Andren et al. 1994).
* `arm`: Shallow cumulus over land (Brown et al., 2002).
* `bomex`: BOMEX trade wind cumulus case (Siebesma et al., 2003).
* `drycblles`: Dry convective boundary layer growing against linear stratification.
* `dycoms`: DYCOMS stratocumulus case.
* `eady`: Eady's baroclinic instability on an f-plane.
* `gabls1`: GABLS1 LES intercomparison (Beare et al. 2006).
* `gabls4s3`: GABLS4 LES intercomparison (Couvreux et al. 2020).
* `lasso`: LASSO cases, based on ARM observations 2015.
* `rcemip`: RCEMIP radiative convective equilibrium case (Wing et al., 2018).
* `rico`: RICO precipitating cumulus case (van Zanten et al., 2013).
* `sullivan2011`: Dry convective boundary layer following Sullivan and Patton (2011).

## Immersed boundary terrain and open boundaries
These cases test the immersed boundary (IB) terrain and the open lateral boundaries of the `develop_openbc` branch. Each folder has a README with the set-up, how to run it on a laptop or a SLURM cluster, and how to compare the results with the reference. The input keys are listed in `doc/immersed_boundary_openbc.md`.
* `andren1994_ib`: The neutral boundary layer of Andren et al. (1994) over a flat immersed boundary, against the standard surface.
* `axelsen2009_lbc`: Katabatic case P1 of Axelsen and van Dop (2009) on a plateau-slope-valley IB terrain with open boundaries.
* `axelsen2009_rotated`: Katabatic case P1 of Axelsen and van Dop (2009) in the rotated slope frame: MicroHH's surface model against a flat IB.
* `fedorovich2017_b13`: Nocturnal low-level jet over a gentle IB slope (Fedorovich et al., 2017, case B13).
* `katabatic_idealised`: Greenland katabatic flow down an IB slope with open boundaries, against the rotated slope frame.
* `maass1994_hill`: Channel flow over a wavy IB wall (Maass and Schumann, 1994).

## DNS
* `ekman`: Ekman spiral with constant viscosity.
* `moser180`: Turbulent channel flow at Re_tau = 180 (Moser et al., 1999).
* `moser590`: Turbulent channel flow at Re_tau = 590 (Moser et al., 1999).
* `taylorgreen`: Taylor-Greene vortex case to test spatial accuracy of schemes.
* `vanHeerwaarden2016`: Dry convective boundary layer over surface with constant temperature (van Heerwaarden and Mellado, 2016).
