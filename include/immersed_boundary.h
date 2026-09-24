/*
 * MicroHH
 * Copyright (c) 2011-2024 Chiel van Heerwaarden
 * Copyright (c) 2011-2024 Thijs Heus
 * Copyright (c) 2014-2024 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef IMMERSED_BOUNDARY_H
#define IMMERSED_BOUNDARY_H

#include "boundary.h"
#include "boundary_cyclic.h"

class Master;
class Input;
template<typename> class Grid;
template<typename> class Fields;
template<typename> class Timeloop;
template<typename> class Cross;
template<typename> class Column;
template<typename> class Thermo;
template<typename> class Stats;
template<typename> class Radiation;
class Netcdf_handle;

enum class IB_type {Disabled, DEM, User};

enum class IB_wall_type {Disabled, Neutral, Most};

template<typename TF>
struct Wall_cells
{
    int n;

    std::vector<int> i;      // the FLUID cell that owns the face
    std::vector<int> j;
    std::vector<int> k;
    std::vector<int> axis;   // 0 = wall normal to x, 1 = to y, 2 = to z (floor)
    std::vector<int> sign;   // -1 solid on the low side, +1 on the high side

    std::vector<TF> dn;      // wall-normal distance of the cell centre  [m]
    std::vector<TF> da;      // cell extent along the normal
    std::vector<TF> z0m;
    std::vector<TF> z0h;

    std::vector<TF> obuk;    // kept between substeps as the iteration's guess
    std::vector<TF> ustar;
    std::vector<TF> utan;    // tangential speed used, for diagnostics
};

// Ghost cell info on staggered grid
template<typename TF>
struct Ghost_cells
{
    int nghost;

    //
    // CPU
    //

    // Indices of IB ghost cells:
    std::vector<int> i;     // size = number of ghost cells
    std::vector<int> j;
    std::vector<int> k;

    // Nearest location of IB to ghost cell:
    std::vector<TF> xb;     // size = number of ghost cells
    std::vector<TF> yb;
    std::vector<TF> zb;

    // Location of interpolation point outside IB:
    std::vector<TF> xi;     // size = number of ghost cells
    std::vector<TF> yi;
    std::vector<TF> zi;

    std::vector<TF> di; // Distance ghost cell to interpolation point

    // Points outside IB used for IDW interpolation:
    std::vector<int> ip_i;     // size = number of ghost cells x n_idw_points
    std::vector<int> ip_j;
    std::vector<int> ip_k;
    std::vector<TF>  ip_d;  // Distance to interpolation point

    // Interpolation coefficients
    std::vector<TF> c_idw;     // size = number of ghost cells x n_idw_points
    std::vector<TF> c_idw_sum; // size = number of ghost cells

    // Spatially varying scalar (and momentum..) boundary conditions
    std::map<std::string, std::vector<TF>> sbot;
    std::vector<TF> mbot;

    std::vector<TF> a_wall;
    std::vector<TF> nsq;
    std::vector<int> wall_idx;
    
    //
    // GPU 
    //

    // Indices of IB ghost cells:
    int* i_g;
    int* j_g;
    int* k_g;

    // Nearest location of IB to ghost cell:
    TF* xb_g;
    TF* yb_g;
    TF* zb_g;

    // Location of interpolation point outside IB:
    TF* xi_g;
    TF* yi_g;
    TF* zi_g;

    TF* di_g;  // Distance ghost cell to interpolation point

    // Points outside IB used for IDW interpolation:
    int* ip_i_g;
    int* ip_j_g;
    int* ip_k_g;
    TF* ip_d_g;

    // Interpolation coefficients
    TF* c_idw_g;
    TF* c_idw_sum_g;

    // Spatially varying scalar (and momentum..) boundary conditions
    std::map<std::string, TF*> sbot_g;
    TF* mbot_g;
};

// Convenience struct to simplify sorting
template<typename TF>
struct Neighbour
{
    int i;
    int j;
    int k;
    TF distance;
};

template<typename TF>
class Immersed_boundary
{
    public:
        Immersed_boundary(Master&, Grid<TF>&, Fields<TF>&, Input&);
        ~Immersed_boundary();

        void init(Input&, Cross<TF>&);
        void create(Netcdf_handle&);

        void exec_momentum();
        void exec_scalars();
        void exec_scalar_flux(Thermo<TF>&, Radiation<TF>&, Stats<TF>&);
        // Tundra as a canopy: the resistance-limited surface humidity.
        // See apply_ib_vegetation.py.
        void exec_vegetation(Thermo<TF>&, Radiation<TF>&);
        void exec_momentum_flux(Stats<TF>&);
        void exec_strain_most();

        void exec_impermeable();
        // Air-only scalar advection next to the terrain: [IB] sw_advec_wall.
        void exec_advec_wall();
        // Keep the inside of the terrain inert: see [IB] sw_blank_solid.
        void blank_solid_momentum();
        void blank_solid_scalars();
        void update_time_dependent(Timeloop<TF>&);
        // Re-interpolate the current sbot_2d onto the ghost cells' wall points.
        void sbot_2d_to_ghosts();
        void exec_wall_model(Thermo<TF>&, Stats<TF>&);

        void exec_cross(Cross<TF>&, unsigned long);
        // Column output: the surface diagnostics as time series at the
        // [column] locations, and the IB mask of that column as a profile.
        void create_column(Column<TF>&);
        void exec_column(Column<TF>&, Thermo<TF>&);


        bool has_mask(std::string);
        void get_mask(Stats<TF>&, std::string);

        void prepare_device();
        void clear_device();

        IB_type get_switch() const { return sw_ib; }

    private:
        Master& master;
        Grid<TF>& grid;
        Fields<TF>& fields;
        Field3d_io<TF> field3d_io;
        Boundary_cyclic<TF> boundary_cyclic;

        IB_type sw_ib;

        int n_idw_points;       // Number of interpolation points in IDW interpolation
        bool sw_dem_halo_replicate;  // Fill the DEM halo at the OUTER domain edges
                                     // by replication instead of cyclically. Correct
                                     // when the lateral boundaries are open.

        // Boundary conditions for scalars
        Boundary_type sbcbot;
        std::map<std::string, TF> sbc;
        std::vector<std::string> sbot_spatial_list;
        // Monin-Obukhov wall model.
        IB_wall_type sw_wall_model;
        TF z0m_ib;
        TF z0h_ib;
        TF tPr_wm;                  // [diff] tPr, for the scalar diffusivity
        Wall_cells<TF> wall;
        bool sw_momentum_flux;
        bool sw_momentum_flux_vertical;
        bool mom_flux_reported;
        Wall_cells<TF> wall_u;
        Wall_cells<TF> wall_v;
        std::vector<int> wall_floor_ij;
        bool sw_strain_most;
        bool sw_strain_most_vertical;
        bool strain_most_reported;

        // One-shot diffusion-number probe. See apply_ib_dnum_probe.py.
        bool dn_probe_done;
        TF dnmax_ib;

        // Wall damping of the Smagorinsky length at the IB. See
        // apply_ib_wall_mlen.py.
        bool sw_strain_mlen;
        bool sw_mason_ib;
        TF cs_ib;
        bool strain_most_built;
        TF strain_most_min;
        std::vector<int> strain_most_m;   // one face index per unique wall cell
        std::vector<int> wall_u_floor;
        std::vector<int> wall_v_floor;
        TF diag_z_mom;
        TF diag_z_scalar;
        std::vector<TF> hfss_ij;
        std::vector<TF> hfls_ij;
        bool sw_scalar_flux;
        std::map<std::string, std::vector<TF>> wall_value_face;
        bool sw_scalar_flux_vertical;
        bool wall_flux_reported;
        int  n_zl_clamped_last;
        bool sw_wall_stability_vertical;
        bool sw_impermeable;
        std::map<std::string, std::vector<TF>> wall_flux;
        std::vector<std::string> crosslist_wall;
        /*
         * The single implementation of every terrain-following surface
         * diagnostic. Fills `out` (ijcells) and returns false if the name is
         * not one it knows. exec_cross and exec_column BOTH call it, so a
         * cross-section and a column time series of the same quantity are the
         * same number by construction.
         */
        bool calc_surface_diag(const std::string&, TF* const restrict);

        // What [IB] columnlist asked for, already resolved to canonical names.
        std::vector<std::string> columnlist;

        /*
         * The base-state exner, cached the first time a Thermo is in reach
         * (exec_wall_model and exec_column both have one; exec_cross does
         * not). Only T_2m_ib needs it. The anelastic base state does not
         * change in time here - swphydro_3d is off - so one copy is enough.
         */
        std::vector<TF> exner_ref;


        std::map<std::string, std::vector<TF>> sbot_2d;

        // [IB] sw_advec_wall - apply_ib_advec_wall.py. One list per axis of
        // the faces whose 2i5 stencil touches a solid cell, each face stored
        // as the index of the cell on its HIGH side (the face lies between
        // c-d and c), with what to replace its flux by.
        bool sw_advec_wall;
        bool advec_wall_reported;
        std::vector<std::string> advec_wall_limited;   // [advec] fluxlimit_list
        std::vector<int> advec_face_ijk[3];
        std::vector<signed char> advec_face_cls[3];

        bool sw_blank_solid;
        std::vector<int> blank_s;
        std::vector<int> blank_s_ij;
        std::vector<int> blank_u;
        std::vector<int> blank_v;
        std::vector<int> blank_w;

        // [diff] tPr, so <scalar>_fluxbot_ib can use the SGS diffusivity.
        TF tPr_ib;
        bool sw_timedep_sbot;
        bool sbot_timedep_init;
        unsigned long iloadtime_sbot;
        unsigned long itime_sbot_prev;
        unsigned long itime_sbot_next;
        std::map<std::string, std::vector<TF>> sbot_2d_prev;
        std::map<std::string, std::vector<TF>> sbot_2d_next;

        /*
         * [IB] sw_vegetation - apply_ib_vegetation.py.
         *
         * The IB surface as a canopy rather than as free water. Everything
         * here is per column (ijcells) except the soil profile, which is one
         * column shared by the whole domain, and the van Genuchten table,
         * which is one entry per soil class.
         */
        bool sw_vegetation;
        bool veg_reported;
        int  veg_soil_ktot;

        std::vector<TF> veg_c_veg;        // vegetation cover fraction   [-]
        std::vector<TF> veg_lai;          // leaf area index        [m2 m-2]
        std::vector<TF> veg_rs_veg_min;   // min canopy resistance   [s m-1]
        std::vector<TF> veg_rs_soil_min;  // min soil resistance     [s m-1]
        std::vector<TF> veg_gD;           // VPD response             [Pa-1]

        // Diagnostics, one per column, rebuilt every substep.
        std::vector<TF> veg_ra;           // aerodynamic resistance  [s m-1]
        std::vector<TF> veg_rs;           // effective surface resistance
        std::vector<TF> veg_f1;           // radiation stress            [-]
        std::vector<TF> veg_f3;           // VPD stress                  [-]
        std::vector<TF> veg_vpd;          // vapour pressure deficit    [Pa]
        std::vector<TF> veg_qt_bot;       // the surface value actually used

        // The soil, PER COLUMN: (ktot, ijcells), bottom-up, so index
        // (sk-1)*ijcells + ij is the top layer - the same order the MicroHH
        // LSM and soil_grid.cxx use. Static in patch 17 (RACMO's state at
        // t=0, held); patch 18 integrates it.
        std::vector<TF> veg_t_soil;
        std::vector<TF> veg_theta_soil;
        std::vector<TF> veg_root_frac;    // 1-D over the soil levels
        std::vector<int> veg_soil_index;  // 1-D over the soil levels
        // f2 (root-zone moisture) and f2b (top-layer moisture), per column.
        // Scalars until the soil became per column; now one value each per
        // column, still computed once because the soil does not move.
        std::vector<TF> veg_f2;
        std::vector<TF> veg_f2b;

        // IB input from DEM
        std::vector<TF> dem;
        std::vector<unsigned int> k_dem;

        // Terrain-following xy cross-sections, <var>_tf<h>. See
        // apply_ib_tf_cross.py.
        bool sw_tf_cross;
        std::vector<TF> tf_cross_heights;

        // All ghost cell properties
        std::map<std::string, Ghost_cells<TF>> ghost;

        // Statistics
        std::vector<std::string> available_masks;

        // Cross-sections
        std::vector<std::string> crosslist;
};

#endif
