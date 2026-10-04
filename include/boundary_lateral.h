/*
 * MicroHH
 * Copyright (c) 2011-2020 Chiel van Heerwaarden
 * Copyright (c) 2011-2020 Thijs Heus
 * Copyright (c) 2014-2020 Bart van Stratum
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

#ifndef BOUNDARY_LATERAL_H
#define BOUNDARY_LATERAL_H

#include <vector>
#include <string>
#include <map>

#include "master.h"
#include "grid.h"

class Master;
class Input;
template<typename> class Grid;
template<typename> class Fields;
template<typename> class Timeloop;
template<typename> class Stats;
template<typename> class Field3d_io;

enum class Lbc_location {West, East, South, North};

template<typename TF>
using Lbc_map = std::map<std::string, std::vector<TF>>;


template<typename TF>
class Boundary_lateral
{
    public:
        Boundary_lateral(Master&, Grid<TF>&, Fields<TF>&, Input&);
        ~Boundary_lateral();

        void init();
        void create(Input&, Timeloop<TF>&, Stats<TF>&, const std::string&);
        void set_ghost_cells(Timeloop<TF>&);
        void set_ghost_cells_scalars();   // apply_lbc_ib_halos.py
        void exec_lateral_sponge(Stats<TF>&);
        void update_time_dependent(Timeloop<TF>&, const bool pres_fix=false);
        unsigned long get_time_limit(unsigned long);

        // Net mass flux from the IB terrain into the air (kg/s, x and y
        // part), from Immersed_boundary::get_rock_flux. Under patch 28 it is
        // let out of the rock through the edge faces (apply_ib_wall_kinematic.py).
        void set_rock_flux(const TF fx, const TF fy) { rock_flux[0] = fx; rock_flux[1] = fy; }

    private:
        Master& master;
        Grid<TF>& grid;
        Fields<TF>& fields;
        Field3d_io<TF> field3d_io;

        void read_lbc(TF&, TF&, Lbc_map<TF>&, Lbc_map<TF>&, Lbc_map<TF>&, Lbc_map<TF>&, const int);
        void read_xy_slice(
                std::vector<TF>&, const std::string&, const int);

        // Terrain-following periodic LBCs (patch 28, apply_lbc_tf_periodic.py).
        void refresh_lbc_tf_periodic(const TF);

        bool sw_openbc;
        bool sw_openbc_uv;
        bool sw_openbc_w;
        bool sw_neumann_w;
        bool sw_timedep;
        bool sw_wtop_2d;

        std::vector<std::string> slist;

        // Sponge/diffusion layer:
        bool sw_sponge;
        int n_sponge;
        TF tau_sponge;
        TF w_diff;

        // Turbulence recycling.
        std::map<Lbc_location, bool> sw_recycle;
        std::vector<std::string> recycle_list;
        TF tau_recycle;
        int recycle_offset;

        // Current (constant or time interpolated) BCs:
        Lbc_map<TF> lbc_w;
        Lbc_map<TF> lbc_s;
        Lbc_map<TF> lbc_e;
        Lbc_map<TF> lbc_n;

        // Previous and next BCs, for time dependent LBCs.
        Lbc_map<TF> lbc_w_prev;
        Lbc_map<TF> lbc_e_prev;
        Lbc_map<TF> lbc_s_prev;
        Lbc_map<TF> lbc_n_prev;

        Lbc_map<TF> lbc_w_next;
        Lbc_map<TF> lbc_e_next;
        Lbc_map<TF> lbc_s_next;
        Lbc_map<TF> lbc_n_next;

        unsigned int loadfreq;
        unsigned long prev_itime;
        unsigned long next_itime;

        // Spatially constant (calculated) `w_top`.
        TF w_top;
        TF w_top_prev;
        TF w_top_next;

        // Spatially varying (input) `w_top`.
        std::vector<TF> w_top_2d;
        std::vector<TF> w_top_2d_prev;
        std::vector<TF> w_top_2d_next;

        // Terrain-following periodic LBCs: each edge buffer is refilled from
        // the opposite edge, at the same height above the local surface.
        bool sw_lbc_tf_periodic;
        bool lbc_tf_initialized;                    // Buffers filled at least once.
        bool sw_lbc_rock_export = false;            // [IB] sw_wall_kinematic under patch 28.
        bool rock_export_reported = false;          // apply_ib_wall_kinematic.py log line printed.
        std::string lbc_tf_dem_file;
        std::map<std::string, TF> lbc_tf_gradient;  // Environmental d(phi)/dz per scalar.
        std::vector<TF> dem_tf;                     // Surface height, local columns incl. halo.
        TF rock_flux[2] = {TF(0), TF(0)};          // Air <- terrain mass flux, x and y part.

        const std::string tend_name = "lbc_sponge";
        const std::string tend_longname = "Lateral sponge layer";
};
#endif