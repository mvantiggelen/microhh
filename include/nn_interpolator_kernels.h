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

#ifndef BOUNDARY_LATERAL_KERNELS_H
#define BOUNDARY_LATERAL_KERNELS_H

#include <vector>
#include <cmath>
#include <algorithm>


namespace NN_interpolator_kernels
{
    template<typename TF>
    std::vector<TF> arange(
            const TF start,
            const TF stop,
            const TF step)
    {
        if (step <= 0)
            throw std::runtime_error("Step needs to be positive (in this implementation..)");

        std::vector<TF> result;

        TF value = start;
        const TF epsilon = std::numeric_limits<TF>::epsilon() * std::abs(start + stop + step);

        while (value + epsilon < stop)
        {
            result.push_back(value);
            value += step;
        }

        return result;
    }


    template<typename TF>
    bool intersects_mpi_subdomain(
        const std::vector<TF>& x,
        const std::vector<TF>& y,
        const TF x_min, const TF x_max,
        const TF y_min, const TF y_max)
    {
        // First simple check on bounds.
        if (x.back() < x_min || x.front() > x_max)
            return false;

        if (y.back() < y_min || y.front() > y_max)
            return false;

        // Check coordinate pairs.
        for (int i=0; i<x.size(); ++i)
            for (int j=0; j<y.size(); ++j)
                if (x[i] >= x_min && x[i] < x_max && y[j] >= y_min && y[j] < y_max)
                    return true;

        return false;
    }


    template<typename TF>
    std::pair<int, int> get_start_end_indexes(
        const std::vector<TF>& x,
        const TF x0,
        const TF x1)
    {
        auto start_it = std::lower_bound(x.begin(), x.end(), x0);
        auto end_it = std::lower_bound(x.begin(), x.end(), x1);

        int start_idx = std::distance(x.begin(), start_it);
        int end_idx = std::distance(x.begin(), end_it);

        return std::make_pair(start_idx, end_idx);
    }


    template<typename TF>
    std::vector<int> get_nn_indexes(
        const std::vector<TF>& x_target,
        const std::vector<TF>& x_all)
    {
        std::vector<int> nn_index(x_target.size());
        int nearest_index;

        for (int i=0; i<nn_index.size(); ++i)
        {
            auto it = std::lower_bound(x_all.begin(), x_all.end(), x_target[i]);

            if (it == x_all.begin())
                nearest_index = 0;
            else if (it == x_all.end())
                nearest_index = x_all.size() - 1;
            else
            {
                // std::lower_bound returns the first element that is larger, not the nearest...
                const int upper_index = std::distance(x_all.begin(), it);
                const int lower_index = upper_index - 1;

                TF dist_to_lower = x_target[i] - x_all[lower_index];
                TF dist_to_upper = x_all[upper_index] - x_target[i];

                nearest_index = (dist_to_lower <= dist_to_upper) ? lower_index : upper_index;
            }

            nn_index[i] = nearest_index;
        }

        return nn_index;
    }


    // Offset of each target coordinate from the coordinate of its nearest
    // neighbour (patch 61, [subdomain] interpolation=linear).
    template<typename TF>
    std::vector<TF> get_nn_offsets(
        const std::vector<TF>& x_target,
        const std::vector<TF>& x_all,
        const std::vector<int>& nn_index)
    {
        std::vector<TF> offset(x_target.size());

        for (int i=0; i<offset.size(); ++i)
            offset[i] = x_target[i] - x_all[nn_index[i]];

        return offset;
    }


    // Monotonised-central limited slope (van Leer 1977) of a parent cell
    // from its two neighbours; dm and dp are the distances to them.
    template<typename TF>
    inline TF mc_slope(
        const TF fm, const TF fc, const TF fp,
        const TF dm, const TF dp)
    {
        const TF sl = (fc - fm) / dm;
        const TF sr = (fp - fc) / dp;

        if (sl*sr <= TF(0))
            return TF(0);

        const TF sc = (fp - fm) / (dm + dp);
        const TF smin = std::min(std::abs(sc), TF(2)*std::min(std::abs(sl), std::abs(sr)));

        return std::copysign(smin, sc);
    }


    // Conservative piecewise-linear refinement (patch 61): in every parent
    // cell f = f_c + s_x dx + s_y dy + s_z dz, with limited slopes. For child
    // points placed symmetrically in the parent cell the child mean equals
    // the parent value, so boundary-normal mass fluxes are those of the parent.
    template<typename TF>
    void linear_interpolate(
        TF* const restrict fld_out,
        const TF* const restrict fld_in,
        const int* const restrict nn_i,
        const int* const restrict nn_j,
        const int* const restrict nn_k,
        const TF* const restrict off_i,
        const TF* const restrict off_j,
        const TF* const restrict off_k,
        const TF* const restrict x_in,
        const TF* const restrict y_in,
        const TF* const restrict z_in,
        const int isize,
        const int jsize,
        const int ksize,
        const int istride_out,
        const int jstride_out,
        const int kstride_out,
        const int istride_in,
        const int jstride_in,
        const int kstride_in,
        const int kstart_in,
        const int kend_in)
    {
        for (int k=0; k<ksize; ++k)
            for (int j=0; j<jsize; ++j)
                for (int i=0; i<isize; ++i)
                {
                    const int ip = nn_i[i];
                    const int jp = nn_j[j];
                    const int kp = nn_k[k];

                    const int ijk_in = ip*istride_in + jp*jstride_in + kp*kstride_in;
                    const int ijk_out = i*istride_out + j*jstride_out + k*kstride_out;

                    const TF fc = fld_in[ijk_in];

                    const TF sx = mc_slope(
                            fld_in[ijk_in-istride_in], fc, fld_in[ijk_in+istride_in],
                            x_in[ip]-x_in[ip-1], x_in[ip+1]-x_in[ip]);

                    const TF sy = mc_slope(
                            fld_in[ijk_in-jstride_in], fc, fld_in[ijk_in+jstride_in],
                            y_in[jp]-y_in[jp-1], y_in[jp+1]-y_in[jp]);

                    // No slope in the first and last parent layer: the
                    // ghost levels hold boundary values, not air.
                    const TF sz = (kp > kstart_in && kp < kend_in-1) ?
                        mc_slope(
                            fld_in[ijk_in-kstride_in], fc, fld_in[ijk_in+kstride_in],
                            z_in[kp]-z_in[kp-1], z_in[kp+1]-z_in[kp]) : TF(0);

                    fld_out[ijk_out] = fc + sx*off_i[i] + sy*off_j[j] + sz*off_k[k];
                }
    }


    template<typename TF>
    void nn_interpolate(
        TF* const restrict fld_out,
        const TF* const restrict fld_in,
        const int* const restrict nn_i,
        const int* const restrict nn_j,
        const int* const restrict nn_k,
        const int isize,
        const int jsize,
        const int ksize,
        const int istride_out,
        const int jstride_out,
        const int kstride_out,
        const int istride_in,
        const int jstride_in,
        const int kstride_in)
    {
        for (int k=0; k<ksize; ++k)
            for (int j=0; j<jsize; ++j)
                for (int i=0; i<isize; ++i)
                {
                    const int ijk_in = nn_i[i]*istride_in + nn_j[j]*jstride_in + nn_k[k]*kstride_in;
                    const int ijk_out = i*istride_out + j*jstride_out + k*kstride_out;

                    fld_out[ijk_out] = fld_in[ijk_in];
                }
    }
}
#endif
