#
# ECMWF Atos HPC2020. AMD EPYC Rome (Zen 2), the same microarchitecture as
# Snellius's CPU partition, so the flags come straight from snellius.cmake.
#
#   module load prgenv/gnu gcc/11.2.0 hpcx-openmpi
#   module load netcdf4 hdf5 fftw
#
# ECMWF's modules do NOT put headers on the compiler's default search path.
# They export <PKG>_DIR / <PKG>_INCLUDE / <PKG>_LIB instead, which is why a
# config copied from another machine fails with "netcdf.h: No such file or
# directory" even though the module is loaded. The find_path/find_library
# blocks below pick the paths up from whichever of those variables exists.
#

if(USEMPI)
    set(ENV{CXX} mpicxx)
    set(ENV{FC} mpif90)
else()
    set(ENV{CXX} g++)
    set(ENV{FC} gfortran)
endif()

set(USER_CXX_FLAGS "")
set(USER_CXX_FLAGS_RELEASE "-O3 -march=znver2 -mtune=znver2 -mfma -mavx2 -fomit-frame-pointer")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(USER_FC_FLAGS "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
set(USER_FC_FLAGS_RELEASE "-DNDEBUG -O3 -march=znver2 -mtune=znver2 -mfma -mavx2 -fomit-frame-pointer")

# Header locations. ENV <VAR> lets find_path read a module variable that holds
# a bare directory; the explicit <DIR>/include entries cover the rest.
find_path(NETCDF_INCLUDE_DIR netcdf.h
    HINTS ENV NETCDF4_INCLUDE ENV NETCDF_INCLUDE
          $ENV{NETCDF4_DIR}/include $ENV{NETCDF_DIR}/include)

find_path(FFTW_INCLUDE_DIR fftw3.h
    HINTS ENV FFTW_INCLUDE $ENV{FFTW_DIR}/include)

find_path(HDF5_INCLUDE_DIR hdf5.h
    HINTS ENV HDF5_INCLUDE $ENV{HDF5_DIR}/include)

if(NOT NETCDF_INCLUDE_DIR)
    message(FATAL_ERROR
        "netcdf.h not found. Load the module and delete CMakeCache.txt:\n"
        "    module load netcdf4\n"
        "    rm CMakeCache.txt && cmake .. -DSYST=atos -DUSEMPI=TRUE -DCMAKE_BUILD_TYPE=Release\n"
        "If it is loaded, print what it exports with 'module show netcdf4' and\n"
        "add that include directory to INCLUDE_DIRS in config/atos.cmake.")
endif()
if(NOT FFTW_INCLUDE_DIR)
    message(FATAL_ERROR "fftw3.h not found. Try 'module load fftw'.")
endif()

set(INCLUDE_DIRS ${NETCDF_INCLUDE_DIR} ${FFTW_INCLUDE_DIR})
if(HDF5_INCLUDE_DIR)
    list(APPEND INCLUDE_DIRS ${HDF5_INCLUDE_DIR})
endif()
list(REMOVE_DUPLICATES INCLUDE_DIRS)
message(STATUS "Include dirs: ${INCLUDE_DIRS}")

# Library locations, same idea.
find_library(NETCDF_C_LIB NAMES netcdf
    HINTS ENV NETCDF4_LIB ENV NETCDF_LIB
          $ENV{NETCDF4_DIR}/lib $ENV{NETCDF4_DIR}/lib64
          $ENV{NETCDF_DIR}/lib  $ENV{NETCDF_DIR}/lib64)

find_library(FFTW_C_LIB  NAMES fftw3  HINTS ENV FFTW_LIB $ENV{FFTW_DIR}/lib $ENV{FFTW_DIR}/lib64)
find_library(FFTWF_C_LIB NAMES fftw3f HINTS ENV FFTW_LIB $ENV{FFTW_DIR}/lib $ENV{FFTW_DIR}/lib64)
find_library(HDF5_C_LIB  NAMES hdf5   HINTS ENV HDF5_LIB $ENV{HDF5_DIR}/lib $ENV{HDF5_DIR}/lib64)

# Fall back to the plain -l name if the module keeps the library somewhere
# find_library does not look; the linker's own search path may still have it.
if(NOT NETCDF_C_LIB)
    set(NETCDF_C_LIB "netcdf")
endif()
if(NOT FFTW_C_LIB)
    set(FFTW_C_LIB "fftw3")
endif()
if(NOT FFTWF_C_LIB)
    set(FFTWF_C_LIB "fftw3f")
endif()
if(NOT HDF5_C_LIB)
    set(HDF5_C_LIB "hdf5")
endif()

set(NETCDF_LIB_C ${NETCDF_C_LIB})
set(FFTW_LIB     ${FFTW_C_LIB})
set(FFTWF_LIB    ${FFTWF_C_LIB})
set(HDF5_LIB     ${HDF5_C_LIB})

# No szip here: snellius.cmake links -lsz, which Atos's HDF5 does not
# necessarily provide. Add it back only if the link fails on a szip symbol.
set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB} m z curl)
message(STATUS "Libraries: ${LIBS}")

add_definitions(-DRESTRICTKEYWORD=__restrict__)
add_definitions(-DRTE_USE_CBOOL)

# Snellius disables collective MPI-IO for cross sections because it misbehaves
# on GPFS. Atos is Lustre, where collective writes are usually much faster, so
# it is left ON. Uncomment if cross-section files come out truncated.
# add_definitions(-DDISABLE_2D_MPIIO=1)

