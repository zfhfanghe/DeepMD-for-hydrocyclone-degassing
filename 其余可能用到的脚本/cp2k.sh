#!/bin/bash
#SBATCH -n 96
#SBATCH -N 1   
#SBATCH -p intel96


module load CP2K/2024.3-toolchain-plumed
mpirun  cp2k.psmp cp2k.inp >cp2k.out 2>cp2k.err
