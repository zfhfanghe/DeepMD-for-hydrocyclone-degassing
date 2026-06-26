# DeepMD for Hydrocyclone Degassing

This repository contains computational input files, example data, trained DeepMD potential models, and post-processing scripts associated with the paper **Machine learning potentials reveal non-equilibrium dynamics in hydrocyclone degassing for boosting photocatalytic hydrogen production**.

The workflow combines CP2K ab initio molecular dynamics (AIMD), DeePMD-kit, DP-GEN active learning, LAMMPS-PLUMED non-equilibrium molecular dynamics (NEMD), and structural descriptor/PCA analysis to investigate the microscopic dynamics of shear-assisted H2 nanobubble detachment at the Pt-COF/water interface.

## Repository Contents

| Directory | Purpose | Key files |
|---|---|---|
| `1-matlab-气泡半径/` | Estimate the equilibrium radius of an H2 nanobubble using a curvature-corrected Laplace pressure relation. | `rad.m` |
| `2-packmol-建模/` | Build the initial COF/H2O/H2 three-phase configuration using Packmol. | `build.inp`, `cof33.pdb`, `h2.pdb`, `h2o.pdb`, `cof_water_H2.pdb` |
| `3-cp2k-aimd/` | CP2K-PLUMED AIMD input files and example PLUMED collective-variable output. | `cp2k.inp`, `plumed.dat`, `输出结果/colvar.out` |
| `4-dpdata-转化训练集/` | Convert CP2K AIMD output into DeePMD training and validation datasets. | `cp2k.out`, `dpslice.py`, `conv.slurm`, `readme.txt` |
| `5-dpgen-训练势函数/` | DP-GEN concurrent-learning configuration for Deep Potential training and model-deviation sampling. | `param.json`, `machine.json`, `cof3.lmp`, `plumed.lammps`, `plumed.dat` |
| `6-势函数评估-误差图/` | Validate the trained potential and plot energy/force prediction-error density maps. | `frozen_model.pb`, `plot-density.py`, `plot-density-f.py`, `e.slurm`, `f.slurm` |
| `7-势函数评估-pca降维/` | Calculate H2-bubble structural descriptors, perform PCA projection, and export Chemiscope visualization data. | `pca.py`, `apply.py`, `unified-cv-analysis-step1.json.gz`, `轨迹参考.png` |
| `8-lammps-大规模模拟/` | Run large-scale LAMMPS-PLUMED shear NEMD simulations using the trained DeepMD potential. | `shear.lammps`, `input.plumed`, `h50.lmp`, `frozen_model.pb`, `sub.v100` |
| `9-cp2k-plumed-气泡拉脱表面/` | Run CP2K-PLUMED pulling simulations for bubble detachment from the surface. | `cp2k.restart`, `plumed.dat` |
| `其余可能用到的脚本/` | Auxiliary scripts for job submission and PDB-to-LAMMPS data conversion. | `cp2k.sh`, `pdb转lammps.txt` |

## Software Environment

The input files in this repository are mainly intended for Linux/HPC environments and are submitted through Slurm.
All Linux jobs were submitted and completed on the Gengzi supercomputing server.
All workflows in this project refer to the tutorial video by Prof. Jincheng Liu (BV1JZ4y1E7hp).
Author email: `zhoufh@mail.ecust.edu.cn`. Discussions and exchanges are welcome.

## Workflow Overview

The repository directories are ordered according to the computational workflow:

1. Estimate the target H2 bubble radius.
2. Build the COF/H2O/H2 three-phase initial configuration.
3. Run CP2K-PLUMED AIMD to generate quantum-mechanical reference data.
4. Convert CP2K output into DeePMD training and validation data.
5. Use DP-GEN active learning to train and improve the Deep Potential model.
6. Validate the potential accuracy with DFT energies and forces.
7. Analyze structural phase-space coverage using bubble descriptors and PCA.
8. Run shear-driven large-scale NEMD with LAMMPS-PLUMED.
9. Run CP2K-PLUMED bubble-detachment pulling simulations.

## 1. Bubble Radius Estimation

Directory:

```bash
1-matlab-气泡半径/
```

`rad.m` solves the equilibrium bubble radius by fixed-point iteration using the ideal-gas relation and a curvature-corrected Laplace pressure. The main parameters in the example file are:

```matlab
N      = 60;
k_B    = 1.380649e-23;
gamma  = 0.072;
T      = 298.15;
P_out  = 111457;
delta  = 0.3e-9;
```

Run with MATLAB.

The script outputs the converged radius and plots the convergence curves for the radius and residual.

## 2. Packmol Initial Configuration

Directory:

```bash
2-packmol-建模/
```

`build.inp` is used to build the COF/H2O/H2 three-phase initial configuration:

- COF slab: `cof33.pdb`
- H2 molecular template: `h2.pdb`
- H2O molecular template: `h2o.pdb`
- Output structure: `cof_water_H2.pdb`

Run:

```bash
packmol < build.inp
```

This example input places a three-layer COF slab, constructs a bubble containing 60 H2 molecules near the Pt center, and fills the region outside the bubble with 1702 water molecules.

## 3. CP2K-PLUMED AIMD

Directory:

```bash
3-cp2k-aimd/
```

Main files:

- `cp2k.inp`: CP2K AIMD input file with embedded coordinates.
- `plumed.dat`: PLUMED collective-variable definitions.
- `输出结果/colvar.out`: example PLUMED output.

Main settings in `cp2k.inp` include:

- `RUN_TYPE MD`
- Quickstep/GPW
- Periodic cell of approximately `38.981 x 22.506 x 30.000 Å`
- PBE functional
- NVT ensemble
- 1 fs timestep
- PLUMED enabled through `PLUMED_INPUT_FILE "plumed.dat"`

`plumed.dat` records the compactness of the H2 bubble and the distance between the bubble center of mass and the Pt atom:

```plumed
hc: CENTER ATOMS=1933-1972
dists: DISTANCES GROUPA=hc GROUPB=1933-1972 NOPBC HIGHEST LOWMEM
uwall: UPPER_WALLS ARG=dists.highest AT=0.53 KAPPA=5000.0
d2: DISTANCE ATOMS=hc,131
PRINT ARG=dists.highest,d2 STRIDE=1 FILE=colvar.out
```

Example run:

```bash
sbatch cp2k.sh
```

The submission script is located in:

```bash
其余可能用到的脚本/
```

The example `colvar.out` has the following fields:

```text
#! FIELDS time dists.highest d2
```

## 4. Convert CP2K Output to DeePMD Data

Directory:

```bash
4-dpdata-转化训练集/
```

This directory contains a `cp2k.out` file extracted from a real trajectory, which can be used to test the data-conversion workflow. The conversion requires the `dpdata` and `cp2kdata` plugins to be installed on the supercomputing system.

Submit with Slurm:

```bash
sbatch conv.slurm
```

The script generates:

- `training_data/`
- `validation_data/`

The current example script randomly selects 10 frames from the first 100 frames as the validation set, and uses the remaining frames as the training set.

## 5. DP-GEN Active Learning and Potential Training

It is recommended to perform this step together with the tutorial video.

Directory:

```bash
5-dpgen-训练势函数/
```

Main configuration files:

- `param.json`: DP-GEN workflow, DeePMD network, model-deviation exploration, and CP2K labeling parameters.
- `machine.json`: Slurm resource configuration and command templates.
- `cof3.lmp`: LAMMPS structure used for model-deviation exploration.
- `plumed.lammps`: LAMMPS template used by DP-GEN.
- `plumed.dat`: PLUMED template used for model-deviation trajectories.

Run:

```bash
cd 5-dpgen-训练势函数
dpgen run param.json machine.json
```

Key settings in `param.json`:

```json
"type_map": ["O", "N", "C", "H", "Pt"],
"numb_models": 4,
"model_devi_f_trust_lo": 1.00,
"model_devi_f_trust_hi": 2.00,
"fp_style": "cp2k"
```

The default DeepMD model uses the `se_e2_a` descriptor with `rcut = 6.5`, and the fitting network is `[240, 240, 240]`.

## 6. Potential Accuracy Validation

It is recommended to perform this step together with the tutorial video.

Directory:

```bash
6-势函数评估-误差图/
```

Use `dp test` to compare the trained model with the DeePMD validation dataset:

```bash
cd 6-势函数评估-误差图
dp test -m frozen_model.pb -s ./validation_data -n 100 -d detail_file
```

This command generates:

- `detail_file.e.out`
- `detail_file.f.out`

Plot the energy and force prediction-error density maps:

```bash
python plot-density.py
python plot-density-f.py
```

If the trajectory files are large, submitting Slurm scripts is recommended:

```bash
sbatch e.slurm
sbatch f.slurm
```

Output files:

- `e.png`: DFT energy vs DP-predicted energy.
- `f.png`: DFT force vs DP-predicted force.

## 7. PCA Analysis of Bubble Descriptors

Directory:

```bash
7-势函数评估-pca降维/
```

`pca.py` calculates bubble structural descriptors from DeePMD-format trajectories and performs PCA projection. The script also exports a Chemiscope-compatible `.json.gz` file.

Main descriptors include:

- `bubble_rg`: radius of gyration
- `bubble_sphericity`: sphericity
- `bubble_surface_area`: surface area
- `bubble_volume`: volume
- `bubble_density`: density
- `h_coord_numbers`: H coordination number
- `avg_h2_distance`: average H-H distance
- `bubble_eccentricity`: eccentricity
- `bubble_to_pt_distance`: bubble-to-Pt distance
- `h2_orientation_order`: H2 orientational order

Before running, edit the configuration block at the beginning of `pca.py`:

```python
DATA_DIRS = [
    "/path/to/traindata/tr1/set.000",
    "/path/to/traindata/tr2/set.000"
]
BUBBLE_INDICES = list(range(3834, 3894))
SAMPLING_STEP = 1
MAX_FRAMES = 1000
START_FRAME = 0
```

Run:

```bash
cd 7-势函数评估-pca降维
python pca.py
```

Typical outputs:

- `reference_pca_model_step*.npz`
- `mul-pca-step*.json.gz`
- `mul_bubble_pca_step*.dat`

The included `unified-cv-analysis-step1.json.gz` is a precomputed Chemiscope dataset containing 468 structures and the following properties:

- energy
- frame index
- bubble descriptors
- coordination numbers
- `bubble_PCA_X`
- `bubble_PCA_Y`

To project different trajectories into the same coordinate system, first run `pca.py` to generate the reference PCA coordinate system, then edit and run `apply.py`:

```bash
python apply.py
```

The generated `unified-cv-analysis-step1.json.gz` can be post-processed at `https://chemiscope.org/`.

## 8. LAMMPS-PLUMED Shear NEMD

Directory:

```bash
8-lammps-大规模模拟/
```

Main files:

- `h50.lmp`: LAMMPS structure file containing 6046 atoms.
- `frozen_model.pb`: trained DeepMD potential.
- `shear.lammps`: LAMMPS input for shear-driven NEMD.
- `input.plumed`: PLUMED collective variables and restraints.
- `sub.v100`: Slurm submission script for the V100 GPU queue.

Key settings in `shear.lammps`:

```lammps
units           metal
boundary        p p p
atom_style      atomic
read_data       h50.lmp
pair_style      deepmd ./frozen_model.pb
group           substrate id <= 792
group           h2        id 793:892
group           water     id >= 893
fix             shear_drive middle_fluid addforce 0.001 0.0 0.0
fix             velocity_profile mobile ave/chunk 5 200 1000 z_chunks vx file velocity.dat
```

Submit:

```bash
sbatch sub.v100
```

The PLUMED file records the H2 bubble radius and center-of-mass height:

```plumed
WHOLEMOLECULES ENTITY0=793-892
hc: CENTER ATOMS=793-892
dists: DISTANCES GROUPA=hc GROUPB=793-892 HIGHEST LOWMEM
uwall: UPPER_WALLS ARG=dists.highest AT=0.78 KAPPA=5000.0
p: POSITION ATOM=hc NOPBC
PRINT ARG=dists.highest,p.z STRIDE=10 FILE=COLVAR
```

Main outputs:

- `lmp.out`
- `lmp.err`
- `COLVAR`
- `velocity.dat`
- `shear.lammpstrj`
- `output.plumed`

## 9. CP2K-PLUMED Bubble Pulling Simulation

Directory:

```bash
9-cp2k-plumed-气泡拉脱表面/
```

This directory contains a CP2K restart-format input file and a PLUMED moving-restraint setup for pulling the H2 bubble away from the surface.

Main files:

- `cp2k.restart`: CP2K restart input containing the structure and MD settings.
- `plumed.dat`: PLUMED restraint and output definitions.

Main settings in `cp2k.restart`:

- `RUN_TYPE MD`
- NVT ensemble
- `STEPS 2000`
- timestep about 1 fs
- temperature `298.15 K`
- fixed atoms: `1..396`
- PLUMED input: `plumed.dat`

The PLUMED moving restraint acts on the z coordinate of the H2 bubble center of mass:

```plumed
hc: CENTER ATOMS=397-436
dists: DISTANCES GROUPA=hc GROUPB=397-436 NOPBC HIGHEST LOWMEM
uwall: UPPER_WALLS ARG=dists.highest AT=0.58 KAPPA=5000.0
p: POSITION ATOM=hc NOPBC

MOVINGRESTRAINT ...
  LABEL=pull
  ARG=p.z
  STEP0=1000 AT0=1.66 KAPPA0=3000.0
  STEP1=1500 AT1=2.16 KAPPA1=3000.0
  STEP2=2000 AT2=2.16
... MOVINGRESTRAINT

PRINT ARG=dists.highest,p.z,pull.bias STRIDE=1 FILE=COLVAR
```

Example CP2K run:

```bash
sbatch cp2k.sh
```

## Atom Types and Index Conventions

The element type order used by DeepMD/LAMMPS is:

| Element | LAMMPS type |
|---|---:|
| O | 1 |
| N | 2 |
| C | 3 |
| H | 4 |
| Pt | 5 |

DP-GEN uses the same order:

```json
"type_map": ["O", "N", "C", "H", "Pt"]
```

Index conventions:

| Context | Convention | Example |
|---|---|---|
| PLUMED | 1-based atom ID | `793-892`, `397-436` |
| LAMMPS group | 1-based atom ID | `group h2 id 793:892` |
| Python/NumPy | 0-based array index | `range(3834, 3894)` corresponds to atom IDs `3835-3894` |

When converting structures with VMD/TopoTools, keep the element-to-LAMMPS-type mapping unchanged. The helper file `其余可能用到的脚本/pdb转lammps.txt` provides the mapping commands used in this repository.

## Output Files

Common output files are listed below:

| Output file | Source | Description |
|---|---|---|
| `colvar.out` or `COLVAR` | PLUMED | Bubble radius, distance, height, and restraint-related collective variables. |
| `training_data/`, `validation_data/` | `dpslice.py` | DeePMD datasets converted from CP2K output. |
| `iter.*/` | DP-GEN | Active-learning iteration directories. |
| `frozen_model.pb` | DeePMD-kit | Frozen Deep Potential model. |
| `detail_file.e.out`, `detail_file.f.out` | `dp test` | DFT vs DP energy/force comparison data. |
| `e.png`, `f.png` | plotting scripts | Energy/force prediction-error density maps. |
| `velocity.dat` | LAMMPS `fix ave/chunk` | z-binned x-velocity profile. |
| `shear.lammpstrj` | LAMMPS dump | Atomistic trajectory from shear NEMD. |
| `*.json.gz` | Chemiscope/PCA scripts | Interactive visualization data for structural descriptors. |

## Citation

If you use this repository, please cite the corresponding paper and the software packages used in your workflow, including CP2K, PLUMED, DeePMD-kit, DP-GEN, LAMMPS, Packmol, VMD/TopoTools, ASE, and Chemiscope.

Last updated: 2026-06-26
