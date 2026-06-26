# DeepMD for Hydrocyclone Degassing

本仓库包含论文 **Machine learning potentials reveal non-equilibrium dynamics in hydrocyclone degassing for boosting photocatalytic hydrogen production** 相关的计算输入文件、示例数据、训练好的 DeepMD 势函数模型和后处理脚本。

本工作流结合 CP2K 从头算分子动力学（AIMD）、DeePMD-kit、DP-GEN 主动学习、LAMMPS-PLUMED 非平衡分子动力学（NEMD）以及结构描述符/PCA 分析，用于研究 Pt-COF/水界面上剪切作用辅助 H2 纳米气泡脱附的微观动力学。

## 仓库内容

| 目录 | 用途 | 主要文件 |
|---|---|---|
| `1-matlab-气泡半径/` | 基于曲率修正的 Laplace 压力关系估算 H2 纳米气泡平衡半径。 | `rad.m` |
| `2-packmol-建模/` | 使用 Packmol 构建 COF/H2O/H2 三相初始构型。 | `build.inp`, `cof33.pdb`, `h2.pdb`, `h2o.pdb`, `cof_water_H2.pdb` |
| `3-cp2k-aimd/` | CP2K-PLUMED AIMD 输入文件和 PLUMED 集体变量示例输出。 | `cp2k.inp`, `plumed.dat`, `输出结果/colvar.out` |
| `4-dpdata-转化训练集/` | 将 CP2K AIMD 输出转换为 DeePMD 训练集和验证集。 | `cp2k.out`, `dpslice.py`, `conv.slurm`, `readme.txt` |
| `5-dpgen-训练势函数/` | DP-GEN 并发学习配置，用于 Deep Potential 训练和模型偏差采样。 | `param.json`, `machine.json`, `cof3.lmp`, `plumed.lammps`, `plumed.dat` |
| `6-势函数评估-误差图/` | 验证训练好的势函数并绘制能量/力预测误差密度图。 | `frozen_model.pb`, `plot-density.py`, `plot-density-f.py`, `e.slurm`, `f.slurm` |
| `7-势函数评估-pca降维/` | 计算 H2 气泡结构描述符，进行 PCA 投影，并导出 Chemiscope 可视化数据。 | `pca.py`, `apply.py`, `unified-cv-analysis-step1.json.gz`, `轨迹参考.png` |
| `8-lammps-大规模模拟/` | 使用训练好的 DeepMD 势函数运行 LAMMPS-PLUMED 大规模剪切 NEMD。 | `shear.lammps`, `input.plumed`, `h50.lmp`, `frozen_model.pb`, `sub.v100` |
| `9-cp2k-plumed-气泡拉脱表面/` | 使用 CP2K-PLUMED 进行气泡脱离表面的拉脱模拟。 | `cp2k.restart`, `plumed.dat` |
| `其余可能用到的脚本/` | 作业提交和 PDB 转 LAMMPS 数据文件的辅助脚本。 | `cp2k.sh`, `pdb转lammps.txt` |

## 软件环境

本仓库中的输入文件主要面向 Linux/HPC 环境，并使用 Slurm 提交任务。
所有的linux作业任务均在庚子超算服务器中提交并完成。
本项目中所有工作流程参考刘锦程老师视频（BV1JZ4y1E7hp）
作者邮箱：`zhoufh@mail.ecust.edu.cn` 欢迎互相交流学习。

## 工作流概览

仓库目录按照计算流程排序：

1. 估算目标 H2 气泡半径。
2. 构建 COF/H2O/H2 三相初始构型。
3. 运行 CP2K-PLUMED AIMD，生成量子力学参考数据。
4. 将 CP2K 输出转换为 DeePMD 训练和验证数据。
5. 使用 DP-GEN 主动学习训练并改进 Deep Potential 势函数。
6. 用 DFT 能量和力验证势函数精度。
7. 使用气泡结构描述符和 PCA 分析结构相空间覆盖。
8. 使用 LAMMPS-PLUMED 运行剪切驱动的大规模 NEMD。
9. 使用 CP2K-PLUMED 运行气泡脱附拉脱模拟。

## 1. 气泡半径估算

目录：

```bash
1-matlab-气泡半径/
```

`rad.m` 通过理想气体关系和曲率修正 Laplace 压力的固定点迭代求解气泡平衡半径。示例文件的主要参数为：

```matlab
N      = 60;
k_B    = 1.380649e-23;
gamma  = 0.072;
T      = 298.15;
P_out  = 111457;
delta  = 0.3e-9;
```

使用 MATLAB 运行：

脚本会输出收敛半径，并绘制半径和残差的收敛曲线。

## 2. Packmol 初始构型

目录：

```bash
2-packmol-建模/
```

`build.inp` 用于构建 COF/H2O/H2 三相初始构型：

- COF 片层：`cof33.pdb`
- H2 分子模板：`h2.pdb`
- H2O 分子模板：`h2o.pdb`
- 输出结构：`cof_water_H2.pdb`

运行：

```bash
packmol < build.inp
```

该输入示例文件放置一个 3层COF 片层，在 Pt 中心附近构造含 60 个 H2 分子的气泡，并在气泡区域外填充 1702 个水分子。

## 3. CP2K-PLUMED AIMD

目录：

```bash
3-cp2k-aimd/
```

主要文件：

- `cp2k.inp`：包含坐标的 CP2K AIMD 输入文件。
- `plumed.dat`：PLUMED 集体变量定义。
- `输出结果/colvar.out`：PLUMED 示例输出。

`cp2k.inp` 中的主要设置包括：

- `RUN_TYPE MD`
- Quickstep/GPW
- 周期性晶胞约为 `38.981 x 22.506 x 30.000 Å`
- PBE 泛函
- NVT 系综
- 1 fs 时间步长
- 通过 `PLUMED_INPUT_FILE "plumed.dat"` 启用 PLUMED

`plumed.dat` 记录 H2 气泡紧凑性以及气泡质心到 Pt 原子的距离：

```plumed
hc: CENTER ATOMS=1933-1972
dists: DISTANCES GROUPA=hc GROUPB=1933-1972 NOPBC HIGHEST LOWMEM
uwall: UPPER_WALLS ARG=dists.highest AT=0.53 KAPPA=5000.0
d2: DISTANCE ATOMS=hc,131
PRINT ARG=dists.highest,d2 STRIDE=1 FILE=colvar.out
```

运行示例：

```bash
sbatch cp2k.sh
```

提交任务脚本见
```bash
其余可能用到的脚本/
```

示例 `colvar.out` 的字段为：

```text
#! FIELDS time dists.highest d2
```

## 4. CP2K 输出转 DeePMD 数据

目录：

```bash
4-dpdata-转化训练集/
```

该目录包含一个从真实轨迹中截取的 `cp2k.out`，可用于测试数据转换流程。转换过程要求超算安装 `dpdata` 和 `cp2kdata` 插件：

使用 Slurm 提交：

```bash
sbatch conv.slurm
```

脚本会生成：

- `training_data/`
- `validation_data/`

当前示例脚本为从前 100 帧中随机选取 10 帧作为验证集，其余帧作为训练集。

## 5. DP-GEN 主动学习与势函数训练

此步骤建议配合视频操作。
目录：

```bash
5-dpgen-训练势函数/
```

主要配置文件：

- `param.json`：DP-GEN 工作流、DeePMD 网络、模型偏差探索和 CP2K 标注参数。
- `machine.json`：Slurm 资源配置和命令模板。
- `cof3.lmp`：模型偏差探索阶段使用的 LAMMPS 结构。
- `plumed.lammps`：DP-GEN 使用的 LAMMPS 模板。
- `plumed.dat`：模型偏差轨迹使用的 PLUMED 模板。

运行：

```bash
cd 5-dpgen-训练势函数
dpgen run param.json machine.json
```

`param.json` 中的关键设置：

```json
"type_map": ["O", "N", "C", "H", "Pt"],
"numb_models": 4,
"model_devi_f_trust_lo": 1.00,
"model_devi_f_trust_hi": 2.00,
"fp_style": "cp2k"
```

默认 DeepMD 模型使用 `se_e2_a` 描述符，`rcut = 6.5`，拟合网络为 `[240, 240, 240]`。

## 6. 势函数精度验证

此步骤建议配合视频操作。
目录：

```bash
6-势函数评估-误差图/
```

使用 `dp test` 将训练好的模型与 DeePMD 验证集进行比较：

```bash
cd 6-势函数评估-误差图
dp test -m frozen_model.pb -s ./validation_data -n 100 -d detail_file
```

该命令会生成：

- `detail_file.e.out`
- `detail_file.f.out`

绘制能量和力的预测误差密度图：

```bash
python plot-density.py
python plot-density-f.py
```

若轨迹文件过大建议提交 Slurm 脚本：

```bash
sbatch e.slurm
sbatch f.slurm
```

输出文件：

- `e.png`：DFT 能量与 DP 预测能量对比图。
- `f.png`：DFT 力与 DP 预测力对比图。

## 7. 气泡描述符 PCA 分析

目录：

```bash
7-势函数评估-pca降维/
```

`pca.py` 从 DeePMD 格式轨迹中计算气泡结构描述符，并进行 PCA 投影。脚本同时会导出 Chemiscope 兼容的 `.json.gz` 文件。

主要描述符包括：

- `bubble_rg`：回转半径
- `bubble_sphericity`：球形度
- `bubble_surface_area`：表面积
- `bubble_volume`：体积
- `bubble_density`：密度
- `h_coord_numbers`：H 配位数
- `avg_h2_distance`：平均 H-H 距离
- `bubble_eccentricity`：偏心率
- `bubble_to_pt_distance`：气泡到 Pt 的距离
- `h2_orientation_order`：H2 取向有序度

运行前需要修改 `pca.py` 文件开头的配置区：

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

运行：

```bash
cd 7-势函数评估-pca降维
python pca.py
```

典型输出：

- `reference_pca_model_step*.npz`
- `mul-pca-step*.json.gz`
- `mul_bubble_pca_step*.dat`

当前目录中的 `unified-cv-analysis-step1.json.gz` 是预计算的 Chemiscope 数据集，包含 468 个结构以及以下属性：

- energy
- frame index
- bubble descriptors
- coordination numbers
- `bubble_PCA_X`
- `bubble_PCA_Y`

如需将不同的轨迹投影同一坐标系，应在运行`pca.py`生成基准 PCA 坐标系后，编辑并运行 `apply.py`：

```bash
python apply.py
```

生成的`unified-cv-analysis-step1.json.gz`在`https://chemiscope.org/` 后处理。

## 8. LAMMPS-PLUMED 剪切 NEMD

目录：

```bash
8-lammps-大规模模拟/
```

主要文件：

- `h50.lmp`：LAMMPS 结构文件，包含 6046 个原子。
- `frozen_model.pb`：训练好的 DeepMD 势函数。
- `shear.lammps`：剪切驱动 NEMD 的 LAMMPS 输入。
- `input.plumed`：PLUMED 集体变量和约束。
- `sub.v100`：V100 GPU 队列的 Slurm 提交脚本。

`shear.lammps` 中的关键设置：

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

提交运行：

```bash
sbatch sub.v100
```

PLUMED 文件记录 H2 气泡半径和质心高度：

```plumed
WHOLEMOLECULES ENTITY0=793-892
hc: CENTER ATOMS=793-892
dists: DISTANCES GROUPA=hc GROUPB=793-892 HIGHEST LOWMEM
uwall: UPPER_WALLS ARG=dists.highest AT=0.78 KAPPA=5000.0
p: POSITION ATOM=hc NOPBC
PRINT ARG=dists.highest,p.z STRIDE=10 FILE=COLVAR
```

主要输出：

- `lmp.out`
- `lmp.err`
- `COLVAR`
- `velocity.dat`
- `shear.lammpstrj`
- `output.plumed`

## 9. CP2K-PLUMED 气泡拉脱模拟

目录：

```bash
9-cp2k-plumed-气泡拉脱表面/
```

该目录包含一个 CP2K restart 格式输入文件，以及用于将 H2 气泡从表面拉离的 PLUMED moving-restraint 设置。

主要文件：

- `cp2k.restart`：包含结构和 MD 设置的 CP2K restart 输入。
- `plumed.dat`：PLUMED 约束和输出定义。

`cp2k.restart` 中的主要设置：

- `RUN_TYPE MD`
- NVT 系综
- `STEPS 2000`
- 时间步长约 1 fs
- 温度 `298.15 K`
- 固定原子：`1..396`
- PLUMED 输入：`plumed.dat`

PLUMED moving restraint 作用于 H2 气泡质心的 z 坐标：

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

CP2K 运行示例：

```bash
sbatch cp2k.sh
```


## 原子类型和索引约定

DeepMD/LAMMPS 的元素类型顺序为：

| 元素 | LAMMPS type |
|---|---:|
| O | 1 |
| N | 2 |
| C | 3 |
| H | 4 |
| Pt | 5 |

DP-GEN 中使用相同顺序：

```json
"type_map": ["O", "N", "C", "H", "Pt"]
```

索引规则：

| 使用场景 | 规则 | 示例 |
|---|---|---|
| PLUMED | 1-based 原子编号 | `793-892`, `397-436` |
| LAMMPS group | 1-based 原子编号 | `group h2 id 793:892` |
| Python/NumPy | 0-based 数组索引 | `range(3834, 3894)` 对应原子编号 `3835-3894` |

使用 VMD/TopoTools 转换结构时，应保持元素到 LAMMPS type 的映射不变。辅助文件 `其余可能用到的脚本/pdb转lammps.txt` 给出了本仓库使用的映射命令。

## 输出文件说明

常见输出文件如下：

| 输出文件 | 来源 | 说明 |
|---|---|---|
| `colvar.out` 或 `COLVAR` | PLUMED | 气泡半径、距离、高度和约束相关集体变量。 |
| `training_data/`, `validation_data/` | `dpslice.py` | 由 CP2K 输出转换得到的 DeePMD 数据集。 |
| `iter.*/` | DP-GEN | 主动学习迭代目录。 |
| `frozen_model.pb` | DeePMD-kit | 冻结后的 Deep Potential 模型。 |
| `detail_file.e.out`, `detail_file.f.out` | `dp test` | DFT 与 DP 的能量/力对比数据。 |
| `e.png`, `f.png` | 绘图脚本 | 能量/力预测误差密度图。 |
| `velocity.dat` | LAMMPS `fix ave/chunk` | 沿 z 方向分箱的 x 方向速度剖面。 |
| `shear.lammpstrj` | LAMMPS dump | 剪切 NEMD 原子轨迹。 |
| `*.json.gz` | Chemiscope/PCA 脚本 | 结构描述符交互式可视化数据。 |


## 引用

如果使用本仓库，请引用对应论文，并同时引用实际使用的软件包，包括 CP2K、PLUMED、DeePMD-kit、DP-GEN、LAMMPS、Packmol、VMD/TopoTools、ASE 和 Chemiscope。

最后更新：2026-06-26
