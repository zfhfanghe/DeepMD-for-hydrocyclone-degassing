#!/bin/bash
#SBATCH -N 1 #调用节点数
#SBATCH -n 64 #调用总核数
#SBATCH -p intel64 #队列名称
#SBATCH -o output #转存标准输出至文件
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate deepmd-2.2.9-gpu

cd "$SLURM_SUBMIT_DIR"
python dpslice.py
