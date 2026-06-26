#!/bin/bash
#SBATCH -N 1 #调用节点数
#SBATCH -n 120 #调用总核数
#SBATCH -p res_zhoufh #队列名称
#SBATCH -o output #转存标准输出至文件
module load anaconda3
cd "$SLURM_SUBMIT_DIR"
python plot-density.py

