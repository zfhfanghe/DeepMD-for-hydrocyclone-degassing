第六步：检验模型的好坏。 终端输入：
dp test -m graph-compress.pb -s ./validation_data -n 100 -d detail_file 回车，
训练集改到所在目录(第4步产生的验证集） 

利用detail_file.f.out+plot-density-f.py绘制力的误差图、detail_file.e.out+plot-density.py绘制能量的误差图，直接提交脚本：
sbatch f.slurm