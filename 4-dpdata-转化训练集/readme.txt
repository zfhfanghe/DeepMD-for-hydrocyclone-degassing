在DeePMD环境下对数据集 cp2k.out 进行数据转化，借助dpdata程序实现；转化完成后利用脚本(dpslice.py)对数据集进行切片。

终端直接提交命令(需要有cp2kdata插件，庚子超算已安装)。
sbatch conv.slurm
注意：dpslice.py 需要根据训练集和验证集大小分别调整参数


具体原理（古法操作步骤）
1. 需要先进入DeePMD环境：module load anaconda3 进入python环境；
2. conda activate deepmd-2.2.9-gpu 进入DeePMD-kit环境；
3. python3 回车；
4. import dpdata 回车；
5. data = dpdata.LabeledSystem( ".",  fmt="cp2kdata/md", cp2k_output_name="cp2k.out") 大约需要数十分钟，参考时间：5000帧，1972个原子的体系，用时约2小时，因此不推荐本地运算，直接提交任务到队列。

目录下的cp2k.out为示例轨迹，从真实轨迹中截取，用作程序测试。