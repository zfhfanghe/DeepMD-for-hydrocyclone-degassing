#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
应用参考PCA模型到新轨迹
"""

import numpy as np
import os
from pca import *  # 导入你原始脚本的所有函数

def load_pca_model(model_file):
    """加载PCA模型参数"""
    if not os.path.exists(model_file):
        print(f"❌ 找不到PCA模型文件: {model_file}")
        print("请先运行 pca.py 生成参考模型！")
        return None
        
    data = np.load(model_file, allow_pickle=True)
    return {
        'mean': data['mean'],
        'std': data['std'],
        'eigenvectors': data['eigenvectors'],
        'explained_variance': data['explained_variance'],
        'cv_names': data['cv_names'].tolist()
    }

def apply_pca_model(cv_matrix, pca_model):
    """使用预训练的PCA模型变换新数据"""
    # 确保std不为0
    data_std = pca_model['std'].copy()
    data_std[data_std == 0] = 1
    
    # 使用参考模型的参数进行标准化
    data_scaled = (cv_matrix - pca_model['mean']) / data_std
    
    # 投影到参考主成分空间
    pca_result = np.dot(data_scaled, pca_model['eigenvectors'][:, :2])
    
    return pca_result

def main_apply():
    """使用参考PCA模型分析新轨迹"""
    
    # ===========================================
    # 配置新轨迹参数 - 在这里修改！
    # ===========================================
    NEW_DATA_DIR = "/home/zhoufh/zfh/tq/rad/dp2/traindata/tr3/set.000"  # 👈 修改这里
    NEW_BUBBLE_INDICES = list(range(3834, 3893))   # 👈 修改这里（如果需要）
    NEW_SUFFIX = "_tr3"                          # 👈 修改这里
    
    # 参考模型文件名
    REFERENCE_MODEL = "reference_pca_model_step10.npz"  # 👈 根据需要修改
    
    # ===========================================
    
    print("=== 应用参考PCA模型到新轨迹 ===")
    
    # 1. 加载参考PCA模型
    pca_model = load_pca_model(REFERENCE_MODEL)
    if pca_model is None:
        return
        
    print(f"✅ 成功加载参考PCA模型: {REFERENCE_MODEL}")
    print(f"📊 模型解释方差: PC1={pca_model['explained_variance'][0]:.3f}, PC2={pca_model['explained_variance'][1]:.3f}")
    
    # 2. 更新全局变量（适配多轨迹版本）
    global DATA_DIRS, BUBBLE_INDICES
    original_data_dirs = DATA_DIRS
    original_bubble_indices = BUBBLE_INDICES
    
    # 临时设置为单轨迹
    DATA_DIRS = [NEW_DATA_DIR]
    BUBBLE_INDICES = NEW_BUBBLE_INDICES
    
    print(f"🔄 分析新轨迹: {NEW_DATA_DIR}")
    print(f"🫧 新气泡范围: {BUBBLE_INDICES[0]+1}-{BUBBLE_INDICES[-1]+1}")
    
    try:
        # 3. 使用多轨迹加载函数（但只有一个轨迹）
        frames, frame_indices, trajectory_labels, trajectory_info = load_multiple_trajectories(DATA_DIRS)
        
        if len(frames) == 0:
            print("❌ 没有成功加载轨迹!")
            return
        
        print(f"成功加载 {len(frames)} 个结构")
        
        # 4. 计算CV
        all_cvs = calculate_unified_cvs_optimized(frames, frame_indices)
        
        # 5. 构建CV矩阵（必须与参考模型顺序一致）
        expected_cv_names = pca_model['cv_names']
        cv_matrix = np.array([all_cvs[name] for name in expected_cv_names]).T
        
        print(f"CV矩阵形状: {cv_matrix.shape}")
        print(f"期望CV顺序: {expected_cv_names}")
        
        # 6. 应用参考PCA模型
        print("应用参考PCA模型...")
        pca_result = apply_pca_model(cv_matrix, pca_model)
        
        # 7. 保存结果
        energies_data = np.array(all_cvs["energies"]) if all_cvs["energies"] else np.zeros(len(pca_result))
        
        output_data = np.column_stack([pca_result[:, 0], pca_result[:, 1], energies_data])
        output_file = f"bubble_pca_step{SAMPLING_STEP}{NEW_SUFFIX}.dat"
        
        np.savetxt(output_file, output_data,
                   header="bubble_PCA_X\tbubble_PCA_Y\tenergy",
                   fmt='%.6f', delimiter='\t')
        
        print(f"💾 新轨迹PCA结果已保存: {output_file}")
        print(f"✅ 该结果与参考轨迹在同一PCA坐标系中，可直接比较！")
        
        # 8. 生成对比建议
        print(f"\n🎯 对比可视化建议:")
        reference_file = f"multi_traj_bubble_pca_step{SAMPLING_STEP}.dat"
        if os.path.exists(reference_file):
            print(f"参考文件: {reference_file}")
            print(f"新轨迹文件: {output_file}")
            print(f"可以将两个文件的PCA_X, PCA_Y列放在同一张图中比较")
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复原始全局变量
        DATA_DIRS = original_data_dirs
        BUBBLE_INDICES = original_bubble_indices

if __name__ == "__main__":
    main_apply()