#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一CV分析：气泡特性 + 全体配位数 (优化版)
作者: 基于前面讨论整合
功能: 计算气泡CV、全体配位数，并进行PCA降维分析
"""

import numpy as np
import chemiscope
from ase import Atoms
import os
import time

# ===========================================
# 配置参数
# ===========================================
DATA_DIRS = [
    "/home/zhoufh/zfh/tq/rad/dp2/traindata/tr1/set.000",
    "/home/zhoufh/zfh/tq/rad/dp2/traindata/tr2/set.000"
    ]
BUBBLE_INDICES = list(range(3834, 3894))  # 3835-3894 → 3834-3893

# 采样参数
SAMPLING_STEP = 1      # 每隔10步提取一次
MAX_FRAMES = 1000        # 最大帧数限制
START_FRAME = 0         # 起始帧

# 配位数计算的截断距离 (单位: Å)
H_H_CUTOFF = 2.5        # H-H配位截断距离
O_H_CUTOFF = 2.5        # O-H配位截断距离
O_O_CUTOFF = 2.5        # O-O配位截断距离

# ===========================================
# 核心计算函数
# ===========================================

def simple_pca(data, n_components=2):
    """简单的PCA实现"""
    data_centered = data - np.mean(data, axis=0)
    cov_matrix = np.cov(data_centered.T)
    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
    
    idx = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[idx]
    eigenvecs = eigenvecs[:, idx]
    
    result = np.dot(data_centered, eigenvecs[:, :n_components])
    explained_variance_ratio = eigenvals[:n_components] / np.sum(eigenvals)
    
    return result, explained_variance_ratio, eigenvecs

def calculate_coordination_numbers_fast(positions, symbols, verbose=False):
    """快速计算配位数 - 使用向量化操作"""
    if verbose:
        print("    计算全体配位数...")
        start_time = time.time()
    
    # 分离原子索引
    h_indices = np.array([i for i, s in enumerate(symbols) if s == 'H'])
    o_indices = np.array([i for i, s in enumerate(symbols) if s == 'O'])
    
    if verbose:
        print(f"    H原子数量: {len(h_indices)}, O原子数量: {len(o_indices)}")
    
    # O原子配位数计算 (向量化)
    if len(o_indices) > 0 and len(h_indices) > 0:
        o_positions = positions[o_indices]  # (n_o, 3)
        h_positions = positions[h_indices]  # (n_h, 3)
        
        # 计算O-H距离矩阵 (n_o, n_h)
        o_expanded = o_positions[:, np.newaxis, :]  # (n_o, 1, 3)
        h_expanded = h_positions[np.newaxis, :, :]  # (1, n_h, 3)
        oh_distances = np.linalg.norm(o_expanded - h_expanded, axis=2)  # (n_o, n_h)
        
        # 计算每个O原子的配位数
        o_coordination_counts = np.sum(oh_distances < O_H_CUTOFF, axis=1)
        avg_o_coordination = float(np.mean(o_coordination_counts))
    else:
        avg_o_coordination = 0.0
    
    if verbose:
        print(f"    O配位数计算完成，耗时: {time.time() - start_time:.2f}s")
    
    # H原子配位数计算 (采样计算，避免过大计算量)
    if len(h_indices) > 100:
        # 如果H原子太多，随机采样100个进行计算
        sampled_h_indices = np.random.choice(h_indices, size=100, replace=False)
        if verbose:
            print(f"    H原子太多，随机采样100个计算配位数")
    else:
        sampled_h_indices = h_indices
    
    if len(sampled_h_indices) > 0:
        h_coord_counts = []
        
        for i, h_idx in enumerate(sampled_h_indices):
            h_pos = positions[h_idx]
            
            # 与其他H原子的距离
            other_h_indices = h_indices[h_indices != h_idx]
            if len(other_h_indices) > 0:
                h_distances = np.linalg.norm(positions[other_h_indices] - h_pos, axis=1)
                h_h_neighbors = np.sum(h_distances < H_H_CUTOFF)
            else:
                h_h_neighbors = 0
            
            # 与O原子的距离
            if len(o_indices) > 0:
                o_distances = np.linalg.norm(positions[o_indices] - h_pos, axis=1)
                h_o_neighbors = np.sum(o_distances < O_H_CUTOFF)
            else:
                h_o_neighbors = 0
            
            h_coord_counts.append(h_h_neighbors + h_o_neighbors)
            
            # 只在verbose模式下显示详细进度
        
        avg_h_coordination = float(np.mean(h_coord_counts))
    else:
        avg_h_coordination = 0.0
    
    if verbose:
        total_time = time.time() - start_time
        print(f"    配位数计算完成，总耗时: {total_time:.2f}s")
    
    return avg_o_coordination, avg_h_coordination
    
def calculate_bubble_cvs_fast(bubble_positions):
    """快速计算气泡CV"""
    n_atoms = len(bubble_positions)
    bubble_com = np.mean(bubble_positions, axis=0)
    
    # 1. 气泡回转半径
    distances_from_com = np.linalg.norm(bubble_positions - bubble_com, axis=1)
    rg = float(np.sqrt(np.mean(distances_from_com**2)))
    
    # 2. 球形度
    bubble_centered = bubble_positions - bubble_com
    if n_atoms > 3:
        cov_matrix = np.cov(bubble_centered.T)
        eigenvals = np.linalg.eigvals(cov_matrix)
        eigenvals = np.sort(eigenvals)[::-1]
        sphericity = float(eigenvals[2] / eigenvals[0]) if eigenvals[0] > 0 else 0.0
    else:
        sphericity = 1.0
    
    # 3. 表面积和体积
    max_radius = float(np.max(distances_from_com)) if len(distances_from_com) > 0 else 0.0
    surface_area = float(4 * np.pi * max_radius**2)
    volume = float((4/3) * np.pi * max_radius**3)
    
    # 4. 密度
    density = float(n_atoms / volume) if volume > 0 else 0.0
    
    # 5. H配位数 (向量化计算)
    if n_atoms > 1:
        # 计算所有原子间的距离矩阵
        pos_expanded_i = bubble_positions[:, np.newaxis, :]  # (n, 1, 3)
        pos_expanded_j = bubble_positions[np.newaxis, :, :]  # (1, n, 3)
        distance_matrix = np.linalg.norm(pos_expanded_i - pos_expanded_j, axis=2)  # (n, n)
        
        # 排除自身，计算配位数
        np.fill_diagonal(distance_matrix, np.inf)  # 避免自身距离为0
        coord_counts = np.sum(distance_matrix < H_H_CUTOFF, axis=1)
        avg_coord = float(np.mean(coord_counts))
    else:
        avg_coord = 0.0
    
    # 6. 平均H2间距
    if n_atoms > 1:
        upper_triangle = np.triu_indices(n_atoms, k=1)
        all_distances = distance_matrix[upper_triangle]
        all_distances = all_distances[all_distances != np.inf]
        avg_dist = float(np.mean(all_distances)) if len(all_distances) > 0 else 0.0
    else:
        avg_dist = 0.0
    
    # 7. 偏心率
    if n_atoms > 3:
        eigenvals = np.sort(np.linalg.eigvals(np.cov(bubble_centered.T)))[::-1]
        if eigenvals[0] > 0 and eigenvals[1] > 0:
            eccentricity = float(np.sqrt(1 - eigenvals[1]**2 / eigenvals[0]**2))
        else:
            eccentricity = 0.0
    else:
        eccentricity = 0.0
    
    return {
        'rg': rg,
        'sphericity': sphericity,
        'surface_area': surface_area,
        'volume': volume,
        'density': density,
        'avg_coord': avg_coord,
        'avg_distance': avg_dist,
        'eccentricity': eccentricity
    }

def calculate_h2_orientation_order_fast(bubble_positions):
    """快速计算H2取向有序度"""
    n_atoms = len(bubble_positions)
    
    if n_atoms < 4:
        return 0.0
    
    # 简化方法：每两个相邻H原子看作一个H2分子
    h2_vectors = []
    
    for i in range(0, n_atoms-1, 2):
        if i+1 < n_atoms:
            h1_pos = bubble_positions[i]
            h2_pos = bubble_positions[i+1]
            
            dist = np.linalg.norm(h2_pos - h1_pos)
            if dist < 2.0:  # 合理的H2键长
                h2_vector = h2_pos - h1_pos
                h2_vector = h2_vector / np.linalg.norm(h2_vector)
                h2_vectors.append(h2_vector)
    
    if len(h2_vectors) < 2:
        return 0.0
    
    # 向量化计算所有H2向量间的点积
    h2_vectors = np.array(h2_vectors)  # (n_h2, 3)
    
    # 计算所有向量对的点积
    dot_products = np.abs(np.dot(h2_vectors, h2_vectors.T))
    
    # 排除对角线元素
    mask = ~np.eye(len(h2_vectors), dtype=bool)
    orientations = dot_products[mask]
    
    return float(np.mean(orientations)) if len(orientations) > 0 else 0.0

def calculate_unified_cvs_optimized(frames, selected_frame_indices):
    """优化版CV计算 - 每50帧输出一次"""
    print(f"计算统一CV指标 (优化版)...")
    print(f"气泡区域: 原子 {BUBBLE_INDICES[0]+1}-{BUBBLE_INDICES[-1]+1}")
    print(f"分析 {len(frames)} 帧")
    
    # 初始化所有CV列表
    bubble_rg = []
    bubble_sphericity = []
    bubble_surface_area = []
    bubble_volume = []
    bubble_density = []
    h_coord_numbers = []
    avg_h2_distance = []
    bubble_eccentricity = []
    bubble_to_pt_distance = []
    h2_orientation_order = []
    o_coordination_numbers = []
    h_coordination_numbers = []
    energies = []
    frame_indices = []
    
    total_start_time = time.time()
    
    for frame_idx, frame in enumerate(frames):
        frame_start_time = time.time()
        
        positions = frame.get_positions()
        symbols = frame.get_chemical_symbols()
        
        # 记录信息
        original_frame_idx = selected_frame_indices[frame_idx]
        frame_indices.append(original_frame_idx)
        
        if 'energy' in frame.info:
            energies.append(float(frame.info['energy']))
        
        # 只在特定帧输出详细信息
        verbose = (frame_idx + 1) % 200 == 0 or frame_idx == 0
        
        if verbose:
            print(f"  处理第 {frame_idx + 1}/{len(frames)} 帧 (原始第 {original_frame_idx} 帧)")
        
        # === 气泡CV计算 ===
        bubble_positions = positions[BUBBLE_INDICES]
        bubble_com = np.mean(bubble_positions, axis=0)
        
        # 快速计算气泡基础CV
        bubble_cvs = calculate_bubble_cvs_fast(bubble_positions)
        
        bubble_rg.append(bubble_cvs['rg'])
        bubble_sphericity.append(bubble_cvs['sphericity'])
        bubble_surface_area.append(bubble_cvs['surface_area'])
        bubble_volume.append(bubble_cvs['volume'])
        bubble_density.append(bubble_cvs['density'])
        h_coord_numbers.append(bubble_cvs['avg_coord'])
        avg_h2_distance.append(bubble_cvs['avg_distance'])
        bubble_eccentricity.append(bubble_cvs['eccentricity'])
        
        # 气泡到Pt距离
        pt_indices = [i for i, s in enumerate(symbols) if s == 'Pt']
        if pt_indices:
            pt_positions = positions[pt_indices]
            pt_distances = np.linalg.norm(pt_positions - bubble_com, axis=1)
            min_pt_distance = float(np.min(pt_distances))
        else:
            min_pt_distance = 0.0
        bubble_to_pt_distance.append(min_pt_distance)
        
        # H2取向有序度
        orientation_order = calculate_h2_orientation_order_fast(bubble_positions)
        h2_orientation_order.append(orientation_order)
        
        # === 全体原子配位数计算 ===
        avg_o_coord, avg_h_coord = calculate_coordination_numbers_fast(positions, symbols, verbose=verbose)
        o_coordination_numbers.append(avg_o_coord)
        h_coordination_numbers.append(avg_h_coord)
        
        # 输出进度信息
        if verbose:
            frame_time = time.time() - frame_start_time
            print(f"    第 {frame_idx + 1} 帧完成，耗时: {frame_time:.2f}s")
            
            # 估算剩余时间
            if frame_idx > 0:
                avg_time_per_frame = (time.time() - total_start_time) / (frame_idx + 1)
                remaining_frames = len(frames) - frame_idx - 1
                estimated_remaining = avg_time_per_frame * remaining_frames
                print(f"    预计剩余时间: {estimated_remaining:.1f}s")
            print()  # 空行分隔
        else:
            # 简单进度显示
            if (frame_idx + 1) % 10 == 0:
                print(f"  进度: {frame_idx + 1}/{len(frames)} 帧", end='\r')
    
    total_time = time.time() - total_start_time
    print(f"\n所有CV计算完成！总耗时: {total_time:.2f}s")
    
    return {
        "bubble_rg": bubble_rg,
        "bubble_sphericity": bubble_sphericity,
        "bubble_surface_area": bubble_surface_area,
        "bubble_volume": bubble_volume,
        "bubble_density": bubble_density,
        "h_coord_numbers": h_coord_numbers,
        "avg_h2_distance": avg_h2_distance,
        "bubble_eccentricity": bubble_eccentricity,
        "bubble_to_pt_distance": bubble_to_pt_distance,
        "h2_orientation_order": h2_orientation_order,
        "o_coordination_numbers": o_coordination_numbers,
        "h_coordination_numbers": h_coordination_numbers,
        "energies": energies,
        "frame_indices": frame_indices
    }

def analyze_pca_components(cv_matrix, cv_names):
    """PCA主成分分析"""
    data_centered = cv_matrix - np.mean(cv_matrix, axis=0)
    data_std = np.std(cv_matrix, axis=0)
    data_std[data_std == 0] = 1
    data_scaled = data_centered / data_std
    
    pca_result, explained_variance_ratio, eigenvecs = simple_pca(data_scaled, n_components=2)
    loadings = eigenvecs[:, :2]
    
    print(f"\n🔍 PCA主成分分析:")
    print(f"PC1解释方差: {explained_variance_ratio[0]:.3f} ({explained_variance_ratio[0]*100:.1f}%)")
    print(f"PC2解释方差: {explained_variance_ratio[1]:.3f} ({explained_variance_ratio[1]*100:.1f}%)")
    
    print(f"\n📊 主成分载荷:")
    print(f"{'变量':<25} {'PC1载荷':<12} {'PC2载荷':<12}")
    print("-" * 50)
    
    pc1_interpretation = []
    pc2_interpretation = []
    
    for i, cv_name in enumerate(cv_names):
        pc1_loading = loadings[i, 0]
        pc2_loading = loadings[i, 1]
        print(f"{cv_name:<25} {pc1_loading:>8.3f}    {pc2_loading:>8.3f}")
        
        if abs(pc1_loading) > 0.3:
            pc1_interpretation.append(f"{cv_name}({pc1_loading:+.2f})")
        if abs(pc2_loading) > 0.3:
            pc2_interpretation.append(f"{cv_name}({pc2_loading:+.2f})")
    
    print(f"\n🎯 物理解释:")
    print(f"PC1: {', '.join(pc1_interpretation)}")
    print(f"PC2: {', '.join(pc2_interpretation)}")
    
    return pca_result, explained_variance_ratio, pc1_interpretation, pc2_interpretation

def save_pca_model(cv_matrix, cv_names, model_file="reference_pca_model.npz"):
    """保存PCA模型参数"""
    # 标准化（与analyze_pca_components中保持一致）
    data_centered = cv_matrix - np.mean(cv_matrix, axis=0)
    data_std = np.std(cv_matrix, axis=0)
    data_std[data_std == 0] = 1
    data_scaled = data_centered / data_std
    
    # PCA计算
    pca_result, explained_variance_ratio, eigenvecs = simple_pca(data_scaled, n_components=2)
    
    # 保存模型参数
    np.savez(model_file,
             mean=np.mean(cv_matrix, axis=0),
             std=np.std(cv_matrix, axis=0),
             eigenvectors=eigenvecs,
             explained_variance=explained_variance_ratio,
             cv_names=np.array(cv_names))  # 转为numpy数组保存
    
    print(f"\n🔧 PCA模型已保存至: {model_file}")
    print(f"📊 该模型可用于其他轨迹的一致性PCA分析")
    
    return pca_result, explained_variance_ratio

def read_type_raw(type_file, expected_atoms):
    """读取type.raw"""
    try:
        with open(type_file, 'r') as f:
            content = f.read().strip()
            if content:
                data = np.array([int(x) for x in content.split()])
                if len(data) == expected_atoms:
                    return data
    except:
        pass
    return None

def load_single_trajectory(data_dir):
    """加载单个轨迹的数据"""
    print(f"  📂 加载轨迹: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"    ❌ 目录不存在: {data_dir}")
        return None
    
    # 加载数据
    coords = np.load(os.path.join(data_dir, 'coord.npy'))
    boxes = np.load(os.path.join(data_dir, 'box.npy'))
    
    energies = None
    if os.path.exists(os.path.join(data_dir, 'energy.npy')):
        energies = np.load(os.path.join(data_dir, 'energy.npy'))
    
    n_total_frames = coords.shape[0]
    n_atoms = coords.shape[1] // 3
    
    print(f"    📊 数据: {n_total_frames} 帧, {n_atoms} 原子")
    
    # 检查气泡索引
    if max(BUBBLE_INDICES) >= n_atoms:
        print(f"    ❌ 气泡原子索引超出范围!")
        return None
    
    # 采样索引
    end_frame = min(n_total_frames, START_FRAME + MAX_FRAMES * SAMPLING_STEP)
    selected_indices = list(range(START_FRAME, end_frame, SAMPLING_STEP))
    selected_indices = [i for i in selected_indices if i < n_total_frames]
    
    print(f"    📋 选择了 {len(selected_indices)} 帧")
    
    # 处理原子类型
    symbols = ['C'] * n_atoms
    
    type_map_file = os.path.join(data_dir, 'type_map.raw')
    type_file = os.path.join(data_dir, 'type.raw')
    
    if os.path.exists(type_map_file) and os.path.exists(type_file):
        with open(type_map_file, 'r') as f:
            elements = f.read().strip().split()
            type_map = {i: element for i, element in enumerate(elements)}
        
        atom_types = read_type_raw(type_file, n_atoms)
        if atom_types is not None:
            symbols = [type_map.get(t, 'C') for t in atom_types]
    
    # 重塑数据
    coords = coords.reshape(n_total_frames, n_atoms, 3)
    boxes = boxes.reshape(n_total_frames, 3, 3)
    
    # 创建采样结构
    frames = []
    frame_indices = []
    
    for i in selected_indices:
        atoms = Atoms(symbols=symbols, positions=coords[i], 
                     cell=boxes[i], pbc=True)
        if energies is not None:
            atoms.info['energy'] = float(energies[i])
        frames.append(atoms)
        frame_indices.append(i)
    
    return {
        'frames': frames,
        'frame_indices': frame_indices,
        'trajectory_name': os.path.basename(os.path.dirname(data_dir))  # 提取轨迹名称
    }

def load_multiple_trajectories(data_dirs):
    """加载多个轨迹并合并"""
    print(f"🔄 加载 {len(data_dirs)} 条轨迹...")
    
    all_frames = []
    all_frame_indices = []
    all_trajectory_labels = []
    trajectory_info = []
    
    for traj_idx, data_dir in enumerate(data_dirs):
        traj_data = load_single_trajectory(data_dir)
        
        if traj_data is None:
            print(f"    ⚠️  跳过轨迹 {traj_idx+1}")
            continue
        
        # 合并数据
        all_frames.extend(traj_data['frames'])
        
        # 为frame_indices添加轨迹前缀，避免重复
        prefixed_indices = [f"traj{traj_idx+1}_frame{idx}" for idx in traj_data['frame_indices']]
        all_frame_indices.extend(prefixed_indices)
        
        # 添加轨迹标签
        traj_label = f"traj_{traj_idx+1}"
        all_trajectory_labels.extend([traj_label] * len(traj_data['frames']))
        
        trajectory_info.append({
            'index': traj_idx + 1,
            'name': traj_data['trajectory_name'],
            'frames_count': len(traj_data['frames']),
            'label': traj_label
        })
        
        print(f"    ✅ 轨迹 {traj_idx+1}: {len(traj_data['frames'])} 帧")
    
    print(f"🎯 合并完成: 总共 {len(all_frames)} 帧")
    for info in trajectory_info:
        print(f"    {info['label']}: {info['frames_count']} 帧")
    
    return all_frames, all_frame_indices, all_trajectory_labels, trajectory_info
# ===========================================
# 主函数
# ===========================================

def main():
    """主函数"""
    print("=== 优化版统一CV分析：气泡 + 全体配位数 ===")
    print(f"采样参数: 起始={START_FRAME}, 步长={SAMPLING_STEP}, 最大={MAX_FRAMES}")
    print(f"分析轨迹数: {len(DATA_DIRS)}")
    
    # 加载多个轨迹
    frames, frame_indices, trajectory_labels, trajectory_info = load_multiple_trajectories(DATA_DIRS)
    
    if len(frames) == 0:
        print("❌ 没有成功加载任何轨迹!")
        return
    
    # 检查原子类型分布（使用第一个frame）
    symbols = frames[0].get_chemical_symbols()
    unique_symbols = list(set(symbols))
    print(f"\n原子类型: {unique_symbols}")
    for symbol in unique_symbols:
        count = symbols.count(symbol)
        print(f"  {symbol}: {count} 个")
    
    # 检查气泡区域原子类型
    bubble_symbols = [symbols[i] for i in BUBBLE_INDICES]
    print(f"气泡区域原子类型: {set(bubble_symbols)}")
    print(f"氢原子数量: {bubble_symbols.count('H')}/{len(BUBBLE_INDICES)}")  
    
    # 使用优化版CV计算
    all_cvs = calculate_unified_cvs_optimized(frames, frame_indices)
    
    # PCA分析 (只对气泡CV进行降维)
    bubble_cv_names = [
        "bubble_rg", "bubble_sphericity", "bubble_surface_area", 
        "bubble_volume", "bubble_density", "h_coord_numbers",
        "avg_h2_distance", "bubble_eccentricity", "bubble_to_pt_distance",
        "h2_orientation_order"
    ]
    
    bubble_cv_matrix = np.array([all_cvs[name] for name in bubble_cv_names]).T
    
    print("执行气泡CV的PCA分析...")
    pca_result, explained_variance, pc1_interp, pc2_interp = analyze_pca_components(
        bubble_cv_matrix, bubble_cv_names)

    print("\n保存PCA模型用于其他轨迹...")
    model_filename = f"reference_pca_model_step{SAMPLING_STEP}.npz"
    save_pca_model(bubble_cv_matrix, bubble_cv_names, model_filename)
    
    # 构建properties
    properties = {}
    
    # 基础属性
    if all_cvs["energies"]:
        properties["energy"] = {
            "target": "structure",
            "values": all_cvs["energies"],
            "description": "Total energy",
            "units": "eV"
        }
    
    properties["frame_index"] = {
        "target": "structure",
        "values": all_cvs["frame_indices"],
        "description": "Original frame index",
        "units": "dimensionless"
    }
    
    properties["trajectory"] = {
    "target": "structure", 
    "values": trajectory_labels,
    "description": "Trajectory source",
    "units": "dimensionless"
    }
    
    # 气泡CV
    bubble_descriptions = {
        "bubble_rg": ("H2 bubble gyration radius", "Å"),
        "bubble_sphericity": ("H2 bubble sphericity", "dimensionless"),
        "bubble_surface_area": ("H2 bubble surface area", "Å²"),
        "bubble_volume": ("H2 bubble volume", "Å³"),
        "bubble_density": ("H2 bubble density", "atoms/Å³"),
        "h_coord_numbers": ("H coordination in bubble", "count"),
        "avg_h2_distance": ("Average H-H distance in bubble", "Å"),
        "bubble_eccentricity": ("H2 bubble eccentricity", "dimensionless"),
        "bubble_to_pt_distance": ("Bubble center to Pt distance", "Å"),
        "h2_orientation_order": ("H2 molecular orientation order", "dimensionless")
    }
    
    for cv_name in bubble_cv_names:
        if cv_name in all_cvs and cv_name in bubble_descriptions:
            desc, unit = bubble_descriptions[cv_name]
            properties[cv_name] = {
                "target": "structure",
                "values": all_cvs[cv_name],
                "description": desc,
                "units": unit
            }
    
    # 全体原子配位数
    properties["o_coordination"] = {
        "target": "structure",
        "values": all_cvs["o_coordination_numbers"],
        "description": "Average O coordination number",
        "units": "count"
    }
    
    properties["h_coordination"] = {
        "target": "structure",
        "values": all_cvs["h_coordination_numbers"],
        "description": "Average H coordination number",
        "units": "count"
    }
    
    # PCA结果
    properties["bubble_PCA_X"] = {
        "target": "structure",
        "values": [float(x) for x in pca_result[:, 0]],
        "description": f"Bubble PC1: {', '.join(pc1_interp[:2])} (var: {explained_variance[0]:.3f})",
        "units": "dimensionless"
    }
    
    properties["bubble_PCA_Y"] = {
        "target": "structure",
        "values": [float(x) for x in pca_result[:, 1]],
        "description": f"Bubble PC2: {', '.join(pc2_interp[:2])} (var: {explained_variance[1]:.3f})",
        "units": "dimensionless"
    }
    
    # 生成文件
    output_file = f"mul-pca-step{SAMPLING_STEP}.json.gz"
    
    chemiscope.write_input(
        path=output_file,
        frames=frames,
        properties=properties,
        meta={
            "name": "Multi-Trajectory CV Analysis: Bubble + Coordination",
            "description": f"Complete CV analysis from {len(DATA_DIRS)} trajectories: bubble properties and coordination numbers, {len(frames)} total frames"
        }
    )
    
    print(f"\n🎉 生成文件: {output_file}")
    print(f"📊 包含 {len(frames)} 帧和 {len(properties)} 种CV")
    print(f"🔢  来源: {len(DATA_DIRS)} 条轨迹，总共 {len(frames)} 帧")
    print(f"\n📈 轨迹统计:")
    for info in trajectory_info:
        print(f"  {info['label']}: {info['frames_count']} 帧")
    
    # 提取PCA数据
    bubble_pca_x = pca_result[:, 0]
    bubble_pca_y = pca_result[:, 1]
    
    # 处理能量数据
    if all_cvs["energies"] and len(all_cvs["energies"]) == len(bubble_pca_x):
        energies_data = np.array(all_cvs["energies"])
    else:
        energies_data = np.zeros(len(bubble_pca_x))
        print("⚠️  警告：能量数据不可用，将用0填充")
    
    # 组合成三列数据
    pca_data = np.column_stack([bubble_pca_x, bubble_pca_y, energies_data])
    
    # 生成输出文件名
    pca_output_file = f"mul_bubble_pca_step{SAMPLING_STEP}.dat"
    
    # 保存为dat文件
    np.savetxt(pca_output_file, pca_data, 
               header="bubble_PCA_X\tbubble_PCA_Y\tenergy", 
               fmt='%.6f', 
               delimiter='\t')
    
    print(f"💾 PCA+能量数据已保存至: {pca_output_file}")
    
    print(f"\n📋 推荐的可视化组合:")
    print(f"🫧 气泡大小演化: X=frame_index, Y=bubble_rg")
    print(f"🫧 气泡-能量关系: X=bubble_rg, Y=energy")
    print(f"🫧 气泡形状分析: X=bubble_sphericity, Y=bubble_eccentricity")
    print(f"🫧 气泡-催化剂: X=bubble_to_pt_distance, Y=energy")
    print(f"🫧 配位数分析: X=o_coordination, Y=h_coordination")
    print(f"🫧 PCA降维: X=bubble_PCA_X, Y=bubble_PCA_Y, Color=energy")
    print(f"🫧 H2结构: X=h2_orientation_order, Y=avg_h2_distance")
    print(f"\n🎯 PCA模型使用说明:")
    print(f"1. 当前轨迹的PCA模型已保存为: reference_pca_model_step{SAMPLING_STEP}.npz")
    print(f"2. 对其他轨迹使用相同PCA坐标系的方法:")
    print(f"   - 修改 BUBBLE_INDICES 为新轨迹的气泡范围")
    print(f"   - 修改 DATA_DIR 为新轨迹路径") 
    print(f"   - 运行脚本生成新的PCA结果")
    print(f"   - 所有结果将在同一坐标系中，可直接比较")
if __name__ == "__main__":
    main()