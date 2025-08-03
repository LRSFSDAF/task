'''
Description: 
Author: Damocles_lin
Date: 2025-07-29 20:25:16
LastEditTime: 2025-08-01 12:38:50
LastEditors: Damocles_lin
'''
import os
import sys
import subprocess
import numpy as np
import open3d as o3d
import pycolmap
import logging
from typing import Dict, Tuple, Optional, List
from utils import setup_logger, CAMERA_MODEL_NAMES, Timer
import time

# 全局logger初始化为None，将在主函数中初始化
logger = None

def run_colmap_command(command: List[str], description: str) -> Tuple[bool, float]:
    """
    运行COLMAP命令并检查结果
    
    参数:
        command (List[str]): 要执行的命令列表
        description (str): 命令描述
        
    返回:
        Tuple[bool, float]: 
            - 命令执行是否成功
            - 执行耗时（秒）
    """
    global logger
    logger.info(f"正在执行: {description}")
    logger.debug(f"命令: {' '.join(command)}")
    
    start_time = time.time()
    try:
        # 直接输出到控制台，同时捕获输出用于日志记录
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stdout和stderr
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )
        
        # 实时输出到控制台
        output_lines = []
        for line in process.stdout:
            sys.stdout.write(line)  # 实时输出到控制台
            sys.stdout.flush()
            output_lines.append(line)
        
        # 等待进程完成
        return_code = process.wait()
        elapsed = time.time() - start_time
        
        if return_code == 0:
            logger.info(f"{description} 完成，耗时: {elapsed:.2f}秒")
            # 将完整输出记录到debug日志
            full_output = ''.join(output_lines)
            logger.debug(f"命令输出:\n{full_output}")
            return True, elapsed
        else:
            logger.error(f"{description} 失败，错误码: {return_code}")
            logger.error(f"失败耗时: {elapsed:.2f}秒")
            # 将完整输出记录到error日志
            full_output = ''.join(output_lines)
            logger.error(f"错误输出:\n{full_output}")
            return False, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"{description} 发生未知错误: {str(e)}")
        return False, elapsed

def run_colmap_pipeline(image_dir: str, output_dir: str, time_log_file: str) -> Optional[str]:
    """
    执行完整的COLMAP重建流程
    
    参数:
        image_dir (str): 输入图像目录
        output_dir (str): 输出目录
        time_log_file (str): 耗时日志文件路径
        
    返回:
        Optional[str]: 成功时返回稠密重建目录，失败时返回None
    """
    global logger
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")
    
    # 记录步骤耗时
    step_times = {}
    
    # 1. 特征提取
    success, time_fe = run_colmap_command([
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--ImageReader.single_camera', '1',
    ], "特征提取")
    step_times['特征提取'] = time_fe
    if not success:
        return None
    
    # 2. 特征匹配
    success, time_fm = run_colmap_command([
        'colmap', 'exhaustive_matcher',
        '--database_path', database_path,
    ], "特征匹配")
    step_times['特征匹配'] = time_fm
    if not success:
        return None
    
    # 3. 稀疏重建
    os.makedirs(sparse_dir, exist_ok=True)
    success, time_sfm = run_colmap_command([
        'colmap', 'mapper',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--output_path', sparse_dir,
    ], "稀疏重建")
    step_times['稀疏重建'] = time_sfm
    if not success:
        return None
    
    # 4. 稠密重建
    os.makedirs(dense_dir, exist_ok=True)
    success, time_undistort = run_colmap_command([
        'colmap', 'image_undistorter',
        '--image_path', image_dir,
        '--input_path', os.path.join(sparse_dir, "0"),
        '--output_path', dense_dir,
        '--output_type', 'COLMAP'
    ], "图像去畸变")
    step_times['图像去畸变'] = time_undistort
    if not success:
        return None
    
    success, time_patch = run_colmap_command([
        'colmap', 'patch_match_stereo',
        '--workspace_path', dense_dir,
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true',
    ], "稠密匹配")
    step_times['稠密匹配'] = time_patch
    if not success:
        return None
    
    # 5. 生成点云和网格
    fused_path = os.path.join(dense_dir, "fused.ply")
    meshed_path = os.path.join(dense_dir, "meshed.ply")
    
    success, time_fusion = run_colmap_command([
        'colmap', 'stereo_fusion',
        '--workspace_path', dense_dir,
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', fused_path
    ], "点云融合")
    step_times['点云融合'] = time_fusion
    if not success:
        return None
    
    success, time_mesh = run_colmap_command([
        'colmap', 'poisson_mesher',
        '--input_path', fused_path,
        '--output_path', meshed_path
    ], "网格生成")
    step_times['网格生成'] = time_mesh
    if not success:
        return None
    
    # 记录总耗时
    total_time = sum(step_times.values())
    step_times['总耗时'] = total_time
    
    # 保存耗时记录到文件
    os.makedirs(os.path.dirname(time_log_file), exist_ok=True)
    with open(time_log_file, 'w') as f:
        f.write(f"COLMAP重建流程耗时统计 ({time.strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write("=" * 50 + "\n")
        for step, t in step_times.items():
            f.write(f"{step}: {t:.2f}秒\n")
        f.write("=" * 50 + "\n")
        f.write(f"总耗时: {total_time:.2f}秒\n")
    
    logger.info(f"重建流程耗时统计已保存到: {time_log_file}")
    return dense_dir

def parse_colmap_data(sparse_dir: str) -> Tuple[Dict, Dict]:
    """
    解析COLMAP输出的相机参数
    
    参数:
        sparse_dir (str): 稀疏重建目录路径
        
    返回:
        Tuple[Dict, Dict]: 
            - 相机参数字典
            - 图像位姿字典
    """
    global logger
    # 查找模型目录
    model_dirs = [
        d for d in os.listdir(sparse_dir) 
        if os.path.isdir(os.path.join(sparse_dir, d)) and d.isdigit()
    ]
    
    if not model_dirs:
        raise FileNotFoundError(f"在 {sparse_dir} 中未找到重建模型")
    
    # 使用最新的模型目录
    latest_model_dir = os.path.join(sparse_dir, max(model_dirs, key=int))
    
    try:
        reconstruction = pycolmap.Reconstruction(latest_model_dir)
    except Exception as e:
        logger.error(f"加载重建模型失败: {str(e)}")
        raise
    
    # 解析相机参数
    cameras = {}
    for camera_id, camera in reconstruction.cameras.items():
        cameras[camera_id] = {
            'model': int(camera.model),
            'width': camera.width,
            'height': camera.height,
            'params': camera.params,
        }
    
    # 解析图像位姿
    images = {}
    for image_id, image in reconstruction.images.items():
        cam_from_world = image.cam_from_world()
        
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_from_world.rotation.matrix()
        extrinsic[:3, 3] = cam_from_world.translation
        
        images[image.name] = {
            'camera_id': image.camera_id,
            'extrinsic': extrinsic
        }
    
    return cameras, images

def save_reconstruction_data(
    dense_dir: str, 
    sparse_dir: str, 
    results_path: str
) -> bool:
    """
    保存重建结果数据到NPZ文件
    
    参数:
        dense_dir (str): 稠密重建目录路径
        sparse_dir (str): 稀疏重建目录路径
        results_path (str): 结果文件保存路径
        
    返回:
        bool: 保存是否成功
    """
    global logger
    try:
        # 加载点云
        fused_path = os.path.join(dense_dir, "fused.ply")
        if os.path.exists(fused_path):
            point_cloud = o3d.io.read_point_cloud(fused_path)
        else:
            logger.warning(f"点云文件不存在: {fused_path}")
            point_cloud = None            
        
        # 加载网格
        meshed_path = os.path.join(dense_dir, "meshed.ply")
        mesh = None
        if os.path.exists(meshed_path):
            try:
                mesh = o3d.io.read_triangle_mesh(meshed_path)
                if mesh.has_vertices():
                    mesh.compute_vertex_normals()
            except Exception as e:
                logger.warning(f"加载网格失败: {str(e)}")
        
        # 解析相机参数
        try:
            cameras, images = parse_colmap_data(sparse_dir)
        except Exception as e:
            logger.error(f"解析相机参数失败: {str(e)}")
            cameras, images = {}, {}
        
        # 准备保存数据
        save_data = {
            'cameras': cameras,
            'images': images
        }
        
        if point_cloud and point_cloud.has_points():
            save_data['points'] = np.asarray(point_cloud.points)
            if point_cloud.has_colors():
                save_data['colors'] = np.asarray(point_cloud.colors)

        if mesh and mesh.has_vertices():
            save_data['vertices'] = np.asarray(mesh.vertices)
            if mesh.has_triangles():
                save_data['triangles'] = np.asarray(mesh.triangles)
            if mesh.has_vertex_colors():
                save_data['vertex_colors'] = np.asarray(mesh.vertex_colors)

        # 保存到NPZ文件
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        np.savez_compressed(results_path, **save_data)
        logger.info(f"重建数据已保存到 {results_path}")
        return True
    except Exception as e:
        logger.error(f"保存重建数据失败: {str(e)}")
        return False

def run_reconstruction_pipeline(
    image_dir: str, 
    output_dir: str, 
    results_dir: str,
) -> bool:
    """
    运行完整的重建流程
    
    参数:
        image_dir (str): 输入图像目录
        output_dir (str): 输出目录
        results_dir (str): 结果保存目录
        
    返回:
        bool: 重建流程是否成功
    """
    global logger
    logger.info(f"开始重建流程，输入目录: {image_dir}, 输出目录: {output_dir}, 结果目录：{results_dir}")
    
    # 创建时间日志文件路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    time_log_dir = os.path.join(results_dir, "log")
    time_log_file = os.path.join(time_log_dir, f"reconstruction_times_{timestamp}.txt")
    
    # 执行COLMAP流程
    dense_dir = run_colmap_pipeline(image_dir, output_dir, time_log_file)
    if not dense_dir:
        logger.error("COLMAP重建流程失败")
        return False
    
    # 检查稀疏重建目录
    sparse_dir = os.path.join(output_dir, "sparse")
    if not os.path.exists(sparse_dir):
        logger.error(f"稀疏重建目录不存在: {sparse_dir}")
        return False
    
    # 保存重建数据
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "reconstruction_data.npz")
    if not save_reconstruction_data(dense_dir, sparse_dir, results_path):
        return False
    
    logger.info("重建流程成功完成")
    return True

if __name__ == "__main__":

    # 设置文件路径
    input_dir = "./data/images-little-prince"
    output_dir = "./output"
    results_dir = "./results"

    try:
        # 初始化带时间戳的日志文件
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(results_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"reconstruction_log_{timestamp}.txt")
        
        # 配置全局logger
        logger = setup_logger('reconstruction', log_file=log_file)
        
        logger.info(f"日志文件已创建: {log_file}")
        success = run_reconstruction_pipeline(input_dir, output_dir, results_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception("重建过程中发生未处理的错误")
        sys.exit(1)