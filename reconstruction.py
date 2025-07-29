'''
Description: 
Author: Damocles_lin
Date: 2025-07-29 20:25:16
LastEditTime: 2025-07-29 20:35:27
LastEditors: Damocles_lin
'''
import os
import sys
import subprocess
import numpy as np
import open3d as o3d
import pycolmap
from typing import List, Optional, Tuple, Dict
from utils import setup_logger, CAMERA_MODEL_NAMES

logger = setup_logger('reconstruction')

def run_colmap_command(command: List[str], description: str) -> bool:
    """运行COLMAP命令并检查结果"""
    logger.info(f"正在执行: {description}")
    logger.debug(f"命令: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.debug(f"命令输出:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} 失败，错误码: {e.returncode}")
        logger.error(f"错误输出:\n{e.stderr}")
        return False
    except Exception as e:
        logger.error(f"{description} 发生未知错误: {str(e)}")
        return False

def run_colmap_pipeline(image_dir: str, output_dir: str) -> Optional[str]:
    """执行完整的COLMAP重建流程"""
    os.makedirs(output_dir, exist_ok=True)
    database_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse")
    dense_dir = os.path.join(output_dir, "dense")
    
    # 1. 特征提取
    if not run_colmap_command([
        'colmap', 'feature_extractor',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--ImageReader.single_camera', '1'
    ], "特征提取"):
        return None
    
    # 2. 特征匹配
    if not run_colmap_command([
        'colmap', 'exhaustive_matcher',
        '--database_path', database_path
    ], "特征匹配"):
        return None
    
    # 3. 稀疏重建
    os.makedirs(sparse_dir, exist_ok=True)
    if not run_colmap_command([
        'colmap', 'mapper',
        '--database_path', database_path,
        '--image_path', image_dir,
        '--output_path', sparse_dir
    ], "稀疏重建"):
        return None
    
    # 4. 稠密重建
    os.makedirs(dense_dir, exist_ok=True)
    if not run_colmap_command([
        'colmap', 'image_undistorter',
        '--image_path', image_dir,
        '--input_path', os.path.join(sparse_dir, "0"),
        '--output_path', dense_dir,
        '--output_type', 'COLMAP'
    ], "图像去畸变"):
        return None
    
    if not run_colmap_command([
        'colmap', 'patch_match_stereo',
        '--workspace_path', dense_dir,
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true'
    ], "稠密匹配"):
        return None
    
    # 5. 生成点云和网格
    fused_path = os.path.join(dense_dir, "fused.ply")
    meshed_path = os.path.join(dense_dir, "meshed.ply")
    
    if not run_colmap_command([
        'colmap', 'stereo_fusion',
        '--workspace_path', dense_dir,
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', fused_path
    ], "点云融合"):
        return None
    
    if not run_colmap_command([
        'colmap', 'poisson_mesher',
        '--input_path', fused_path,
        '--output_path', meshed_path
    ], "网格生成"):
        return None
    
    return dense_dir

def parse_colmap_data(sparse_dir: str) -> Tuple[Dict, Dict]:
    """解析COLMAP输出的相机参数"""
    # 查找模型目录
    model_dirs = [
        d for d in os.listdir(sparse_dir) 
        if os.path.isdir(os.path.join(sparse_dir, d)) and d.isdigit()
    ]
    
    if not model_dirs:
        raise FileNotFoundError(f"在 {sparse_dir} 中未找到重建模型")
    
    # 使用最新的模型目录
    latest_model_dir = os.path.join(sparse_dir, max(model_dirs, key=int))
    
    # 使用pycolmap加载重建结果
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
    output_path: str
) -> Tuple[Optional[o3d.geometry.PointCloud], Optional[o3d.geometry.TriangleMesh]]:
    """保存重建结果数据到NPZ文件"""
    fused_path = os.path.join(dense_dir, "fused.ply")
    meshed_path = os.path.join(dense_dir, "meshed.ply")
    
    # 加载点云
    if not os.path.exists(fused_path):
        logger.error(f"点云文件不存在: {fused_path}")
        return None, None
    
    try:
        point_cloud = o3d.io.read_point_cloud(fused_path)
    except Exception as e:
        logger.error(f"加载点云失败: {str(e)}")
        point_cloud = None
    
    # 加载网格
    mesh = None
    if os.path.exists(meshed_path):
        try:
            mesh = o3d.io.read_triangle_mesh(meshed_path)
            if mesh.has_vertices():
                mesh.compute_vertex_normals()
        except Exception as e:
            logger.error(f"加载网格失败: {str(e)}")
    
    # 解析相机参数
    try:
        cameras, images = parse_colmap_data(sparse_dir)
    except Exception as e:
        logger.error(f"解析相机参数失败: {str(e)}")
        return None, None
    
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
    
    # 保存到NPZ文件
    try:
        np.savez_compressed(output_path, **save_data)
        logger.info(f"重建数据已保存到 {output_path}")
    except Exception as e:
        logger.error(f"保存重建数据失败: {str(e)}")
    
    return point_cloud, mesh

def run_reconstruction_pipeline(
    image_dir: str, 
    output_dir: str, 
    output_file: str = "reconstruction_data.npz"
) -> bool:
    """运行完整的重建流程"""
    logger.info(f"开始重建流程，输入目录: {image_dir}, 输出目录: {output_dir}")
    
    # 执行COLMAP流程
    dense_dir = run_colmap_pipeline(image_dir, output_dir)
    if not dense_dir:
        logger.error("COLMAP重建流程失败")
        return False
    
    # 检查稀疏重建目录
    sparse_dir = os.path.join(output_dir, "sparse")
    if not os.path.exists(sparse_dir):
        logger.error(f"稀疏重建目录不存在: {sparse_dir}")
        return False
    
    # 保存重建数据
    output_path = os.path.join(output_dir, output_file)
    point_cloud, mesh = save_reconstruction_data(dense_dir, sparse_dir, output_path)
    
    # 可视化结果
    if point_cloud:
        logger.info("可视化点云...")
        try:
            o3d.visualization.draw_geometries([point_cloud])
        except Exception as e:
            logger.error(f"点云可视化失败: {str(e)}")
    
    if mesh and mesh.has_vertices():
        logger.info("可视化网格...")
        try:
            o3d.visualization.draw_geometries([mesh])
        except Exception as e:
            logger.error(f"网格可视化失败: {str(e)}")
    
    return True

if __name__ == "__main__":
    try:
        success = run_reconstruction_pipeline("./images", "./output")
        if not success:
            logger.error("重建流程失败")
            sys.exit(1)
        logger.info("重建流程成功完成")
    except Exception as e:
        logger.exception("重建过程中发生未处理的错误")
        sys.exit(1)