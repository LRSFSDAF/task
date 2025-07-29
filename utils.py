'''
Description: 
Author: Damocles_lin
Date: 2025-07-29 20:22:45
LastEditTime: 2025-07-29 20:24:18
LastEditors: Damocles_lin
'''
import os
import sys
import logging
import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Optional, List, Any

# 配置日志
def setup_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    """设置并返回配置好的日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# 相机模型映射
CAMERA_MODEL_NAMES = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE",
}

def create_intrinsic_matrix(camera_info: Dict[str, Any]) -> np.ndarray:
    """
    根据相机信息创建内参矩阵
    
    参数:
        camera_info: 相机信息字典，包含'model'和'params'字段
        
    返回:
        np.array: 3x3 内参矩阵
    """
    model_id = camera_info['model']
    params = camera_info['params']
    
    if model_id == 0:   # SIMPLE_PINHOLE
        focal, cx, cy = params
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    elif model_id == 1:    # 'PINHOLE'
        fx, fy, cx, cy = params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif model_id in [2, 3]:    # RADIAL类型
        focal, cx, cy, *_ = params
        return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    elif model_id == 4:    # OPENCV
        fx, fy, cx, cy, *_ = params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        if len(params) >= 4:
            # 取前四个参数，假设为fx, fy, cx, cy
            fx, fy, cx, cy = params[:4]
            return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        # 不支持的模型返回单位矩阵
        model_name = CAMERA_MODEL_NAMES.get(model_id, f"未知模型({model_id})")
        raise ValueError(f"不支持的相机模型 '{model_name}'，参数不足")

def project_points_to_image(
    points3d: np.ndarray, 
    intrinsic: np.ndarray, 
    extrinsic: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将3D点投影到2D图像平面
    
    参数:
        points3d: 3D点坐标 (N,3)
        intrinsic: 相机内参矩阵 (3,3)
        extrinsic: 相机外参矩阵 (4,4)
        
    返回:
        points2d: 投影后的2D坐标 (N,2)
        valid: 有效点的布尔掩码
    """
    # 转换为齐次坐标
    points_homo = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
    
    # 应用外参: 世界坐标 -> 相机坐标
    camera_coords = (extrinsic @ points_homo.T).T
    
    # 过滤相机后面的点
    valid = camera_coords[:, 2] > 0
    camera_coords = camera_coords[valid]
    
    # 应用内参: 相机坐标 -> 图像坐标
    image_coords = (intrinsic @ camera_coords[:, :3].T).T
    image_coords = image_coords[:, :2] / image_coords[:, 2, None]
    
    return image_coords, valid

def load_colmap_data(path: str) -> Dict[str, Any]:
    """
    加载COLMAP重建数据
    
    参数:
        path: npz文件路径
        
    返回:
        dict: 包含点云、网格和相机参数的数据
    """
    data = np.load(path, allow_pickle=True)
    return {
        'points': data['points'],
        'colors': data['colors'],
        'vertices': data.get('vertices', None),
        'triangles': data.get('triangles', None),
        'cameras': data['cameras'].item(),
        'images': data['images'].item()
    }

def visualize_geometry(
    geometry: o3d.geometry.Geometry, 
    window_name: str = "Open3D Viewer",
    save_path: Optional[str] = None
) -> bool:
    """
    可视化Open3D几何对象
    
    参数:
        geometry: Open3D几何对象
        window_name: 窗口名称
        save_path: 可选，保存可视化结果路径
        
    返回:
        bool: 是否成功可视化
    """
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=800, height=600)
        vis.add_geometry(geometry)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        if isinstance(geometry, o3d.geometry.PointCloud):
            render_option.point_size = 2.0
        render_option.light_on = True
        
        # 运行可视化
        vis.run()
        
        # 保存截图
        if save_path:
            vis.capture_screen_image(save_path)
        
        vis.destroy_window()
        return True
    except Exception as e:
        logging.error(f"可视化错误: {str(e)}")
        return False