'''
Description: 
Author: Damocles_lin
Date: 2025-07-29 20:22:45
LastEditTime: 2025-07-31 15:21:03
LastEditors: Damocles_lin
'''
import os
import sys
import logging
import numpy as np
import open3d as o3d
from typing import Dict, Tuple, Optional, List, Any
import time  # 新增时间模块

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

def setup_logger(name: str, log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """配置并返回日志记录器，支持输出到文件"""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # 清除现有处理器避免重复
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 创建文件处理器（如果提供了日志文件路径）
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

class Timer:
    """计时器类"""
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.elapsed_time = 0.0
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.elapsed_time = end_time - self.start_time
        
    def get_elapsed(self) -> float:
        """获取经过的时间（秒）"""
        return self.elapsed_time

def create_intrinsic_matrix(camera_info: Dict[str, Any]) -> np.ndarray:
    """根据相机信息创建内参矩阵"""
    model_id = camera_info['model']
    params = camera_info['params']
    
    try:
        if model_id == 0:   # SIMPLE_PINHOLE
            focal, cx, cy = params
            return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        elif model_id == 1:    # PINHOLE
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
                fx, fy, cx, cy = params[:4]
                return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            model_name = CAMERA_MODEL_NAMES.get(model_id, f"未知模型({model_id})")
            raise ValueError(f"不支持的相机模型 '{model_name}'，参数不足")
    except (ValueError, TypeError) as e:
        logging.error(f"创建内参矩阵失败: {str(e)}")
        raise

def project_points_to_image(
    points3d: np.ndarray, 
    intrinsic: np.ndarray, 
    extrinsic: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """将3D点投影到2D图像平面"""
    try:
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
    except Exception as e:
        logging.error(f"点投影失败: {str(e)}")
        raise

def load_colmap_data(path: str) -> Dict[str, Any]:
    """加载COLMAP重建数据"""
    try:
        data = np.load(path, allow_pickle=True)
        return {
            'points': data['points'],
            'colors': data['colors'],
            'vertices': data.get('vertices', None),
            'triangles': data.get('triangles', None),
            'vertex_colors': data.get('vertex_colors', None),
            'cameras': data['cameras'].item(),
            'images': data['images'].item()
        }
    except Exception as e:
        logging.error(f"加载COLMAP数据失败: {str(e)}")
        raise

def visualize_geometry(
    geometry: o3d.geometry.Geometry, 
    window_name: str = "Open3D Viewer",
    save_path: Optional[str] = None
) -> bool:
    """可视化Open3D几何对象"""
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1200, height=900)
        
        # 统一处理单个或多个几何体
        geometries = [geometry] if not isinstance(geometry, list) else geometry
        for g in geometries:
            vis.add_geometry(g)
        
        # 配置渲染选项
        render_option = vis.get_render_option()
        if any(isinstance(g, o3d.geometry.PointCloud) for g in geometries):
            render_option.point_size = 1.5
        render_option.light_on = True
        render_option.background_color = np.array([0.1, 0.1, 0.1])
        
        # 运行可视化
        vis.run()
        
        # 保存截图
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vis.capture_screen_image(save_path)
        
        vis.destroy_window()
        return True
    except Exception as e:
        logging.error(f"可视化错误: {str(e)}")
        return False