'''
Description: 支持点云、网格模型和相机位姿的可视化
Author: Damocles_lin
Date: 2025-07-29 20:26:58
LastEditTime: 2025-08-06 19:52:08
LastEditors: Damocles_lin
'''
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from typing import Optional, List
from utils import setup_logger, load_colmap_data, create_intrinsic_matrix, project_points_to_image, visualize_geometry

logger = setup_logger('visualization')

def create_point_cloud(points: np.ndarray, colors: np.ndarray) -> o3d.geometry.PointCloud:
    """
    从点坐标和颜色数据创建Open3D点云对象
    
    参数:
        points (np.ndarray): Nx3的点坐标数组
        colors (np.ndarray): Nx3的点颜色数组
        
    返回:
        o3d.geometry.PointCloud: 创建的点云对象
    """
    if points.size == 0 or colors.size == 0:
        raise ValueError("点云数据为空")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def create_mesh(vertices: np.ndarray, triangles: np.ndarray, vertex_colors: Optional[np.ndarray] = None) -> o3d.geometry.TriangleMesh:
    """
    从顶点、三角形和颜色数据创建Open3D网格对象
    
    参数:
        vertices (np.ndarray): Mx3的顶点坐标数组
        triangles (np.ndarray): Kx3的三角形索引数组
        vertex_colors (Optional[np.ndarray]): Mx3的顶点颜色数组
        
    返回:
        o3d.geometry.TriangleMesh: 创建的网格对象
    """
    if vertices is None or triangles is None:
        raise ValueError("网格数据无效")
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 处理顶点颜色
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        if np.max(vertex_colors) > 1.0:
            vertex_colors = vertex_colors / 255.0
        mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    else:
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
    
    mesh.compute_vertex_normals()
    return mesh

def visualize_camera_poses(extrinsics: List[np.ndarray], size: float = 0.1) -> o3d.geometry.LineSet:
    """
    创建表示相机位姿的坐标系集合
    
    参数:
        extrinsics (List[np.ndarray]): 相机外参矩阵列表
        size (float): 坐标系大小，默认为0.1
        
    返回:
        o3d.geometry.LineSet: 包含相机坐标系的线集对象
    """
    camera_poses = o3d.geometry.LineSet()
    points_all = []
    lines_all = []
    colors_all = []
    
    for idx, extrinsic in enumerate(extrinsics):
        # 计算相机在世界坐标系中的位置
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        camera_center = -R.T @ t
        
        # 坐标系点
        points = [camera_center]
        points.append(camera_center + R[:, 0] * size)  # X轴
        points.append(camera_center + R[:, 1] * size)  # Y轴
        points.append(camera_center + R[:, 2] * size)  # Z轴
        
        # 创建线
        lines = [[0,1], [0,2], [0,3]]
        colors = [[1,0,0], [0,1,0], [0,0,1]]  # 红、绿、蓝
        
        # 添加到总集合
        n = len(points_all)
        points_all.extend(points)
        lines_all.extend([[n+i, n+j] for i,j in lines])
        colors_all.extend(colors)
    
    camera_poses.points = o3d.utility.Vector3dVector(np.array(points_all))
    camera_poses.lines = o3d.utility.Vector2iVector(np.array(lines_all))
    camera_poses.colors = o3d.utility.Vector3dVector(np.array(colors_all))
    
    return camera_poses

def visualize_projection(image_path: str, points2d: np.ndarray, save_path: str) -> bool:
    """
    在2D图像上可视化投影点
    
    参数:
        image_path (str): 输入图像路径
        points2d (np.ndarray): Nx2的2D点坐标数组
        save_path (str): 结果保存路径
        
    返回:
        bool: 可视化是否成功
    """
    if not os.path.exists(image_path):
        logger.error(f"图像文件不存在: {image_path}")
        return False
    
    try:
        image = plt.imread(image_path)
        plt.figure(figsize=(16, 10))
        plt.imshow(image)
        
        # 限制显示点数
        max_points = 5000
        if points2d.shape[0] > max_points:
            indices = np.random.choice(points2d.shape[0], max_points, replace=False)
            points2d = points2d[indices]
        
        plt.scatter(points2d[:, 0], points2d[:, 1], s=2, c='red', alpha=0.7)
        plt.title("3D Point Projection")
        plt.axis('off')
        
        # 保存图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"投影结果已保存到 {save_path}")
        return True
    except Exception as e:
        logger.error(f"可视化投影失败: {str(e)}")
        return False

def run_visualization_pipeline(data_path: str, image_dir: str, output_dir: str) -> bool:
    """
    运行完整的可视化流程
    
    参数:
        data_path (str): 重建数据文件路径
        image_dir (str): 原始图像目录路径
        output_dir (str): 输出结果目录路径
        
    返回:
        bool: 可视化流程是否成功
    """
    logger.info(f"开始可视化流程，数据文件: {data_path}")
    
    # 加载重建数据
    try:
        data = load_colmap_data(data_path)
    except Exception as e:
        logger.error(f"加载重建数据失败: {str(e)}")
        return False
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 可视化点云和相机位姿
    if 'points' in data and data['points'].size > 0:
        logger.info("可视化点云...")
        try:
            pcd = create_point_cloud(data['points'], data.get('colors', np.zeros_like(data['points'])))
            
            # 获取相机位姿
            extrinsics = [img['extrinsic'] for img in data['images'].values()]
            camera_poses = visualize_camera_poses(extrinsics, size=0.1)
            
            # 可视化点云和相机
            visualize_geometry(
                [pcd, camera_poses], 
                "3D Point Cloud with Cameras", 
                os.path.join(output_dir, "point_cloud_with_cameras.png")
            )
        except Exception as e:
            logger.error(f"点云可视化失败: {str(e)}")
    
    # 可视化网格
    if 'vertices' in data and data['vertices'] is not None:
        logger.info("可视化网格...")
        try:
            mesh = create_mesh(
                data['vertices'], 
                data['triangles'], 
                data.get('vertex_colors', None)
            )
            
            visualize_geometry(
                mesh, 
                "3D Mesh", 
                os.path.join(output_dir, "mesh.png")
            )
        except Exception as e:
            logger.error(f"网格可视化失败: {str(e)}")
    
    # 投影验证
    if 'images' in data and data['images'] and 'points' in data and data['points'].size > 0:
        logger.info("执行投影验证...")
        
        # 选择一张图像
        image_name = list(data['images'].keys())[10]
        image_info = data['images'][image_name]
        camera_id = image_info['camera_id']
        
        if camera_id in data['cameras']:
            try:
                camera_info = data['cameras'][camera_id]
                intrinsic = create_intrinsic_matrix(camera_info)
                extrinsic = image_info['extrinsic']
                
                # 投影点云
                points2d, valid = project_points_to_image(
                    data['points'], 
                    intrinsic, 
                    extrinsic
                )
                
                # 可视化投影结果
                image_path = os.path.join(image_dir, image_name)
                visualize_projection(image_path, points2d, os.path.join(output_dir, "projection_result.png"))
            except Exception as e:
                logger.error(f"投影过程中发生错误: {str(e)}")
        else:
            logger.error(f"找不到相机ID {camera_id} 的信息")
    else:
        logger.warning("缺少点云或图像数据，跳过投影验证")
    
    return True

if __name__ == "__main__":

    # 数据路径
    data_path = "./results/reconstruction_data.npz"
    images_dir = "./data/images-little-prince"
    output_dir = "./results/png"
    try:
        success = run_visualization_pipeline(data_path, images_dir, output_dir)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception("可视化过程中发生未处理的错误")
        sys.exit(1)