'''
Description: 
Author: Damocles_lin
Date: 2025-07-29 20:26:58
LastEditTime: 2025-07-29 20:43:55
LastEditors: Damocles_lin
'''
import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils import setup_logger, load_colmap_data, create_intrinsic_matrix, project_points_to_image, visualize_geometry

logger = setup_logger('visualization')

def visualize_point_cloud_3d(points: np.ndarray, colors: np.ndarray) -> bool:
    """
    可视化3D点云
    
    参数:
        points: 点云坐标 (N,3)
        colors: 点云颜色 (N,3)
        
    返回:
        bool: 是否成功可视化
    """
    if points.size == 0 or colors.size == 0:
        logger.warning("点云数据为空")
        return False
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    success = visualize_geometry(pcd, "3D Point Cloud", "point_cloud.png")
    if not success:
        logger.warning("点云可视化失败")
    return success

def visualize_mesh_3d(vertices: np.ndarray, triangles: np.ndarray) -> bool:
    """
    可视化3D网格
    
    参数:
        vertices: 顶点坐标 (M,3)
        triangles: 三角形面片 (K,3)
        
    返回:
        bool: 是否成功可视化
    """
    if vertices is None or triangles is None:
        logger.warning("网格数据无效")
        return False
    
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    success = visualize_geometry(mesh, "3D Mesh", "mesh.png")
    if not success:
        logger.warning("网格可视化失败")
    return success

def visualize_projection(image_path: str, points2d: np.ndarray) -> bool:
    """
    在2D图像上可视化投影点
    
    参数:
        image_path: 图像文件路径
        points2d: 2D点坐标 (N,2)
        
    返回:
        bool: 是否成功可视化
    """
    if not os.path.exists(image_path):
        logger.error(f"图像文件不存在: {image_path}")
        return False
    
    try:
        image = plt.imread(image_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        
        # 仅绘制部分点避免过度拥挤
        num_points = points2d.shape[0]
        if num_points > 5000:
            indices = np.random.choice(num_points, 5000, replace=False)
            points2d = points2d[indices]
        
        plt.scatter(points2d[:, 0], points2d[:, 1], s=1, c='red', alpha=0.5)
        plt.title("3D Point Projection")
        plt.axis('off')
        
        # 保存图像
        save_path = "projection_result.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
        
        logger.info(f"投影结果已保存到 {save_path}")
        return True
    except Exception as e:
        logger.error(f"可视化投影失败: {str(e)}")
        return False

def run_visualization_pipeline(data_path: str, image_dir: str = "images") -> bool:
    """
    运行完整的可视化流程
    
    参数:
        data_path: NPZ数据文件路径
        image_dir: 图像目录
        
    返回:
        bool: 是否成功执行
    """
    logger.info(f"开始可视化流程，数据文件: {data_path}")
    
    # 加载重建数据
    try:
        data = load_colmap_data(data_path)
    except Exception as e:
        logger.error(f"加载重建数据失败: {str(e)}")
        return False
    
    # 可视化点云
    if 'points' in data and data['points'].size > 0:
        logger.info("可视化点云...")
        visualize_point_cloud_3d(data['points'], data['colors'])
    else:
        logger.warning("没有可用的点云数据")
    
    # 可视化网格
    if 'vertices' in data and data['vertices'] is not None:
        logger.info("可视化网格...")
        visualize_mesh_3d(data['vertices'], data['triangles'])
    else:
        logger.warning("没有可用的网格数据")
    
    # 投影验证
    if 'images' in data and data['images']:
        logger.info("执行投影验证...")
        
        # 选择第一张图像
        image_name = list(data['images'].keys())[0]
        image_info = data['images'][image_name]
        camera_id = image_info['camera_id']
        
        if camera_id in data['cameras']:
            camera_info = data['cameras'][camera_id]
            
            try:
                intrinsic = create_intrinsic_matrix(camera_info)
                extrinsic = image_info['extrinsic']
                
                logger.debug(f"内参矩阵:\n{intrinsic}")
                logger.debug(f"外参矩阵:\n{extrinsic}")
                
                # 投影点云
                if 'points' in data and data['points'].size > 0:
                    points2d, valid = project_points_to_image(
                        data['points'], 
                        intrinsic, 
                        extrinsic
                    )
                    
                    # 可视化投影结果
                    image_path = os.path.join(image_dir, image_name)
                    visualize_projection(image_path, points2d)
                else:
                    logger.error("没有点云数据用于投影")
            except Exception as e:
                logger.error(f"投影过程中发生错误: {str(e)}")
        else:
            logger.error(f"找不到相机ID {camera_id} 的信息")
    else:
        logger.warning("没有可用的图像数据")
    
    return True

if __name__ == "__main__":
    try:
        success = run_visualization_pipeline("reconstruction_data.npz")
        if not success:
            logger.error("可视化流程失败")
            sys.exit(1)
        logger.info("可视化流程成功完成")
    except Exception as e:
        logger.exception("可视化过程中发生未处理的错误")
        sys.exit(1)