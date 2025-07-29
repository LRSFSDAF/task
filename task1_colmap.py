import os
import subprocess
import numpy as np
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import pycolmap

def run_colmap_pipeline(image_dir, output_dir):
    """执行COLMAP重建流程"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 特征提取
    subprocess.run([
        'colmap', 'feature_extractor',
        '--database_path', f'{output_dir}/database.db',
        '--image_path', image_dir,
        '--ImageReader.single_camera', '1'
    ])
    
    # 2. 特征匹配
    subprocess.run([
        'colmap', 'exhaustive_matcher',
        '--database_path', f'{output_dir}/database.db'
    ])
    
    # 3. 稀疏重建
    sparse_dir = f'{output_dir}/sparse'
    os.makedirs(sparse_dir, exist_ok=True)
    subprocess.run([
        'colmap', 'mapper',
        '--database_path', f'{output_dir}/database.db',
        '--image_path', image_dir,
        '--output_path', sparse_dir
    ])
    
    # 4. 稠密重建
    dense_dir = f'{output_dir}/dense'
    os.makedirs(dense_dir, exist_ok=True)
    subprocess.run([
        'colmap', 'image_undistorter',
        '--image_path', image_dir,
        '--input_path', f'{sparse_dir}/0',
        '--output_path', dense_dir,
        '--output_type', 'COLMAP'
    ])
    subprocess.run([
        'colmap', 'patch_match_stereo',
        '--workspace_path', dense_dir,
        '--workspace_format', 'COLMAP',
        '--PatchMatchStereo.geom_consistency', 'true'
    ])
    
    # 5. 生成点云和网格
    subprocess.run([
        'colmap', 'stereo_fusion',
        '--workspace_path', dense_dir,
        '--workspace_format', 'COLMAP',
        '--input_type', 'geometric',
        '--output_path', f'{dense_dir}/fused.ply'
    ])
    subprocess.run([
        'colmap', 'poisson_mesher',
        '--input_path', f'{dense_dir}/fused.ply',
        '--output_path', f'{dense_dir}/meshed.ply'
    ])
    
    return dense_dir

def parse_colmap_data(sparse_dir):
    """解析COLMAP输出的相机参数 - 使用pycolmap处理二进制格式"""
    # 查找模型目录（通常是0, 1等数字命名的子目录）
    model_dirs = [d for d in os.listdir(sparse_dir) 
                 if os.path.isdir(os.path.join(sparse_dir, d)) and d.isdigit()]
    
    if not model_dirs:
        raise FileNotFoundError(f"No reconstruction model found in {sparse_dir}")
    
    # 使用最新的模型目录
    latest_model_dir = os.path.join(sparse_dir, max(model_dirs, key=int))
    
    # 使用pycolmap加载重建结果
    reconstruction = pycolmap.Reconstruction(latest_model_dir)  
    
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
        # 获取相机到世界坐标系的变换矩阵
        cam_from_world = image.cam_from_world() # 4x4矩阵

        # 构建外参矩阵 [R|t]
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = cam_from_world.rotation.matrix()
        extrinsic[:3, 3] = cam_from_world.translation
        
        images[image.name] = {
            'camera_id': image.camera_id,
            'extrinsic': extrinsic
        }
    
    return cameras, images

def save_reconstruction_data(dense_dir, sparse_dir, output_path):
    """保存重建结果数据"""
    # 加载点云和网格
    fused_path = f'{dense_dir}/fused.ply'
    meshed_path = f'{dense_dir}/meshed.ply'
    
    if not os.path.exists(fused_path):
        raise FileNotFoundError(f"稠密点云文件不存在: {fused_path}")
    
    point_cloud = o3d.io.read_point_cloud(fused_path)
    
    mesh = None
    if os.path.exists(meshed_path):
        mesh = o3d.io.read_triangle_mesh(meshed_path)
    
    # 解析相机参数
    cameras, images = parse_colmap_data(sparse_dir)
    
    # 保存为npz文件
    save_data = {
        'points': np.asarray(point_cloud.points) if point_cloud.has_points() else np.array([]),
        'colors': np.asarray(point_cloud.colors) if point_cloud.has_colors() else np.array([]),
        'cameras': cameras,
        'images': images
    }
    
    if mesh:
        if mesh.has_vertices():
            save_data['vertices'] = np.asarray(mesh.vertices)
        if mesh.has_triangles():
            save_data['triangles'] = np.asarray(mesh.triangles)
    
    np.savez_compressed(output_path, **save_data)
    
    return point_cloud, mesh

# 示例使用
if __name__ == "__main__":
    try:
        # 假设图像存储在 images/ 目录下
        dense_dir = run_colmap_pipeline("./images", "./output")
        
        # 检查输出目录是否存在
        if not os.path.exists("./output/sparse"):
            raise FileNotFoundError("稀疏重建目录不存在，请检查COLMAP是否运行成功")
        
        point_cloud, mesh = save_reconstruction_data(
            dense_dir, "./output/sparse", "./reconstruction_data.npz"
        )
        
        # 可视化重建结果
        if point_cloud and point_cloud.has_points():
            o3d.visualization.draw_geometries([point_cloud])
        else:
            print("警告：点云数据为空或无效")
            
        if mesh and mesh.has_vertices():
            o3d.visualization.draw_geometries([mesh])
        else:
            print("警告：网格数据为空或无效")
            
    except Exception as e:
        print(f"重建过程中发生错误: {str(e)}")
        # 打印详细的错误信息
        import traceback
        traceback.print_exc()