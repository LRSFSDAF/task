import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

def load_colmap_data(path):
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
        'vertices': data['vertices'] if 'vertices' in data else None,
        'triangles': data['triangles'] if 'triangles' in data else None,
        'cameras': data['cameras'].item(),
        'images': data['images'].item()
    }

def visualize_point_cloud_3d(points, colors):
    """
    可视化3D点云
    
    参数:
        points: 点云坐标 (N,3)
        colors: 点云颜色 (N,3)
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    try:
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"无法可视化点云: {str(e)}")
        print("尝试保存点云图像...")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.capture_screen_image("point_cloud_visualization.png")
        vis.destroy_window()
        print("点云图像已保存为 'point_cloud_visualization.png'")

def visualize_mesh_3d(vertices, triangles):
    """
    可视化3D网格
    
    参数:
        vertices: 顶点坐标 (M,3)
        triangles: 三角形面片 (K,3)
    """
    if vertices is None or triangles is None:
        print("警告: 网格数据无效")
        return
        
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    
    try:
        o3d.visualization.draw_geometries([mesh])
    except Exception as e:
        print(f"无法可视化网格: {str(e)}")
        print("尝试保存网格图像...")
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.capture_screen_image("mesh_visualization.png")
        vis.destroy_window()
        print("网格图像已保存为 'mesh_visualization.png'")

def project_points_to_image(points3d, intrinsic, extrinsic):
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

def visualize_projection(image_path, points2d):
    """
    在2D图像上可视化投影点
    
    参数:
        image_path: 图像文件路径
        points2d: 2D点坐标 (N,2)
    """
    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 {image_path}")
        return
        
    try:
        image = plt.imread(image_path)
        plt.figure(figsize=(10, 8))
        plt.imshow(image)
        plt.scatter(points2d[:, 0], points2d[:, 1], s=1, c='red', alpha=0.5)
        plt.title("3D Point Projection")
        plt.axis('off')
        plt.savefig("projection_result.png", bbox_inches='tight')
        plt.close()
        print("投影结果已保存为 'projection_result.png'")
    except Exception as e:
        print(f"无法可视化投影: {str(e)}")

def create_intrinsic_matrix(camera_info):
    """
    根据相机信息创建内参矩阵
    
    参数:
        camera_info: 相机信息字典
        
    返回:
        np.array: 3x3 内参矩阵
    """
    model_id = camera_info['model']
    model_names = {
        0: "SIMPLE_PINHOLE",
        1: "PINHOLE",
        2: "SIMPLE_RADIAL",
        3: "RADIAL",
        4: "OPENCV",
        5: "OPENCV_FISHEYE"
    }
    model_name = model_names.get(model_id, f"未知模型({model_id})")
    params = camera_info['params']
    
    if model_id == 0:   # SIMPLE_PINHOLE
        f, cx, cy = params
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model_id == 1:    # 'PINHOLE'
        fx, fy, cx, cy = params
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif model_id == 2:    # 'SIMPLE_RADIAL'
        f, cx, cy, k = params
        return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    else:
        print(f"警告: 不支持的相机模型 '{model_name}'，使用单位矩阵代替")
        return np.eye(3)

# 示例使用
if __name__ == "__main__":
    try:
        # 加载重建数据
        data = load_colmap_data("reconstruction_data.npz")
        print("重建数据加载成功")
        
        # 可视化点云
        if 'points' in data and data['points'].size > 0:
            print("可视化点云...")
            visualize_point_cloud_3d(data['points'], data['colors'])
        else:
            print("警告: 没有可用的点云数据")
            
        # 可视化网格
        if 'vertices' in data and data['vertices'] is not None:
            print("可视化网格...")
            visualize_mesh_3d(data['vertices'], data['triangles'])
        else:
            print("警告: 没有可用的网格数据")
            
        # 投影验证
        if 'images' in data and len(data['images']) > 0:
            print("执行投影验证...")
            # 选择第一张图像和对应的相机参数
            image_name = list(data['images'].keys())[0]
            image_info = data['images'][image_name]
            camera_id = image_info['camera_id']
            
            if camera_id in data['cameras']:
                camera_info = data['cameras'][camera_id]
                
                # 创建内参矩阵
                intrinsic = create_intrinsic_matrix(camera_info)
                extrinsic = image_info['extrinsic']
                
                print(f"内参矩阵:\n{intrinsic}")
                print(f"外参矩阵:\n{extrinsic}")
                
                # 投影点云
                if 'points' in data and data['points'].size > 0:
                    points2d, valid = project_points_to_image(
                        data['points'], 
                        intrinsic, 
                        extrinsic
                    )
                    
                    # 可视化投影结果
                    image_path = os.path.join("images", image_name)
                    visualize_projection(image_path, points2d)
                else:
                    print("错误: 没有点云数据用于投影")
            else:
                print(f"错误: 找不到相机ID {camera_id} 的信息")
        else:
            print("警告: 没有可用的图像数据")
            
    except Exception as e:
        print(f"执行过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()