'''
Description: 
Author: Damocles_lin
Date: 2025-07-28 18:39:30
LastEditTime: 2025-07-29 13:02:34
LastEditors: Damocles_lin
'''
# import numpy as np

# # 加载npz文件
# data = np.load('./reconstruction_data.npz', allow_pickle=True)

# # 查看包含的所有数组的名称
# print("包含的数组名称：", data.files)

# # 遍历所有数组并查看
# for name in data.files:
#     print("="*50)
#     print(f"\n数组'{name}'的形状：", data[name].shape)
#     print(f"数组'{name}'的内容：\n", data[name])

# # 关闭文件（可选，在上下文管理器中会自动关闭）
# data.close()
import numpy as np
import os
import json

def load_and_export_npz(npz_path, output_txt_path):
    """
    加载NPZ文件并将内容导出为可读的文本文件
    
    参数:
        npz_path: NPZ文件路径
        output_txt_path: 输出文本文件路径
    """
    try:
        # 加载NPZ文件
        data = np.load(npz_path, allow_pickle=True)
        
        # 创建输出文件
        with open(output_txt_path, 'w') as f:
            # 写入文件头
            f.write("=" * 80 + "\n")
            f.write(f"三维重建数据解析报告\n")
            f.write(f"原始文件: {npz_path}\n")
            f.write(f"生成时间: {np.datetime64('now')}\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. 点云数据
            if 'points' in data:
                points = data['points']
                f.write("=" * 80 + "\n")
                f.write("点云数据 (Point Cloud)\n")
                f.write("=" * 80 + "\n")
                f.write(f"总点数: {len(points):,}\n")
                
                # 输出点云摘要
                f.write("\n点云摘要:\n")
                f.write(f"  X范围: [{np.min(points[:, 0]):.4f}, {np.max(points[:, 0]):.4f}]\n")
                f.write(f"  Y范围: [{np.min(points[:, 1]):.4f}, {np.max(points[:, 1]):.4f}]\n")
                f.write(f"  Z范围: [{np.min(points[:, 2]):.4f}, {np.max(points[:, 2]):.4f}]\n")
                f.write(f"  中心点: [{np.mean(points[:, 0]):.4f}, {np.mean(points[:, 1]):.4f}, {np.mean(points[:, 2]):.4f}]\n")
                
                # 输出前10个点
                f.write("\n前10个点 (x, y, z):\n")
                for i, point in enumerate(points[:10]):
                    f.write(f"  Point {i+1}: [{point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f}]\n")
            
            # 2. 颜色数据
            if 'colors' in data:
                colors = data['colors']
                f.write("\n" + "=" * 80 + "\n")
                f.write("颜色数据 (Colors)\n")
                f.write("=" * 80 + "\n")
                f.write(f"颜色点数: {len(colors):,}\n")
                
                # 输出前10个颜色
                f.write("\n前10个颜色 (r, g, b):\n")
                for i, color in enumerate(colors[:10]):
                    f.write(f"  Color {i+1}: [{color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f}]\n")
            
            # 3. 网格顶点
            if 'vertices' in data:
                vertices = data['vertices']
                f.write("\n" + "=" * 80 + "\n")
                f.write("网格顶点数据 (Mesh Vertices)\n")
                f.write("=" * 80 + "\n")
                f.write(f"顶点数量: {len(vertices):,}\n")
                
                # 输出顶点摘要
                f.write("\n顶点摘要:\n")
                f.write(f"  X范围: [{np.min(vertices[:, 0]):.4f}, {np.max(vertices[:, 0]):.4f}]\n")
                f.write(f"  Y范围: [{np.min(vertices[:, 1]):.4f}, {np.max(vertices[:, 1]):.4f}]\n")
                f.write(f"  Z范围: [{np.min(vertices[:, 2]):.4f}, {np.max(vertices[:, 2]):.4f}]\n")
                
                # 输出前10个顶点
                f.write("\n前10个顶点 (x, y, z):\n")
                for i, vertex in enumerate(vertices[:10]):
                    f.write(f"  Vertex {i+1}: [{vertex[0]:.6f}, {vertex[1]:.6f}, {vertex[2]:.6f}]\n")
            
            # 4. 网格三角形
            if 'triangles' in data:
                triangles = data['triangles']
                f.write("\n" + "=" * 80 + "\n")
                f.write("网格三角形数据 (Mesh Triangles)\n")
                f.write("=" * 80 + "\n")
                f.write(f"三角形数量: {len(triangles):,}\n")
                
                # 输出前10个三角形
                f.write("\n前10个三角形 (顶点索引):\n")
                for i, triangle in enumerate(triangles[:10]):
                    f.write(f"  Triangle {i+1}: [{triangle[0]}, {triangle[1]}, {triangle[2]}]\n")
            
            # 5. 相机参数
            if 'cameras' in data:
                cameras = data['cameras'].item()
                f.write("\n" + "=" * 80 + "\n")
                f.write("相机参数 (Cameras)\n")
                f.write("=" * 80 + "\n")
                f.write(f"相机数量: {len(cameras)}\n\n")
                
                for cam_id, cam_data in cameras.items():
                    f.write(f"相机 ID: {cam_id}\n")
                    f.write(f"  模型: {cam_data['model']}\n")
                    f.write(f"  宽度: {cam_data['width']}\n")
                    f.write(f"  高度: {cam_data['height']}\n")
                    f.write(f"  参数: {cam_data['params']}\n\n")
            
            # 6. 图像参数
            if 'images' in data:
                images = data['images'].item()
                f.write("\n" + "=" * 80 + "\n")
                f.write("图像参数 (Images)\n")
                f.write("=" * 80 + "\n")
                f.write(f"图像数量: {len(images)}\n\n")
                
                for img_name, img_data in images.items():
                    f.write(f"图像名称: {img_name}\n")
                    f.write(f"  相机 ID: {img_data['camera_id']}\n")
                    
                    # 格式化外参矩阵
                    extrinsic = img_data['extrinsic']
                    f.write("  外参矩阵:\n")
                    for row in extrinsic:
                        f.write("    [")
                        f.write(", ".join([f"{val:.6f}" for val in row]))
                        f.write("]\n")
                    f.write("\n")
            
            # 添加文件结尾
            f.write("\n" + "=" * 80 + "\n")
            f.write("数据解析完成\n")
            f.write("=" * 80 + "\n")
        
        print(f"数据已成功导出到: {output_txt_path}")
        return True
    
    except Exception as e:
        print(f"加载和导出数据时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 输入文件路径
    input_npz = "reconstruction_data.npz"
    output_txt = "reconstruction_data_report.txt"
    
    # 检查文件是否存在
    if not os.path.exists(input_npz):
        print(f"错误: 输入文件不存在 {input_npz}")
        print("请确保已运行 task1_colmap.py 生成重建数据")
        exit(1)
    
    # 加载并导出数据
    if load_and_export_npz(input_npz, output_txt):
        print("操作成功完成!")
    else:
        print("操作失败，请检查错误信息")
