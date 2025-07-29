import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QFileDialog, QLabel, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import open3d as o3d
import threading

class VisualizationThread(QThread):
    """用于在独立线程中运行Open3D可视化的线程"""
    finished = pyqtSignal()
    
    def __init__(self, geometry, window_name="Open3D Viewer"):
        super().__init__()
        self.geometry = geometry
        self.window_name = window_name
        self.vis = o3d.visualization.Visualizer()
        
    def run(self):
        try:
            # 使用正确的参数创建窗口
            self.vis.create_window(window_name=self.window_name, width=800, height=600)
            self.vis.add_geometry(self.geometry)
            
            # 设置渲染选项
            render_option = self.vis.get_render_option()
            render_option.point_size = 2.0
            render_option.light_on = True
            
            # 运行可视化循环
            self.vis.run()
            self.vis.destroy_window()
        except Exception as e:
            print(f"可视化错误: {str(e)}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    """主窗口 - 修复版"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D重建查看器")
        self.setGeometry(100, 100, 800, 600)
        
        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignTop)
        
        # 标题
        title_label = QLabel("三维重建可视化工具")
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        self.btn_load_pcd = QPushButton("加载点云")
        self.btn_load_pcd.setMinimumHeight(50)
        self.btn_load_pcd.clicked.connect(self.load_point_cloud)
        button_layout.addWidget(self.btn_load_pcd)
        
        self.btn_load_mesh = QPushButton("加载网格")
        self.btn_load_mesh.setMinimumHeight(50)
        self.btn_load_mesh.clicked.connect(self.load_mesh)
        button_layout.addWidget(self.btn_load_mesh)
        
        self.btn_help = QPushButton("帮助")
        self.btn_help.setMinimumHeight(50)
        self.btn_help.clicked.connect(self.show_help)
        button_layout.addWidget(self.btn_help)
        
        main_layout.addLayout(button_layout)
        
        # 状态区域
        self.status_label = QLabel("就绪: 请加载点云或网格文件")
        self.status_label.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.status_label)
        
        # 存储当前可视化线程
        self.visualization_threads = []
    
    def load_point_cloud(self):
        """加载点云文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "打开点云文件", "", "PLY文件 (*.ply);;所有文件 (*)"
            )
            if file_path:
                self.status_label.setText(f"正在加载点云: {os.path.basename(file_path)}...")
                QApplication.processEvents()  # 更新UI
                
                pcd = o3d.io.read_point_cloud(file_path)
                if not pcd.has_points():
                    self.status_label.setText("错误: 无法加载点云")
                    return
                
                # 启动可视化线程
                window_name = f"点云查看器: {os.path.basename(file_path)}"
                thread = VisualizationThread(pcd, window_name)
                thread.finished.connect(lambda: self.on_visualization_finished(thread))
                thread.start()
                
                self.visualization_threads.append(thread)
                self.status_label.setText(f"已加载: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_label.setText(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载点云失败:\n{str(e)}")
    
    def load_mesh(self):
        """加载网格文件"""
        try:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "打开网格文件", "", "PLY文件 (*.ply);;OBJ文件 (*.obj);;所有文件 (*)"
            )
            if file_path:
                self.status_label.setText(f"正在加载网格: {os.path.basename(file_path)}...")
                QApplication.processEvents()  # 更新UI
                
                mesh = o3d.io.read_triangle_mesh(file_path)
                if not mesh.has_vertices():
                    self.status_label.setText("错误: 无法加载网格")
                    return
                
                mesh.compute_vertex_normals()
                
                # 启动可视化线程
                window_name = f"网格查看器: {os.path.basename(file_path)}"
                thread = VisualizationThread(mesh, window_name)
                thread.finished.connect(lambda: self.on_visualization_finished(thread))
                thread.start()
                
                self.visualization_threads.append(thread)
                self.status_label.setText(f"已加载: {os.path.basename(file_path)}")
        except Exception as e:
            self.status_label.setText(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载网格失败:\n{str(e)}")
    
    def on_visualization_finished(self, thread):
        """可视化线程完成时的处理"""
        try:
            if thread in self.visualization_threads:
                self.visualization_threads.remove(thread)
                self.status_label.setText("可视化窗口已关闭")
        except:
            pass
    
    def show_help(self):
        """显示帮助信息"""
        help_text = (
            "3D可视化工具使用说明\n\n"
            "1. 加载点云或网格文件:\n"
            "   - 点击'加载点云'按钮打开PLY格式的点云文件\n"
            "   - 点击'加载网格'按钮打开PLY或OBJ格式的网格文件\n\n"
            "2. 3D窗口交互操作:\n"
            "   - 鼠标左键拖动: 旋转视图\n"
            "   - 鼠标右键拖动: 平移视图\n"
            "   - 鼠标滚轮: 缩放视图\n"
            "   - K键: 锁定/解锁视角\n"
            "   - R键: 重置视图\n\n"
            "3. 关闭窗口:\n"
            "   - 直接关闭3D窗口即可\n\n"
            "注意: 每个文件会在单独的窗口中打开"
        )
        QMessageBox.information(self, "帮助", help_text)
    
    def closeEvent(self, event):
        """关闭主窗口时清理所有资源"""
        # 尝试停止所有可视化线程
        for thread in self.visualization_threads:
            try:
                if thread.isRunning():
                    thread.terminate()
            except:
                pass
        
        event.accept()

# 应用入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())