'''
Description: 
Author: Damocles_lin
Date: 2025-07-29 20:27:35
LastEditTime: 2025-07-29 22:31:10
LastEditors: Damocles_lin
'''
import sys
import os
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QLabel, QMessageBox, QGroupBox,
    QGridLayout, QFrame, QSizePolicy, QTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
from utils import setup_logger, load_colmap_data, visualize_geometry

logger = setup_logger('gui')

class VisualizationThread(QThread):
    """用于在独立线程中运行Open3D可视化的线程"""
    finished = pyqtSignal()
    error = pyqtSignal(str)
    
    def __init__(self, geometry, window_name="Open3D Viewer"):
        super().__init__()
        self.geometry = geometry
        self.window_name = window_name
        
    def run(self):
        try:
            success = visualize_geometry(self.geometry, self.window_name)
            if not success:
                self.error.emit("可视化失败")
        except Exception as e:
            self.error.emit(f"可视化错误: {str(e)}")
        finally:
            self.finished.emit()

class MainWindow(QMainWindow):
    """主窗口 - 3D重建查看器"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维重建可视化工具")
        self.setGeometry(100, 100, 800, 600)
        
        # 设置应用图标
        if hasattr(sys, '_MEIPASS'):
            icon_path = os.path.join(sys._MEIPASS, 'icon.ico')
        else:
            icon_path = 'icon.ico'
        
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setAlignment(Qt.AlignTop)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # 标题区域
        title_frame = QFrame()
        title_frame.setFrameShape(QFrame.StyledPanel)
        title_frame.setStyleSheet("background-color: #f0f0f0; border-radius: 5px;")
        title_layout = QVBoxLayout(title_frame)
        
        title_label = QLabel("三维重建可视化工具")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; padding: 10px;")
        
        subtitle_label = QLabel("加载和查看点云、网格和重建数据")
        subtitle_label.setFont(QFont("Arial", 10))
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #7f8c8d; padding-bottom: 10px;")
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        main_layout.addWidget(title_frame)
        
        # 文件操作区域
        file_group = QGroupBox("文件操作")
        file_group.setFont(QFont("Arial", 10, QFont.Bold))
        file_group.setStyleSheet("QGroupBox { border: 1px solid #bdc3c7; border-radius: 5px; margin-top: 10px; }"
                                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        
        grid_layout = QGridLayout(file_group)
        grid_layout.setSpacing(15)
        
        # 创建按钮
        self.btn_load_pcd = self.create_button("加载点云", "#3498db", "point_cloud.png")
        self.btn_load_pcd.clicked.connect(self.load_point_cloud)
        
        self.btn_load_mesh = self.create_button("加载网格", "#2ecc71", "mesh.png")
        self.btn_load_mesh.clicked.connect(self.load_mesh)
        
        self.btn_load_npz = self.create_button("加载重建数据", "#9b59b6", "reconstruction.png")
        self.btn_load_npz.clicked.connect(self.load_reconstruction_data)
        
        self.btn_help = self.create_button("帮助", "#e74c3c", "help.png")
        self.btn_help.clicked.connect(self.show_help)
        
        # 添加到网格布局
        grid_layout.addWidget(self.btn_load_pcd, 0, 0)
        grid_layout.addWidget(self.btn_load_mesh, 0, 1)
        grid_layout.addWidget(self.btn_load_npz, 1, 0)
        grid_layout.addWidget(self.btn_help, 1, 1)
        
        main_layout.addWidget(file_group)
        
        # 文件路径显示区域
        path_group = QGroupBox("当前文件")
        path_group.setFont(QFont("Arial", 10, QFont.Bold))
        path_group.setStyleSheet("QGroupBox { border: 1px solid #bdc3c7; border-radius: 5px; }"
                                "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        
        path_layout = QVBoxLayout(path_group)
        self.file_path_label = QLabel("未加载任何文件")
        self.file_path_label.setFont(QFont("Arial", 9))
        self.file_path_label.setStyleSheet("padding: 8px; background-color: #f8f9fa; border-radius: 3px;")
        self.file_path_label.setWordWrap(True)
        path_layout.addWidget(self.file_path_label)
        
        main_layout.addWidget(path_group)
        
        # 状态区域
        status_group = QGroupBox("状态信息")
        status_group.setFont(QFont("Arial", 10, QFont.Bold))
        status_group.setStyleSheet("QGroupBox { border: 1px solid #bdc3c7; border-radius: 5px; }"
                                  "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        
        status_layout = QVBoxLayout(status_group)
        self.status_display = QTextEdit()
        self.status_display.setFont(QFont("Arial", 9))
        self.status_display.setReadOnly(True)
        self.status_display.setStyleSheet("background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 3px;")
        self.status_display.setPlaceholderText("状态信息将显示在这里...")
        status_layout.addWidget(self.status_display)
        
        main_layout.addWidget(status_group)
        
        # 底部状态栏
        self.statusBar().setFont(QFont("Arial", 8))
        self.statusBar().showMessage("就绪")
        
        # 存储当前可视化线程
        self.visualization_threads = []
        
        # 设置主窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QPushButton {
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
                color: white;
            }
            QPushButton:hover {
                opacity: 0.9;
            }
            QPushButton:pressed {
                padding: 9px;
            }
        """)
    
    def create_button(self, text, color, icon_name=None):
        """创建带样式的按钮"""
        button = QPushButton(text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setMinimumHeight(70)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                font-size: 12px;
            }}
        """)
        
        # 尝试添加图标
        if icon_name:
            icon_path = None
            if hasattr(sys, '_MEIPASS'):
                icon_path = os.path.join(sys._MEIPASS, icon_name)
            else:
                icon_path = icon_name
            
            if icon_path and os.path.exists(icon_path):
                button.setIcon(QIcon(icon_path))
                button.setIconSize(QSize(24, 24))
        
        return button
    
    def update_status(self, message):
        """更新状态显示"""
        self.status_display.append(f"> {message}")
        self.statusBar().showMessage(message)
        QApplication.processEvents()
    
    def load_point_cloud(self):
        """加载点云文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开点云文件", "", "PLY文件 (*.ply);;所有文件 (*)"
        )
        if not file_path:
            return
            
        self.update_status(f"正在加载点云: {os.path.basename(file_path)}...")
        self.file_path_label.setText(file_path)
        
        try:
            pcd = o3d.io.read_point_cloud(file_path)
            if not pcd.has_points():
                self.update_status("错误: 点云文件无效")
                QMessageBox.warning(self, "错误", "无法加载点云文件或文件为空")
                return
            
            # 启动可视化线程
            window_name = f"点云查看器: {os.path.basename(file_path)}"
            thread = VisualizationThread(pcd, window_name)
            thread.finished.connect(lambda: self.on_visualization_finished(thread))
            thread.error.connect(self.handle_visualization_error)
            thread.start()
            
            self.visualization_threads.append(thread)
            self.update_status(f"已加载: {os.path.basename(file_path)}")
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载点云失败:\n{str(e)}")
            logger.error(f"加载点云失败: {str(e)}")
    
    def load_mesh(self):
        """加载网格文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开网格文件", "", "PLY文件 (*.ply);;OBJ文件 (*.obj);;所有文件 (*)"
        )
        if not file_path:
            return
            
        self.update_status(f"正在加载网格: {os.path.basename(file_path)}...")
        self.file_path_label.setText(file_path)
        
        try:
            mesh = o3d.io.read_triangle_mesh(file_path)
            if not mesh.has_vertices():
                self.update_status("错误: 网格文件无效")
                QMessageBox.warning(self, "错误", "无法加载网格文件或文件为空")
                return
            
            mesh.compute_vertex_normals()
            
            # 启动可视化线程
            window_name = f"网格查看器: {os.path.basename(file_path)}"
            thread = VisualizationThread(mesh, window_name)
            thread.finished.connect(lambda: self.on_visualization_finished(thread))
            thread.error.connect(self.handle_visualization_error)
            thread.start()
            
            self.visualization_threads.append(thread)
            self.update_status(f"已加载: {os.path.basename(file_path)}")
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载网格失败:\n{str(e)}")
            logger.error(f"加载网格失败: {str(e)}")
    
    def load_reconstruction_data(self):
        """加载重建数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开重建数据文件", "", "NPZ文件 (*.npz);;所有文件 (*)"
        )
        if not file_path:
            return
            
        self.update_status(f"正在加载重建数据: {os.path.basename(file_path)}...")
        self.file_path_label.setText(file_path)
        
        try:
            data = load_colmap_data(file_path)
            
            # 创建选择对话框
            choice = QMessageBox.question(
                self,
                "选择可视化类型",
                "请选择要可视化的内容:",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes
            )
            
            if choice == QMessageBox.Cancel:
                self.update_status("操作取消")
                return
                
            if choice == QMessageBox.Yes and 'points' in data and data['points'].size > 0:
                # 可视化点云
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data['points'])
                if 'colors' in data:
                    pcd.colors = o3d.utility.Vector3dVector(data['colors'])
                
                window_name = f"重建点云: {os.path.basename(file_path)}"
                thread = VisualizationThread(pcd, window_name)
                self.update_status(f"可视化点云: {os.path.basename(file_path)}")
            
            elif choice == QMessageBox.No and 'vertices' in data and data['vertices'] is not None:
                # 可视化网格
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(data['vertices'])
                if 'triangles' in data:
                    mesh.triangles = o3d.utility.Vector3iVector(data['triangles'])
                mesh.compute_vertex_normals()
                
                window_name = f"重建网格: {os.path.basename(file_path)}"
                thread = VisualizationThread(mesh, window_name)
                self.update_status(f"可视化网格: {os.path.basename(file_path)}")
            else:
                self.update_status("错误: 没有可用的可视化数据")
                QMessageBox.warning(self, "错误", "重建数据中没有点云或网格信息")
                return
            
            thread.finished.connect(lambda: self.on_visualization_finished(thread))
            thread.error.connect(self.handle_visualization_error)
            thread.start()
            self.visualization_threads.append(thread)
            
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载重建数据失败:\n{str(e)}")
            logger.error(f"加载重建数据失败: {str(e)}")
    
    def on_visualization_finished(self, thread):
        """可视化线程完成时的处理"""
        try:
            if thread in self.visualization_threads:
                self.visualization_threads.remove(thread)
                self.update_status("可视化窗口已关闭")
        except Exception as e:
            logger.error(f"处理可视化完成时出错: {str(e)}")
    
    def handle_visualization_error(self, message):
        """处理可视化错误"""
        self.update_status(f"错误: {message}")
        QMessageBox.critical(self, "可视化错误", message)
    
    def show_help(self):
        """显示帮助信息"""
        help_text = (
            "3D可视化工具使用说明\n\n"
            "1. 加载点云或网格文件:\n"
            "   - 点击'加载点云'按钮打开PLY格式的点云文件\n"
            "   - 点击'加载网格'按钮打开PLY或OBJ格式的网格文件\n"
            "   - 点击'加载重建数据'按钮打开COLMAP生成的NPZ文件\n\n"
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

def run_gui():
    """启动GUI应用程序"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # 设置全局样式
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(50, 50, 50))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipText, QColor(50, 50, 50))
    palette.setColor(QPalette.Text, QColor(50, 50, 50))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(50, 50, 50))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(52, 152, 219))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_gui()