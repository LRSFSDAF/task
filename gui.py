'''
Description: 3D可视化工具 - 使用PyOpenGL嵌入渲染
Author: Damocles_lin
Date: 2025-07-29 20:27:35
LastEditTime: 2025-08-01 01:51:11
LastEditors: Damocles_lin
'''
import sys
import os
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QFileDialog, QLabel, QMessageBox, QGroupBox,
    QGridLayout, QFrame, QSizePolicy, QTextEdit, QSplitter, 
    QOpenGLWidget
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import (
    QIcon, QFont, QPalette, QColor, QOpenGLVersionProfile,
    QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLShaderProgram, QOpenGLShader
)
from utils import setup_logger, load_colmap_data, create_intrinsic_matrix, project_points_to_image
import logging
from OpenGL import GL as gl
from OpenGL import GLU as glu

logger = setup_logger('gui')

class OpenGLRenderer(QOpenGLWidget):
    """使用PyOpenGL渲染3D场景的Widget"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)
        
        # 场景数据
        self.point_cloud = None
        self.mesh = None
        self.camera_poses = None
        
        # 相机参数
        self.camera_distance = 5.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.camera_translation = [0.0, 0.0, 0.0]
        
        # 鼠标交互
        self.last_mouse_pos = None
        self.rotation_sensitivity = 0.5
        self.translation_sensitivity = 0.01
        self.zoom_sensitivity = 0.1
        
        # OpenGL对象
        self.shader_program = None
        self.vao = None
        self.vbo = None

    def initializeGL(self):
        """初始化OpenGL上下文"""
        # 设置基本OpenGL状态
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glPointSize(2.0)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        
        # 创建着色器程序
        self.create_shaders()
        
        # 创建顶点数组对象
        self.vao = QOpenGLVertexArrayObject()
        if self.vao.create():
            self.vao.bind()
        else:
            logger.error("无法创建顶点数组对象")
        
        # 初始化模型视图矩阵
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glTranslatef(0, 0, -5)  # 初始相机位置

    def create_shaders(self):
        """创建着色器程序"""
        # 顶点着色器
        vertex_shader = """
            #version 330 core
            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            out vec3 fragColor;
            void main() {
                gl_Position = projection * view * model * vec4(position, 1.0);
                fragColor = color;
            }
        """
        
        # 片段着色器
        fragment_shader = """
            #version 330 core
            in vec3 fragColor;
            out vec4 FragColor;
            void main() {
                FragColor = vec4(fragColor, 1.0);
            }
        """
        
        # 编译着色器
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader)
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader)
        self.shader_program.link()
        
        if not self.shader_program.isLinked():
            logger.error("着色器链接失败: " + self.shader_program.log())
            return False
        return True

    def resizeGL(self, w, h):
        """调整窗口大小"""
        gl.glViewport(0, 0, w, h)
        aspect = w / h if h > 0 else 1.0
        
        # 设置投影矩阵
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        glu.gluPerspective(45, aspect, 0.1, 100.0)
        
        # 切换回模型视图矩阵
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

    def paintGL(self):
        """渲染场景"""
        # 清除缓冲区
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        
        # 设置模型视图矩阵
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        
        # 应用相机变换
        gl.glTranslatef(*self.camera_translation)
        gl.glTranslatef(0, 0, -self.camera_distance)
        gl.glRotatef(self.camera_rotation_x, 1, 0, 0)
        gl.glRotatef(self.camera_rotation_y, 0, 1, 0)
        
        # 绘制坐标轴
        self.draw_axes()
        
        # 绘制点云
        if self.point_cloud:
            self.draw_point_cloud()
        
        # 绘制网格
        if self.mesh:
            self.draw_mesh()
        
        # 绘制相机位姿
        if self.camera_poses:
            self.draw_camera_poses()

    def draw_axes(self):
        """绘制坐标轴"""
        gl.glBegin(gl.GL_LINES)
        # x轴 - 红色
        gl.glColor3f(1, 0, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(1, 0, 0)
        
        # y轴 - 绿色
        gl.glColor3f(0, 1, 0)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 1, 0)
        
        # z轴 - 蓝色
        gl.glColor3f(0, 0, 1)
        gl.glVertex3f(0, 0, 0)
        gl.glVertex3f(0, 0, 1)
        gl.glEnd()

    def draw_point_cloud(self):
        """绘制点云"""
        if 'points' not in self.point_cloud or 'colors' not in self.point_cloud:
            return
            
        points = self.point_cloud['points']
        colors = self.point_cloud['colors']
        
        gl.glBegin(gl.GL_POINTS)
        for i in range(len(points)):
            gl.glColor3f(colors[i][0], colors[i][1], colors[i][2])
            gl.glVertex3f(points[i][0], points[i][1], points[i][2])
        gl.glEnd()

    def draw_mesh(self):
        """绘制网格"""
        if 'vertices' not in self.mesh or 'triangles' not in self.mesh:
            return
            
        vertices = self.mesh['vertices']
        triangles = self.mesh['triangles']
        colors = self.mesh.get('colors', np.ones_like(vertices) * 0.7)
        
        # 绘制网格面
        gl.glBegin(gl.GL_TRIANGLES)
        for tri in triangles:
            for idx in tri:
                if idx < len(colors):  # 确保索引有效
                    gl.glColor3f(colors[idx][0], colors[idx][1], colors[idx][2])
                else:
                    gl.glColor3f(0.7, 0.7, 0.7)  # 默认颜色
                if idx < len(vertices):  # 确保索引有效
                    gl.glVertex3f(vertices[idx][0], vertices[idx][1], vertices[idx][2])
        gl.glEnd()
        
        # 绘制网格线 - 使用线框模式
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        gl.glLineWidth(1.0)
        gl.glColor3f(0.2, 0.2, 0.2)  # 线框颜色
        gl.glBegin(gl.GL_TRIANGLES)
        for tri in triangles:
            for idx in tri:
                if idx < len(vertices):
                    gl.glVertex3f(vertices[idx][0], vertices[idx][1], vertices[idx][2])
        gl.glEnd()
        gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def draw_camera_poses(self):
        """绘制相机位姿"""
        for extrinsic in self.camera_poses:
            # 计算相机在世界坐标系中的位置
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            camera_center = -R.T @ t
            
            # 坐标系大小
            size = 0.1
            
            # 计算坐标系方向
            x_dir = R[:, 0] * size
            y_dir = R[:, 1] * size
            z_dir = R[:, 2] * size
            
            # 绘制相机坐标系
            gl.glBegin(gl.GL_LINES)
            # x轴 - 红色
            gl.glColor3f(1, 0, 0)
            gl.glVertex3f(*camera_center)
            gl.glVertex3f(*(camera_center + x_dir))
            
            # y轴 - 绿色
            gl.glColor3f(0, 1, 0)
            gl.glVertex3f(*camera_center)
            gl.glVertex3f(*(camera_center + y_dir))
            
            # z轴 - 蓝色
            gl.glColor3f(0, 0, 1)
            gl.glVertex3f(*camera_center)
            gl.glVertex3f(*(camera_center + z_dir))
            gl.glEnd()

    def set_point_cloud(self, points, colors):
        """设置点云数据"""
        # 计算点云中心并调整位置
        if len(points) > 0:
            center = np.mean(points, axis=0)
            # 将点云移动到原点附近
            points = points - center
        else:
            center = np.array([0, 0, 0])
        
        self.point_cloud = {
            'points': points,
            'colors': colors,
            'center': center
        }
        self.update()

    def set_mesh(self, vertices, triangles, colors=None):
        """设置网格数据"""
        if colors is None:
            colors = np.ones_like(vertices) * 0.7
            
        # 计算网格中心并调整位置
        if len(vertices) > 0:
            center = np.mean(vertices, axis=0)
            # 将网格移动到原点附近
            vertices = vertices - center
        else:
            center = np.array([0, 0, 0])
        
        self.mesh = {
            'vertices': vertices,
            'triangles': triangles,
            'colors': colors,
            'center': center
        }
        self.update()

    def set_camera_poses(self, extrinsics):
        """设置相机位姿"""
        self.camera_poses = extrinsics
        self.update()

    def reset_view(self):
        """重置视图"""
        self.camera_distance = 5.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.camera_translation = [0.0, 0.0, 0.0]
        self.update()

    def mousePressEvent(self, event):
        """鼠标按下事件"""
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        """鼠标移动事件"""
        if self.last_mouse_pos is None:
            return
            
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        
        if event.buttons() & Qt.LeftButton:
            # 旋转
            self.camera_rotation_x += dy * self.rotation_sensitivity
            self.camera_rotation_y += dx * self.rotation_sensitivity
        elif event.buttons() & Qt.RightButton:
            # 平移
            self.camera_translation[0] += dx * self.translation_sensitivity
            self.camera_translation[1] -= dy * self.translation_sensitivity
        
        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
        """鼠标滚轮事件 - 缩放"""
        delta = event.angleDelta().y()
        self.camera_distance += delta * self.zoom_sensitivity * -0.1
        self.camera_distance = max(0.1, min(self.camera_distance, 50.0))
        self.update()

class MainWindow(QMainWindow):
    """主窗口 - 3D重建查看器"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三维重建可视化工具")
        self.setGeometry(100, 100, 1200, 800)
        
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
        
        # 创建OpenGL渲染器
        self.gl_widget = OpenGLRenderer()
        
        # 创建控制按钮
        control_frame = QFrame()
        control_layout = QHBoxLayout(control_frame)
        
        self.btn_reset_view = QPushButton("重置视图")
        self.btn_reset_view.setFixedHeight(40)
        self.btn_reset_view.clicked.connect(self.gl_widget.reset_view)
        
        control_layout.addWidget(self.btn_reset_view)
        control_layout.addStretch()
        
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
        
        # 创建分割器
        splitter = QSplitter(Qt.Vertical)
        
        # 创建上部分组件
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.addWidget(file_group)
        top_layout.addWidget(path_group)
        top_layout.addWidget(status_group)
        
        # 添加组件到分割器
        splitter.addWidget(top_widget)
        splitter.addWidget(self.gl_widget)
        splitter.setSizes([300, 600])
        
        # 添加到主布局
        main_layout.addWidget(control_frame)
        main_layout.addWidget(splitter)
        
        # 底部状态栏
        self.statusBar().setFont(QFont("Arial", 8))
        self.statusBar().showMessage("就绪")
        
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
            
            # 获取点云数据
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(points) * 0.7
            
            # 设置到OpenGL渲染器
            self.gl_widget.set_point_cloud(points, colors)
            self.gl_widget.reset_view()
            
            self.update_status(f"已加载: {os.path.basename(file_path)}")
            self.update_status(f"点数: {len(points):,}")
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
            
            # 获取网格数据
            vertices = np.asarray(mesh.vertices)
            triangles = np.asarray(mesh.triangles)
            colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else np.ones_like(vertices) * 0.7
            
            # 设置到OpenGL渲染器
            self.gl_widget.set_mesh(vertices, triangles, colors)
            self.gl_widget.reset_view()
            
            self.update_status(f"已加载: {os.path.basename(file_path)}")
            self.update_status(f"顶点数: {len(vertices):,}, 面片数: {len(triangles):,}")
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
                points = data['points']
                colors = data.get('colors', np.ones_like(points) * 0.7)
                
                # 获取相机位姿
                extrinsics = [img['extrinsic'] for img in data['images'].values()]
                
                # 设置到OpenGL渲染器
                self.gl_widget.set_point_cloud(points, colors)
                self.gl_widget.set_camera_poses(extrinsics)
                self.gl_widget.reset_view()
                
                self.update_status(f"可视化点云: {os.path.basename(file_path)}")
                self.update_status(f"点数: {len(points):,}, 相机数: {len(extrinsics)}")
            
            elif choice == QMessageBox.No and 'vertices' in data and data['vertices'] is not None:
                # 可视化网格
                vertices = data['vertices']
                triangles = data['triangles']
                colors = data.get('vertex_colors', np.ones_like(vertices) * 0.7)
                
                # 设置到OpenGL渲染器
                self.gl_widget.set_mesh(vertices, triangles, colors)
                self.gl_widget.reset_view()
                
                self.update_status(f"可视化网格: {os.path.basename(file_path)}")
                self.update_status(f"顶点数: {len(vertices):,}, 面片数: {len(triangles):,}")
            else:
                self.update_status("错误: 没有可用的可视化数据")
                QMessageBox.warning(self, "错误", "重建数据中没有点云或网格信息")
                return
            
        except Exception as e:
            self.update_status(f"错误: {str(e)}")
            QMessageBox.critical(self, "错误", f"加载重建数据失败:\n{str(e)}")
            logger.error(f"加载重建数据失败: {str(e)}")
    
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
            "   - 点击'重置视图'按钮恢复初始视角\n\n"
            "3. 相机位姿显示:\n"
            "   - 加载重建数据时选择点云选项，会同时显示相机位姿\n"
            "   - 红色轴: X轴, 绿色轴: Y轴, 蓝色轴: Z轴\n\n"
            "4. 状态信息:\n"
            "   - 底部状态栏显示当前操作状态\n"
            "   - 状态信息区域显示详细日志"
        )
        QMessageBox.information(self, "帮助", help_text)
    
    def closeEvent(self, event):
        """关闭主窗口时清理所有资源"""
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