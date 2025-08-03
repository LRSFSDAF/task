'''
Description: 3D可视化工具 - 使用PyOpenGL嵌入渲染
Author: Damocles_lin
Date: 2025-07-29 20:27:35
LastEditTime: 2025-08-04 00:25:01
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
    QIcon, QFont, QPalette, QColor,
    QOpenGLBuffer, QOpenGLVertexArrayObject, QOpenGLShaderProgram, QOpenGLShader,
    QSurfaceFormat
)
from utils import setup_logger, load_colmap_data, create_intrinsic_matrix, project_points_to_image
import logging
from OpenGL import GL as gl
from OpenGL import GLU as glu

logger = setup_logger('gui')

class OpenGLRenderer(QOpenGLWidget):
    """使用PyOpenGL渲染3D场景的Widget（使用VBO）"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 600)

        # 设置OpenGL版本为3.3核心模式
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        self.setFormat(fmt)

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
        self.vao_point = None
        self.vbo_point = None
        self.n_points = 0

        self.vao_mesh = None
        self.vbo_mesh = None
        self.n_mesh_indices = 0
        
        # 新增：坐标轴和相机位姿的VBO
        self.vao_axes = None
        self.vbo_axes = None
        self.vao_cameras = None
        self.vbo_cameras = None
        self.n_camera_vertices = 0

    def initializeGL(self):
        gl.glClearColor(0.1, 0.1, 0.1, 1.0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        self.create_shaders()
        self.create_axes_vbo()  # 创建坐标轴的VBO

    def create_shaders(self):
        vertex_shader = """
            #version 330 core
            layout (location = 0) in vec3 position;
            layout (location = 1) in vec3 color;
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            uniform float pointSize;  // 添加点大小控制
            out vec3 fragColor;
            void main() {
                gl_Position = projection * view * model * vec4(position, 1.0);
                fragColor = color;
                gl_PointSize = pointSize;  // 设置点大小
            }
        """
        fragment_shader = """
            #version 330 core
            in vec3 fragColor;
            out vec4 FragColor;
            void main() {
                FragColor = vec4(fragColor, 1.0);
            }
        """
        self.shader_program = QOpenGLShaderProgram()
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.Vertex, vertex_shader)
        self.shader_program.addShaderFromSourceCode(QOpenGLShader.Fragment, fragment_shader)
        self.shader_program.link()
        if not self.shader_program.isLinked():
            logger.error("着色器链接失败: " + self.shader_program.log())
            return False
        return True

    def create_axes_vbo(self):
        """创建坐标轴的VBO"""
        # 坐标轴数据：位置和颜色
        # 每个顶点6个float：位置(3) + 颜色(3)
        vertices = [
            # X轴 (红色)
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            1.0, 0.0, 0.0, 1.0, 0.0, 0.0,
            
            # Y轴 (绿色)
            0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
            
            # Z轴 (蓝色)
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 1.0
        ]
        vertices = np.array(vertices, dtype=np.float32)
        
        self.vao_axes = QOpenGLVertexArrayObject()
        self.vao_axes.create()
        self.vao_axes.bind()
        
        self.vbo_axes = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_axes.create()
        self.vbo_axes.bind()
        self.vbo_axes.allocate(vertices.tobytes(), vertices.nbytes)
        
        # 设置顶点属性
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(12))
        
        self.vbo_axes.release()
        self.vao_axes.release()

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        if not self.shader_program or not self.shader_program.isLinked():
            return

        self.shader_program.bind()
        # 构建模型、视图、投影矩阵
        projection = self.projection_matrix()
        model = self.model_matrix()
        view = self.view_matrix()

        self.shader_program.setUniformValue("model", model)
        self.shader_program.setUniformValue("view", view)
        self.shader_program.setUniformValue("projection", projection)

        # 绘制坐标轴（使用VBO）
        if self.vao_axes:
            self.vao_axes.bind()
            gl.glDrawArrays(gl.GL_LINES, 0, 6)  # 6个顶点
            self.vao_axes.release()

        # 绘制点云
        if self.vao_point and self.n_points > 0:
            # 确保着色器程序已绑定
            self.shader_program.bind()
            self.shader_program.setUniformValue("pointSize", 2.0)  # 设置点大小
            
            self.vao_point.bind()
            gl.glDrawArrays(gl.GL_POINTS, 0, self.n_points)
            self.vao_point.release()

        # 绘制网格
        if self.vao_mesh and self.n_mesh_indices > 0:
            # 确保着色器程序已绑定
            self.shader_program.bind()
            
            self.vao_mesh.bind()
            gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self.n_mesh_indices)
            self.vao_mesh.release()

        # 绘制相机位姿（使用VBO）
        if self.vao_cameras and self.n_camera_vertices > 0:
            # 确保着色器程序已绑定
            self.shader_program.bind()
            
            self.vao_cameras.bind()
            gl.glDrawArrays(gl.GL_LINES, 0, self.n_camera_vertices)
            self.vao_cameras.release()
        
        self.shader_program.release()

    def projection_matrix(self):
        import PyQt5.QtGui as QtGui
        w, h = self.width(), self.height()
        aspect = w / h if h > 0 else 1.0
        perspective = QtGui.QMatrix4x4()
        perspective.perspective(45, aspect, 0.1, 100.0)
        return perspective

    def model_matrix(self):
        import PyQt5.QtGui as QtGui
        model = QtGui.QMatrix4x4()
        return model

    def view_matrix(self):
        import PyQt5.QtGui as QtGui
        view = QtGui.QMatrix4x4()
        # 平移
        view.translate(self.camera_translation[0], self.camera_translation[1], -self.camera_distance)
        # 先绕X轴再绕Y轴
        view.rotate(self.camera_rotation_x, 1, 0, 0)
        view.rotate(self.camera_rotation_y, 0, 1, 0)
        return view

    def set_point_cloud(self, points, colors):
        # 归一化和中心化
        if len(points) > 0:
            center = np.mean(points, axis=0)
            points = points - center
        else:
            center = np.array([0, 0, 0])

        self.point_cloud = {
            'points': points,
            'colors': colors,
            'center': center
        }
        self.update_pointcloud_vbo(points, colors)
        self.update()

    def update_pointcloud_vbo(self, points, colors):
        # 清理旧的
        if self.vbo_point:
            self.vbo_point.destroy()
            self.vbo_point = None
        if self.vao_point:
            self.vao_point.destroy()
            self.vao_point = None

        if points is None or len(points) == 0:
            self.n_points = 0
            return

        # 合并到一个数组：N,6
        vertex_data = np.hstack([points.astype(np.float32), colors.astype(np.float32)])
        self.n_points = len(points)
        
        # 确保在OpenGL上下文中创建对象
        self.makeCurrent()
        
        self.vao_point = QOpenGLVertexArrayObject()
        self.vao_point.create()
        self.vao_point.bind()
        
        self.vbo_point = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_point.create()
        self.vbo_point.bind()
        self.vbo_point.allocate(vertex_data.tobytes(), vertex_data.nbytes)

        # 设置属性
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(12))

        self.vbo_point.release()
        self.vao_point.release()
        
        # 确保操作完成
        self.doneCurrent()

    def set_mesh(self, vertices, triangles, colors=None):
        if colors is None:
            colors = np.ones_like(vertices) * 0.7
        if len(vertices) > 0:
            center = np.mean(vertices, axis=0)
            vertices = vertices - center
        else:
            center = np.array([0, 0, 0])
        self.mesh = {
            'vertices': vertices,
            'triangles': triangles,
            'colors': colors,
            'center': center
        }
        self.update_mesh_vbo(vertices, triangles, colors)
        self.update()

    def update_mesh_vbo(self, vertices, triangles, colors):
        if self.vbo_mesh:
            self.vbo_mesh.destroy()
            self.vbo_mesh = None
        if self.vao_mesh:
            self.vao_mesh.destroy()
            self.vao_mesh = None

        if vertices is None or len(vertices) == 0 or triangles is None or len(triangles) == 0:
            self.n_mesh_indices = 0
            return

        # 将三角形顶点展开为N*3行，每行6个float
        flat_vertices = vertices[triangles.reshape(-1)]
        flat_colors = colors[triangles.reshape(-1)]
        vertex_data = np.hstack([flat_vertices.astype(np.float32), flat_colors.astype(np.float32)])
        self.n_mesh_indices = len(flat_vertices)
        
        # 确保在OpenGL上下文中创建对象
        self.makeCurrent()
        
        self.vao_mesh = QOpenGLVertexArrayObject()
        self.vao_mesh.create()
        self.vao_mesh.bind()
        
        self.vbo_mesh = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_mesh.create()
        self.vbo_mesh.bind()
        self.vbo_mesh.allocate(vertex_data.tobytes(), vertex_data.nbytes)

        # 设置属性
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(12))

        self.vbo_mesh.release()
        self.vao_mesh.release()
        
        # 确保操作完成
        self.doneCurrent()

    def set_camera_poses(self, extrinsics):
        self.camera_poses = extrinsics
        self.update_cameras_vbo(extrinsics)  # 更新相机位姿的VBO
        self.update()

    def update_cameras_vbo(self, extrinsics):
        """更新相机位姿的VBO"""
        # 清理旧的
        if self.vbo_cameras:
            self.vbo_cameras.destroy()
            self.vbo_cameras = None
        if self.vao_cameras:
            self.vao_cameras.destroy()
            self.vao_cameras = None

        if extrinsics is None or len(extrinsics) == 0:
            self.n_camera_vertices = 0
            return

        # 创建顶点数据数组
        vertices = []
        for extrinsic in extrinsics:
            R = extrinsic[:3, :3]
            t = extrinsic[:3, 3]
            camera_center = -R.T @ t
            size = 0.1
            x_dir = R[:, 0] * size
            y_dir = R[:, 1] * size
            z_dir = R[:, 2] * size
            
            # 每个相机位姿需要6个顶点（3条线）
            # X轴 (红色)
            vertices.extend([camera_center[0], camera_center[1], camera_center[2], 1.0, 0.0, 0.0])
            vertices.extend([camera_center[0] + x_dir[0], camera_center[1] + x_dir[1], camera_center[2] + x_dir[2], 1.0, 0.0, 0.0])
            
            # Y轴 (绿色)
            vertices.extend([camera_center[0], camera_center[1], camera_center[2], 0.0, 1.0, 0.0])
            vertices.extend([camera_center[0] + y_dir[0], camera_center[1] + y_dir[1], camera_center[2] + y_dir[2], 0.0, 1.0, 0.0])
            
            # Z轴 (蓝色)
            vertices.extend([camera_center[0], camera_center[1], camera_center[2], 0.0, 0.0, 1.0])
            vertices.extend([camera_center[0] + z_dir[0], camera_center[1] + z_dir[1], camera_center[2] + z_dir[2], 0.0, 0.0, 1.0])
        
        vertices = np.array(vertices, dtype=np.float32)
        self.n_camera_vertices = len(vertices) // 6  # 每个顶点6个值
        
        # 确保在OpenGL上下文中创建对象
        self.makeCurrent()
        
        # 创建VAO和VBO
        self.vao_cameras = QOpenGLVertexArrayObject()
        self.vao_cameras.create()
        self.vao_cameras.bind()
        
        self.vbo_cameras = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self.vbo_cameras.create()
        self.vbo_cameras.bind()
        self.vbo_cameras.allocate(vertices.tobytes(), vertices.nbytes)
        
        # 设置顶点属性
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, gl.ctypes.c_void_p(12))
        
        self.vbo_cameras.release()
        self.vao_cameras.release()
        
        # 确保操作完成
        self.doneCurrent()

    def reset_view(self):
        self.camera_distance = 5.0
        self.camera_rotation_x = 0.0
        self.camera_rotation_y = 0.0
        self.camera_translation = [0.0, 0.0, 0.0]
        self.update()
        
    def clear_scene(self):
        self.point_cloud = None
        self.mesh = None
        self.camera_poses = None
        self.reset_view()
        
        # 确保在OpenGL上下文中释放资源
        self.makeCurrent()
        
        # 清除VBO
        if self.vbo_point:
            self.vbo_point.destroy()
            self.vbo_point = None
        if self.vao_point:
            self.vao_point.destroy()
            self.vao_point = None
        if self.vbo_mesh:
            self.vbo_mesh.destroy()
            self.vbo_mesh = None
        if self.vao_mesh:
            self.vao_mesh.destroy()
            self.vao_mesh = None
        if self.vbo_cameras:
            self.vbo_cameras.destroy()
            self.vbo_cameras = None
        if self.vao_cameras:
            self.vao_cameras.destroy()
            self.vao_cameras = None
        
        self.doneCurrent()
        
        self.n_points = 0
        self.n_mesh_indices = 0
        self.n_camera_vertices = 0
        self.update()

    def mousePressEvent(self, event):
        self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_mouse_pos is None:
            return
        dx = event.x() - self.last_mouse_pos.x()
        dy = event.y() - self.last_mouse_pos.y()
        if abs(dx) < 1 and abs(dy) < 1:
            return  # 忽略微小移动，提高流畅度
        if event.buttons() & Qt.LeftButton:
            self.camera_rotation_x += dy * self.rotation_sensitivity
            self.camera_rotation_y += dx * self.rotation_sensitivity
        elif event.buttons() & Qt.RightButton:
            self.camera_translation[0] += dx * self.translation_sensitivity
            self.camera_translation[1] -= dy * self.translation_sensitivity
        self.last_mouse_pos = event.pos()
        self.update()

    def wheelEvent(self, event):
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
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # 创建OpenGL渲染器
        self.gl_widget = OpenGLRenderer()

        # 左侧控制面板
        left_panel = QWidget()
        left_panel.setFixedWidth(300)  # 固定宽度
        left_layout = QVBoxLayout(left_panel)
        left_layout.setAlignment(Qt.AlignTop)
        left_layout.setSpacing(15)
        
        # 视图控制按钮
        view_control_group = QGroupBox("视图控制")
        view_control_layout = QHBoxLayout(view_control_group)
        
        self.btn_reset_view = QPushButton("重置视图")
        self.btn_reset_view.setFixedHeight(40)
        self.btn_reset_view.clicked.connect(self.gl_widget.reset_view)
        
        self.btn_clear_view = QPushButton("清除视图")
        self.btn_clear_view.setFixedHeight(40)
        self.btn_clear_view.clicked.connect(self.clear_view)
        self.btn_clear_view.setStyleSheet("background-color: #e74c3c; color: white;")  # 红色按钮突出显示
        
        view_control_layout.addWidget(self.btn_reset_view)
        view_control_layout.addWidget(self.btn_clear_view)
        
        # 文件操作区域
        file_group = QGroupBox("文件操作")
        file_layout = QGridLayout(file_group)
        file_layout.setSpacing(15)
        
        # 创建按钮
        self.btn_load_pcd = self.create_button("加载点云", "point_cloud.png")
        self.btn_load_pcd.clicked.connect(self.load_point_cloud)
        
        self.btn_load_mesh = self.create_button("加载网格", "mesh.png")
        self.btn_load_mesh.clicked.connect(self.load_mesh)
        
        self.btn_load_npz = self.create_button("加载重建数据", "reconstruction.png")
        self.btn_load_npz.clicked.connect(self.load_reconstruction_data)
        
        self.btn_help = self.create_button("帮助", "help.png")
        self.btn_help.clicked.connect(self.show_help)
        
        # 添加到网格布局
        file_layout.addWidget(self.btn_load_pcd, 0, 0)
        file_layout.addWidget(self.btn_load_mesh, 0, 1)
        file_layout.addWidget(self.btn_load_npz, 1, 0)
        file_layout.addWidget(self.btn_help, 1, 1)
        
        # 信息显示区域
        info_group = QGroupBox("信息显示")
        info_layout = QVBoxLayout(info_group)
        
        self.info_display = QTextEdit()
        self.info_display.setFont(QFont("Arial", 9))
        self.info_display.setReadOnly(True)
        self.info_display.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa; 
                border: 1px solid #e0e0e0; 
                border-radius: 3px;
                padding: 8px;
            }
        """)
        self.info_display.setPlaceholderText("文件信息和状态信息将显示在这里...")
        info_layout.addWidget(self.info_display)
        
        # 添加组件到左侧面板
        left_layout.addWidget(view_control_group)
        left_layout.addWidget(file_group)
        left_layout.addWidget(info_group)
        
        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.gl_widget)
        
        # 底部状态栏
        self.statusBar().setFont(QFont("Arial", 8))
        self.statusBar().showMessage("就绪")
        
        # 设置主窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ecf0f1;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
    
    def create_button(self, text, icon_name=None):
        """创建带样式的按钮"""
        button = QPushButton(text)
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        button.setMinimumHeight(70)
        button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                padding: 9px;
                background-color: #1c638e;
            }
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
        self.info_display.append(f"> {message}")
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
            "   - 点击'重置视图'按钮恢复初始视角\n"
            "   - 点击'清除视图'按钮移除所有对象并重置视图\n\n"
            "3. 相机位姿显示:\n"
            "   - 加载重建数据时选择点云选项，会同时显示相机位姿\n"
            "   - 红色轴: X轴, 绿色轴: Y轴, 蓝色轴: Z轴\n\n"
            "4. 信息显示:\n"
            "   - 底部状态栏显示当前操作状态\n"
            "   - 信息框显示详细日志和文件信息"
        )
        QMessageBox.information(self, "帮助", help_text)
    
    def clear_view(self):
        """清除当前视图中的所有对象"""
        self.gl_widget.clear_scene()
        self.update_status("已清除所有对象并重置视图")
    
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