from PySide6.QtWidgets import QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

from core.render import Render

class RenderWindow(QMainWindow):
    def __init__(self, render_engine: Render):
        super().__init__()
        self.render_engine = render_engine
        self.setWindowTitle("MLX Render")
        self.setGeometry(100, 100, 1024, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.toggle_button = QPushButton("Stop", self)
        self.toggle_button.clicked.connect(self.toggle_generation)
        layout.addWidget(self.toggle_button)

        # Swapped width and height to match the original dimensions
        self.image_shape = self.render_engine.image_buffer.shape
        self.generator_thread = self.render_engine
        self.generator_thread.image_ready.connect(self.update_image)
        self.generator_thread.start()

    def update_image(self, image_data):
        width, height, channel = self.image_shape
        bytes_per_line = 3 * width
        qimage = QImage(image_data.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def toggle_generation(self):
        if self.generator_thread.running:
            self.generator_thread.stop()
            self.toggle_button.setText("Start")
        else:
            self.generator_thread = self.render_engine
            self.generator_thread.image_ready.connect(self.update_image)
            self.generator_thread.start()
            self.toggle_button.setText("Stop")

    def closeEvent(self, event):
        self.generator_thread.stop()
        self.generator_thread.wait()
        super().closeEvent(event)
