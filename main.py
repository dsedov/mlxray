import sys
import mlx.core as mx
import numpy as np

from PySide6.QtWidgets import QApplication
from core.render import Render
from core.image import ImageBuffer
from core.camera import Camera
from ui.render_window import RenderWindow
from usd.loader import UsdLoader
from usd.camera import UsdCamera
from usd.light import UsdLight
from usd.geo import UsdGeo


if __name__ == "__main__":
    usd_loader = UsdLoader("cornell_box.usda")
    camera = UsdCamera.load_camera(usd_loader)
    lights = UsdLight.load_lights(usd_loader)
    geos, norms = UsdGeo.load_geos(usd_loader)

    image_buffer = ImageBuffer(1024, 768)
    
    render = Render(image_buffer, camera, lights, geos, norms)

    app = QApplication(sys.argv)
    window = RenderWindow(render)
    window.show()
    sys.exit(app.exec())