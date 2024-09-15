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
from usd.material import UsdMaterial


if __name__ == "__main__":
    usd_loader = UsdLoader("cornell_box.usda")
    camera = UsdCamera.load_camera(usd_loader)
    lights = UsdLight.load_lights(usd_loader)
    materials = UsdMaterial.load_materials(usd_loader)
    geos, norms, mats = UsdGeo.load_geos(usd_loader, materials)
    print(f"Loaded {len(geos)} geos, {len(norms)} normals, {len(mats)} materials")
    print(f"Material: {mats}")

    image_buffer = ImageBuffer(1024, 1024)
    
    render = Render(image_buffer, camera, lights, geos, norms, mats)

    app = QApplication(sys.argv)
    window = RenderWindow(render)
    window.show()
    sys.exit(app.exec())