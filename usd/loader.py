import pxr
from pxr import Usd, UsdGeom, UsdShade, UsdLux

class UsdLoader:
    def __init__(self, path):
        print(f"Loading USD file: {path}")
        self.stage = Usd.Stage.Open(path)
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(self.stage)
        print(f"Meters per unit: {self.meters_per_unit}")

    def find_camera(self):
        for prim in self.stage.Traverse():
            if prim.IsA(UsdGeom.Camera):
                return prim
        print("No camera found")
        return None

    def find_lights(self):
        lights = []
        for prim in self.stage.TraverseAll():
            if prim.IsA(UsdLux.RectLight):
                lights.append(prim)
        return lights

    def find_geos(self):
        geos = []
        for prim in self.stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                geos.append(prim)
        return geos

    def find_materials(self):
        materials = []
        for prim in self.stage.TraverseAll():
            if prim.IsA(UsdShade.Material):
                materials.append(prim)
        return materials