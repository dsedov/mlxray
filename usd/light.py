from pxr import UsdGeom, Usd, UsdLux
from pxr import Gf
from core.light import Light
from usd.loader import UsdLoader
import mlx.core as mx
from math import atan
class UsdLight:
    def load_lights(usd_loader: UsdLoader):
        lights = []
        for light_prim in usd_loader.find_lights():
            if light_prim.IsA(UsdLux.RectLight):
                print(f"\nLoaded RectLight: {light_prim.GetPath()}")
                light = Light()
                xform = UsdGeom.Xformable(light_prim).ComputeLocalToWorldTransform(time=Usd.TimeCode.Default())

                color_attr = UsdLux.RectLight(light_prim).GetColorAttr()
                if color_attr:
                    light.color = mx.array(color_attr.Get())
                    print(f"Color: {light.color}")
                else:
                    print(f"No color attribute found for {light_prim.GetPath()}")
                    light.color = mx.array([1.0, 1.0, 1.0])
                
                intensity_attr = UsdLux.RectLight(light_prim).GetIntensityAttr()
                if intensity_attr:
                    light.intensity = intensity_attr.Get()
                    print(f"Intensity: {light.intensity}")
                else:
                    print(f"No intensity attribute found for {light_prim.GetPath()}")
          
                width_attr = UsdLux.RectLight(light_prim).GetWidthAttr()
                if width_attr:
                    light.width = width_attr.Get()
                    print(f"Width: {light.width}")
                else:
                    print(f"No width attribute found for {light_prim.GetPath()}")
                    light.width = 1.0
                
                height_attr = UsdLux.RectLight(light_prim).GetHeightAttr()
                if height_attr:
                    light.height = height_attr.Get()
                    print(f"Height: {light.height}")
                else:
                    print(f"No height attribute found for {light_prim.GetPath()}")
                    light.height = 1.0

                v0 = xform.Transform(Gf.Vec3f(light.width/2.0,  -light.height/2.0, 0.0))
                v1 = xform.Transform(Gf.Vec3f(-light.width/2.0, -light.height/2.0, 0.0))
                v2 = xform.Transform(Gf.Vec3f(-light.width/2.0,  light.height/2.0, 0.0))
                v3 = xform.Transform(Gf.Vec3f(light.width/2.0,   light.height/2.0, 0.0))
                light.vertices = mx.array([
                    [v0[0], v0[1], v0[2]],
                    [v1[0], v1[1], v1[2]],
                    [v2[0], v2[1], v2[2]],
                    [v3[0], v3[1], v3[2]]
                ])
                print(f"Vertices: {light.vertices}")
                light.Q = mx.array([v0[0], v0[1], v0[2]])
                light.u = mx.array([v1[0], v1[1], v1[2]]) - light.Q
                light.v = mx.array([v3[0], v3[1], v3[2]]) - light.Q
                print(f"Q: {light.Q}")
                print(f"u: {light.u}")
                print(f"v: {light.v}")
                lights.append(light)
        return lights
                
                