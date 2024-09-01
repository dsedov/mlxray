from pxr import UsdGeom, Usd
from pxr import Gf
from core.camera import Camera
from usd.loader import UsdLoader
import mlx.core as mx
from math import atan
class UsdCamera:
    def load_camera(usd_loader: UsdLoader):
        camera = Camera()
        camera_prim = usd_loader.find_camera()
        usd_camera = UsdGeom.Camera(camera_prim)
        horizontal_aperture = usd_camera.GetHorizontalApertureAttr().Get()
        vertical_aperture = usd_camera.GetVerticalApertureAttr().Get()

        horizontal_aperture *= usd_loader.meters_per_unit * 100.0
        vertical_aperture *= usd_loader.meters_per_unit * 100.0

        if usd_camera.GetFocalLengthAttr().IsAuthored():
            focal_length = usd_camera.GetFocalLengthAttr().Get()
            focal_length *= usd_loader.meters_per_unit * 100.0
            camera.fov = 2.0 * atan((horizontal_aperture * 0.5) / focal_length)
            camera.fov = camera.fov * (180.0 / mx.pi)
        local_to_world = usd_camera.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        camera.center = mx.array(local_to_world.ExtractTranslation())
        forward = Gf.Vec3f(0,0,-1)
        forward = forward * local_to_world.ExtractRotationMatrix()
        camera.look_at = mx.array(
            [camera.center[0] + forward[0],
             camera.center[1] + forward[1],
             camera.center[2] + forward[2]]
        )
        up = Gf.Vec3f(0,1,0)
        up = up * local_to_world.ExtractRotationMatrix()
        camera.look_up = mx.array(
            [camera.center[0] + up[0],
             camera.center[1] + up[1],
             camera.center[2] + up[2]]
        )
        camera.recompute()
        print(f"\nLoaded camera : {usd_camera.GetPath()}")
        print(f"Camera center : {camera.center}")
        print(f"Camera look_at: {camera.look_at}")
        print(f"Camera look_up: {camera.look_up}")
        print(f"Camera fov    : {camera.fov}")
        return camera

