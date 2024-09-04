from .image import ImageBuffer
from .camera import Camera
import mlx.core as mx
import numpy as np
from kernels.render_kernel import render_kernel
from PySide6.QtCore import QThread, Signal
from .vector import *
from .bvh import BVH  # Assuming you have a BVH class implemented

class Render(QThread):
    image_ready = Signal(np.ndarray)

    def __init__(self, image_buffer: ImageBuffer, camera: Camera, lights: list, geos: list, is_vertical_fov = False, fov_in_degrees = True):
        super().__init__()

        self.running = True
        self.image_buffer = image_buffer
        self.camera = camera
        self.lights = lights
        self.geos = geos

        print("Initialized render engine")
        focal_length = mx.linalg.norm(self.camera.center - self.camera.look_at)
        fov_radians = mx.radians(self.camera.fov) if fov_in_degrees else self.camera.fov
        if is_vertical_fov:
            viewport_height = 2.0 * focal_length * mx.tan(fov_radians / 2.0)
            viewport_width = viewport_height * self.image_buffer.width / self.image_buffer.height
        else:
            viewport_width  = 2.0 * focal_length * mx.tan(fov_radians / 2.0)
            viewport_height = viewport_width / (self.image_buffer.width / self.image_buffer.height)

        w = (camera.center - camera.look_at).norm()
        u = cross(camera.look_up, w).norm()
        v = cross(w, u)

        viewport_u = viewport_width * u
        viewport_v = viewport_height * -v

        self.pixel_delta_u = viewport_u / self.image_buffer.width
        self.pixel_delta_v = viewport_v / self.image_buffer.height

        viewport_upper_left = camera.center - (focal_length * w) - viewport_u/2.0 - viewport_v/2.0
        self.pixel00_loc = viewport_upper_left + 0.5 * (self.pixel_delta_u + self.pixel_delta_v)

    def run(self):
        print("Preparing geos")
        all_geos = None 
        for geo in self.geos:
            if all_geos is None:
                all_geos = geo
            else:
                all_geos = np.vstack((all_geos, geo))
        geos = mx.array(all_geos)
        
        print("Building BVH")
        bvh = BVH(geos)
        bvh.print_bvh()

        bboxes = mx.array(bvh.get_bboxes())
        indices = mx.array(bvh.get_indices())
        
        #bvh.print_bvh()

        print(f"Rendering geos with shape {geos.shape}")
        print(f"Rendering image with shape {self.image_buffer.data.shape}")
        print(f"pixel00_loc: {self.pixel00_loc}")
        print(f"pixel_delta_u: {self.pixel_delta_u}")
        print(f"pixel_delta_v: {self.pixel_delta_v}")

        sample = 256
        np_image_buffer = None
        from tqdm import tqdm

        import time

        start_time = time.time()

        print(f"geos shape: {geos.shape}")
        print(f"bboxes shape: {bboxes.shape}")
        print(f"indices shape: {indices.shape}")


        print(f"bboxes: {bboxes}")


        for i in tqdm(range(sample), desc="Rendering", unit="sample"):
            if not self.running:
                break 
            self.image_buffer.data = render_kernel(
                image_buffer  = self.image_buffer.data, 
                camera_center = self.camera.center,
                pixel00_loc   = self.pixel00_loc, 
                pixel_delta_u = self.pixel_delta_u, 
                pixel_delta_v = self.pixel_delta_v,
                geos          = geos,
                bboxes        = bboxes,
                indices       = indices,
            )
            # show image buffer
            if np_image_buffer is None:
                np_image_buffer = np.array(self.image_buffer.data)
                image_data = (np_image_buffer * 255).astype(np.uint8)
                self.image_ready.emit(image_data)
            else:
                np_image_buffer += np.array(self.image_buffer.data)

                image_data = ((np_image_buffer / float(i+1)) * 255).astype(np.uint8)
                self.image_ready.emit(image_data)
            
            # Update progress bar
            tqdm.write(f"Completed {i+1}/{sample} samples")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total rendering time: {elapsed_time:.2f} seconds")

    def stop(self):
        self.running = False