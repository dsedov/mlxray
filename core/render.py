from .image import ImageBuffer
from .camera import Camera
import mlx.core as mx
import numpy as np
from kernels.render_kernel import render_kernel
from kernels.sharpen_kernel import sharpen_kernel
from PySide6.QtCore import QThread, Signal
from .vector import *
from .bvh import BVH
import time
from tqdm import tqdm
from tools.bluenoise import BlueNoiseGenerator
from scipy.ndimage import gaussian_filter

class Render(QThread):
    image_ready = Signal(np.ndarray)

    def __init__(self, image_buffer: ImageBuffer, camera: Camera, lights: list, geos: list, norms: list, mats: list, is_vertical_fov = False, fov_in_degrees = True):
        super().__init__()

        self.running = True
        self.image_buffer = image_buffer
        self.camera = camera
        self.lights = lights
        self.geos = geos
        self.norms = norms
        self.mats = mats

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
        all_norms = None
        all_mats = None
        for geo in self.geos:
            if all_geos is None:
                all_geos = geo
            else:
                all_geos = np.vstack((all_geos, geo))
        geos = mx.array(all_geos)

        print("Preparing normals")
        for norm in self.norms:
            if all_norms is None:
                all_norms = norm
            else:
                all_norms = np.vstack((all_norms, norm))
        norms = mx.array(all_norms)


        print("Preparing materials")
        for mat in self.mats:
            if all_mats is None:
                all_mats = mat
            else:
                all_mats = np.vstack((all_mats, mat))
        mats = mx.array(all_mats, dtype=mx.int32)  

        print("Building BVH")
        bvh_start_time = time.time()
        bvh = BVH(geos)
        bvh_end_time = time.time()
        print(f"BVH construction time: {bvh_end_time - bvh_start_time:.2f} seconds")
     

        bboxes = mx.array(bvh.get_bboxes())
        indices = mx.array(bvh.get_indices())
        polygon_indices = mx.array(bvh.get_polygon_indices())

        samples = 1024
        np_image_buffer = None

        start_time = time.time()

        blue_noise_generator = BlueNoiseGenerator(256, 100, 5)
        blue_noise_texture = blue_noise_generator.load_noise(filename="512x512x4_3d_blue_noise.npy")
        blue_noise_texture_size = blue_noise_texture.shape[0]

        for i in tqdm(range(samples), desc="Rendering", unit="sample"):
            if not self.running:
                break 
            self.image_buffer.data = render_kernel(
                image_buffer  = self.image_buffer.data, 
                camera_center = self.camera.center,
                pixel00_loc   = self.pixel00_loc, 
                pixel_delta_u = self.pixel_delta_u, 
                pixel_delta_v = self.pixel_delta_v,
                sample        = i,
                samples       = samples,
                geos          = geos,
                norms         = norms,
                mats          = mats,
                bboxes        = bboxes,
                indices       = indices,
                polygon_indices = polygon_indices,
                blue_noise_texture  = blue_noise_texture,
                blue_noise_texture_size = blue_noise_texture_size,
            )

            self.image_buffer.data = sharpen_kernel(self.image_buffer.data)

            if np_image_buffer is None:
                np_image_buffer = np.array(self.image_buffer.data)
            else:
                np_image_buffer += np.array(self.image_buffer.data)

            # Calculate average image
            avg_image = np_image_buffer / float(i+1)

            # Ensure the image is in the correct format for display
            image_data = (avg_image * 255).clip(0, 255).astype(np.uint8)

            self.image_ready.emit(image_data)
            
            # Update progress bar
            tqdm.write(f"Completed {i+1}/{samples} samples")

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Total rendering time: {elapsed_time:.2f} seconds")

    def stop(self):
        self.running = False