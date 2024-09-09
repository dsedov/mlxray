from mlx import core as mx
class Material:
    def __init__(self):
        self.name = ""
        self.albedo = mx.array([1.0, 1.0, 1.0])
        self.roughness = 0.0
        self.metallic = 0.0
        self.ior = 1.5
        self.transmission = 0.0
        self.absorption = mx.array([0.0, 0.0, 0.0])
        self.emission = mx.array([0.0, 0.0, 0.0])