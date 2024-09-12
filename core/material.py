import mlx.core as mx

class Material:
    def __init__(self):
        self.name = ""
        self.base = 0.0
        self.base_color = mx.array([1.0, 1.0, 1.0])
        self.roughness = 0.0
        self.metalness = 0.0
        self.transmission = 0.0
        self.specular = 0.0
        self.specular_roughness = 0.0
        self.ior = 1.0