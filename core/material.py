import mlx.core as mx

class Material:
    def __init__(self, name, base_weight, base_color, metalness, transmission, specular, specular_roughness, ior):
        self.name = name
        self.base_weight = base_weight
        self.base_color = base_color
        self.metalness = metalness
        self.transmission = transmission
        self.specular = specular
        self.specular_roughness = specular_roughness
        self.ior = ior