import mlx.core as mx
class ImageBuffer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.shape = [width, height, 3]
        self.data = mx.zeros(self.shape, dtype=mx.float32)

    def set_pixel(self, x: int, y: int, r: int, g: int, b: int):
        self.data[x, y] = [r, g, b]