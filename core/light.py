import mlx.core as mx

class Light:
    def __init__(self):

        self.width = 0.0
        self.height = 0.0

        self.Q = mx.array([0.0, 0.0, 0.0])
        self.u = mx.array([0.0, 0.0, 0.0])
        self.v = mx.array([0.0, 0.0, 0.0])
        self.intensity = 0.0
        self.color = mx.array([0.0, 0.0, 0.0])

        self.vertices = mx.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
