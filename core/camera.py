import mlx.core as mx

class Camera:
    def __init__(self, fov = 60.0, center = mx.array([0.0, 0.0, 0.0]), look_at = mx.array([0.0, 0.0, -1.0]), look_up = mx.array([0.0, 1.0, 0.0])):
        self.fov = fov
        self.center = center
        self.look_at = look_at
        self.look_up = look_up
        self.data = mx.array([self.fov, 
                              self.center[0], self.center[1], self.center[2],
                              self.look_at[0], self.look_at[1], self.look_at[2],
                              self.look_up[0], self.look_up[1], self.look_up[2]])

    def recompute(self):
        self.data = mx.array([self.fov, 
                              self.center[0], self.center[1], self.center[2],
                              self.look_at[0], self.look_at[1], self.look_at[2],
                              self.look_up[0], self.look_up[1], self.look_up[2]])