import mlx.core as mx

def cross(a, b):
    return mx.array([a[1] * b[2] - a[2] * b[1],
                     a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])

def extend_array_methods():
    def norm(self):
        """Return a unit vector."""
        return self / mx.linalg.norm(self)

    mx.array.norm = norm

extend_array_methods()