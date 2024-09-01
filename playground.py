import numpy as np
import mlx.core as mx

class BVHNode:
    def __init__(self, box_min, box_max, left_child, right_child, primitive_offset, primitive_count):
        self.box_min = box_min
        self.box_max = box_max
        self.left_child = left_child
        self.right_child = right_child
        self.primitive_offset = primitive_offset
        self.primitive_count = primitive_count

def serialize_bvh(root_node):
    flat_nodes = []
    
    def flatten(node):
        node_index = len(flat_nodes)
        flat_nodes.append(node)
        
        if node.left_child is not None:
            node.left_child = flatten(node.left_child)
        if node.right_child is not None:
            node.right_child = flatten(node.right_child)
        
        return node_index
    
    flatten(root_node)
    
    # Create a structured numpy array
    dtype = np.dtype([
        ('box_min', np.float32, (3,)),
        ('box_max', np.float32, (3,)),
        ('left_child', np.int32),
        ('right_child', np.int32),
        ('primitive_offset', np.int32),
        ('primitive_count', np.int32)
    ])
    
    serialized = np.empty(len(flat_nodes), dtype=dtype)
    
    for i, node in enumerate(flat_nodes):
        serialized[i] = (
            node.box_min,
            node.box_max,
            node.left_child if node.left_child is not None else -1,
            node.right_child if node.right_child is not None else -1,
            node.primitive_offset,
            node.primitive_count
        )
    
    return serialized

# Example usage:
root = BVHNode(
    box_min=np.array([-1, -1, -1], dtype=np.float32),
    box_max=np.array([1, 1, 1], dtype=np.float32),
    left_child=None,
    right_child=None,
    primitive_offset=0,
    primitive_count=0
)

serialized_bvh = serialize_bvh(root)

print(serialized_bvh)

# Convert to MLX array
mlx_bvh = mx.array(serialized_bvh)