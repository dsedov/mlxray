import mlx.core as mx
from typing import List, Tuple

class BVHNode:
    def __init__(self, start: int, end: int, bbox: mx.array):
        self.left = None
        self.right = None
        self.start = start
        self.end = end
        self.bbox = bbox

class BVH:
    def __init__(self, geos: mx.array):
        self.geos = geos
        self.nodes: List[BVHNode] = []
        self.indices: List[int] = []
        self.geo_pointers: List[int] = []
        self.geo_pointers_count: List[int] = []
        self.bboxes: List[mx.array] = []

        # Build the BVH
        self._build()

    def _build(self):
        # Initialize with all triangles
        triangles = [(i, self._compute_bbox(i)) for i in range(self.geos.shape[0] // 6)]
        root = self._recursive_build(triangles, 0)
        self._flatten_bvh(root, -1, 0)

    def _recursive_build(self, triangles: List[Tuple[int, mx.array]], depth: int) -> BVHNode:
        start, end = 0, len(triangles)
        
        # Compute bounding box for this node
        bbox = self._compute_node_bbox([t[1] for t in triangles])
        
        node = BVHNode(start, end, bbox)
        
        if end - start <= 4 or depth > 20:  # Leaf node
            self.geo_pointers.append(start)
            self.geo_pointers_count.append(end - start)
            return node
        
        # Choose split axis (alternate between x, y, z)
        axis = depth % 3
        
        # Sort triangles based on centroid along the chosen axis
        triangles.sort(key=lambda t: mx.mean(t[1][:, axis]).item())
        
        mid = (start + end) // 2
        node.left = self._recursive_build(triangles[:mid], depth + 1)
        node.right = self._recursive_build(triangles[mid:], depth + 1)
        
        return node

    def _flatten_bvh(self, node: BVHNode, parent_idx: int, depth: int) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(node)
        self.bboxes.append(node.bbox)
        
        if node.left is None and node.right is None:  # Leaf node
            self.indices.extend([node_idx, -1, parent_idx, depth])
        else:
            self.indices.extend([node_idx, node_idx, parent_idx, depth])
            left_idx = self._flatten_bvh(node.left, node_idx, depth + 1)
            right_idx = self._flatten_bvh(node.right, node_idx, depth + 1)
            self.indices[node_idx * 4] = left_idx
            self.indices[node_idx * 4 + 1] = right_idx
        
        return node_idx

    def _compute_bbox(self, triangle_idx: int) -> mx.array:
        triangle = self.geos[triangle_idx * 6: (triangle_idx + 1) * 6, :3]
        min_bounds = mx.min(triangle, axis=0)
        max_bounds = mx.max(triangle, axis=0)
        return mx.stack([min_bounds, max_bounds])

    def _compute_node_bbox(self, bboxes: List[mx.array]) -> mx.array:
        all_bboxes = mx.stack(bboxes)
        min_bounds = mx.min(all_bboxes[:, 0], axis=0)
        max_bounds = mx.max(all_bboxes[:, 1], axis=0)
        return mx.stack([min_bounds, max_bounds])

    def get_bboxes(self) -> mx.array:
        return mx.concatenate(self.bboxes)

    def get_indices(self) -> mx.array:
        return mx.array(self.indices, dtype=mx.int32)

    def get_geo_pointers(self) -> mx.array:
        return mx.array(self.geo_pointers, dtype=mx.int32)

    def get_geo_pointers_count(self) -> mx.array:
        return mx.array(self.geo_pointers_count, dtype=mx.int32)
