import mlx.core as mx
from typing import List, Tuple
from tqdm import tqdm

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
        self.bboxes: List[mx.array] = []

        # Build the BVH
        self._build()

    def _build(self):
        triangles = [(i, self._compute_bbox(i)) for i in tqdm(range(self.geos.shape[0] // 3), desc="Computing bounding boxes")]
        root = self._recursive_build(triangles, 0)
        self._flatten_bvh(root, -1, 0)
        print("BVH construction completed")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Total leaf nodes: {sum(1 for i in range(0, len(self.indices), 5) if self.indices[i+4] == 1)}")
        print(f"BVH depth: {max(self.indices[3::5])}")
        print(f"Number of triangles: {len(triangles)}")

    def _recursive_build(self, triangles: List[Tuple[int, mx.array]], depth: int) -> BVHNode:
        bbox = self._compute_node_bbox([t[1][:3] for t in triangles])
        start = min(t[0] for t in triangles)
        end = max(t[0] for t in triangles) + 1
        node = BVHNode(start, end, bbox)

        if len(triangles) <= 4 or depth > 100:  # Leaf node
            return node

        best_axis = 0
        best_cost = float('inf')
        best_split = None

        for axis in range(3):
            # Sort triangles based on their centroid along the current axis
            sorted_triangles = sorted(triangles, key=lambda t: mx.mean(t[1][:3, axis]).item())
            mid = len(sorted_triangles) // 2

            # Compute bounding boxes for left and right children
            left_bbox = self._compute_node_bbox([t[1][:3] for t in sorted_triangles[:mid]])
            right_bbox = self._compute_node_bbox([t[1][:3] for t in sorted_triangles[mid:]])

            # Compute the cost of this split (combination of overlap and total volume)
            overlap = self._compute_overlap(left_bbox, right_bbox)
            total_volume = self._compute_volume(left_bbox) + self._compute_volume(right_bbox)
            cost = overlap + total_volume

            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split = (sorted_triangles[:mid], sorted_triangles[mid:])

        # Use the best split found
        node.left = self._recursive_build(best_split[0], depth + 1)
        node.right = self._recursive_build(best_split[1], depth + 1)
        node.start = min(node.left.start, node.right.start)
        node.end = max(node.left.end, node.right.end)

        return node
    def _compute_overlap(self, bbox1: mx.array, bbox2: mx.array) -> float:
        overlap = mx.maximum(0, mx.minimum(bbox1[1], bbox2[1]) - mx.maximum(bbox1[0], bbox2[0]))
        return mx.prod(overlap).item()

    def _compute_volume(self, bbox: mx.array) -> float:
        extents = bbox[1] - bbox[0]
        return mx.prod(extents).item()
    def _flatten_bvh(self, node: BVHNode, parent_idx: int, depth: int) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(node)
        self.bboxes.append(node.bbox)

        is_leaf = node.left is None and node.right is None
        if is_leaf:
            self.indices.extend([node.start, node.end - node.start, parent_idx, depth, 1])
        else:
            self.indices.extend([0, 0, parent_idx, depth, 0])
            left_idx = self._flatten_bvh(node.left, node_idx, depth + 1)
            right_idx = self._flatten_bvh(node.right, node_idx, depth + 1)
            
            self.indices[node_idx * 5] = left_idx
            self.indices[node_idx * 5 + 1] = right_idx
        
        return node_idx

    def _compute_bbox(self, triangle_idx: int) -> mx.array:
        triangle = self.geos[triangle_idx * 3: (triangle_idx * 3 + 3), :3]
        min_bounds = mx.min(triangle, axis=0)
        max_bounds = mx.max(triangle, axis=0)
        return mx.stack([min_bounds, max_bounds])

    def _compute_node_bbox(self, bboxes: List[mx.array]) -> mx.array:
        all_bboxes = mx.stack(bboxes)
        min_bounds = mx.min(all_bboxes[:, 0], axis=0)
        max_bounds = mx.max(all_bboxes[:, 1], axis=0)
        padding = 0.001
        min_bounds -= padding
        max_bounds += padding

        return mx.stack([min_bounds, max_bounds])

    def get_bboxes(self) -> mx.array:
        return mx.concatenate(self.bboxes)

    def get_indices(self) -> mx.array:
        return mx.array(self.indices, dtype=mx.int32)

    def get_geo_pointers(self) -> mx.array:
        return mx.array(self.geo_pointers, dtype=mx.int32)

    def get_geo_pointers_count(self) -> mx.array:
        return mx.array(self.geo_pointers_count, dtype=mx.int32)
    
    def print_bvh(self, show_non_leaf_nodes=True):
        def print_node(index, depth):
            if index == -1 or index >= len(self.nodes):
                print(f"{'  ' * depth}Invalid node index: {index}")
                return
            
            indent = "  " * depth
            bbox = self.bboxes[index]
            
            node_data = self.indices[index * 5 : index * 5 + 5]
            child1, child2, parent, node_depth, is_leaf = node_data
            

            if is_leaf:
                print(f"{indent}  {depth} Points: {child2}")
            else:
                
                print(f"{indent}  Left Child:")
                print_node(child1, depth + 1)
                print(f"{indent}  Right Child:") 
                print_node(child2, depth + 1)

        print("BVH Tree Structure:")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Indices: {self.indices}")
        print_node(0, 0)