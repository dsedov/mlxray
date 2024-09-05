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
        self.polygon_indices = []  # New attribute to store polygon indices

class BVH:
    def __init__(self, geos: mx.array):
        self.geos = geos
        self.nodes: List[BVHNode] = []
        self.indices: List[int] = []
        self.bboxes: List[mx.array] = []
        self.polygon_indices: List[int] = []

        # Build the BVH
        self._build()

    def _build(self):
        num_triangles = self.geos.shape[0] // 3
        triangles = mx.arange(num_triangles)
        
        bboxes_min = mx.stack([mx.min(self.geos[i*3:(i+1)*3, :3], axis=0) for i in range(num_triangles)])
        bboxes_max = mx.stack([mx.max(self.geos[i*3:(i+1)*3, :3], axis=0) for i in range(num_triangles)])
        bboxes = mx.concatenate([bboxes_min, bboxes_max], axis=1)
        
        root = self._recursive_build(triangles, bboxes, 0)
        self._flatten_bvh(root, -1, 0)
        print("BVH construction completed")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Total leaf nodes: {sum(1 for i in range(0, len(self.indices), 5) if self.indices[i+4] == 1)}")
        print(f"BVH depth: {max(self.indices[3::5])}")
        print(f"Number of triangles: {num_triangles}")

    def _recursive_build(self, triangle_indices: mx.array, bboxes: mx.array, depth: int) -> BVHNode:
        bbox = mx.concatenate([mx.min(bboxes[:, :3], axis=0), mx.max(bboxes[:, 3:], axis=0)])
        node = BVHNode(0, len(triangle_indices), bbox)
        node.polygon_indices = triangle_indices.tolist()

        if len(triangle_indices) <= 8 or depth > 30:
            return node

        centroids = (bboxes[:, :3] + bboxes[:, 3:]) * 0.5
        centroid_min = mx.min(centroids, axis=0)
        centroid_max = mx.max(centroids, axis=0)
        centroid_extent = centroid_max - centroid_min

        best_axis = mx.argmax(centroid_extent).item()
        if centroid_extent[best_axis] < 1e-6:
            return node

        # Sort triangles based on their centroid along the best axis
        sorted_indices = mx.argsort(centroids[:, best_axis])
        triangle_indices = triangle_indices[sorted_indices]
        bboxes = bboxes[sorted_indices]

        # Try multiple split positions
        best_cost = float('inf')
        best_split = None

        for ratio in [0.5, 0.3, 0.7]:
            split = int(len(triangle_indices) * ratio)
            if split == 0 or split == len(triangle_indices):
                continue

            left_bbox = mx.concatenate([mx.min(bboxes[:split, :3], axis=0), mx.max(bboxes[:split, 3:], axis=0)])
            right_bbox = mx.concatenate([mx.min(bboxes[split:, :3], axis=0), mx.max(bboxes[split:, 3:], axis=0)])

            left_cost = self._compute_surface_area(left_bbox) * split
            right_cost = self._compute_surface_area(right_bbox) * (len(triangle_indices) - split)
            total_cost = left_cost + right_cost

            if total_cost < best_cost:
                best_cost = total_cost
                best_split = split

        # Use the best split found
        if best_split is None:
            best_split = len(triangle_indices) // 2

        node.left = self._recursive_build(triangle_indices[:best_split], bboxes[:best_split], depth + 1)
        node.right = self._recursive_build(triangle_indices[best_split:], bboxes[best_split:], depth + 1)
        node.start = min(node.left.start, node.right.start)
        node.end = max(node.left.end, node.right.end)

        return node

    def _compute_surface_area(self, bbox: mx.array) -> float:
        extents = bbox[3:] - bbox[:3]
        return 2 * (extents[0] * extents[1] + extents[1] * extents[2] + extents[2] * extents[0]).item()

    def _flatten_bvh(self, node: BVHNode, parent_idx: int, depth: int) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(node)
        self.bboxes.append(node.bbox)

        is_leaf = node.left is None and node.right is None
        if is_leaf:
            start = len(self.polygon_indices)
            count = len(node.polygon_indices)
            self.polygon_indices.extend(node.polygon_indices)
            self.indices.extend([start, count, parent_idx, depth, 1])
        else:
            self.indices.extend([0, 0, parent_idx, depth, 0])
            left_idx = self._flatten_bvh(node.left, node_idx, depth + 1)
            right_idx = self._flatten_bvh(node.right, node_idx, depth + 1)
            
            self.indices[node_idx * 5] = left_idx
            self.indices[node_idx * 5 + 1] = right_idx
        
        return node_idx

    def get_bboxes(self) -> mx.array:
        return mx.concatenate(self.bboxes)

    def get_indices(self) -> mx.array:
        return mx.array(self.indices, dtype=mx.int32)

    def get_polygon_indices(self) -> mx.array:
        return mx.array(self.polygon_indices, dtype=mx.int32)

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
                print(f"{indent}Leaf Node (depth {depth}): {child2} triangles")
            elif show_non_leaf_nodes:
                print(f"{indent}Internal Node (depth {depth}):")
                print_node(child1, depth + 1)
                print_node(child2, depth + 1)
            else:
                print_node(child1, depth + 1)
                print_node(child2, depth + 1)

        print("BVH Tree Structure:")
        print(f"Total nodes: {len(self.nodes)}")
        leaf_nodes = [node for node in self.nodes if node.left is None and node.right is None]
        print(f"Leaf nodes: {len(leaf_nodes)}")
        print(f"Max depth: {max(self.indices[3::5])}")
        print(f"Average triangles per leaf: {sum(node.end - node.start for node in leaf_nodes) / len(leaf_nodes):.2f}")
        print(f"Max triangles in a leaf: {max(node.end - node.start for node in leaf_nodes)}")
        print_node(0, 0)