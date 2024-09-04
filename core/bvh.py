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
        self.geo_pointers: List[int] = []
        self.geo_pointers_count: List[int] = []
        self.bboxes: List[mx.array] = []

        # Sanity check
        expected_triangle_count = geos.shape[0] // 6
        print(f"Expected triangle count: {expected_triangle_count}")

        # Build the BVH
        self._build()

    def _build(self):
        # Initialize with all triangles
        triangles = [(i, self._compute_bbox(i)) for i in tqdm(range(self.geos.shape[0] // 6), desc="Computing bounding boxes")]
        root = self._recursive_build(triangles, 0)
        self._flatten_bvh(root, -1, 0)
        print("BVH construction completed")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Total leaf nodes: {sum(1 for node in self.nodes if node.left is None and node.right is None)}")
        print(f"BVH depth: {max(self.indices[3::4])}")
        print(f"Number of triangles: {len(triangles)}")  # Use len(triangles) instead of len(self.geo_pointers_count)
        print(f"Total triangles in leaves: {sum(self.geo_pointers_count)}")

    def _recursive_build(self, triangles: List[Tuple[int, mx.array]], depth: int) -> BVHNode:
        # Compute bounding box for this node
        bbox = self._compute_node_bbox([t[1] for t in triangles])
        
        # Create node with correct start and end indices
        start = min(t[0] for t in triangles)
        end   = max(t[0] for t in triangles) + 1
        node  = BVHNode(start, end, bbox)
        
        if len(triangles) <= 4 or depth > 20:  # Leaf node
            self.geo_pointers.append(start)
            self.geo_pointers_count.append(len(triangles))
            return node
        
        # Choose split axis (alternate between x, y, z)
        axis = depth % 3
        
        # Sort triangles based on centroid along the chosen axis
        triangles.sort(key=lambda t: mx.mean(t[1][:, axis]).item())
        
        mid = len(triangles) // 2
        node.left = self._recursive_build(triangles[:mid], depth + 1)
        node.right = self._recursive_build(triangles[mid:], depth + 1)
        
        # Update node's start and end based on children
        node.start = min(node.left.start, node.right.start)
        node.end = max(node.left.end, node.right.end)
        
        return node

    def _flatten_bvh(self, node: BVHNode, parent_idx: int, depth: int) -> int:
        node_idx = len(self.nodes)
        self.nodes.append(node)
        self.bboxes.append(node.bbox)

        if node.left is None and node.right is None:  # Leaf node
            self.indices.extend([node.start, node.end - node.start, parent_idx, depth])
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
    
    def print_bvh(self, show_non_leaf_nodes=False):
        def print_node(index, depth):
            if index == -1 or index >= len(self.nodes):
                print(f"{'  ' * depth}Invalid node index: {index}")
                return
            
            node = self.nodes[index]
            indent = "  " * depth
            bbox = self.bboxes[index]
            
            is_leaf = node.left is None and node.right is None
            
            if is_leaf or show_non_leaf_nodes:
                print(f"{indent}Node {index}:")
                print(f"{indent}  Depth: {depth}")
                print(f"{indent}  Bounding Box: Min {bbox[0]}, Max {bbox[1]}")
                print(f"{indent}  Start: {node.start}, End: {node.end}")
            
            if is_leaf:  # Leaf node
                print(f"{indent}  Leaf Node")
                print(f"{indent}  Triangles: {node.end - node.start}")
                for indx in range(node.start, node.end):
                    print(f"{indent}  Triangle {indx}: {self.geos[indx * 6: (indx + 1) * 6]}")
            else:
                left_index = self.indices[index * 4]
                right_index = self.indices[index * 4 + 1]
                if show_non_leaf_nodes:
                    print(f"{indent}  Left Child: {left_index}")
                    print(f"{indent}  Right Child: {right_index}")
                print_node(left_index, depth + 1)
                print_node(right_index, depth + 1)

        print("BVH Tree Structure:")
        print(f"Total nodes: {len(self.nodes)}")
        print(f"Indices: {self.indices}")
        print_node(0, 0)