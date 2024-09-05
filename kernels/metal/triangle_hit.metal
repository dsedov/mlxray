/*
 BVH INDEX LAYOUT
  [0] - left index
  [1] - right index
  [2] - parent index
  [3] - depth
  [4] - is_leaf
*/
class AABB {
public:
    AABB() {}
    AABB(float3 a, float3 b) : minimum(a), maximum(b) {}

    float3 minimum;
    float3 maximum;
};

bool intersect_aabb(Ray ray, AABB aabb, Interval ray_t) {
    float3 t_min = (aabb.minimum - ray.origin) / ray.direction;
    float3 t_max = (aabb.maximum - ray.origin) / ray.direction;
    
    float3 t1 = min(t_min, t_max);
    float3 t2 = max(t_min, t_max);
    
    float t_near = max(max(t1.x, t1.y), t1.z);
    float t_far = min(min(t2.x, t2.y), t2.z);
    
    return t_near <= t_far && t_far >= ray_t.min && t_near <= ray_t.max;
}

class BVH {
public:
    BVH() : index(-1), geos(nullptr), bboxes(nullptr), indices(nullptr), polygon_indices(nullptr) {}
    
    BVH(
        int index, 
        const device float* geos, 
        const device float* bboxes, 
        const device int*   indices, 
        const device int*   polygon_indices
    ) {
        this->index = index;
        this->geos = geos;
        this->bboxes = bboxes;
        this->indices = indices;
        this->polygon_indices = polygon_indices;
    }

    void init(
        int index, 
        const device float* geos, 
        const device float* bboxes, 
        const device int*   indices, 
        const device int*   polygon_indices
    ) {
        this->index = index;
        this->geos = geos;
        this->bboxes = bboxes;
        this->indices = indices;
        this->polygon_indices = polygon_indices;
    }

    BVH left() {
        return BVH(indices[index * 5], geos, bboxes, indices, polygon_indices);
    }

    BVH right() {
        return BVH(indices[index * 5 + 1], geos, bboxes, indices, polygon_indices);
    }

    bool is_leaf() {
        return indices[index * 5 + 4] == 1;
    }

    AABB get_bbox() {
        return AABB(
            float3(bboxes[index * 6], bboxes[index * 6 + 1], bboxes[index * 6 + 2]),
            float3(bboxes[index * 6 + 3], bboxes[index * 6 + 4], bboxes[index * 6 + 5])
        );
    }

    int get_polygon_index_start() {
        return indices[index * 5];
    }

    int get_polygon_count() {
        return indices[index * 5 + 1];
    }

    int get_index() {
        return index;
    }

private:
    int index;
    const device float* geos;
    const device float* bboxes;
    const device int* indices;
    const device int* polygon_indices;
};

HitRecord triangle_hit(Ray ray, Interval ray_t, float3 v0, float3 v1, float3 v2, float3 n0, float3 n1, float3 n2) {
    HitRecord hit_record;
    hit_record.hit = false;

    float EPSILON = 1e-9;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(ray.direction, edge2);
    float a = dot(edge1, h);

    if (a > -EPSILON && a < EPSILON){
        return hit_record;
    }
    
    float f = 1.0 / a;
    float3 s = ray.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return hit_record;
    }
    float3 q = cross(s, edge1);
    float v = f * dot(ray.direction, q);

    if (v < 0.0 || u + v > 1.0){
        return hit_record;
    }

    float t = f * dot(edge2, q);
    if(t > ray_t.min && t < ray_t.max) {
        
        hit_record.hit = true;
        hit_record.t = t;
        hit_record.p = ray.origin + t * ray.direction;
        float w = 1.0 - u - v;
        hit_record.normal = normalize(w * n0 + u * n1 + v * n2);
        hit_record.front_face = dot(ray.direction, hit_record.normal) < 0.0;
        return hit_record;
    }

    return hit_record;
}

HitRecord hit(Ray ray, Interval ray_t, const device float* geos, const device float* norms, const device float* bboxes, const device int* indices, const device int* polygon_indices) {
    BVH root;
    root.init(0, geos, bboxes, indices, polygon_indices);
    HitRecord global_hit_record;
    global_hit_record.hit = false;

    // Stack-based traversal
    BVH stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = root;

    while (stack_ptr > 0) {
        BVH node = stack[--stack_ptr];
        
        if (!intersect_aabb(ray, node.get_bbox(), ray_t)) {
            continue;
        }

        if (node.is_leaf()) {
            int polygon_index_start = node.get_polygon_index_start();
            int polygon_count = node.get_polygon_count();

            for (int i = 0; i < polygon_count; i++) {
                int idx = polygon_indices[polygon_index_start + i];
                float3 v0 = float3(geos[idx * 9],     geos[idx * 9 + 1],  geos[idx * 9 + 2]);
                float3 v1 = float3(geos[idx * 9 + 3], geos[idx * 9 + 4],  geos[idx * 9 + 5]);
                float3 v2 = float3(geos[idx * 9 + 6], geos[idx * 9 + 7],  geos[idx * 9 + 8]);
                float3 n0 = float3(norms[idx * 9],     norms[idx * 9 + 1],  norms[idx * 9 + 2]);
                float3 n1 = float3(norms[idx * 9 + 3], norms[idx * 9 + 4],  norms[idx * 9 + 5]);
                float3 n2 = float3(norms[idx * 9 + 6], norms[idx * 9 + 7],  norms[idx * 9 + 8]);

                HitRecord hit_record = triangle_hit(ray, ray_t, v0, v1, v2, n0, n1, n2);
                if (hit_record.hit && hit_record.t < ray_t.max) {
                    ray_t.max = hit_record.t;
                    global_hit_record = hit_record;
                }
            }
        } else {
            stack[stack_ptr++] = node.right();
            stack[stack_ptr++] = node.left();
        }
    }

    return global_hit_record;
}