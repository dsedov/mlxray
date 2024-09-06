
#define MAX_DEPTH 20
struct HitRecordStack {
    HitRecord hit_records[MAX_DEPTH];
    uint count;
};

float3 ray_color(Ray ray, const device float* geos, const device float* norms, const device float* bboxes, const device int* indices, const device int* polygon_indices, MetalRandom rand, BlueNoiseRandom blue_noise_rand) { 
    HitRecordStack hit_record_stack;
    hit_record_stack.count = 0;

    HitRecord hit_record = hit(ray, Interval{0.1, 10000.0}, geos, norms, bboxes, indices, polygon_indices);
    if (!hit_record.hit) {
        return float3(0.0, 0.0, 0.0);
    }

    hit_record_stack.hit_records[hit_record_stack.count++] = hit_record;

    for (uint i = 1; i < MAX_DEPTH; i++) {
        float3 direction = blue_noise_rand.random_on_hemisphere(-hit_record.normal);
        hit_record = hit(Ray{hit_record.p, direction}, Interval{0.0001, 10000.0}, geos, norms, bboxes, indices, polygon_indices);
        if (hit_record.hit) {
            hit_record_stack.hit_records[hit_record_stack.count++] = hit_record;
        } else {
            break;
        }
    }

    float3 color = float3(1.0, 1.0, 1.0);
    for (uint i = hit_record_stack.count; i > 0; i--) {
        color *= 0.8;
    }
    return color;
}