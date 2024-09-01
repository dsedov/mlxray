float3 ray_color(Ray ray, const device float* geos, uint geos_count, uint depth) {  
    uint i = 0;

    HitRecord hit_record = hit(ray, Interval{0.1, 10000.0}, geos, geos_count);
    if(!hit_record.hit) {
        return float3(0.0, 0.0, 0.0);
    }
    return 0.5 * (hit_record.normal + float3(1.0, 1.0, 1.0));
}