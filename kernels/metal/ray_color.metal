float3 ray_color(Ray ray, const device float* geos, uint geos_count, uint depth) {  
    uint i = 0;
    for(i = 0; i < geos_count; i++) {
        float3 v0 = float3(geos[i * 18],     geos[i * 18 + 1], geos[i * 18 + 2]);
        float3 v1 = float3(geos[i * 18 + 3], geos[i * 18 + 4], geos[i * 18 + 5]);
        float3 v2 = float3(geos[i * 18 + 6], geos[i * 18 + 7], geos[i * 18 + 8]);
        float3 n0 = float3(geos[i * 18 + 9], geos[i * 18 + 10], geos[i * 18 + 11]);
        float3 n1 = float3(geos[i * 18 + 12], geos[i * 18 + 13], geos[i * 18 + 14]);
        float3 n2 = float3(geos[i * 18 + 15], geos[i * 18 + 16], geos[i * 18 + 17]);
        Interval ray_t {0.001, FLT_MAX};
        HitRecord hit_record = triangle_hit(ray, ray_t, v0, v1, v2, n0, n1, n2);
        if(hit_record.hit) {
            return abs(hit_record.normal);
        }
    }
    return float3(0.0, 0.0, 0.0);
}