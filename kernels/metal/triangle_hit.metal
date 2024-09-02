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

HitRecord hit(Ray ray, Interval ray_t, const device float* geos, uint geos_count) {
    uint i = 0;
    HitRecord global_hit_record;

    for(i = 0; i < geos_count; i++) {
        float3 v0 = float3(geos[i * 18],      geos[i * 18 + 1],  geos[i * 18 + 2]);
        float3 v1 = float3(geos[i * 18 + 3],  geos[i * 18 + 4],  geos[i * 18 + 5]);
        float3 v2 = float3(geos[i * 18 + 6],  geos[i * 18 + 7],  geos[i * 18 + 8]);
        float3 n0 = float3(geos[i * 18 + 9],  geos[i * 18 + 10], geos[i * 18 + 11]);
        float3 n1 = float3(geos[i * 18 + 12], geos[i * 18 + 13], geos[i * 18 + 14]);
        float3 n2 = float3(geos[i * 18 + 15], geos[i * 18 + 16], geos[i * 18 + 17]);

        HitRecord hit_record = triangle_hit(ray, ray_t, v0, v1, v2, n0, n1, n2);
        if(hit_record.hit) {
            ray_t.max = hit_record.t;
            global_hit_record = hit_record;
        }
    }
    if(global_hit_record.hit) {
        return global_hit_record;
    }
    return HitRecord();
}