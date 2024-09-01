HitRecord triangle_hit(Ray ray, Interval ray_t, float3 v0, float3 v1, float3 v2, float3 n0, float3 n1, float3 n2) {
    HitRecord hit_record;
    hit_record.hit = false;

    float EPSILON = 1e-8;
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