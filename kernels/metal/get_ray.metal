Ray get_ray(float2 uv, float3 camera_center, float3 pixel00_loc, float3 pixel_delta_u, float3 pixel_delta_v, uint random_seed){
    Ray ray;
    random rand(uint(uv.x), uint(uv.y), random_seed);
    uv += float2(-0.5 +rand.rand(), -0.5 +rand.rand());
    float3 pixel_sample = pixel00_loc + uv.x * pixel_delta_u + uv.y * pixel_delta_v;
    ray.depth = 0;
    ray.origin = camera_center;
    ray.direction = pixel_sample - ray.origin;
    return ray;
}