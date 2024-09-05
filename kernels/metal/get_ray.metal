Ray get_ray(float2 uv, float3 camera_center, float3 pixel00_loc, float3 pixel_delta_u, float3 pixel_delta_v, uint sample, uint samples, MetalRandom rand){
    Ray ray;

    // Calculate stratum size
    int sqrt_samples = ceil(sqrt(float(samples)));
    float stratum_size = 1.0f / sqrt_samples;
    
    // Calculate stratum indices
    int stratum_x = sample % sqrt_samples;
    int stratum_y = sample / sqrt_samples;
    
    // Generate sample within the stratum
    float px = (stratum_x + rand.rand_float()) * stratum_size - 0.5f;
    float py = (stratum_y + rand.rand_float()) * stratum_size - 0.5f;
    uv += float2(px, py);
    float3 pixel_sample = pixel00_loc + uv.x * pixel_delta_u + uv.y * pixel_delta_v;
    ray.depth = 0;
    ray.origin = camera_center;
    ray.direction = pixel_sample - ray.origin;
    return ray;
}