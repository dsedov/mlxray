Ray get_ray(float2 uv, float3 camera_center, float3 pixel00_loc, float3 pixel_delta_u, float3 pixel_delta_v, uint sample, uint samples, MetalRandom rand, const device float* blue_noise_texture, BlueNoiseRandom blue_noise_rand){
    Ray ray;

    // Calculate stratum size
    int sqrt_samples = ceil(sqrt(float(samples)));
    float stratum_size = 1.0f / sqrt_samples;
    float2 blue_noise = get_blue_noise_sample(sample, samples, blue_noise_texture);
    
    // Calculate stratum indices
    int stratum_x = sample % sqrt_samples;
    int stratum_y = sample / sqrt_samples;

    // Generate sample within the stratum
    float px = (stratum_x + blue_noise_rand.rand()) * stratum_size - 0.5f;
    float py = (stratum_y + blue_noise_rand.rand()) * stratum_size - 0.5f;
    px += blue_noise.x - 0.5f;
    py += blue_noise.y - 0.5f;
    px /= 2.0;
    py /= 2.0;

    uv += float2(px, py);
    float3 pixel_sample = pixel00_loc + uv.x * pixel_delta_u + uv.y * pixel_delta_v;
    ray.depth = 0;
    ray.origin = camera_center;
    ray.direction = pixel_sample - ray.origin;
    return ray;
}