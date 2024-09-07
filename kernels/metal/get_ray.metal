Ray get_ray(float2 uv, 
            float3 camera_center, 
            float3 pixel00_loc, 
            float3 pixel_delta_u, 
            float3 pixel_delta_v, 
            uint sample, 
            const device float* blue_noise_texture){
    Ray ray;

    float2 blue_noise_offset = get_blue_noise_sample(sample, blue_noise_texture);
    // Use blue noise offset directly
    float px = blue_noise_offset.x - 0.5f;
    float py = blue_noise_offset.y - 0.5f;

    uv += float2(px, py);
    float3 pixel_sample = pixel00_loc + uv.x * pixel_delta_u + uv.y * pixel_delta_v;
    ray.depth = 0;
    ray.origin = camera_center;
    ray.direction = pixel_sample - ray.origin;
    return ray;
}