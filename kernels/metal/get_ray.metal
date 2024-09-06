Ray get_ray(float2 uv, 
            float3 camera_center, 
            float3 pixel00_loc, 
            float3 pixel_delta_u, 
            float3 pixel_delta_v, 
            uint sample, 
            uint samples,
            uint random_seed,
            const device float* blue_noise_texture,
            uint blue_noise_texture_size){
    Ray ray;

    // Calculate the index in the blue noise texture
    uint x = (sample + random_seed) % blue_noise_texture_size;
    uint y = ((sample + random_seed) / blue_noise_texture_size) % blue_noise_texture_size;
    uint blue_noise_index = (y * blue_noise_texture_size + x) * 3;

    // Extract float2 from blue noise texture
    float2 blue_noise_offset = float2(blue_noise_texture[blue_noise_index],
                                      blue_noise_texture[blue_noise_index + 1]);

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