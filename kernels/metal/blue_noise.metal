
thread float2 get_blue_noise_sample(uint sample, const device float* blue_noise_texture) {
    const int TEXTURE_SIZE = 256; // Size of the blue noise texture

    // Use the sample index to look up in the blue noise texture
    uint x = sample % TEXTURE_SIZE;
    uint y = (sample / TEXTURE_SIZE) % TEXTURE_SIZE;
    uint index = y * TEXTURE_SIZE + x;

    // Fetch the blue noise value
    float2 blue_noise = float2(blue_noise_texture[index * 2], blue_noise_texture[index * 2 + 1]);
    return blue_noise;



    // Apply scrambling to avoid repeating patterns
    float2 scramble = float2(
        fract(sin(float(sample) * 12.9898) * 43758.5453),
        fract(sin(float(sample) * 78.233) * 43758.5453)
    );

    // Combine blue noise with scrambling
    return fract(blue_noise + scramble);
}
thread float3 get_blue_noise_sample_3d(uint sample, const device float* blue_noise_texture) {
    uint TEXTURE_SIZE = 256; // Size of the blue noise texture

    // Use the sample index to look up in the blue noise texture
    uint x = sample % TEXTURE_SIZE;
    uint y = (sample / TEXTURE_SIZE) % TEXTURE_SIZE;

    uint index = (x + y * TEXTURE_SIZE) * 3;

    // Fetch the blue noise value
    float3 blue_noise = float3(blue_noise_texture[index], blue_noise_texture[index + 1], blue_noise_texture[index + 2]);
    return blue_noise;
}
thread float3 get_blue_noise_in_unit_sphere(uint sample, const device float* blue_noise_texture) {
    float3 p;
    uint k = sample;
    do {    
        float3 blue_noise = get_blue_noise_sample_3d(k, blue_noise_texture);
        p = 2.0 * blue_noise - 1.0;
        k += 1;
    } while (dot(p, p) >= 1.0);
    return p;
}
thread float3 get_blue_noise_unit_vector(uint sample, const device float* blue_noise_texture) {
    return normalize(get_blue_noise_in_unit_sphere(sample, blue_noise_texture));
}
thread float3 get_blue_noise_on_hemisphere(float3 normal, uint sample, const device float* blue_noise_texture) {
    float3 blue_noise = get_blue_noise_unit_vector(sample, blue_noise_texture);
    if (dot(blue_noise, normal) > 0.0) {
        return blue_noise;
    } else {
        return -blue_noise;
    }
}