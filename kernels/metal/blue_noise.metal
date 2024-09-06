
float2 get_blue_noise_sample(uint sample, uint samples, const device float2* blue_noise_texture) {
    const int TEXTURE_SIZE = 128; // Size of the blue noise texture

    // Use the sample index to look up in the blue noise texture
    uint x = sample % TEXTURE_SIZE;
    uint y = (sample / TEXTURE_SIZE) % TEXTURE_SIZE;
    uint index = y * TEXTURE_SIZE + x;

    // Fetch the blue noise value
    float2 blue_noise = blue_noise_texture[index];

    // Apply scrambling to avoid repeating patterns
    float2 scramble = float2(
        fract(sin(float(sample) * 12.9898) * 43758.5453),
        fract(sin(float(sample) * 78.233) * 43758.5453)
    );

    // Combine blue noise with scrambling
    return fract(blue_noise + scramble);
}