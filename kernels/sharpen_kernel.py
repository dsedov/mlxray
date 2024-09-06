import mlx.core as mx

def sharpen_kernel(image_buffer: mx.array):
    source = """
    float sharpening_kernel[9] = {
        0.0f, -0.2f,  0.0f,
       -0.2f,  1.8f, -0.2f,
        0.0f, -0.2f,  0.0f
    };
    
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;
    uint width = threads_per_grid.x;
    uint height = threads_per_grid.y;
    
    if (x < width && y < height) {
        float3 sum = float3(0, 0, 0);
        
        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                int px = clamp(int(x) + kx, 0, int(width) - 1);
                int py = clamp(int(y) + ky, 0, int(height) - 1);
                uint idx = (px + py * width) * 3;
                float3 pixel = float3(image_buffer[idx], image_buffer[idx+1], image_buffer[idx+2]);
                sum += pixel * sharpening_kernel[(ky+1)*3 + (kx+1)];
            }
        }
        
        uint elem = (x + y * width) * 3;
        out[elem]     = clamp(sum.r, 0.0f, 1.0f);
        out[elem + 1] = clamp(sum.g, 0.0f, 1.0f);
        out[elem + 2] = clamp(sum.b, 0.0f, 1.0f);
    }
    """
    kernel = mx.fast.metal_kernel(
        name="sharpen_kernel",
        source=source
    )

    outputs = kernel(
        inputs={
                "image_buffer": image_buffer, 
                }, 
        template={"T": mx.float32}, 
        grid=(image_buffer.shape[0], image_buffer.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": image_buffer.shape},
        output_dtypes={"out": image_buffer.dtype},
    )
    return outputs["out"]