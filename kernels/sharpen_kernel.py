import mlx.core as mx

def sharpen_kernel(image_buffer: mx.array):
    source = """
    uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;
    
    out[elem]     = image_buffer[elem];
    out[elem + 1] = image_buffer[elem + 1];
    out[elem + 2] = image_buffer[elem + 2];
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