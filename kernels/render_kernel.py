import mlx.core as mx

def render_kernel(image_buffer: mx.array, camera_center: mx.array, pixel00_loc: mx.array, pixel_delta_u: mx.array, pixel_delta_v: mx.array, geos: mx.array):

    structures_source = ""
    get_ray_source = ""
    ray_color_source = ""
    triangle_hit_source = ""

    with open("kernels/metal/structures.metal", "r") as f:
        structures_source = f.read()
    with open("kernels/metal/get_ray.metal", "r") as f:
        get_ray_source = f.read()
    with open("kernels/metal/ray_color.metal", "r") as f:
        ray_color_source = f.read()
    with open("kernels/metal/triangle_hit.metal", "r") as f:    
        triangle_hit_source = f.read()
    header = structures_source + get_ray_source + triangle_hit_source + ray_color_source


    source = """
    uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;
    uint x = thread_position_in_grid.x;
    uint y = thread_position_in_grid.y;

    Ray ray = get_ray(  float2(float(x), float(y)), 
                        float3(camera_center[0], camera_center[1], camera_center[2]), 
                        float3(pixel00_loc[0], pixel00_loc[1], pixel00_loc[2]), 
                        float3(pixel_delta_u[0], pixel_delta_u[1], pixel_delta_u[2]), 
                        float3(pixel_delta_v[0], pixel_delta_v[1], pixel_delta_v[2]));

    float3 color = ray_color(ray, geos, geos_count, 0);

    out[elem]     = color[0];
    out[elem + 1] = color[1];
    out[elem + 2] = color[2];
    """
    kernel = mx.fast.metal_kernel(
        name="render_kernel",
        source=source,
        header=header,
    )
 

    outputs = kernel(
        inputs={
                "image_buffernp": image_buffer, 
                "camera_center" : camera_center,
                "pixel00_loc"   : pixel00_loc,
                "pixel_delta_u" : pixel_delta_u,
                "pixel_delta_v" : pixel_delta_v,
                "geos"          : geos,
                "geos_count"    : geos.shape[0],
                }, 
        template={"T": mx.float32}, 
        grid=(image_buffer.shape[0], image_buffer.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": image_buffer.shape},
        output_dtypes={"out": image_buffer.dtype},
        #verbose=True
    )
    return outputs["out"]
