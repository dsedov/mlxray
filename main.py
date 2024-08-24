import mlx.core as mx
def exp_elementwise(a: mx.array) :
    source = """
        uint elem = thread_position_in_grid.x;
        T tmp = inp[elem];
        out[elem] = metal::exp(tmp);
    """

    kernel = mx.fast.metal_kernel(
        name="myexp",
        source=source,
    )
    outputs = kernel(
        inputs={"inp": a}, 
        template={"T": mx. float32}, 
        grid=(a.size, 1, 1), 
        threadgroup=(256, 1, 1), 
        output_shapes={"out": a.shape},
        output_dtypes={"out": a.dtype},
    )
    return outputs ["out"]
a = mx.random.normal(shape=(4, 16)).astype(mx. float16)
b = exp_elementwise(a)
assert mx.allclose(b, mx.exp(a))