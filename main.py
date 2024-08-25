import mlx.core as mx
import numpy as np
def random_colors(a: mx.array):
    source = """
        uint seed = thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x ;
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;

        float r = float(thread_position_in_grid.x) / 1024.0;
        float g = float(thread_position_in_grid.y) / 768.0;
        float b = float((seed >> 16) & 0xff) / 255.0;

        uint elem = (thread_position_in_grid.x + thread_position_in_grid.y * threads_per_grid.x) * 3;

        out[elem] = r;
        out[elem + 1] = g;
        out[elem + 2] = b;
    """
    kernel = mx.fast.metal_kernel(
        name="fill_image",
        source=source,
    )
    outputs = kernel(
        inputs={"inp": a}, 
        template={"T": mx. float32}, 
        grid=(a.shape[0], a.shape[1], 1), 
        threadgroup=(256,1, 1), 
        output_shapes={"out": a.shape},
        output_dtypes={"out": a.dtype},
    )
    return outputs ["out"]

image_buffer = mx.zeros([1024,768,3], dtype=mx.float32)
print(image_buffer.shape)

image_buffer = random_colors(image_buffer)
np_image_buffer = np.array(image_buffer)

from PIL import Image

# Convert the numpy array to uint8 and reshape it
image_data = (np_image_buffer * 255).astype(np.uint8)
image_data = image_data.reshape((768, 1024, 3))

# Create a PIL Image from the numpy array
image = Image.fromarray(image_data)

# Display the image
image.show()


print(np_image_buffer)
