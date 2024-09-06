import numpy as np
import matplotlib.pyplot as plt
def generate_bluenoise_texture(x):
    """
    Generates a blue noise texture of size x by x.

    Parameters:
    x (int): The size of the texture.

    Returns:
    np.ndarray: A float array of size x by x representing the blue noise texture.
    """
    # Initialize the texture with random values
    texture = np.random.rand(x, x).astype(np.float32)
    
    # Apply a high-pass filter to emphasize high-frequency components
    # This is a simple example; more sophisticated methods can be used
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
    bluenoise_texture = np.clip(np.convolve(texture.flatten(), kernel.flatten(), 'same').reshape(x, x), 0, 1)
    
    return bluenoise_texture


if __name__ == "__main__":
    texture = generate_bluenoise_texture(128)
    # show the texture
    plt.imshow(texture, cmap='gray')
    plt.show()