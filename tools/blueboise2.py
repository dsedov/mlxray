import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def create_blue_noise(size, k, r):
    # Step 1: Create a size x size floating point array of zeroes
    texture = np.zeros((size, size), dtype=float)

    # Step 2: Create a vector of numbers from 0 to 1.0 of length size * size
    values = np.linspace(0, 1, size * size)
    #np.random.shuffle(values)

    # Create a mask to track empty spots
    mask = np.ones((size, size), dtype=bool)

    # Step 3: Place k first values randomly in the array
    print(f"Placing initial {k} points...")
    for i in range(k):
        while True:
            x, y = np.random.randint(0, size, 2)
            if mask[x, y]:
                texture[x, y] = values[i]
                mask[x, y] = False
                break
        if (i + 1) % 100 == 0:
            print(f"Placed {i + 1}/{k} initial points")

    # Display the initial placement
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(texture, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Initial {k} points")
    plt.colorbar()

    # Step 4: Place remaining values in the most empty spots
    total_points = size * size
    print(f"Placing remaining {total_points - k} points...")
    for i in range(k, total_points):
        # Apply Gaussian blur
        blurred = gaussian_filter(texture, sigma=r/3)  # sigma = r/3 is a good approximation

        # Find the most empty spot (lowest value in blurred image where mask is True)
        blurred[~mask] = np.inf  # Set non-empty spots to infinity
        x, y = np.unravel_index(np.argmin(blurred), blurred.shape)

        texture[x, y] = values[i]
        mask[x, y] = False  # Update the mask

        if (i + 1) % 100 == 0:
            print(f"Placed {i + 1}/{total_points} points ({((i + 1) / total_points * 100):.2f}%)")

    print("Blue noise generation complete!")

    # Display the final result
    plt.subplot(122)
    plt.imshow(texture, cmap='gray', vmin=0, vmax=1)
    plt.title("Final blue noise texture")
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    return texture

if __name__ == "__main__":
    size = 128  # Size of the texture
    k = 500    # Number of initial random placements
    r = 5     # Radius for Gaussian blur

    blue_noise = create_blue_noise(size, k, r)