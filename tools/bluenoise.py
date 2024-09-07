import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os

class BlueNoiseGenerator:
    def __init__(self, size, k, r):
        self.size = size
        self.k = k
        self.r = r

    def create_blue_noise(self):
        texture = np.zeros((self.size, self.size), dtype=float)
        values = np.linspace(0, 1, self.size * self.size)
        mask = np.ones((self.size, self.size), dtype=bool)

        print(f"Placing initial {self.k} points...")
        for i in range(self.k):
            while True:
                x, y = np.random.randint(0, self.size, 2)
                if mask[x, y]:
                    texture[x, y] = values[i]
                    mask[x, y] = False
                    break
            if (i + 1) % 100 == 0:
                print(f"Placed {i + 1}/{self.k} initial points")

        total_points = self.size * self.size
        print(f"Placing remaining {total_points - self.k} points...")
        for i in range(self.k, total_points):
            blurred = gaussian_filter(texture, sigma=self.r/3)
            blurred[~mask] = np.inf
            x, y = np.unravel_index(np.argmin(blurred), blurred.shape)
            texture[x, y] = values[i]
            mask[x, y] = False
            if (i + 1) % 100 == 0:
                print(f"Placed {i + 1}/{total_points} points ({((i + 1) / total_points * 100):.2f}%)")

        print("Blue noise generation complete!")
        return texture

    def create_color_noise(self):
        red = self.create_blue_noise()
        green = self.create_blue_noise()
        blue = self.create_blue_noise()
        return np.stack([red, green, blue], axis=-1)

    def save_noise(self, noise, filename):
        np.save(filename, noise)
        print(f"Noise saved as {filename}")

    def load_noise(self, filename):
        return np.load(filename)

    def display_noise(self, noise, title):
        plt.figure(figsize=(10, 10))
        if noise.ndim == 2:
            plt.imshow(noise, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(noise)
        plt.title(title)
        plt.colorbar()
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    size = 256
    k = 100
    r = 5

    generator = BlueNoiseGenerator(size, k, r)

    # Generate and display grayscale noise
    color_noise = generator.create_color_noise()
    generator.display_noise(color_noise, "Blue Noise")
    generator.save_noise(color_noise, f"{size}x{size}x3_blue_noise.npy")

    # Generate and display grayscale noise
    gray_noise = generator.create_blue_noise()
    generator.display_noise(gray_noise, "Grayscale Blue Noise")
    generator.save_noise(gray_noise, f"{size}x{size}_grayscale_blue_noise.npy")
