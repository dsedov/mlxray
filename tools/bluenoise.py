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
    def create_color_noise_x(self, depth):
        red = self.create_blue_noise_x(depth)
        green = self.create_blue_noise_x(depth)
        blue = self.create_blue_noise_x(depth)
        return np.stack([red, green, blue], axis=-1)
    def create_color_noise_3d(self):
        red = self.create_blue_noise_3d()
        green = self.create_blue_noise_3d()
        blue = self.create_blue_noise_3d()
        return np.stack([red, green, blue], axis=-1)
    def create_blue_noise_3d(self):
        texture = np.zeros((self.size, self.size, self.size), dtype=float)
        values = np.linspace(0, 1, self.size * self.size * self.size)
        mask = np.ones((self.size, self.size, self.size), dtype=bool)

        print(f"Placing initial {self.k} points in 3D...")
        for i in range(self.k):
            while True:
                x, y, z = np.random.randint(0, self.size, 3)
                if mask[x, y, z]:
                    texture[x, y, z] = values[i]
                    mask[x, y, z] = False
                    break
            if (i + 1) % 100 == 0:
                print(f"Placed {i + 1}/{self.k} initial points")

        total_points = self.size * self.size * self.size
        print(f"Placing remaining {total_points - self.k} points...")
        for i in range(self.k, total_points):
            blurred = gaussian_filter(texture, sigma=self.r/3)
            blurred[~mask] = np.inf
            x, y, z = np.unravel_index(np.argmin(blurred), blurred.shape)
            texture[x, y, z] = values[i]
            mask[x, y, z] = False
            if (i + 1) % 1000 == 0:
                print(f"Placed {i + 1}/{total_points} points ({((i + 1) / total_points * 100):.2f}%)")

        print("3D Blue noise generation complete!")
        return texture
    def create_blue_noise_x(self, depth):
        texture = np.zeros((self.size, self.size, depth), dtype=float)
        values = np.linspace(0, 1, self.size * self.size * depth)
        mask = np.ones((self.size, self.size, depth), dtype=bool)

        print(f"Placing initial {self.k} points in 3D...")
        for i in range(self.k):
            while True:
                x, y = np.random.randint(0, self.size, 2)
                z = np.random.randint(0, depth)
                if mask[x, y, z]:
                    texture[x, y, z] = values[i]
                    mask[x, y, z] = False
                    break
            if (i + 1) % 100 == 0:
                print(f"Placed {i + 1}/{self.k} initial points")

        total_points = self.size * self.size * depth
        print(f"Placing remaining {total_points - self.k} points...")
        for i in range(self.k, total_points):
            blurred = gaussian_filter(texture, sigma=self.r/3)
            blurred[~mask] = np.inf
            x, y, z = np.unravel_index(np.argmin(blurred), blurred.shape)
            texture[x, y, z] = values[i]
            mask[x, y, z] = False
            if (i + 1) % 1000 == 0:
                print(f"Placed {i + 1}/{total_points} points ({((i + 1) / total_points * 100):.2f}%)")

        print("3D Blue noise generation complete!")
        return texture
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

    def display_noise_3d(self, noise, title):
        if noise.ndim != 3:
            raise ValueError("Input noise must be 3D")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(title)

        # XY plane (middle slice)
        ax1.imshow(noise[:, :, noise.shape[2]//2], cmap='gray', vmin=0, vmax=1)
        ax1.set_title("XY Plane (Middle Slice)")
        ax1.axis('off')

        # XZ plane (middle slice)
        ax2.imshow(noise[:, noise.shape[1]//2, :], cmap='gray', vmin=0, vmax=1)
        ax2.set_title("XZ Plane (Middle Slice)")
        ax2.axis('off')

        # YZ plane (middle slice)
        ax3.imshow(noise[noise.shape[0]//2, :, :], cmap='gray', vmin=0, vmax=1)
        ax3.set_title("YZ Plane (Middle Slice)")
        ax3.axis('off')

        plt.tight_layout()
        plt.show()
        

if __name__ == "__main__":
    size = 64
    depth = 10
    k = 50
    r = 6

    generator = BlueNoiseGenerator(size, k, r)

    # Generate and save 3D blue noise
    noise_3d = generator.create_blue_noise_x(10)
    generator.save_noise(noise_3d, f"{size}x{size}x{depth}_3d_blue_noise.npy")
    print(f"3D Blue noise shape: {noise_3d.shape}")

    # Display three cross-sections of the 3D noise
    generator.display_noise_3d(noise_3d, "3D Blue Noise Cross-sections")

    exit(0)

    # Generate and display grayscale noise
    gray_noise = generator.create_blue_noise()
    generator.display_noise(gray_noise, "Grayscale Blue Noise")
    generator.save_noise(gray_noise, f"{size}x{size}_grayscale_blue_noise.npy")

    # Generate and display grayscale noise
    color_noise = generator.create_color_noise()
    generator.display_noise(color_noise, "Blue Noise")
    generator.save_noise(color_noise, f"{size}x{size}x3_blue_noise.npy")

    

    # Display a slice of the 3D noise
    generator.display_noise(noise_3d[:, :, size // 2], "3D Blue Noise (Middle Slice)")

