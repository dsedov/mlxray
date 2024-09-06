import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def generate_bluenoise_texture(size, r=2, k=30):
    """
    Generates a blue noise texture using Poisson disk sampling.

    Parameters:
    size (int): The size of the texture (size x size).
    r (float): The minimum distance between points.
    k (int): Number of attempts to place each sample.

    Returns:
    np.ndarray: A float array of shape (size, size, 2) representing the blue noise texture.
    """
    def get_cell_indices(pt):
        return int(pt[0] // cell_size), int(pt[1] // cell_size)

    def get_neighbours(grid, pt):
        cell_x, cell_y = get_cell_indices(pt)
        for i in range(max(0, cell_x - 2), min(cell_x + 3, grid_size)):
            for j in range(max(0, cell_y - 2), min(cell_y + 3, grid_size)):
                if grid[i][j] is not None:
                    yield grid[i][j]

    def point_valid(pt, grid, r, width, height):
        if not (0 <= pt[0] < width and 0 <= pt[1] < height):
            return False
        cell_x, cell_y = get_cell_indices(pt)
        for nx, ny in get_neighbours(grid, pt):
            if np.hypot(nx - pt[0], ny - pt[1]) < r:
                return False
        return True

    width = height = size
    cell_size = r / np.sqrt(2)
    grid_size = int(np.ceil(size / cell_size))
    grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]

    points = []
    active = []

    # First point
    pt = (np.random.uniform(0, width), np.random.uniform(0, height))
    points.append(pt)
    active.append(pt)
    cell_x, cell_y = get_cell_indices(pt)
    grid[cell_x][cell_y] = pt

    # Generate points
    while active:
        idx = np.random.randint(0, len(active))
        pt = active[idx]
        found = False
        for _ in range(k):
            new_pt = (
                pt[0] + np.random.uniform(r, 2*r) * np.cos(np.random.uniform(0, 2*np.pi)),
                pt[1] + np.random.uniform(r, 2*r) * np.sin(np.random.uniform(0, 2*np.pi))
            )
            if point_valid(new_pt, grid, r, width, height):
                points.append(new_pt)
                active.append(new_pt)
                cell_x, cell_y = get_cell_indices(new_pt)
                grid[cell_x][cell_y] = new_pt
                found = True
                break
        if not found:
            active.pop(idx)

    # Convert points to a texture of float2 values
    points = np.array(points)
    tree = cKDTree(points)
    
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    pixel_coords = np.c_[yy.ravel(), xx.ravel()]
    
    _, indices = tree.query(pixel_coords)
    closest_points = points[indices]
    
    texture = closest_points.reshape(size, size, 2) / size  # Normalize to [0, 1] range

    # Add small random offsets to ensure uniqueness
    texture += np.random.uniform(-0.001, 0.001, texture.shape)
    texture = np.clip(texture, 0, 1)  # Ensure values stay in [0, 1] range

    return texture

def generate_1d_bluenoise(N, r=0.1):
    """
    Generates a 1D array of random floating-point numbers with blue noise distribution.

    Parameters:
    N (int): The length of the output array.
    r (float): The minimum distance between points (in the range [0, 1]).

    Returns:
    np.ndarray: A float array of length N with values between 0 and 1.
    """
    points = []
    active = []

    # First point
    pt = np.random.uniform(0, 1)
    points.append(pt)
    active.append(pt)

    while len(points) < N:
        if not active:
            # If we run out of active points, add a random point
            pt = np.random.uniform(0, 1)
            points.append(pt)
            active.append(pt)
            continue

        idx = np.random.randint(0, len(active))
        pt = active[idx]

        for _ in range(30):  # 30 attempts to place a new point
            new_pt = (pt + np.random.uniform(r, 2*r)) % 1  # Wrap around to [0, 1]
            if all(abs(new_pt - p) % 1 > r for p in points):
                points.append(new_pt)
                active.append(new_pt)
                break
        else:
            active.pop(idx)

    # If we don't have enough points, fill the rest randomly
    while len(points) < N:
        points.append(np.random.uniform(0, 1))

    # Shuffle the points to avoid any bias from the generation process
    np.random.shuffle(points)

    return np.array(points)

if __name__ == "__main__":
    # Test the 2D texture generation
    texture = generate_bluenoise_texture(128)
    print(f"2D Texture shape: {texture.shape}")
    print(f"texture[0, 0]: {texture[0, 0]}")
    print(f"texture[0, 1]: {texture[0, 1]}")
    print(f"texture[1, 0]: {texture[1, 0]}")

    # Test the 1D blue noise generation
    N = 1000
    blue_noise_1d = generate_1d_bluenoise(N)
    print(f"\n1D Blue Noise array length: {len(blue_noise_1d)}")
    print(f"First 5 values: {blue_noise_1d[:5]}")

    # Visualize the 1D blue noise
    plt.figure(figsize=(15, 5))
    
    plt.subplot(121)
    plt.hist(blue_noise_1d, bins=50, edgecolor='black')
    plt.title('1D Blue Noise Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    
    plt.subplot(122)
    plt.scatter(range(N), blue_noise_1d, s=1, alpha=0.5)
    plt.title('1D Blue Noise Scatter Plot')
    plt.xlabel('Index')
    plt.ylabel('Value')
    
    plt.tight_layout()
    plt.show()