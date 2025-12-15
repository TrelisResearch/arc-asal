"""
Lenia - A continuous cellular automaton simulation
"""

import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Lenia:
    """Lenia continuous cellular automaton."""

    def __init__(self, size=128, dt=0.1, params=None):
        """
        Initialize Lenia simulation.

        Args:
            size: Grid size (size x size)
            dt: Time step
            params: Dictionary of parameters (mu, sigma, beta for growth function,
                   kernel_radius, kernel_peaks for kernel)
        """
        self.size = size
        self.dt = dt

        # Default parameters - tuned for interesting dynamics
        default_params = {
            'mu': 0.14,      # Growth function center
            'sigma': 0.028,  # Growth function width (wider for more activity)
            'kernel_radius': 13,  # Kernel radius
            'kernel_peaks': [1.0],  # Kernel peak values (for multi-ring kernels)
            'kernel_beta': [0.3],   # Kernel ring widths
        }

        self.params = default_params if params is None else {**default_params, **params}

        # Initialize grid with seed pattern
        self.grid = np.zeros((size, size))
        self._init_seed()

        # Build convolution kernel
        self.kernel = self._build_kernel()

    def _init_seed(self, seed_type='disk', seed_radius=12):
        """
        Initialize with a fixed geometric seed pattern in the center.

        Args:
            seed_type: Type of seed ('disk', 'ring', 'gaussian')
            seed_radius: Radius of the seed pattern
        """
        center = self.size // 2
        y, x = np.ogrid[:self.size, :self.size]
        dist = np.sqrt((x - center)**2 + (y - center)**2)

        if seed_type == 'disk':
            # Solid disk with smooth edges
            self.grid = np.clip(1.0 - dist / seed_radius, 0, 1)
        elif seed_type == 'ring':
            # Ring/annulus shape
            inner = seed_radius * 0.5
            outer = seed_radius
            ring = np.exp(-((dist - (inner + outer) / 2) ** 2) / (seed_radius * 0.3) ** 2)
            self.grid = ring
        elif seed_type == 'gaussian':
            # Gaussian blob
            self.grid = np.exp(-(dist ** 2) / (2 * (seed_radius * 0.5) ** 2))

    def _build_kernel(self):
        """Build the convolution kernel."""
        r = self.params['kernel_radius']
        size = 2 * r + 1

        # Create distance matrix
        x, y = np.meshgrid(np.arange(size) - r, np.arange(size) - r)
        dist = np.sqrt(x**2 + y**2) / r

        # Multi-ring kernel using exponential bumps
        kernel = np.zeros((size, size))
        peaks = self.params['kernel_peaks']
        betas = self.params['kernel_beta']

        n_rings = len(peaks)
        for i, (peak, beta) in enumerate(zip(peaks, betas)):
            # Ring center position (0 to 1)
            ring_pos = (i + 0.5) / n_rings
            # Exponential bump at ring position
            kernel += peak * np.exp(-((dist - ring_pos) ** 2) / (2 * beta ** 2))

        # Mask to circular region
        kernel[dist > 1] = 0

        # Normalize
        kernel = kernel / kernel.sum()

        return kernel

    def growth_function(self, u):
        """Growth function - determines how cells grow based on neighborhood."""
        mu = self.params['mu']
        sigma = self.params['sigma']

        # Gaussian growth function centered at mu
        # Modified to have growth=0 at u=0 (empty cells don't die)
        growth = 2 * np.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1

        # Cells with no neighbors (u~0) should stay dead, not die more
        # This prevents empty space from pulling down filled cells
        return np.where(u < 0.01, 0, growth)

    def step(self):
        """Perform one simulation step."""
        # Compute neighborhood sums via convolution
        potential = convolve2d(self.grid, self.kernel, mode='same', boundary='wrap')

        # Apply growth function
        growth = self.growth_function(potential)

        # Update grid
        self.grid = np.clip(self.grid + self.dt * growth, 0, 1)

        return self.grid

    def run(self, steps=100):
        """Run simulation for given number of steps."""
        for _ in range(steps):
            self.step()
        return self.grid

    def get_snapshot(self):
        """Get current grid state."""
        return self.grid.copy()

    def set_params(self, params):
        """Update parameters and rebuild kernel."""
        self.params.update(params)
        self.kernel = self._build_kernel()

    def reset(self, seed_type='disk', seed_radius=12):
        """Reset grid with a seed pattern."""
        self.grid = np.zeros((self.size, self.size))
        self._init_seed(seed_type, seed_radius)


def visualize_lenia(lenia, steps=500, interval=50, save_path=None):
    """
    Visualize Lenia simulation.

    Args:
        lenia: Lenia instance
        steps: Number of steps to simulate
        interval: Animation interval in ms
        save_path: Optional path to save animation
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(lenia.grid, cmap='viridis', vmin=0, vmax=1)
    ax.set_title('Lenia Simulation')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    def animate(frame):
        lenia.step()
        im.set_array(lenia.grid)
        ax.set_title(f'Lenia Simulation - Step {frame}')
        return [im]

    anim = animation.FuncAnimation(fig, animate, frames=steps, interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer='pillow', fps=20)
        print(f"Animation saved to {save_path}")

    plt.show()
    return anim


def save_snapshot(lenia, path='snapshot.png'):
    """Save current state as image."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(lenia.grid, cmap='viridis', vmin=0, vmax=1)
    ax.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Snapshot saved to {path}")
    return path


if __name__ == '__main__':
    print("Creating Lenia simulation...")

    # Create simulation with default parameters
    lenia = Lenia(size=128, dt=0.1)

    print(f"Grid size: {lenia.size}x{lenia.size}")
    print(f"Parameters: {lenia.params}")

    # Run for a few steps to check it works
    print("\nRunning 100 steps...")
    lenia.run(steps=100)

    print(f"Grid min: {lenia.grid.min():.4f}, max: {lenia.grid.max():.4f}")
    print(f"Grid mean: {lenia.grid.mean():.4f}")

    # Save a snapshot
    save_snapshot(lenia, 'lenia_snapshot.png')

    print("\nSimulation working! Run with --animate flag to see live visualization.")

    # Check for animate flag
    import sys
    if '--animate' in sys.argv:
        print("\nStarting animation...")
        visualize_lenia(lenia, steps=300, interval=50)
