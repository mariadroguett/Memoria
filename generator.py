import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

"""
Generador de politopos a partir de uniones de fibras aleatorias.

Funciones:
- sample_points(...)
- generate_random_fiber(...)
- generate_convex_fiber_union(...)
"""


def sample_points(d, N, scale, dist_type='uniform'):
    """Sample N points in R^d from a specified distribution, scaled accordingly."""
    if dist_type == 'uniform':
        return np.random.uniform(-scale, scale, size=(N, d))
    elif dist_type == 'normal':
        return np.random.normal(0, scale, size=(N, d))
    elif dist_type == 'ball':
        vec = np.random.normal(0, 1, size=(N, d))
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        radii = np.random.uniform(0, scale, size=(N, 1))
        return vec * radii
    elif dist_type == 'simplex':
        exp_samples = np.random.exponential(1.0, size=(N, d + 1))
        points = exp_samples / exp_samples.sum(axis=1, keepdims=True)
        return (points[:, :-1] - 1/d) * scale
    else:
        raise ValueError(f"Unsupported distribution: '{dist_type}'")

def generate_random_fiber(z, d, N, scale, families=None, seed=None):
    """Generate a random fiber at level z in [0, scale]^d, from selected distribution family."""
    if seed is not None:
        np.random.seed(seed)
    if families is None or len(families) == 0:
        families = ['uniform', 'normal', 'ball', 'simplex']

    dist_type = np.random.choice(families)
    points = sample_points(d, N, scale, dist_type)
    return np.hstack([np.full((N, 1), z), points]), dist_type

def generate_convex_fiber_union(
    num_fibers, d, scale_array, N_array=None, families=None,
    seed=None, visualize=False, N_min=5, N_max=15
):
    """
    Generates a convex polytope as the convex hull of union of fibers 
    located at z-levels 0, 1, ..., num_fibers-1 in R^{d+1}.
    """
    if len(scale_array) != num_fibers:
        raise ValueError("Length of scale_array must match num_fibers")

    if families is None or len(families) == 0:
        families = ['uniform', 'normal', 'ball', 'simplex']

    if N_array is None:
        rng = np.random.default_rng(seed + 999 if seed is not None else None)
        N_array = rng.integers(N_min, N_max + 1, size=num_fibers)

    all_points = []
    dist_types_used = []

    for i in range(num_fibers):
        z = i 
        scale_i = scale_array[i]
        N_i = N_array[i]
        seed_i = seed + i if seed is not None else None

        fiber_points, dist_type = generate_random_fiber(z, d, N_i, scale_i, families=families, seed=seed_i)
        all_points.append(fiber_points)
        dist_types_used.append(dist_type)

    all_points = np.vstack(all_points)
    hull = ConvexHull(all_points)
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]

    if visualize and d == 2:
        plot_convex_hull_3d(all_points, hull, N_array)

    return all_points, hull, A, b, dist_types_used, N_array

def plot_convex_hull_3d(points, hull, N_array=None):
    """Plot 3D convex hull for d = 2 case (i.e., R^3)"""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap('tab10')

    start = 0
    for i, N in enumerate(N_array or [len(points)]):
        end = start + N
        color = cmap(i % 10)
        ax.scatter(points[start:end, 0], points[start:end, 1], points[start:end, 2],
                   color=color, label=f'Fiber {i}')
        start = end

    for simplex in hull.simplices:
        tri = points[simplex]
        ax.plot(tri[:, 0], tri[:, 1], tri[:, 2], 'k-', alpha=0.3)

    ax.set_xlabel("z (slice index)")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")
    ax.set_title("Convex hull of fiber union")
    ax.legend()
    plt.tight_layout()
    plt.show()
