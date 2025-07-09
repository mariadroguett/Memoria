import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

def sample_points(d, N, scale, rng=None):
    """Muestra N puntos uniformemente en [-scale, scale]^d"""
    rng = rng or np.random.default_rng()
    return rng.uniform(-scale, scale, size=(N, d))

def generate_random_fiber(z, d, N, scale, rng=None):
    """Genera una fibra en altura z con puntos distribuidos uniformemente."""
    rng = rng or np.random.default_rng()
    points = sample_points(d, N, scale, rng)
    fiber = np.hstack([np.full((N, 1), z), points])
    return fiber

def generate_convex_fiber_union(
    num_fibers, d, scale_array, N_array=None,
    seed=None, visualize=False, N_min=5, N_max=15
):
    """
    Genera un politopo como la envolvente convexa de uniones de fibras aleatorias
    distribuidas uniformemente. Todas las fibras están en niveles z = 0, 1, ..., K.
    """
    if len(scale_array) != num_fibers:
        raise ValueError("scale_array debe tener num_fibers elementos")

    rng = np.random.default_rng(seed)

    if N_array is None:
        N_array = rng.integers(N_min, N_max + 1, size=num_fibers)

    all_points = []

    for i in range(num_fibers):
        z = i
        N_i = N_array[i]
        scale_i = scale_array[i]

        fiber_points = generate_random_fiber(z, d, N_i, scale_i, rng)
        all_points.append(fiber_points)

    all_points = np.vstack(all_points)
    hull = ConvexHull(all_points, qhull_options='QJ' )
    A = hull.equations[:, :-1]
    b = -hull.equations[:, -1]

    if visualize and d == 2:
        plot_convex_hull_3d(all_points, hull, N_array)

    return all_points, hull, A, b, N_array

def plot_convex_hull_3d(points, hull, N_array=None):
    """Grafica la convex hull en 3D para el caso d = 2 (R^3)."""
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fibers', type=int, default=4, help="Número de fibras")
    parser.add_argument('--d', type=int, default=2, help="Dimensión de la fibra")
    parser.add_argument('--scale', type=float, default=1.0, help="Escala para cada fibra")
    parser.add_argument('--seed', type=int, default=42, help="Semilla")
    parser.add_argument('--visualize', action='store_true', help="Visualizar convex hull")
    args = parser.parse_args()

    scale_array = [args.scale] * args.fibers
    generate_convex_fiber_union(
        num_fibers=args.fibers,
        d=args.d,
        scale_array=scale_array,
        seed=args.seed,
        visualize=args.visualize
    )
