from convex_hull import generate_convex_hull,random_vertices_by_fiber
from ortel import ortel
import numpy as np

def main():
    # ----------------------------
    # 1) Politopo aleatorio
    # ----------------------------
    d = 2
    z_vals = [0, 1]
    
    USE_RANDOM = False  # cambia a True para aleatorios

    if USE_RANDOM:
        verts = random_vertices_by_fiber(z_vals, d=d, n_per_z=40, seed=123)
    else:
        verts = np.array([
        [0, 0,   0],
        [0, 0,   1],
        [0, 0.5, 1],
        [1, 0,   0],
        [1, 0,   1],
        [1, 0.5, 1]
        ], dtype=float)

    d = 2
    z_vals = [0, 1]
    A, b = generate_convex_hull(verts)

    bestCP, bestF = ortel(
        A, b, d,
        z_vals=z_vals,
        N_cp=50,
        N_hip=1000,
        N=80000,
        tol=1e-9,
        seed=None,
        batch=5000,    # controla memoria
        guided=False
    )
    print("BestCP:", bestCP)
    print("F(S) ≈", bestF)


if __name__ == "__main__":
    main()

