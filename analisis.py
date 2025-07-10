import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def leer_fibra_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    A, b = [], []
    modo = 'A'
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            if 'b_z' in line:
                modo = 'b'
            continue
        if modo == 'A':
            A.append(list(map(float, line.split())))
        else:
            b = list(map(float, line.split()))
    return np.array(A), np.array(b)

def centro_interior(A, b):
    d = A.shape[1]
    res = linprog(np.zeros(d), A_ub=A, b_ub=b, method='highs')
    return res.x if res.success else None

def leer_Ab_txt(k, iteracion, carpeta):
    ruta = os.path.join(carpeta, f"Ab_k{k:03d}_iter{iteracion:03d}.txt")
    A, b = [], []
    leyendo_b = False
    with open(ruta, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if 'b' in line:
                    leyendo_b = True
                continue
            if leyendo_b:
                b = list(map(float, line.split()))
            else:
                A.append(list(map(float, line.split())))
    return np.array(A), np.array(b)

def graficar_fibras_2d(k, iteracion, carpeta):
    plt.figure(figsize=(8, 6))
    for z in range(k + 1):
        ruta = os.path.join(carpeta, f"fibras_k{k:03d}_iter{iteracion:03d}_z{z:02d}.txt")
        if not os.path.exists(ruta): continue
       A, b = leer_fibra_txt(ruta)
        interior = centro_interior(A, b)
        if interior is None: continue
        hs = np.hstack([-b[:, np.newaxis], A])
        hs_int = HalfspaceIntersection(hs, interior)
        hull = ConvexHull(hs_int.intersections)
        pts = hs_int.intersections[hull.vertices]
        plt.fill(*pts.T, alpha=0.4, label=f'z = {z}')
        plt.plot(*pts.T, 'ko-', alpha=0.5)
    plt.title(f"Fibras $P_z$ proyectadas (k={k}, iter={iteracion})")
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    path = os.path.join(carpeta, f"fig_fibras_k{k}_iter{iteracion}.png")
    plt.savefig(path, dpi=300)
    plt.close()

def graficar_hull_3d(k, iteracion, carpeta):
    A, b = leer_Ab_txt(k, iteracion, carpeta)
    if A.shape[1] != 3: return
    puntos = []
    for z in range(k + 1):
        for _ in range(500):
            p = np.random.rand(2)
            x = np.hstack(([z], p))
            if np.all(A @ x <= b + 1e-8):
                puntos.append(x)
    if not puntos: return
    puntos = np.array(puntos)
    hull = ConvexHull(puntos)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    for simplex in hull.simplices:
        tri = puntos[simplex]
        ax.add_collection3d(Poly3DCollection([tri], color='lightblue', alpha=0.6, edgecolor='gray'))
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], s=4, color='black', alpha=0.3)
    ax.set_xlabel('z')
    ax.set_ylabel('$x_1$')
    ax.set_zlabel('$x_2$')
    ax.set_title(f'Convex hull (k={k}, iter={iteracion})')
    plt.tight_layout()
    path = os.path.join(carpeta, f"fig_hull_k{k}_iter{iteracion}.png")
    plt.savefig(path, dpi=300)
    plt.close()

def graficar_volumenes(k, iteracion, carpeta):
    archivo = os.path.join(carpeta, f"volumenes_k{k:03d}_iter{iteracion:03d}.txt")
    if not os.path.exists(archivo): return
    with open(archivo) as f:
        lines = f.readlines()
    vol_z = []
    for line in lines:
        if 'Volumen fibra' in line:
            partes = line.strip().split(':')
            vol_z.append(float(partes[1]))
    if not vol_z: return
    plt.figure(figsize=(7, 5))
    z_vals = list(range(len(vol_z)))
    plt.bar(z_vals, vol_z, color='cornflowerblue', edgecolor='black')
    plt.title(f'Volumen por fibra (k={k}, iter={iteracion})')
    plt.xlabel('z')
    plt.ylabel('Volumen estimado')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(carpeta, f"fig_volumenes_k{k}_iter{iteracion}.png")
    plt.savefig(path, dpi=300)
    plt.close()

def graficar_histograma(k, carpeta):
    archivo = os.path.join(carpeta, f"resultados_k_{k:03d}.csv")
    if not os.path.exists(archivo): return
    df = pd.read_csv(archivo)
    Fs = df['F'].dropna().to_numpy()
    if len(Fs) == 0: return
    cota_teo = 1 / (2 * np.e)
    plt.figure(figsize=(10, 6))
    plt.hist(Fs, bins=30, color='skyblue', edgecolor='black', alpha=0.9)
    plt.axvline(cota_teo, color='red', linestyle='--', linewidth=2, label=f'Cota 1/(2e) ≈ {cota_teo:.4f}')
    plt.title(f'Distribución de $F(S)$ (k = {k}, n = {len(Fs)})')
    plt.xlabel('$F(S)$')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(carpeta, f"fig_histograma_f_k{k}.png")
    plt.savefig(path, dpi=300)
    plt.close()

def iteraciones_disponibles(k, carpeta):
    archivos = os.listdir(carpeta)
    patron = re.compile(rf'Ab_k{k:03d}_iter(\d+).txt')
    return sorted({int(m.group(1)) for f in archivos if (m := patron.match(f))})

def main(k, carpeta="salida_k"):
    print(f"\n🔍 Analizando resultados para k = {k}")
    iters = iteraciones_disponibles(k, carpeta)
    if not iters:
        print("No se encontraron iteraciones.")
        return
    graficar_histograma(k, carpeta)
    for iteracion in iters:
        graficar_fibras_2d(k, iteracion, carpeta)
        graficar_hull_3d(k, iteracion, carpeta)
        graficar_volumenes(k, iteracion, carpeta)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True)
    args = parser.parse_args()
    main(args.k)


#python analisis.py --k 4 --iter 0 
#python analisis.py --k 3 --iter 0 --plot_3d
#python analisis.py --k 3 --iter 0 --plot_fibras
