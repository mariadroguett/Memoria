import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from scipy.spatial import HalfspaceIntersection, ConvexHull
from scipy.optimize import linprog
from mpl_toolkits.mplot3d import Axes3D  # necesario para proyección 3D


def plot_histograma_F(k):
    archivo = f"salida_k/resultados_k_{k:03d}.csv"
    if not os.path.exists(archivo):
        print(f"No se encontró: {archivo}")
        return

    df = pd.read_csv(archivo)
    if 'F' not in df.columns:
        print(f"No hay columna F en el archivo.")
        return

    Fs = df['F'].dropna().to_numpy()
    n_total = len(Fs)

    promedio = np.mean(Fs)
    mediana = np.median(Fs)
    std_dev = np.std(Fs)
    minimo = np.min(Fs)
    maximo = np.max(Fs)
    q1 = np.percentile(Fs, 25)
    q3 = np.percentile(Fs, 75)

    cota_teorica = 1 / (2 * np.e)
    cota_alta = 0.22
    cota_baja = 0.16

    n_bajo = np.sum(Fs < cota_baja)
    n_medio = np.sum((Fs >= cota_baja) & (Fs <= cota_alta))
    n_alto = np.sum(Fs > cota_alta)

    print(f" Estadísticas para k = {k} ({n_total} muestras):")
    print(f"  ▸ Promedio     : {promedio:.6f}")
    print(f"  ▸ Mediana      : {mediana:.6f}")
    print(f"  ▸ Std. Dev.    : {std_dev:.6f}")
    print(f"  ▸ Mínimo       : {minimo:.6f}")
    print(f"  ▸ Máximo       : {maximo:.6f}")
    print(f"  ▸ Q1 / Q3      : {q1:.6f} / {q3:.6f}")
    print()
    print(f"Resultados para k={k} (n={n_total}):")
    print(f"  F < 0.16: {n_bajo} ({n_bajo/n_total:.1%})")
    print(f"  0.16 ≤ F ≤ 0.22: {n_medio} ({n_medio/n_total:.1%})")
    print(f"  F > 0.22: {n_alto} ({n_alto/n_total:.1%})")
    print(f"  Cota teórica 1/(2e) ≈ {cota_teorica:.4f}")

    plt.figure(figsize=(10, 6))
    plt.hist(Fs, bins=40, color='skyblue', edgecolor='black')
    plt.axvline(cota_teorica, color='red', linestyle='--', linewidth=2, label=f'Cota 1/(2e) ≈ {cota_teorica:.4f}')
    plt.axvline(cota_alta, color='orange', linestyle='--', linewidth=2, label='Cota alta 0.22')
    plt.axvline(cota_baja, color='green', linestyle='--', linewidth=2, label='Cota baja 0.16')
    plt.title(f'Distribución de F para k = {k} (n = {n_total})')
    plt.xlabel('F')
    plt.ylabel('Frecuencia')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def leer_Ab_txt(k, iteracion):
    carpeta = "salida_k"
    ruta = os.path.join(carpeta, f"Ab_k{k:03d}_iter{iteracion:03d}.txt")

    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró: {ruta}")

    A = []
    b = []
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

    A = np.array(A)
    b = np.array(b)

    if A.size == 0 or b.size == 0:
        raise ValueError(f"A o b están vacíos en archivo {ruta} (posible error de escritura)")

    if A.shape[0] != b.size:
        raise ValueError(f"Dimensiones incompatibles entre A ({A.shape}) y b ({b.shape}) en {ruta}")

    return A, b


def centro_interior(A, b):
    d = A.shape[1]
    c = np.zeros(d)
    res = linprog(c, A_ub=A, b_ub=b)
    if res.success:
        return res.x
    else:
        return None


def leer_fibra_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    A = []
    b = []
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


def graficar_fibra_2d(A, b, z):
    interior = centro_interior(A, b)
    if interior is None:
        print(f"No se pudo encontrar punto interior en z = {z}")
        return
    hs = np.hstack([-b[:, np.newaxis], A])
    hs_int = HalfspaceIntersection(hs, interior)
    hull = ConvexHull(hs_int.intersections)
    plt.fill(*hs_int.intersections[hull.vertices].T, alpha=0.5, label=f'z = {z}')
    plt.plot(*hs_int.intersections[hull.vertices].T, 'ko-')


def graficar_todas_fibras_2d(k, iteracion):
    carpeta = "salida_k"
    plt.figure(figsize=(8, 6))
    for z in range(k + 1):
        archivo = os.path.join(carpeta, f"fibras_k{k:03d}_iter{iteracion:03d}_z{z:02d}.txt")
        if not os.path.exists(archivo):
            print(f"No existe: {archivo}")
            continue
        A, b = leer_fibra_txt(archivo)
        graficar_fibra_2d(A, b, z)
    plt.title(f"Lonjas (fibras) proyectadas para K={k}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def graficar_convex_hull_3d(k, iteracion):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    A, b = leer_Ab_txt(k, iteracion)
    d = A.shape[1]
    if d != 3:
        print("Solo se puede graficar en 3D si d=2 (espacio de dimensión 3)")
        return

    # Muestreo aleatorio de puntos dentro del politopo
    puntos = []
    for z in range(k + 1):
        for _ in range(500):
            p = np.random.uniform(0, 1, size=2)
            x = np.hstack(([z], p))
            if np.all(A @ x <= b + 1e-8):
                puntos.append(x)

    puntos = np.array(puntos)
    if len(puntos) == 0:
        print("No se encontraron puntos válidos para graficar.")
        return

    hull = ConvexHull(puntos)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for simplex in hull.simplices:
        tri = puntos[simplex]
        ax.add_collection3d(Poly3DCollection([tri], color='lightblue', edgecolor='gray', linewidths=0.5, alpha=0.6))

    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], s=5, color='black', alpha=0.4)
    ax.set_xlabel('z')
    ax.set_ylabel('x₁')
    ax.set_zlabel('x₂')
    ax.set_title(f'Convex hull del politopo completo (k={k}, iter={iteracion})')
    plt.tight_layout()
    plt.show()


# ==== MAIN ====

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, help="Valor de k a graficar")
    parser.add_argument('--iter', type=int, default=0, help="Número de iteración")
    parser.add_argument('--plot_fibras', action='store_true', help="Graficar lonjas proyectadas (2D)")
    parser.add_argument('--plot_3d', action='store_true', help="Graficar convex hull 3D del politopo")

    args = parser.parse_args()

    plot_histograma_F(args.k)

    if args.plot_fibras:
        graficar_todas_fibras_2d(args.k, args.iter)

    if args.plot_3d:
        graficar_convex_hull_3d(args.k, args.iter)


#python analisis.py --k 4 --iter 0 
#python analisis.py --k 3 --iter 0 --plot_3d
#python analisis.py --k 3 --iter 0 --plot_fibras
