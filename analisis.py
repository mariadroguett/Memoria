import numpy as np
from scipy.spatial import ConvexHull
from experiment import run_global_experiment  
import csv
import os
import time
import argparse
from generator import generar_politopo_k

# def generar_politopo_k(K, d, n_puntos=50):
    # puntos = np.random.rand(n_puntos, d + 1)
    # puntos[:, 0] *= K
    # hull = ConvexHull(puntos,  qhull_options='QJ')
    # A = hull.equations[:, :-1]
    # b = -hull.equations[:, -1]
    # return A, b


def guardar_Ab_txt(A, b, carpeta, K, iteracion):
    nombre = f"Ab_k{K:03d}_iter{iteracion:03d}.txt"
    ruta = os.path.join(carpeta, nombre)
    with open(ruta, 'w') as f:
        f.write("# A\n")
        for fila in A:
            f.write(" ".join(f"{x:.6f}" for x in fila) + "\n")
        f.write("# b\n")
        f.write(" ".join(f"{x:.6f}" for x in b) + "\n")

def guardar_fibras(A, b, K, d, iteracion, carpeta):
    for z in range(K + 1):
        A_p = A[:, 1:]
        b_z = b - A[:, 0] * z
        nombre = f"fibras_k{K:03d}_iter{iteracion:03d}_z{z:02d}.txt"
        ruta = os.path.join(carpeta, nombre)
        with open(ruta, 'w') as f:
            f.write(f"# S_{z} ⊆ ℝ^{d} definida por A_z p ≤ b_z\n")
            for fila in A_p:
                f.write(" ".join(f"{x:.6f}" for x in fila) + "\n")
            f.write("# b_z\n")
            f.write(" ".join(f"{x:.6f}" for x in b_z) + "\n")

def main(K, repeticiones=1, N1=1000, N2=50, N3=10000):
    d = 2
    carpeta = 'salida_k'
    os.makedirs(carpeta, exist_ok=True)

    archivo_csv = os.path.join(carpeta, f"resultados_k_{K:03d}.csv")

    start_iter = 0
    if os.path.exists(archivo_csv):
        with open(archivo_csv, 'r') as f_check:
            start_iter = sum(1 for line in f_check) - 1  

    modo = 'a' if start_iter > 0 else 'w'
    errores = 0
    exitosos = 0
    start_time = time.time()

    with open(archivo_csv, modo, newline='') as f_csv:
        writer = csv.writer(f_csv)
        if modo == 'w':
            writer.writerow(['iteracion', 'cp', 'F'])

        for i in range(start_iter, start_iter + repeticiones):
            try:
                A, b = generar_politopo_k(K, d)
                resultado = run_global_experiment(A, b, K, d, N1=N1, N2=N2, N3=N3)
                cp = resultado['best_cp']
                F = resultado['F']

                if resultado['caso_favorable']:
                    continue  # salta esta iteración

                guardar_Ab_txt(A, b, carpeta, K, i)
                guardar_fibras(A, b, K, d, i, carpeta)

                cp_str = ','.join(f"{x:.6f}" for x in cp) if cp is not None else "null"
                writer.writerow([i, cp_str, F])
                exitosos += 1

                #np.savetxt(os.path.join(carpeta, f"volumen_k{K:03d}_iter{i:03d}.txt"), np.array([F]), fmt='%.8f')

                print(f" Iteración {i}: F = {F:.6f}, cp = {cp_str}")

            except Exception as e:
                errores += 1
                print(f" Error en k={K}, iter={i}: {e}")

    tiempo_total = time.time() - start_time
    print(f"\nFinalizado k={K}: {exitosos}/{repeticiones} exitosos, {errores} errores")
    print(f"Tiempo total: {tiempo_total:.2f}s | Promedio por iteración: {tiempo_total/repeticiones:.3f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, required=True, help='Valor de k (longitud discreta)')
    args = parser.parse_args()
    main(args.k)

