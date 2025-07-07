import numpy as np

def rejection_sampling(d, A, b, z, N):
    """
    Estima la proporción de volumen de la fibra {z} × [0,1]^d
    que cae dentro del politopo Ax ≤ b, usando muestreo por rechazo.
    """
    count = 0
    for _ in range(N):
        p = np.random.uniform(0, 1, size=d)
        x = np.hstack(([z], p))
        if np.all(A @ x <= b + 1e-8):
            count += 1
    return count / N

def run_global_experiment(A, b, K, d, N1=1000, N2=50, N3=10000, seed=None):
    """
    Ejecuta el experimento de visibilidad global sobre el politopo C ⊂ ℝ^{1+d}.

    Busca el punto cp ∈ C desde donde se maximiza la fracción visible del politopo
    tras aplicar cortes direccionales aleatorios.

    Retorna un diccionario con:
        - best_cp: punto con mayor visibilidad (o None)
        - F: fracción de volumen visible desde best_cp
        - caso_favorable: True si alguna fibra tiene > 50% del volumen
        - vol_por_fibra: lista de volúmenes estimados por fibra
        - params: info del experimento
    """
    if seed is not None:
        np.random.seed(seed)

    best_cp = None
    F = 0.0
    directions_used = []

    # Estimar volumen por fibra
    vol_vec = [rejection_sampling(d, A, b, z, N3) for z in range(K + 1)]
    total_vol = sum(vol_vec)
    caso_favorable = any(v > 0.5 * total_vol for v in vol_vec)

    if total_vol == 0:
        print("Volumen total estimado del politopo es 0.")
        return {
            'best_cp': None,
            'F': 0.0,
            'caso_favorable': False,
            'vol_por_fibra': vol_vec,
            'params': {'K': K, 'd': d, 'N1': N1, 'N2': N2, 'N3': N3, 'seed': seed}
        }

    z_coord = 1  # POR MIENTRAAAAAAS

    candidatos_validos = 0
    for _ in range(N1):
        x_coord = np.random.uniform(0, 1, size=d)
        cp = np.hstack(([z_coord], x_coord))

        if not np.all(A @ cp <= b + 1e-8):
            continue

        candidatos_validos += 1
        worst_vol = total_vol

        for _ in range(N2):
            direction = np.random.normal(0, 1, size=d + 1)
            direction /= np.linalg.norm(direction)
            directions_used.append(direction.tolist())

            A_new = np.vstack([A, direction.reshape(1, -1)])
            b_new = np.append(b, np.dot(direction, cp))

            new_vol = sum(rejection_sampling(d, A_new, b_new, z, N3) for z in range(K + 1))
            if new_vol < worst_vol:
                worst_vol = new_vol
                if worst_vol < F:
                    break

        if worst_vol > F:
            F = worst_vol
            best_cp = cp

    if candidatos_validos == 0:
        print("No se encontró ningún punto cp válido dentro del politopo.")

    return {
        'best_cp': best_cp.tolist() if best_cp is not None else None,
        'F': F / total_vol,
        'caso_favorable': caso_favorable,
        'vol_por_fibra': vol_vec,
        'params': {'K': K, 'd': d, 'N1': N1, 'N2': N2, 'N3': N3, 'seed': seed}
    }
