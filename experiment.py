import numpy as np

def rejection_sampling(d, A, b, z, N):
    count = 0
    for _ in range(N):
        p = np.random.uniform(0, 1, size=d)
        x = np.hstack(([z], p))
        if np.all(A @ x <= b + 1e-8):
            count += 1
    return count / N

def casos_no_favorables(K, d, A, b, N3, tol=1e-2):
    vol_vec = [rejection_sampling(d, A, b, z, N3) for z in range(K + 1)]
    total_vol = sum(vol_vec)

    if total_vol == 0:
        print("Volumen total estimado del politopo es 0.")
        return {
            'volumen_total': 0,
            'vol_no_favorables': 0,
            'n_no_favorables': K + 1,
            'vols_no_favorables': vol_vec 
        }

    vols_no_favorables = []
    for vol in vol_vec:
        if not np.isclose(vol, 0.5 * total_vol, rtol=tol):
            vols_no_favorables.append(vol)

    total_vol_no_fav = sum(vols_no_favorables)
    n_no_fav = len(vols_no_favorables)

    return {
        'volumen_total': total_vol,
        'vol_no_favorables': total_vol_no_fav,
        'n_no_favorables': n_no_fav,
        'vols_no_favorables': vols_no_favorables
    }

def run_global_experiment(A, b, K, d, N1=1000, N2=50, N3=10000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    resultado_vol = casos_no_favorables(K, d, A, b, N3)
    total_vol = resultado_vol['volumen_total']
    vol_vec = resultado_vol['vols_no_favorables']

    if total_vol == 0:
        return {
            'best_cp': None,
            'F': 0,
            'caso_favorable': False,
            'vol_por_fibra': vol_vec,
            'params': {'K': K, 'd': d, 'N1': N1, 'N2': N2, 'N3': N3, 'seed': seed}
        }

    best_cp = None
    F = 0.0
    candidatos_validos = 0
    z_coord = 1  # POR MIENTRAAAAAAS

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
        'caso_favorable': False,
        'vol_por_fibra': vol_vec,
        'volumen_total': total_vol,
        'vol_no_favorables': resultado_vol['vol_no_favorables'],
        'n_no_favorables': resultado_vol['n_no_favorables'],
        'params': {'K': K, 'd': d, 'N1': N1, 'N2': N2, 'N3': N3, 'seed': seed}
    }


