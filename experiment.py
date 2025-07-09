import numpy as np
from sklearn.preprocessing import normalize

# A = np.zeros((6,3),dtype=int)
# A[0,0] = 1
# A[1,1] = 1
# A[2,2] = 1
# A[3,0] = -1s
# A[4,1] = -1
# A[5,2] = -1
# b = np.zeros((6,1),dtype=int)
# b[0,0] = 2
# b[1,0] = 1
# b[2,0] = 1

#d =2
#z = 0
#N = 1000

def rejection_sampling(d, A, b, z, N):
    #print(b.shape)
    # A_z = A[:,0]*z.T
    A_z = A[:, 0] * z
    A_rest = A[:,1:] 
   # print(A_rest.shape)
   # print(A_rest)
    #for _ in range(N):
    p = np.random.uniform(0, 1, size=(N,d))
    #x = np.hstack(([z], p))
    x = np.dot(p, A_rest.T)
    #print(x.shape)
    # satisfied = np.all(x + A_z - b <= 1e-8, axis=1)
    satisfied = np.all(x + A_z <= b.flatten() + 1e-8, axis=1)
    #print(b.flatten().shape)    
    #print(satisfied)
    res = float(np.mean(satisfied))
    #("Resultado del rejection_sampling", res)

    return res

    #     if np.all(A @ x <= b + 1e-8):
    #         count += 1
    #         print("aqui")
    # return count / N

#K = 2
#N3 = 10000
# def casos_no_favorables(K, d, A, b, N3, tol=1e-2):
#     vol_vec = [rejection_sampling(d, A, b, z, N3) for z in range(K + 1)]
#     total_vol = sum(vol_vec)
#     #print(vol_vec)
#     #print(total_vol)                                                    

#     if total_vol == 0:
#         print("Volumen total estimado del politopo es 0.")
#         return {
#             'volumen_total': 0,
#             'vol_no_favorables': 0,
#             'n_no_favorables': K + 1,
#             'vols_no_favorables': vol_vec 
#         }

#     vols_no_favorables = []
#     for vol in vol_vec:
#         if vol >= 0.5 * total_vol - tol:  #not np.isclose(vol, 0.5 * total_vol, rtol=tol):
#             vols_no_favorables.append(vol)

#     total_vol_no_fav = sum(vols_no_favorables)
#     print(total_vol_no_fav)
#     n_no_fav = len(vols_no_favorables)
#     #print(n_no_fav)


#     return {
#         'volumen_total': total_vol,
#         'vol_no_favorables': total_vol_no_fav,
#         'n_no_favorables': n_no_fav,
#         'vols_no_favorables': vols_no_favorables
#     }



def run_global_experiment(A, b, K, d, N1=1000, N2=50, N3=10000):
    #if seed is not None:
    #    np.random.seed(seed)

    # resultado_vol = casos_no_favorables(K, d, A, b, N3)
    # total_vol = resultado_vol['volumen_total']
    # vol_vec = resultado_vol['vols_no_favorables']

    vol_vec = [rejection_sampling(d, A, b, z, N3) for z in range(K + 1)]
    total_vol = sum(vol_vec)
    #print("Resultado del rejection_sampling", vol_vec)

    if total_vol == 0:
        return {
            'best_cp': None,
            'F': 0,
            'caso_favorable': False,
            'vol_por_fibra': vol_vec,
            'params': {'K': K, 'd': d, 'N1': N1, 'N2': N2, 'N3': N3}
        }
    caso_favorable = any(v > 0.5 * total_vol for v in vol_vec)
    if caso_favorable:
        print("Caso favorable")
       


    best_cp = None
    F = 0.0
    candidatos_validos = 0
    z_coord = 1  # POR MIENTRAAAAAAS

    for _ in range(N1):
        x_coord = np.random.uniform(0, 1, size=d)
        cp = np.concatenate((np.array([z_coord]), x_coord))
        #print(cp)

        if not np.all(np.dot(A, cp) <= b.flatten() + 1e-8):
            continue

        candidatos_validos += 1
        worst_vol = total_vol

        #for _ in range(N2):
        # direction = np.random.normal(0, 1, size=d + 1)
        # direction /= np.linalg.norm(direction)
        directions = np.random.normal(0, 1, size=(N2, d + 1))
        directions = normalize(directions) 
        #print(directions.shape)


        # A_new = np.vstack([A, direction.reshape(1, -1)])
        # b_new = np.append(b, np.dot(direction, cp))

    #     new_vol = sum(rejection_sampling(d, A_new, b_new, z, N3) for z in range(K + 1))
    #     if new_vol < worst_vol:
    #         worst_vol = new_vol
    #         if worst_vol < F:
    #             break

    #     if worst_vol > F:
    #         F = worst_vol
    #         best_cp = cp

    # if candidatos_validos == 0:
    #     print("No se encontró ningún punto cp válido dentro del politopo.")
        new_vols = []
        for i in range(N2):
            direction = directions[i]
            A_new = np.concatenate((A, direction.reshape(1, -1)), axis=0)
            b_new = np.append(b, np.dot(direction, cp)).reshape(-1, 1)
            vol = 0.0
            for z in range(K + 1):
                vol += rejection_sampling(d, A_new, b_new, z, N3)
            new_vols.append(vol)
            #print("Resultado del rejection_sampling:", new_vols)guardalooo

        worst_vol = min(new_vols)
        #print(worst_vol)
        if worst_vol > F:   
            F = worst_vol
            best_cp = cp
        if candidatos_validos == 0:
            print("No se encontró ningún punto cp válido dentro del politopo")

    return {
        'best_cp': best_cp.tolist() if best_cp is not None else None,
        'F': F / total_vol,
        'caso_favorable': False,
        'vol_por_fibra': vol_vec,
        'volumen_total': total_vol,
        #'vol_no_favorables': resultado_vol['vol_no_favorables'],
        #'n_no_favorables': resultado_vol['n_no_favorables'],
        'params': {'K': K, 'd': d, 'N1': N1, 'N2': N2, 'N3': N3}
    }






