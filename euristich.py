import numpy as np

def ordina_clienti_per_distanza_deposito(clienti, d_drone):
    """
    Ordina una lista di clienti in base alla distanza di andata e ritorno dal deposito (in ordine crescente).
    """
    clienti_con_distanza = [(c, d_drone[c]) for c in clienti]
    clienti_ordinati = sorted(clienti_con_distanza, key=lambda x: x[1])
    return [c for c, _ in clienti_ordinati]

def calcola_savings(clienti_da_servire, d_truck):
    """
    Calcola la matrice dei risparmi S_ij per l'algoritmo di Clarke & Wright.
    S_ij = d(0, i) + d(0, j) - d(i, j)
    """
    risparmi = []
    num_clienti = len(clienti_da_servire)
    for i_idx in range(num_clienti):
        for j_idx in range(i_idx + 1, num_clienti):
            i = clienti_da_servire[i_idx]
            j = clienti_da_servire[j_idx]
            
            s = d_truck[0, i] + d_truck[0, j] - d_truck[i, j]
            risparmi.append((s, i, j))
            
    risparmi.sort(key=lambda x: x[0], reverse=True)
    return risparmi

def calcola_carico_rotta(rotta, weights):
    """Calcola il carico totale di una rotta escludendo il deposito (nodo 0)."""
    return sum(weights[nodo] for nodo in rotta if nodo != 0)

def calcola_tempo_rotta(rotta, t_truck):
    """Calcola il tempo totale di percorrenza di una rotta."""
    tempo = 0
    for idx in range(len(rotta) - 1):
        tempo += t_truck[rotta[idx], rotta[idx+1]]
    return tempo

def solve_heuristic(data):
    """
    Esegue l'euristica completa (Cluster-First, Route-Second + Local Search) per il PDSVRP.
    """
    print("\n" + "="*40)
    print("        AVVIO EURISTICA PDSVRP          ")
    print("="*40)

    # 1. Estrazione Dati
    C_T = data['C_T']
    C_F = data['C_F']
    d_drone = data['d_drone']
    t_drone = data['t_drone']
    d_truck = data['d_truck']
    t_truck = data['t_truck']
    weights = data['weights']
    Q = data['Q_truck']
    T_max_drone = data['T_drone']
    T_max_truck = data['T_truck']
    num_droni_disp = data['m']
    num_camion_disp = data['h']
    costo_km_drone = data['C_cost_drone']
    costo_km_camion = data['C_cost_truck']

    # --- FASE 1: ASSEGNAZIONE DRONI (CLUSTER-FIRST) ---
    print("\n--- FASE 1: ASSEGNAZIONE DRONI ---")
    
    tempi_droni = [0.0] * num_droni_disp
    assegnazioni_droni = {k: [] for k in range(num_droni_disp)}
    clienti_serviti_da_drone = []
    
    C_F_ordinati = ordina_clienti_per_distanza_deposito(C_F, d_drone)
    
    for c in C_F_ordinati:
        assegnato = False
        for k in range(num_droni_disp):
            tempo_volo_c = t_drone[c]
            if tempi_droni[k] + tempo_volo_c <= T_max_drone:
                assegnazioni_droni[k].append(c)
                tempi_droni[k] += tempo_volo_c
                clienti_serviti_da_drone.append(c)
                assegnato = True
                print(f" -> Cliente {c} assegnato al Drone {k+1} (Tempo totale drone: {tempi_droni[k]*60:.1f} min)")
                break
        
    if not clienti_serviti_da_drone:
        print(" Nessun cliente assegnato ai droni in questa fase.")

    # --- FASE 2: INSTRADAMENTO CAMION (ROUTE-SECOND) ---
    print("\n--- FASE 2: INSTRADAMENTO CAMION (Clarke & Wright) ---")
    
    clienti_camion = list(C_T) + [c for c in C_F if c not in clienti_serviti_da_drone]
    print(f" Clienti rimanenti per i camion: {clienti_camion}")

    rotte_finali_camion = []
    if not clienti_camion:
        print(" Tutti i clienti sono stati serviti dai droni!")
    else:
        rotte_camion = [[0, c, 0] for c in clienti_camion]
        rotta_di_cliente = {c: idx for idx, c in enumerate(clienti_camion)}
        risparmi_ordinati = calcola_savings(clienti_camion, d_truck)

        for (risparmio, i, j) in risparmi_ordinati:
            idx_rotta_i = rotta_di_cliente[i]
            idx_rotta_j = rotta_di_cliente[j]

            if idx_rotta_i == idx_rotta_j:
                continue

            rotta_i = rotte_camion[idx_rotta_i]
            rotta_j = rotte_camion[idx_rotta_j]

            if (rotta_i[1] == i or rotta_i[-2] == i) and (rotta_j[1] == j or rotta_j[-2] == j):
                
                nodi_i = rotta_i[1:-1]
                if nodi_i[0] == i:
                    nodi_i.reverse() 
                
                nodi_j = rotta_j[1:-1]
                if nodi_j[-1] == j:
                    nodi_j.reverse() 
                
                rotta_fusa_nodi = nodi_i + nodi_j
                rotta_fusa_completa = [0] + rotta_fusa_nodi + [0]
                
                carico_totale = calcola_carico_rotta(rotta_fusa_completa, weights)
                tempo_totale = calcola_tempo_rotta(rotta_fusa_completa, t_truck)

                if carico_totale <= Q and tempo_totale <= T_max_truck:
                    rotte_camion[idx_rotta_i] = rotta_fusa_completa
                    rotte_camion[idx_rotta_j] = []
                    for nodo in rotta_fusa_nodi:
                        rotta_di_cliente[nodo] = idx_rotta_i

        rotte_finali_camion = [r for r in rotte_camion if r]
        
        if len(rotte_finali_camion) > num_camion_disp:
            print(f" [ATTENZIONE] Il C&W ha generato {len(rotte_finali_camion)} rotte, ma abbiamo solo {num_camion_disp} camion a disposizione!")
        
        for idx, rotta in enumerate(rotte_finali_camion):
            tour_str = " -> ".join(map(str, rotta))
            print(f" -> Camion {idx+1}: {tour_str}")

    # --- FASE 3: RICERCA LOCALE (Shift Drone-Camion / Camion-Drone) ---
    print("\n--- FASE 3: RICERCA LOCALE ---")
    miglioramento = True
    
    while miglioramento:
        miglioramento = False
        
        # 3.1 Spostamento DRONE -> CAMION
        for k in range(num_droni_disp):
            clienti_drone_da_rimuovere = []
            for c in assegnazioni_droni[k]:
                miglior_delta = 0
                miglior_rotta = -1
                miglior_posizione = -1
                
                for idx_r, rotta in enumerate(rotte_finali_camion):
                    for pos in range(1, len(rotta)): 
                        nodo_prec = rotta[pos-1]
                        nodo_succ = rotta[pos]
                        
                        rotta_simulata = rotta[:pos] + [c] + rotta[pos:]
                        carico_simulato = calcola_carico_rotta(rotta_simulata, weights)
                        tempo_simulato = calcola_tempo_rotta(rotta_simulata, t_truck)
                        
                        if carico_simulato <= Q and tempo_simulato <= T_max_truck:
                            risparmio_drone = - (costo_km_drone * d_drone[c])
                            costo_aggiuntivo_camion = costo_km_camion * (d_truck[nodo_prec, c] + d_truck[c, nodo_succ] - d_truck[nodo_prec, nodo_succ])
                            delta_costo = risparmio_drone + costo_aggiuntivo_camion
                            
                            if delta_costo < miglior_delta: 
                                miglior_delta = delta_costo
                                miglior_rotta = idx_r
                                miglior_posizione = pos
                
                if miglior_delta < -0.01:
                    print(f" [Shift DRONE->CAMION] Cliente {c} spostato nel Camion {miglior_rotta+1} (Risparmio: {-miglior_delta:.2f} $)")
                    rotte_finali_camion[miglior_rotta].insert(miglior_posizione, c)
                    clienti_drone_da_rimuovere.append(c)
                    tempi_droni[k] -= t_drone[c]
                    miglioramento = True
                    break 
            
            for c_rimosso in clienti_drone_da_rimuovere:
                assegnazioni_droni[k].remove(c_rimosso)
            if miglioramento: break
            
        if miglioramento: continue 
        
        # 3.2 Spostamento CAMION -> DRONE
        for idx_r, rotta in enumerate(rotte_finali_camion):
            cliente_rimosso = False
            for pos in range(1, len(rotta) - 1): 
                c = rotta[pos]
                
                if c in C_F:
                    nodo_prec = rotta[pos-1]
                    nodo_succ = rotta[pos+1]
                    
                    for k in range(num_droni_disp):
                        if tempi_droni[k] + t_drone[c] <= T_max_drone:
                            costo_aggiuntivo_drone = costo_km_drone * d_drone[c]
                            risparmio_camion = - costo_km_camion * (d_truck[nodo_prec, c] + d_truck[c, nodo_succ] - d_truck[nodo_prec, nodo_succ])
                            delta_costo = costo_aggiuntivo_drone + risparmio_camion
                            
                            if delta_costo < -0.01: 
                                print(f" [Shift CAMION->DRONE] Cliente {c} spostato dal Camion {idx_r+1} al Drone {k+1} (Risparmio: {-delta_costo:.2f} $)")
                                del rotte_finali_camion[idx_r][pos]
                                assegnazioni_droni[k].append(c)
                                tempi_droni[k] += t_drone[c]
                                miglioramento = True
                                cliente_rimosso = True
                                break 
                if cliente_rimosso: break 
            if cliente_rimosso: break 

    # --- CALCOLO COSTI TOTALI FINALI ---
    costo_totale_droni = sum(costo_km_drone * d_drone[c] for k in range(num_droni_disp) for c in assegnazioni_droni[k])
    
    costo_totale_camion = 0
    for rotta in rotte_finali_camion:
        for idx in range(len(rotta) - 1):
            costo_totale_camion += costo_km_camion * d_truck[rotta[idx], rotta[idx+1]]
            
    costo_totale = costo_totale_droni + costo_totale_camion
    
    print("\n--- RIEPILOGO EURISTICA ---")
    print(f"Costo Camion: {costo_totale_camion:.2f} $")
    print(f"Costo Droni:  {costo_totale_droni:.2f} $")
    print(f"COSTO TOTALE: {costo_totale:.2f} $")
    print("="*40 + "\n")
    
    return {
        'costo_totale': costo_totale,
        'assegnazioni_droni': assegnazioni_droni,
        'rotte_camion': rotte_finali_camion
    }