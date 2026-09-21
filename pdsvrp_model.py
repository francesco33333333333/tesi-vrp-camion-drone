from docplex.mp.model import Model
import instance_generator
import disegna_mappe
import euristich
import time  

def build_and_solve_pdsvrp(data):
    """
    Costruisce e risolve il modello matematico MILP esatto per il PDSVRP.
    """
    mdl = Model(name='PDSVRP_Exact_Model')
    
    N = data['nodes']
    C = data['customers']
    C_T = data['C_T']
    C_F = data['C_F']
    D = list(range(1, data['m'] + 1)) 
    
    d_truck = data['d_truck']
    t_truck = data['t_truck']
    d_drone = data['d_drone']
    t_drone = data['t_drone']
    w = data['weights']
    h = data['h']
    Q = data['Q_truck']
    C_truck = data['C_cost_truck']
    C_drone = data['C_cost_drone']
    T_max = data['T_truck']
    T_primo_max = data['T_drone']
    
    # -------------------------------------------------------------
    # VARIABILI DI DECISIONE
    # -------------------------------------------------------------
    x = mdl.binary_var_matrix(N, N, name='x')
    
    # Crea le variabili dei droni SOLO se ci sono droni disponibili (D) e clienti eleggibili (C_F)
    if C_F and D:
        y = mdl.binary_var_matrix(C_F, D, name='y')
    else:
        y = {} # Dizionario vuoto per evitare l'errore "multidict has no keys"
        
    u = mdl.continuous_var_dict(C, lb=0, ub=Q, name='u')
    z = mdl.continuous_var_matrix(N, N, lb=0, ub=T_max, name='z')

    truck_cost = mdl.sum(C_truck * d_truck[i, j] * x[i, j] for i in N for j in N if i != j)
    drone_cost = mdl.sum(C_drone * d_drone[i] * y[i, k] for i in C_F for k in D)
    mdl.minimize(truck_cost + drone_cost)
   
    mdl.add_constraint(mdl.sum(x[0, i] for i in C) <= h, 'MaxTrucksLeavingDepot')
    
    for j in N:
        mdl.add_constraint(
            mdl.sum(x[i, j] for i in N if i != j) - mdl.sum(x[j, i] for i in N if i != j) == 0,
            f'FlowConservation_node_{j}'
        )
        
    for j in C_F:
        # Se D è vuoto, mdl.sum(y...) fa 0, quindi il camion è forzato a visitare anche i clienti Mode-Free
        mdl.add_constraint(
            mdl.sum(x[i, j] for i in N if i != j) + mdl.sum(y[j, k] for k in D) == 1,
            f'ServiceModeFree_cust_{j}'
        )
        
    for j in C_T:
        mdl.add_constraint(
            mdl.sum(x[i, j] for i in N if i != j) == 1,
            f'ServiceTruckOnly_cust_{j}'
        )
        
    for i in C:
        for j in C:
            if i != j:
                mdl.add_constraint(
                    u[i] - u[j] + Q * x[i, j] <= Q - w[j],
                    f'MTZ_Capacity_{i}_{j}'
                )
    
    for k in D:
        mdl.add_constraint(
            mdl.sum(t_drone[j] * y[j, k] for j in C_F) <= T_primo_max,
            f'DroneEndurance_drone_{k}'
        )
        
    for i in C:
        mdl.add_constraint(
            mdl.sum(z[l, i] for l in N if l != i) + mdl.sum(t_truck[i, j] * x[i, j] for j in N if j != i) == mdl.sum(z[i, j] for j in N if j != i),
            f'TruckTimePropagation_node_{i}'
        )
        
    for i in C:
        mdl.add_constraint(z[0, i] == t_truck[0, i] * x[0, i], f'InitialTruckTime_{i}')
        mdl.add_constraint(z[i, 0] <= T_max * x[i, 0], f'MaxTruckDuration_{i}')
        
    for i in N:
        for j in N:
            if i != j:
                mdl.add_constraint(z[i, j] <= T_max * x[i, j], f'TimeBound_{i}_{j}')
                
    for i in N:
        mdl.add_constraint(x[i, i] == 0)
        mdl.add_constraint(z[i, i] == 0)
   
    # LIMITE DI TEMPO CPLEX (attualmente 900 secondi = 15 min)
    mdl.parameters.timelimit = 900 
    
    print(f"\n[INFO] Avvio ottimizzazione esatta CPLEX su istanza con {data['n']} clienti...")
    solution = mdl.solve(log_output=True)
    
    # Inizializzo SEMPRE queste liste, così il "return" non fallisce mai
    active_arcs = []
    drone_assignments = []
    
    if solution:
        print("\n" + "="*40)
        print("         SOLUZIONE OTTIMALE TROVATA          ")
        print("="*40)
        print(f"Costo Totale Operativo di Trasporto: {solution.objective_value:.2f} $")
        print("\n--- ROTTE DEI CAMION ---")
        
        active_arcs = [(i, j) for i in N for j in N if x[i, j].solution_value > 0.5]
        partenze_deposito = [j for (i, j) in active_arcs if i == 0]
        
        for idx, primo_nodo in enumerate(partenze_deposito, 1):
            tour = [0, primo_nodo]
            curr = primo_nodo
            
            while curr != 0:
                prossimo_nodo = [j for (i, j) in active_arcs if i == curr]
                if prossimo_nodo:
                    nodo_successivo = prossimo_nodo[0]
                    tour.append(nodo_successivo)
                    curr = nodo_successivo
                else:
                    break
            
            tour_str = " -> ".join(map(str, tour))
            print(f"\n Camion {idx} - Percorso Sequenziale:")
            print(f"   {tour_str}")
            print("   Dettaglio spostamenti:")
            for i in range(len(tour) - 1):
                u_node = tour[i]
                v_node = tour[i+1]
                print(f"     -> dal Nodo {u_node} al Nodo {v_node} (Distanza: {d_truck[u_node, v_node]:.2f} km)")
            
        print("\n--- ASSEGNAZIONI DRONE ---")
        drone_serviced = False
        
        for i in C_F:
            for k in D:
                if y[i, k].solution_value > 0.5:
                    drone_serviced = True
                    drone_assignments.append((i, k)) 
                    print(f" -> Il Drone {k} serve il Cliente {i} [Peso Pacco: {w[i]:.2f} kg, Tempo Volo: {t_drone[i]*60:.1f} min]")
        if not drone_serviced:
            print(" Nessun drone impiegato in questa istanza.")
        print("="*40 + "\n")
        
    else:
        # CPLEX ti dice il VERO motivo del fallimento
        stato_cplex = mdl.solve_details.status
        print(f"\n[ERRORE] Risoluzione fallita. Motivo reale fornito da CPLEX: {stato_cplex}")
        
    # Restituisco SEMPRE 3 valori per evitare l'errore TypeError
    return solution, active_arcs, drone_assignments

# -------------------------------------------------------------
# BLOCCO MAIN DI ESECUZIONE (SIMULAZIONI )
# -------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("PDSVRP - ESECUZIONE COMPLETA ")
    print("="*60)
    
    numero_clienti = 100 

    istanza_test = instance_generator.generate_instance(
        n=numero_clienti, 
        grid_size=25, 
        depot_pos='c', 
        dist_type='r', 
        num_drones=0,
        num_trucks=4)
    
    # =============================================================
    # FASE 1: PRESENTAZIONE DELL'ISTANZA 
    # =============================================================
    print("\n--- 1. CODICE LATEX: TABELLA DATI ISTANZA ---")
    print("\\begin{table}[htbp]")
    print("    \\centering")
    print("    \\begin{tabular}{clcc}")
    print("        \\toprule")
    print("        \\textbf{Nodo} & \\textbf{Tipo} & \\textbf{Peso (kg)} & \\textbf{Coordinate (X, Y)} \\\\")
    print("        \\midrule")
    
    dep_coord = istanza_test['coords'][0]
    print(f"        0 (Deposito) & - & 0.00 & ({dep_coord[0]:.2f}, {dep_coord[1]:.2f}) \\\\")
    
    for i in range(1, numero_clienti + 1):
        tipo = "Truck-Only" if i in istanza_test['C_T'] else "Mode-Free"
        peso = istanza_test['weights'][i]
        coord = istanza_test['coords'][i]
        print(f"        {i} & {tipo} & {peso:.2f} & ({coord[0]:.2f}, {coord[1]:.2f}) \\\\")
        
    print("        \\bottomrule")
    print("    \\end{tabular}")
    print(f"    \\caption{{Dettaglio dell'istanza iniziale con $N={numero_clienti}$ clienti.}}")
    print(f"    \\label{{tab:istanza_{numero_clienti}}}")
    print("\\end{table}\n")
    
    print(f"[INFO] Generazione della MAPPA INIZIALE (solo nodi)...")
    disegna_mappe.disegna_mappa_percorsi(
        istanza_test, 
        active_arcs=[], 
        drone_assignments=[], 
        titolo=f"Distribuzione Spaziale Iniziale (N={numero_clienti})"
    )

    # =============================================================
    # FASE 2: RISOLUZIONE ESATTA
    # =============================================================
    print("\n>>> INIZIO RISOLUZIONE ESATTA (CPLEX) <<<")
    start_cplex = time.time()
    solution, active_arcs_cplex, drone_assignments_cplex = build_and_solve_pdsvrp(istanza_test)
    tempo_cplex = time.time() - start_cplex
    
    if solution:
        print("[INFO] Generazione della MAPPA CPLEX in corso...")
        disegna_mappe.disegna_mappa_percorsi(
            istanza_test, 
            active_arcs_cplex, 
            drone_assignments_cplex, 
            titolo="PDSVRP - Risoluzione Ottima (CPLEX)"
        )
    
    # =============================================================
    # FASE 3: RISOLUZIONE EURISTICA
    # =============================================================
    print("\n>>> INIZIO RISOLUZIONE EURISTICA <<<")
    start_eur = time.time()
    risultati_euristica = euristich.solve_heuristic(istanza_test)
    tempo_eur = time.time() - start_eur

    active_arcs_euristica = []
    for rotta in risultati_euristica['rotte_camion']:
        for i in range(len(rotta) - 1):
            active_arcs_euristica.append((rotta[i], rotta[i+1]))
            
    drone_assignments_euristica = []
    for k, lista_clienti in risultati_euristica['assegnazioni_droni'].items():
        for c in lista_clienti:
            drone_assignments_euristica.append((c, k + 1))
        
    print("\n[INFO] Generazione della MAPPA EURISTICA in corso...")
    disegna_mappe.disegna_mappa_percorsi(
        istanza_test, 
        active_arcs_euristica, 
        drone_assignments_euristica, 
        titolo="PDSVRP - Risoluzione Euristica"
    )

    # =============================================================
    # FASE 4: STAMPA DEI RISULTATI E DELLE TABELLE LATEX
    # =============================================================
    costo_truck_km = istanza_test['C_cost_truck']
    costo_drone_km = istanza_test['C_cost_drone']
    
    km_truck_cplex = sum(istanza_test['d_truck'][i, j] for i, j in active_arcs_cplex) if solution else 0
    km_drone_cplex = sum(istanza_test['d_drone'][cust] for cust, d_id in drone_assignments_cplex) if solution else 0
    costo_truck_cplex = km_truck_cplex * costo_truck_km
    costo_drone_cplex = km_drone_cplex * costo_drone_km
    costo_totale_cplex = costo_truck_cplex + costo_drone_cplex if solution else float('inf')
    
    km_truck_eur = sum(istanza_test['d_truck'][i, j] for i, j in active_arcs_euristica)
    km_drone_eur = sum(istanza_test['d_drone'][cust] for cust, d_id in drone_assignments_euristica)
    costo_truck_eur = km_truck_eur * costo_truck_km
    costo_drone_eur = km_drone_eur * costo_drone_km
    costo_totale_eur = costo_truck_eur + costo_drone_eur

    if solution and costo_totale_cplex > 0:
        gap_percentuale = ((costo_totale_eur - costo_totale_cplex) / costo_totale_cplex) * 100
    else:
        gap_percentuale = float('nan')

    print("\n--- VERIFICA TEMPI DI GUIDA (EURISTICA) ---")
    for idx, rotta in enumerate(risultati_euristica['rotte_camion']):
        tempo_rotta = sum(istanza_test['t_truck'][rotta[i], rotta[i+1]] for i in range(len(rotta)-1))
        if tempo_rotta > 3.0:
            print(f" [!] Camion {idx+1}: {tempo_rotta:.2f} ore -> ATTENZIONE: SUPERA LE 3 ORE STANDARD")
        else:
            print(f" [OK] Camion {idx+1}: {tempo_rotta:.2f} ore -> Rispetta il turno")

    # TABELLA 1: CONFRONTO
    print("\n--- 2. CODICE LATEX: TABELLA CONFRONTO COSTI E TEMPI ---")
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\resizebox{\\textwidth}{!}{%") 
    print("    \\begin{tabular}{l c c c c c c c}") 
    print("        \\toprule")
    print("        \\textbf{Risolutore} & \\textbf{Km Camion} & \\textbf{Costo Camion} & \\textbf{Km Droni} & \\textbf{Costo Droni} & \\textbf{Costo Totale} & \\textbf{Tempo (s)} & \\textbf{Gap (\\%)} \\\\")
    print("        \\midrule")
    
    if solution:
        print(f"        CPLEX (Esatto) & {km_truck_cplex:.2f} & \\${costo_truck_cplex:.2f} & {km_drone_cplex:.2f} & \\${costo_drone_cplex:.2f} & \\textbf{{\\${costo_totale_cplex:.2f}}} & {tempo_cplex:.2f} & - \\\\")
    else:
        print(f"        CPLEX (Esatto) & - & - & - & - & Timeout & {tempo_cplex:.2f} & - \\\\")
    
    if gap_percentuale != float('nan') and solution:
        print(f"        Euristica & {km_truck_eur:.2f} & \\${costo_truck_eur:.2f} & {km_drone_eur:.2f} & \\${costo_drone_eur:.2f} & \\textbf{{\\${costo_totale_eur:.2f}}} & {tempo_eur:.4f} & {gap_percentuale:.2f} \\% \\\\")
    else:
        print(f"        Euristica & {km_truck_eur:.2f} & \\${costo_truck_eur:.2f} & {km_drone_eur:.2f} & \\${costo_drone_eur:.2f} & \\textbf{{\\${costo_totale_eur:.2f}}} & {tempo_eur:.4f} & - \\\\")
        
    print("        \\bottomrule")
    print("    \\end{tabular}%")
    print("    }")
    print(f"    \\caption{{Riepilogo distanze, costi, tempi e gap di ottimalità per l'istanza con $N={numero_clienti}$.}}")
    print(f"    \\label{{tab:riepilogo_completo_{numero_clienti}}}")
    print("\\end{table}\n")

    # TABELLA 2: UTILIZZO DEI MEZZI
    print("\n--- 3. CODICE LATEX: TABELLA UTILIZZO MEZZI ---")
    print("\\begin{table}[H]")
    print("    \\centering")
    print("    \\begin{tabular}{llcc}")
    print("        \\toprule")
    print("        \\textbf{Risolutore} & \\textbf{Mezzo} & \\textbf{Clienti Serviti} & \\textbf{Tempo Utilizzo (h)} \\\\")
    print("        \\midrule")

    if solution:
        partenze_deposito_cplex = [j for (i, j) in active_arcs_cplex if i == 0]
        for idx, primo_nodo in enumerate(partenze_deposito_cplex, 1):
            tour = [0, primo_nodo]
            curr = primo_nodo
            tempo_rotta = istanza_test['t_truck'][0, primo_nodo]
            while curr != 0:
                prossimi = [j for (i, j) in active_arcs_cplex if i == curr]
                if prossimi:
                    nxt = prossimi[0]
                    tempo_rotta += istanza_test['t_truck'][curr, nxt]
                    tour.append(nxt)
                    curr = nxt
                else:
                    break
            num_clienti = len(tour) - 2 
            print(f"        CPLEX & Camion {idx} & {num_clienti} & {tempo_rotta:.2f} \\\\")
        
        cplex_drone_stats = {}
        for cust, d_id in drone_assignments_cplex:
            if d_id not in cplex_drone_stats:
                cplex_drone_stats[d_id] = {'tempo': 0.0, 'clienti': 0}
            cplex_drone_stats[d_id]['tempo'] += istanza_test['t_drone'][cust]
            cplex_drone_stats[d_id]['clienti'] += 1
            
        for d_id in sorted(cplex_drone_stats.keys()):
            stats = cplex_drone_stats[d_id]
            print(f"        CPLEX & Drone {d_id} & {stats['clienti']} & {stats['tempo']:.2f} \\\\")
    else:
        print("        CPLEX & - & - & Timeout \\\\")

    print("        \\midrule")

    for idx, rotta in enumerate(risultati_euristica['rotte_camion']):
        tempo_rotta = sum(istanza_test['t_truck'][rotta[i], rotta[i+1]] for i in range(len(rotta)-1))
        num_clienti = len(rotta) - 2
        print(f"        Euristica & Camion {idx+1} & {num_clienti} & {tempo_rotta:.2f} \\\\")

    eur_drone_stats = {}
    for cust, d_id in drone_assignments_euristica:
        if d_id not in eur_drone_stats:
            eur_drone_stats[d_id] = {'tempo': 0.0, 'clienti': 0}
        eur_drone_stats[d_id]['tempo'] += istanza_test['t_drone'][cust]
        eur_drone_stats[d_id]['clienti'] += 1

    for d_id in sorted(eur_drone_stats.keys()):
        stats = eur_drone_stats[d_id]
        print(f"        Euristica & Drone {d_id} & {stats['clienti']} & {stats['tempo']:.2f} \\\\")

    print("        \\bottomrule")
    print("    \\end{tabular}")
    print(f"    \\caption{{Dettaglio dell'utilizzo dei mezzi (tempi e clienti serviti) per l'istanza con $N={numero_clienti}$.}}")
    print(f"    \\label{{tab:utilizzo_mezzi_{numero_clienti}}}")
    print("\\end{table}\n")