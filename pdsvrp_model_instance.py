from docplex.mp.model import Model
import instance_generator
import disegna_mappe
import euristich

def build_and_solve_pdsvrp(data):
    """
    Costruisce e risolve il modello matematico MILP esatto per il PDSVRP.
    """
    # Inizializzazione del Modello CPLEX
    mdl = Model(name='PDSVRP_Exact_Model')#assegna un nome al modello per una migliore tracciabilità nei
    #  log di CPLEX
    
    # Estrazione Set e Parametri del Dizionario dell'Istanza
    N = data['nodes']
    C = data['customers']
    C_T = data['C_T']
    C_F = data['C_F']
    D = list(range(1, data['m'] + 1)) # Indici dei droni da 1 a m
    
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
    T_primo_max = data['T_drone']#t' come endurance totale massima per i droni (es. 24 km a 40 km/h = 0.6 ore)
    
    # -------------------------------------------------------------
    # VARIABILI DI DECISIONE
    # -------------------------------------------------------------
    # x_ij = 1 se il camion percorre l'arco (i, j), 0 altrimenti
    x = mdl.binary_var_matrix(N, N, name='x')
    
    # y_i^k = 1 se il cliente i (mode-free) è servito dal drone k, 0 altrimenti
    y = mdl.binary_var_matrix(C_F, D, name='y')
    
    # u_i = carico cumulativo trasportato dal camion fino al nodo i
    u = mdl.continuous_var_dict(C, lb=0, ub=Q, name='u')
    
    # z_ij = durata cumulativa del tour del camion al nodo j dopo essere passato da i
    z = mdl.continuous_var_matrix(N, N, lb=0, ub=T_max, name='z')

    # FUNZIONE OBIETTIVO (Equazione 1)
    truck_cost = mdl.sum(C_truck * d_truck[i, j] * x[i, j] for i in N for j in N if i != j)
    drone_cost = mdl.sum(C_drone * d_drone[i] * y[i, k] for i in C_F for k in D)
    mdl.minimize(truck_cost + drone_cost)
   
    # VINCOLI 
  
    # Vincolo (2): Numero massimo di camion usati dal deposito
    mdl.add_constraint(mdl.sum(x[0, i] for i in C) <= h, 'MaxTrucksLeavingDepot')
    
    # Vincolo (3): Equazioni di conservazione del flusso per ogni nodo
    for j in N:
        mdl.add_constraint(
            mdl.sum(x[i, j] for i in N if i != j) - mdl.sum(x[j, i] for i in N if i != j) == 0,
            f'FlowConservation_node_{j}'
        )
        
    # Vincolo (4): Ogni cliente mode-free deve essere visitato esattamente una volta (Truck o Drone)
    for j in C_F:
        mdl.add_constraint(
            mdl.sum(x[i, j] for i in N if i != j) + mdl.sum(y[j, k] for k in D) == 1,
            f'ServiceModeFree_cust_{j}'
        )
        
    # Vincolo (5): I clienti Truck-Only devono essere visitati obbligatoriamente dai camion
    for j in C_T:
        mdl.add_constraint(
            mdl.sum(x[i, j] for i in N if i != j) == 1,
            f'ServiceTruckOnly_cust_{j}'
        )
        
    # Vincolo (6): Eliminazione dei subtour e capacità del camion (Formulazione Miller-Tucker-Zemlin)
    for i in C:
        for j in C:
            if i != j:
                mdl.add_constraint(
                    u[i] - u[j] + Q * x[i, j] <= Q - w[j],
                    f'MTZ_Capacity_{i}_{j}'
                )
    
    # Vincolo (7): Limite sul tempo massimo operativo di ogni drone (Endurance totale sulla giornata)
    for k in D:
        mdl.add_constraint(
            mdl.sum(t_drone[j] * y[j, k] for j in C_F) <= T_primo_max,
            f'DroneEndurance_drone_{k}'
        )
        
    # Vincoli (8)-(10): Tracciamento e limitazione del tempo di percorrenza dei camion (Fatto da Kara 2011)
    for i in C:
        mdl.add_constraint(
            mdl.sum(z[l, i] for l in N if l != i) + mdl.sum(t_truck[i, j] * x[i, j] for j in N if j != i) == mdl.sum(z[i, j] for j in N if j != i),
            f'TruckTimePropagation_node_{i}'
        )
        
    for i in C:
        # Vincolo (9): Tempo iniziale all'uscita del deposito per il primo nodo servito
        mdl.add_constraint(z[0, i] == t_truck[0, i] * x[0, i], f'InitialTruckTime_{i}')
        # Vincolo (10): Limite massimo del tempo totale al rientro al deposito
        mdl.add_constraint(z[i, 0] <= T_max * x[i, 0], f'MaxTruckDuration_{i}')
        
    # Rimozione esplicita di auto-loop (i, i)
    for i in N:
        mdl.add_constraint(x[i, i] == 0)
        mdl.add_constraint(z[i, i] == 0)
   
    # RISOLUZIONE
    mdl.parameters.timelimit = 300 # Tempo limite di calcolo impostato a 5 minuti per i test
    print(f"\n[INFO] Avvio ottimizzazione esatta CPLEX su istanza con {data['n']} clienti...")
    solution = mdl.solve(log_output=True)
    
    # Controllo e output dei risultati analitici
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
        drone_assignments = [] 
        
        for i in C_F:
            for k in D:
                if y[i, k].solution_value > 0.5:
                    drone_serviced = True
                    drone_assignments.append(i)
                    print(f" -> Il Drone {k} serve il Cliente {i} [Peso Pacco: {w[i]:.2f} kg, Tempo Volo: {t_drone[i]*60:.1f} min]")
        if not drone_serviced:
            print(" Nessun drone impiegato in questa istanza.")
        print("="*40 + "\n")
        
        # <--- GENERAZIONE MAPPA CPLEX SBLOCCATA CON TITOLO MODIFICATO
        print("[INFO] Generazione della mappa spaziale in corso...")
        disegna_mappe.disegna_mappa_percorsi(data, active_arcs, drone_assignments, titolo="PDSVRP - Risoluzione Ottima (CPLEX)")
        
    else:
        print("\n[ERRORE] Il solver non ha trovato una soluzione ammissibile entro il limite di tempo.")
        
    return solution

# -------------------------------------------------------------
# BLOCCO MAIN - ESTRAZIONE SINGOLA ISTANZA PER LA TESI
# -------------------------------------------------------------
if __name__ == "__main__":
    print("="*60)
    print("ESTRAZIONE DATI E MAPPA INIZIALE")
    print("="*60)
    
    # 1. IMPOSTA QUI IL NUMERO DI CLIENTI (es. 5, 10, 15, 20)
    numero_clienti = 5
    
    # 2. Generazione dell'istanza
    istanza_test = instance_generator.generate_instance(
        n=numero_clienti, grid_size=25, depot_pos='e', dist_type='r', num_drones=2, num_trucks=2
    )
    
    # 3. Estrazione dei dati in formato Tabella LaTeX
    print("\n--- COPIA QUESTO CODICE SU OVERLEAF ---")
    print("\\begin{table}[htbp]")
    print("    \\centering")
    print("    \\begin{tabular}{clcc}")
    print("        \\toprule")
    print("        \\textbf{Nodo} & \\textbf{Tipo} & \\textbf{Peso (kg)} & \\textbf{Coordinate (X, Y)} \\\\")
    print("        \\midrule")
    
    # Riga del deposito
    dep_coord = istanza_test['coords'][0]
    print(f"        0 (Deposito) & - & 0.00 & ({dep_coord[0]:.2f}, {dep_coord[1]:.2f}) \\\\")
    
    # Righe dei clienti
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
    print("---------------------------------------\n")
    
    # 4. Disegno della Mappa Iniziale (Solo Nodi)
    print(f"[INFO] Generazione della mappa iniziale per {numero_clienti} clienti...")
    
    # Passando liste vuote [], il grafico stamperà solo i nodi
    disegna_mappe.disegna_mappa_percorsi(
        istanza_test, 
        active_arcs=[], 
        drone_assignments=[], 
        titolo=f"Distribuzione Spaziale Iniziale (N={numero_clienti})"
    )
    
    # NOTE: I solver CPLEX ed Euristica sono temporaneamente disattivati in questo blocco 
    # per permetterti di concentrarti solo sull'estrazione della mappa pulita.