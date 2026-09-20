import numpy as np
import math

def generate_instance(n, grid_size=25, depot_pos='c', dist_type='r', num_drones=5, num_trucks=4):
    """
    Generatore di istanze  basato sulla Sezione 4.1 dell'articolo.
    
    Parametri:
    - n: numero di clienti
    - grid_size: dimensione della griglia quadrata (in km)
    - depot_pos: 'c' (centrale), 'e' (bordo), 'r' (casuale)
    - dist_type: 'r' (uniforme casuale), 'c' (clusterizzato)
    """
    np.random.seed(42) # Seed fisso per avere sempre le stesse istanze durante i test
    
    # 1. Posizionamento del Deposito (Nodo 0)
    if depot_pos == 'c':
        depot_x, depot_y = grid_size / 2.0, grid_size / 2.0
    elif depot_pos == 'e':
        depot_x, depot_y = 0.0, grid_size / 2.0
    else:
        depot_x, depot_y = np.random.uniform(0, grid_size), np.random.uniform(0, grid_size)
    
    #Qui decido dove mettere il deposito, questo influisce sulla risoluzione del problema
        
    # 2. Posizionamento dei Clienti
    if dist_type == 'r':
        clienti_x = np.random.uniform(0, grid_size, n)
        clienti_y = np.random.uniform(0, grid_size, n)
    else:
        # Generazione in 3 macro-cluster distribuiti nella griglia
        num_clusters = 3
        #moltiplico per 0.2 e 0.8 per evitare cluster troppo vicini ai bordi, 
        #che potrebbero essere irrealistici per il PDSVRP
        #uniform permette di generare numeri casuali distribuiti uniformemente in un intervallo specificato, in questo caso tra 0.2*grid_size e 0.8*grid_size,
        #  garantendo così che i centri dei cluster siano posizionati all'interno della griglia ma non troppo
        #  vicini ai bordi.
        centers_x = np.random.uniform(0.2 * grid_size, 0.8 * grid_size, num_clusters)
        centers_y = np.random.uniform(0.2 * grid_size, 0.8 * grid_size, num_clusters)
        
        #  Per ogni cliente, scelgo un cluster e genero una posizione attorno al centro del cluster con una 
        # distribuzione normale
        clienti_x, clienti_y = [], []
        for _ in range(n):
            c = np.random.randint(0, num_clusters)
            #distribuzione normale attorno al centro del cluster con deviazione standard proporzionale alla dimensione d
            # ella griglia
            clienti_x.append(np.clip(np.random.normal(centers_x[c], grid_size / 12.0), 0, grid_size))
            clienti_y.append(np.clip(np.random.normal(centers_y[c], grid_size / 12.0), 0, grid_size))
            #la funzione clip serve a garantire che i clienti non vengano posizionati fuori dalla griglia, 
            # mantenendo così la coerenza dell'istanza generata.
        clienti_x, clienti_y = np.array(clienti_x), np.array(clienti_y)
        #con np array converto le liste di coordinate in array numpy, che sono più efficienti
        #  per le operazioni matematiche e di indicizzazione che seguiranno nella generazione dell'istanza.
   

    #aggoungo in posizione 0 le coordinate del deposito, in modo che all_x[0], all_y[0] rappresentino
    # sempre il deposito,
    all_x = np.insert(clienti_x, 0, depot_x)
    all_y = np.insert(clienti_y, 0, depot_y)
    
    # 3. Generazione dei pesi dei pacchi (Equazione 16 dell'articolo)
    # Con p < 0.86 peso tra 0 e 2.27 kg, altrimenti tra 2.27 e 68 kg
    p = np.random.uniform(0, 1, n)
    #where riceve dei vettori di condizioni e restituisce un array con valori scelti da due opzioni in base alla valutazione di quelle condizioni. 
    # In questo caso, se p < 0.86, viene assegnato un peso casuale tra 0 e 2.27 kg, altrimenti viene assegnato un peso casuale tra 2.27 e 68 kg.
    weights = np.where(p < 0.86, np.random.uniform(0, 2.27, n), np.random.uniform(2.27, 68, n))
    weights = np.insert(weights, 0, 0.0) # Il deposito ha peso zero
    
    # Costanti operative da Tabella 4
    v_truck = 30.0         # km/h
    v_drone = 40.0         # km/h
    max_drone_raggio = 12.0 # Raggio massimo (metà dell'endurance chilometrica totale di 24km)
    drone_max_weight = 2.27
    
    C_T = [] # Truck-only customers
    C_F = [] # Mode-free customers
    
    num_nodes = n + 1
    d_truck = np.zeros((num_nodes, num_nodes))
    t_truck = np.zeros((num_nodes, num_nodes))
    d_drone= np.zeros(num_nodes)
    t_drone = np.zeros(num_nodes)
    
    # 4. Calcolo delle Matrici di Distanza e Tempo e decisione dei Set
    for i in range(num_nodes):
        # Viaggio del drone (andata e ritorno dal deposito usando la distanza Euclidea in linea d'aria)
        dist_drone_0i = math.sqrt((all_x[0] - all_x[i])**2 + (all_y[0] - all_y[i])**2)
        d_drone[i] = 2.0 * dist_drone_0i
        t_drone[i] = d_drone[i] / v_drone
        
        if i > 0:
            # Selezione dell'eleggibilità del cliente basata su peso e distanza radiale
            if weights[i] <= drone_max_weight and dist_drone_0i <= max_drone_raggio:
                C_F.append(i)
            else:
                C_T.append(i)
                
        # OTTIMIZZAZIONE: Calcolo Distanza Manhattan per i camion (Rete Simmetrica)
        # Il ciclo parte da 'i + 1' per saltare la diagonale (i=j) e non ripetere calcoli già fatti
        for j in range(i + 1, num_nodes):
            # Calcolo lo spazio e il tempo una sola volta per ogni coppia di nodi
            dist_camion = abs(all_x[i] - all_x[j]) + abs(all_y[i] - all_y[j])
            tempo_camion = dist_camion / v_truck
            
            # Assegnazione "a specchio": compilo contemporaneamente l'andata e il ritorno
            d_truck[i, j] = dist_camion
            d_truck[j, i] = dist_camion
            
            t_truck[i, j] = tempo_camion
            t_truck[j, i] = tempo_camion

            
    # Definizione del limite temporale dei camion T
    #aumenta il limite temporale dei camion in modo dinamico in base alla distanza massima che un camion potrebbe
    #  dover percorrere per servire un cliente e tornare al deposito, garantendo così che le istanze generate siano 
    # realistiche e che il modello abbia una soluzione ammissibile.
    T_truck_max = 3.0 
    for i in range(1, num_nodes):
        T_truck_max = max(T_truck_max, 2.0 * t_truck[0, i] + 0.5)
        
        
    return {
        'n': n,
        'nodes': list(range(num_nodes)),
        'customers': list(range(1, num_nodes)),
        'C_T': C_T,
        'C_F': C_F,
        'd_truck': d_truck,
        't_truck': t_truck,
        'd_drone': d_drone,
        't_drone': t_drone,
        'weights': weights,
        'h': num_trucks,
        'm': num_drones,
        'Q_truck': 1300.0,       # Tabella 4
        'C_cost_truck': 1.25,    # Costo camion per km
        'C_cost_drone': 0.03,    # Costo drone per km
        'T_truck': T_truck_max,  # Orario massimo turno camion
        'T_drone': 3.0,          # Orario massimo turno drone
        'coords': list(zip(all_x, all_y))#creazione di una lista di tuple che rappresentano le coordinate (x, y) di ogni nodo, inclusi il deposito e i clienti.
    }