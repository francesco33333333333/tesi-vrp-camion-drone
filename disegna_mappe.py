import numpy as np
import math
import matplotlib.pyplot as plt

def disegna_mappa_percorsi(data, active_arcs, drone_assignments, titolo="Mappa Percorsi PDSVRP"):
    """
    
    Mostra il deposito, i clienti e le rotte ottimali di camion e droni.
    """
    coords = data['coords']
    deposito = coords[0]
    C_T = data['C_T']
    C_F = data['C_F']
  
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('#F8F9FA')  
    ax.set_facecolor('#FFFFFF')         
    
    col_deposito = '#D62828'      
    col_ct = '#003049'            
    col_cf = '#2A9D8F'            
    col_rotta_camion = '#0077B6'  
  
    # 1. DISEGNO I NODI DELLA RETE
    
    ax.scatter(deposito[0], deposito[1], c=col_deposito, marker='s', s=220, 
               edgecolors='black', linewidths=1.5, label='Deposito (0)', zorder=5)
    ax.annotate("0", (deposito[0], deposito[1]), fontsize=11, fontweight='bold', 
                color='black', xytext=(8,8), textcoords='offset points')
    
    if C_T:
        cx_T = [coords[i][0] for i in C_T]
        cy_T = [coords[i][1] for i in C_T]
        ax.scatter(cx_T, cy_T, c=col_ct, marker='o', s=110, 
                   edgecolors='black', linewidths=1.2, label='Clienti Truck-Only (C_T)', zorder=4)
        for i in C_T:
            ax.annotate(str(i), (coords[i][0], coords[i][1]), fontsize=10, fontweight='bold',
                        color='#333333', xytext=(6,6), textcoords='offset points')
            
    if C_F:
        cx_F = [coords[i][0] for i in C_F]
        cy_F = [coords[i][1] for i in C_F]
        ax.scatter(cx_F, cy_F, c=col_cf, marker='^', s=130, 
                   edgecolors='black', linewidths=1.2, label='Clienti Mode-Free (C_F)', zorder=4)
        for i in C_F:
            ax.annotate(str(i), (coords[i][0], coords[i][1]), fontsize=10, fontweight='bold',
                        color='#333333', xytext=(6,6), textcoords='offset points')

    # 2. DISEGNO LE ROTTE DEI CAMION
    
    for idx, (i, j) in enumerate(active_arcs):
        x_vals = [coords[i][0], coords[j][0]]
        y_vals = [coords[i][1], coords[j][1]]
        label = 'Rotta Camion' if idx == 0 else ""
        ax.plot(x_vals, y_vals, c=col_rotta_camion, linestyle='-', linewidth=2.5, alpha=0.75, label=label, zorder=2)
        

    # 3. DISEGNO LE ROTTE DEI DRONI 
    if drone_assignments:
        # Trova gli ID unici dei droni usati
        droni_usati = sorted(list(set(d_id for cust, d_id in drone_assignments)))
        
        # Palette di colori moderni (Rosso, Arancione, Viola, Verde, Giallo senape)
        palette_droni = ['#E63946', '#F4A261', '#9B5DE5', '#2A9D8F', '#E9C46A']
        colori_droni = {d_id: palette_droni[i % len(palette_droni)] for i, d_id in enumerate(droni_usati)}
        
        droni_in_legenda = set() # Traccia quali droni abbiamo già messo in legenda
        
        for cust, d_id in drone_assignments:
            x_vals = [deposito[0], coords[cust][0]]
            y_vals = [deposito[1], coords[cust][1]]
            
            # Aggiungiamo l'etichetta solo la prima volta che incontriamo questo specifico drone
            if d_id not in droni_in_legenda:
                label = f'Rotta Drone {d_id}'
                droni_in_legenda.add(d_id)
            else:
                label = ""
                
            ax.plot(x_vals, y_vals, c=colori_droni[d_id], linestyle='--', linewidth=2, alpha=0.85, label=label, zorder=3)
        

    # IMPOSTAZIONI GRAFICHE FINALI
   
    ax.set_title(titolo, fontsize=16, fontweight='bold', color='#1D3557', pad=15)
    ax.set_xlabel('Coordinata X (km)', fontsize=12, fontweight='bold', color='#457B9D')
    ax.set_ylabel('Coordinata Y (km)', fontsize=12, fontweight='bold', color='#457B9D')
    
    ax.grid(True, linestyle='--', color='#CED4DA', linewidth=0.8, alpha=0.7, zorder=0)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ADB5BD')
    ax.spines['bottom'].set_color('#ADB5BD')
    
    # LEGENDA ESTERNA
    legenda = ax.legend(loc='upper left', 
                        bbox_to_anchor=(0.0, -0.15), 
                        ncol=3, 
                        frameon=True, 
                        fancybox=True, 
                        shadow=True, 
                        fontsize=10.5,
                        borderpad=1,
                        facecolor='#FFFFFF')
    legenda.set_zorder(10)
    
    plt.subplots_adjust(bottom=0.25)
    plt.show(block=True)