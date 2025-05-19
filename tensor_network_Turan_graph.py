#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 13:10:29 2024

@author: mplodzien
"""
import random
import quimb as qu
import quimb.tensor as qtn
import numpy as np
import time
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def create_turan_graph_layout(n, r, horizontal_rotation_angle=15, vertical_rotation_angle=15, 
                              horizontal_tilt=0.5, vertical_spacing=1.0, group_spacing=5):
    """
    Creates a custom layout for a Turán graph.

    Parameters:
    - n: Total number of nodes.
    - r: Number of partitions in the Turán graph.
    - horizontal_rotation_angle: Angle (in degrees) to rotate the group's line horizontally.
    - vertical_rotation_angle: Angle (in degrees) to rotate the group's line vertically.
    - horizontal_tilt: Horizontal spacing between nodes in a group.
    - vertical_spacing: Vertical spacing between nodes in a group.
    - group_spacing: Spacing between groups in the grid.

    Returns:
    - pos: A dictionary with node positions.
    """
    
    # Create the Turán graph
    G = nx.turan_graph(n, r)
    
    # Determine the size of each partition
    partition_sizes = [n // r + (1 if x < n % r else 0) for x in range(r)]
    
    # Generate partitions (list of nodes in each partition)
    partitions = []
    current_node = 0
    for size in partition_sizes:
        partitions.append(list(range(current_node, current_node + size)))
        current_node += size

    # Create a layout dictionary to position each group
    pos = {}
    grid_size = int(np.ceil(np.sqrt(r)))  # Arrange partitions in a square grid

    # Convert rotation angles to radians
    horizontal_radians = np.radians(horizontal_rotation_angle)
    vertical_radians = np.radians(vertical_rotation_angle)

    for i, partition in enumerate(partitions):
        # Calculate the grid position for this group
        row = i // grid_size
        col = i % grid_size
        group_center_x = col * group_spacing
        group_center_y = -row * group_spacing  # Negative to keep y-coordinates increasing upwards

        # Arrange nodes in a tilted and rotated line around the group's center
        for j, node in enumerate(partition):
            # Original unrotated positions
            x = j * horizontal_tilt
            y = -j * vertical_spacing
            
            # Apply horizontal rotation (rotation around the vertical axis)
            rotated_x = x * np.cos(horizontal_radians) - y * np.sin(horizontal_radians)
            rotated_y = x * np.sin(horizontal_radians) + y * np.cos(horizontal_radians)
            
            # Apply vertical rotation (rotation around the horizontal axis)
            final_x = rotated_x * np.cos(vertical_radians)
            final_y = rotated_y
            
            # Set final position relative to the group's center
            pos[node] = (group_center_x + final_x, group_center_y + final_y)
    
    return pos
 
#%%
sigma_x = np.array([[0,1.],[1.,0]])   
sigma_y = 1j*np.array([[0,-1.],[1.,0]])    
sigma_z = np.array([[1.,0],[0,-1.]])   

sigma_plus_z = 0.5*(sigma_x + 1j*sigma_y)
sigma_plus_x = 0.5*(sigma_y + 1j*sigma_z)
sigma_plus_y = 0.5*(sigma_z + 1j*sigma_x)
def get_sigma_plus(direction):
    if direction == 'z':  # sigma_plus = 0.5*(X[i] + 1j*Y[i])
        return qu.qu(sigma_plus_z)
    
    if direction == 'x':  # sigma_plus = 0.5*(Y[i] + 1j*Z[i])
        return qu.qu(sigma_plus_x)
    
    if direction == 'y':  # sigma_plus = 0.5*(Z[i] + 1j*X[i])
        return qu.qu(sigma_plus_y)

#%%

# Generate graph

# Parameters (you can change these values)
# n = 9  # Total number of nodes
r = 2  # Number of partitions
m = 12 # number of nodes in group

n = m*r
L = n

horizontal_rotation_angle = 20  # Rotation angle in degrees for horizontal tilt
vertical_rotation_angle = 130  # Rotation angle in degrees for vertical tilt
horizontal_tilt = 0.3  # Horizontal displacement within the group
vertical_spacing = 1  # Vertical spacing between nodes in a group
group_spacing = 5  # Spacing between groups in the grid

# Create the Turán graph
graph = nx.turan_graph(n, r)

 

fig, ax = plt.subplots(1, 1, figsize=(11,10))
# Get the custom layout for the Turán graph
pos = create_turan_graph_layout(n, r, horizontal_rotation_angle, vertical_rotation_angle, 
                                horizontal_tilt, vertical_spacing, group_spacing)

# # Draw the graph with the custom layout
nx.draw(graph, pos, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False)
# plt.show()

filename = "fig_turan_graph_L." + str(L) + "_r." + str(r) + ".png"
path = "./"
plt.savefig(path+filename, format="png", dpi = 500, bbox_inches='tight')

 

#%%

# Generate graph

 

plt.figure(figsize=(12,12))

plt.subplots_adjust(bottom=0.1, right=0.99, top=0.9)

r = 2  # Number of partitions
m = 6 # number of nodes in group

n = m*r
 
 
graph = nx.turan_graph(n, r)
pos = create_turan_graph_layout(n, r, horizontal_rotation_angle, vertical_rotation_angle, 
                                horizontal_tilt, vertical_spacing, group_spacing)
plt.subplot(221)
nx.draw(graph, pos, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False)
 

r = 4  # Number of partitions
m = 3 # number of nodes in group

n = m*r
 
graph = nx.turan_graph(n, r)
pos = create_turan_graph_layout(n, r, horizontal_rotation_angle, vertical_rotation_angle, 
                                horizontal_tilt, vertical_spacing, group_spacing)
plt.subplot(223)
nx.draw(graph, pos, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False)

r = 3  # Number of partitions
m = 6 # number of nodes in group

n = m*r
 
graph = nx.turan_graph(n, r)
pos = create_turan_graph_layout(n, r, horizontal_rotation_angle, vertical_rotation_angle, 
                                horizontal_tilt, vertical_spacing, group_spacing)
plt.subplot(222)
nx.draw(graph, pos, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False)

r = 6  # Number of partitions
m = 2 # number of nodes in group

n = m*r
 
graph = nx.turan_graph(n, r)
pos = create_turan_graph_layout(n, r, horizontal_rotation_angle, vertical_rotation_angle, 
                                horizontal_tilt, vertical_spacing, group_spacing)
plt.subplot(224)
nx.draw(graph, pos, node_size=80, alpha=1, node_color="blue", edge_color="black", with_labels=False)

filename = "fig_turan_graphs" + ".png"
path = "./"
plt.savefig(path+filename, format="png", dpi = 500, bbox_inches='tight')

plt.show()

#%%
 

#%%
K = len(graph.edges())



 

L = len(graph.nodes)
print(f"L = {L}")
# for edge in graph.edges():
    # print(edge)

nodes_to_qubit = {}
for idx, nodes in enumerate(graph.nodes()):
    print(idx)
    nodes_to_qubit[nodes] = idx 

#%% 




#%%


 
# Create the quantum circuit
cluster_state = qtn.Circuit(L)

# Apply Hadamard gates to all qubits
for qubit_i in range(0,L):
    cluster_state.apply_gate('H', qubit_i)
 

 

for edge in graph.edges:

    node_i = nodes_to_qubit[edge[0]] 
    node_j = nodes_to_qubit[edge[1]]  
    
 
    print(node_i, node_j)

 

 
    cluster_state.apply_gate('CZ', node_i, node_j)




#%%
 
cluster_state_ = cluster_state.copy()

 
 


# sigma_plus_vec = ["z",  "x", "y"]
sigma_plus_vec = ["z",  "x"]
# sigma_plus_vec = ["x"]
all_sigma_plus_sets  = [p for p in itertools.product(sigma_plus_vec, repeat=L)]

data_optimized_correlator = []
for idx, sigma_plus_set in enumerate(all_sigma_plus_sets):
    start = time.time()
   
    ########
    
    cluster_state_ = cluster_state.copy()
    for qubit_i in range(0,L):    
        direction = sigma_plus_set[qubit_i]
        cluster_state_.apply_gate(get_sigma_plus(direction), qubit_i)
    
#     # Calculate the inner product between the original state and the modified state
    inner_product = cluster_state.psi.H@cluster_state_.psi

    Epsilon = np.abs(inner_product)**2
    NEpsilon_ent = 4**L*Epsilon
    NEpsilon_bell = 2**L*Epsilon
    
    
    base = 4
    Q_ent = np.emath.logn(base, NEpsilon_ent) 
    

    base = 2
    Q_bell = np.emath.logn(base, NEpsilon_bell)     
    
    
    data_dict_local = {
                        "L"                     : L,
                        "Q_ent"                 : np.around(Q_ent,2),
                        "Q_bell"                : np.around(Q_bell,2),
                        "sigma_plus_set"        : sigma_plus_set,
                        }
    stop = time.time()
    duration = (stop - start)/60
    print("Duration : {:2.2f} [m]".format(duration))
    data_optimized_correlator.append(data_dict_local)
    string = "TN | L = " + str(L) + " | " + str(idx) + "/" + str(len(sigma_plus_vec)**L) + " "
    string = string + "".join(sigma_plus_set) + "\n"
    string = string +  " | Q_ent = " + "{:2.5f}".format(Q_ent)
    string = string +  " | Q_bell = " + "{:2.5f}".format(Q_bell)
    print(string)

data_optimized_correlator = pd.DataFrame(data_optimized_correlator)


Q_ent_max = data_optimized_correlator["Q_ent"].max()

data_Q_ent_max = data_optimized_correlator[ data_optimized_correlator["Q_ent"] == Q_ent_max]

string_optimal_directions = ""
for idx, row in data_Q_ent_max.iterrows():
    Q_bell = row["Q_bell"]
    Q_ent = row["Q_ent"]
    gamma = L - (Q_ent + 1)
    string_optimal_directions = string_optimal_directions + "".join(row["sigma_plus_set"]).lower()
    string_optimal_directions = string_optimal_directions + r"$ | Q_{ent} = $" + str(Q_ent) + " | $Q_{bell} = $" + str(Q_bell) + "\n"
print("=============")
# print("m = ", m)
# print("n = ", n)
print(string_optimal_directions)

#%%
# fig, ax = plt.subplots(1, 1, figsize=(5,5))
fig, ax = plt.subplots(1, 1, figsize=(10,10))


# nx.draw(graph, with_labels=True, node_color='lightblue', ax=ax)
title_string = "TN | L = " + str(L) + " | edges: K = " + str(K) + " | groups: r  = " + str(r)  + " | nodes in group: m = " + str(m)
title_string = title_string + r" | $\gamma = $" + str(gamma) + "\n"
title_string = title_string + " | # = " + str(len(data_Q_ent_max))  + " | \n "
title_string = title_string + string_optimal_directions


ax.set_title(title_string, fontsize = 16)
nx.draw(graph, pos, with_labels=True, node_color='lightblue', ax=ax)

# nx.draw(graph,   with_labels=True, node_color='lightblue', ax=ax)

# parameters = '_L.' + str(L) + '_K.' + str(K) + '_m.' + str(m) + '_n.'+str(n)

# fig.savefig('hexagonal_fig' + parameters + '.png', dpi=fig.dpi)


ax.set_title(title_string, fontsize = 16)
m = 3
# nx.draw(graph, pos, with_labels=True, node_color='lightblue', ax=ax)
# nx.draw(graph, with_labels=True, node_color='lightblue', ax=ax)
plt.savefig("./figures_turan_graphs/fig_TN_turan_graph_L." + str(L)  + "_r." + str(r) + "_m." + str(m) + ".png", format="png", dpi = 200)
plt.show()
#%%




