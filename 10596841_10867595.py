from amplpy import AMPL, Environment
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import distance
import statistics
import networkx as nx
import timeit

INSTANCE_1 = "minimart-I-50"
INSTANCE_2 = "minimart-I-100"

# This function elaborate an optimal solution of the first half of the problem (opening problem).
# It uses a cplex solver in an AMPL environment (using the amplpy library) 
def solve_opening_problem(instance):

    # Environment('full path to the AMPL installation directory')
    ampl = AMPL(Environment("C:\Program Files\\ampl"))
    ampl.set_option('solver', 'cplex')
    # Load the AMPL model from file
    ampl.read("10596841_10867595_opening.mod")
    # Read data
    ampl.read_data(instance + ".dat")
    # Solve the model
    print('Opening Problem')
    ampl.solve()

    # Get the opening problem objective function value
    opening_cost = ampl.get_objective('obj').value()
    # Retrieve the model parameters and variables in order to use them for the next problem
    cx = np.array(list(ampl.get_parameter('Cx').getValues().to_dict().values()), dtype='int')
    cy = np.array(list(ampl.get_parameter('Cy').getValues().to_dict().values()), dtype='int')
    n = int(ampl.get_parameter('n').getValues().to_list()[0])
    x = np.array(list(ampl.get_variable('x').getValues().to_dict().values()), dtype='int')
    vc = int(ampl.get_parameter('Vc').getValues().to_list()[0])
    fc = int(ampl.get_parameter('Fc').getValues().to_list()[0])
    capacity = int(ampl.get_parameter('capacity').getValues().to_list()[0])

    # Creating 3 arrays to store: markets locations, markets x coordinates, markets y coordinates
    markets = []
    cx_markets = []
    cy_markets = []

    for i in range(n):
        if(x[i] == 1):
            markets.append(i + 1)
            cx_markets.append(cx[i])
            cy_markets.append(cy[i])

    return opening_cost, cx, cy, cx_markets, cy_markets, markets, vc, fc, capacity

# This function elaborate an approximate solution of the second half of the problem (truck routing)
def solve_routing_problem(markets, cx, cy, vc, fc, capacity):
    n = len(markets) # n is the number of markets
    n_to_refurbish = n - 1 # n_to_refurbish is the number of market that trucks has to refurbish
    a_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            a_matrix[i][j] = distance.euclidean((cx[i], cy[i]), (cx[j], cy[j]))

    G = nx.from_numpy_matrix(np.matrix(a_matrix), create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {i:markets[i] for i in range(n)})

    ZONE_UP = {markets[i]:'up' for i in range(1,n) if(cy[i] >= cy[0])} 
    ZONE_DOWN = {markets[i]:'down' for i in range(1,n) if(cy[i] < cy[0])}
    nx.set_node_attributes(G, False, "stocked")
    nx.set_node_attributes(G, values=ZONE_UP, name="zone")
    nx.set_node_attributes(G, values=ZONE_DOWN, name="zone")
    nx.set_node_attributes(G, values={1:"root"}, name="zone")

    if(n_to_refurbish % capacity == 0):
        trucks_number = int(n_to_refurbish / capacity)
    else:
        trucks_number = int(n_to_refurbish / capacity + 1)

    trucks_paths = []

    
    for i in range(trucks_number):
        trucks_paths.append([])
        node = 1
        for j in range(capacity + 1):
            trucks_paths[i].append(node)
            G.nodes[node]['stocked'] = True
            l_t, path_t = nx.single_source_dijkstra(G, source = node)
            best_next_node = 0

            if(j >= capacity - 2):
                l_1, path_1 = nx.single_source_dijkstra(G, source = 1)
                
                l_t[1] = 1000000
                for key in l_t.keys():
                    if(key != 1):
                        l_t[key] = statistics.mean([l_t[key], l_1[key]])
                l_t = {k: v for k, v in sorted(l_t.items(), key=lambda item: item[1])}
                

            for key in l_t.keys():
                if(key != 1 and G.nodes[node]['zone'] == G.nodes[key]['zone']):
                    l_t[key] -= min(l_t.keys())
            l_t = {k: v for k, v in sorted(l_t.items(), key=lambda item: item[1])}

            for k in l_t.keys():
                if(not G.nodes[k]['stocked']):
                    best_next_node = k
                    break

            node = best_next_node
            if(node == 0):
                break

        trucks_paths[i].append(1)
    
    # Output formatting data

    trucks_path_dicts = []
    trucks_active_arcs = []

    for k in range(trucks_number):
        trucks_active_arcs.append([])
        trucks_path_dicts.append({})

        for i in range(len(trucks_paths[k]) - 1):
            if(trucks_paths[k][i] == 1):
                if(trucks_paths[k][i + 1] == 1):
                    trucks_path_dicts[k][1] = 1
                    trucks_active_arcs[k].append((1,1))
                else:
                    trucks_path_dicts[k][1] = trucks_paths[k][i + 1]
                    trucks_active_arcs[k].append((1,trucks_paths[k][i + 1]))
            else:
                if(trucks_paths[k][i + 1] == 1):
                    trucks_path_dicts[k][trucks_paths[k][i]] = 1
                    trucks_active_arcs[k].append((trucks_paths[k][i],1))
                else:
                    trucks_path_dicts[k][trucks_paths[k][i]] = trucks_paths[k][i + 1]
                    trucks_active_arcs[k].append((trucks_paths[k][i],trucks_paths[k][i + 1]))

    routing_cost = 0
    for k in range(len(trucks_active_arcs)):
        for (i,j) in trucks_active_arcs[k]:
            routing_cost += G.edges[i,j]['weight']
    routing_cost += trucks_number * fc


    return routing_cost, trucks_number, trucks_active_arcs, trucks_path_dicts

# This function plots the market in their location (using the cx and cy coordinates) and the routes for each truck
def plot_result(instance, cx_original, cy_original, markets, cx_markets, cy_markets, trucks_active_arcs, trucks_number):

    # Market Locations with routes plotting
        plt.figure(figsize=(8,6))
        plt.scatter(cx_markets[1:], cy_markets[1:], c="b")
        plt.scatter(cx_markets[0], cy_markets[0], c="r")
        for i in range(len(markets)):
            plt.annotate(str(markets[i]), (cx_markets[i], cy_markets[i]))
        color = iter(cm.rainbow(np.linspace(0, 1, trucks_number)))
        for active_arcs in trucks_active_arcs:
            c = next(color)
            for (i,j) in active_arcs:
                plt.plot([cx_original[i - 1], cx_original[j - 1]], [cy_original[i - 1], cy_original[j - 1]], c=c, alpha=0.3)
        plt.title(instance + " market locations and refurbish routes")
        plt.show()

# This function write the problem solution in a txt file
def solution_writer(instance, total_cost, opening_cost, routing_cost, markets, trucks_path_dicts, trucks_number):
    
    f = open("solution-" + instance + ".txt","w")
    f.write(str(total_cost) + "\n")
    f.write(str(opening_cost) + "\n")
    f.write(str(routing_cost) + "\n")
    for l in markets:
        if(l > 1):
            f.write("," + str(l))
        else:
            f.write(str(l))
    f.write("\n")
    for i in range(trucks_number):
        elem = 1
        f.write(str(1))
        while(trucks_path_dicts[i][elem] != 1):
            f.write("," + str(trucks_path_dicts[i][elem]))
            elem = trucks_path_dicts[i][elem]
        f.write("," + str(1))
        f.write("\n")

def main():
    for i in range(2):
        
        if i == 0:
            instance = INSTANCE_1
        else:
            instance = INSTANCE_2

        print("\nINSTANCE: " + instance + '\n')

        # Timer to evaluate algorithm execution time
        timer_start = timeit.default_timer()

        # The problem is splitted in 2 sub problems: opening and routing problem. Call functions to solve the 2 sub-problems
        opening_cost, cx_original, cy_original, cx_markets, cy_markets, markets, vc, fc, capacity = solve_opening_problem(instance)
        routing_cost, trucks_number, trucks_active_arcs, trucks_path_dicts = solve_routing_problem(markets, cx_markets, cy_markets, vc, fc, capacity)
        
        timer_stop = timeit.default_timer()
        
        print("Elapsed time to solve the problem: ", timer_stop - timer_start)

        plot_result(instance, cx_original, cy_original, markets, cx_markets, cy_markets, trucks_active_arcs, trucks_number)

        # Total solution

        total_cost = opening_cost + routing_cost

        print("Total Optimized Objective Function value: ", total_cost)

        # Write solution in a txt file
        solution_writer(instance, total_cost, opening_cost, routing_cost, markets, trucks_path_dicts, trucks_number)
        
    
    
if __name__ == '__main__':
    main()