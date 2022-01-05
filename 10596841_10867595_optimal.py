from amplpy import AMPL, Environment
import numpy as np

from scipy.spatial import distance

from matplotlib import pyplot as plt
from matplotlib import cm

import timeit

from vrpy import VehicleRoutingProblem
import networkx as nx
from networkx import relabel_nodes, set_node_attributes

INSTANCE_1 = "minimart-I-50"
INSTANCE_2 = "minimart-I-100"

def solve_opening_problem(instance):

    # --- AMPL optimal solution of the first half of the problem (opening problem) ---
        
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

    opening_cost = ampl.get_objective('obj').value()
    # Retrieve the model parameters and variables in order to use it for the next problem
    cx = np.array(list(ampl.get_parameter('Cx').getValues().to_dict().values()), dtype='int')
    cy = np.array(list(ampl.get_parameter('Cy').getValues().to_dict().values()), dtype='int')
    n = int(ampl.get_parameter('n').getValues().to_list()[0])
    x = np.array(list(ampl.get_variable('x').getValues().to_dict().values()), dtype='int')
    vc = int(ampl.get_parameter('Vc').getValues().to_list()[0])
    fc = int(ampl.get_parameter('Fc').getValues().to_list()[0])
    capacity = int(ampl.get_parameter('capacity').getValues().to_list()[0])

    return opening_cost, cx, cy, n, x, vc, fc, capacity

def solve_routing_problem_optimal(markets, cx, cy, vc, fc, capacity):

    # --- Optimal/Approximated with heuristic solution of the second half of the problem (refurnishing problem) ---

    n = len(markets)
    n_to_refurbish = n - 1

    # Compute the min number of trucks
    if(n_to_refurbish % capacity == 0):
        t = int(n_to_refurbish/capacity)
    else:
        t = int(n_to_refurbish/capacity + 1)

    print("Number of markets to refurbish: ", n_to_refurbish)
    print("Number of trucks: ", t)

    # Environment('full path to the AMPL installation directory')
    ampl = AMPL(Environment("C:\Program Files\\ampl"))
    ampl.set_option('solver', 'cplex')
    # Load the AMPL model from file
    ampl.read("10596841_10867595_refurbishing.mod")

    # Set data

    ampl.get_parameter('n').set_values([n])
    ampl.get_parameter('t').set_values([t])
    ampl.get_parameter('Cx').set_values(cx)
    ampl.get_parameter('Cy').set_values(cy)
    ampl.get_parameter('Vc').set_values([vc])
    ampl.get_parameter('Fc').set_values([fc])
    ampl.get_parameter('capacity').set_values([capacity])

    # Solve the model
    print('\nRouting Problem')
    ampl.solve()

    refurbishing_cost = ampl.get_objective('obj').value()
    y = ampl.get_variable('y').getValues().to_dict()

    trucks_active_arcs = []
    trucks_path_dicts = []

    for i in range(t):
        trucks_active_arcs.append([])
        trucks_path_dicts.append({})
        
    for (i,j,k), value in y.items():
        if(value == 1):
            trucks_active_arcs[int(k) - 1].append((markets[int(i) - 1],markets[int(j) - 1]))
            trucks_path_dicts[int(k) - 1][markets[int(i) - 1]] = markets[int(j) - 1]

    return refurbishing_cost, t, trucks_active_arcs, trucks_path_dicts

def solve_routing_problem_vrpy(markets, cx, cy, vc, fc, capacity):

    print('\nRouting Problem')

    n = len(markets)
    n_to_refurbish = n - 1
    
    print("Number of markets to refurbish: ", n_to_refurbish)

    c = np.zeros((n + 1,n + 1), dtype=[("cost", float)])

    for i in range(n + 1):
        for j in range(n + 1):
            if(i == n or j == 0):
                c[i,j] = 0
            elif(j == n):
                c[i,j] = distance.euclidean((cx[i],cy[i]),(cx[0],cy[0])) * vc
            else:
                c[i,j] = distance.euclidean((cx[i],cy[i]),(cx[j],cy[j])) * vc

    G = nx.from_numpy_matrix(np.matrix(c), create_using=nx.DiGraph)
    DEMAND = {i:1 for i in range(1,n)}
    set_node_attributes(G, values=DEMAND, name="demand")
    G = relabel_nodes(G, {0: "Source", n: "Sink"})
    prob = VehicleRoutingProblem(G, load_capacity=capacity, num_vehicles=n_to_refurbish, fixed_cost= fc)
    prob.solve()
    
    refurbishing_cost = prob.best_value
    t = len(prob.best_routes)
    print("Number of trucks: ", t)
    trucks_active_arcs = []
    trucks_path_dicts = []

    for i in range(t):
        trucks_active_arcs.append([])
        trucks_path_dicts.append({})
    
    for k, value in prob.best_routes.items():
        for i in range(len(value) - 1):
            if(value[i] == 'Source'):
                if(value[i + 1] == 'Sink'):
                    trucks_path_dicts[int(k) - 1][markets[0]] = markets[0]
                    trucks_active_arcs[int(k) - 1].append((markets[0],markets[0]))
                else:
                    trucks_path_dicts[int(k) - 1][markets[0]] = markets[value[i + 1]]
                    trucks_active_arcs[int(k) - 1].append((markets[0],markets[value[i + 1]]))
            else:
                if(value[i + 1] == 'Sink'):
                    trucks_path_dicts[int(k) - 1][markets[value[i]]] = markets[0]
                    trucks_active_arcs[int(k) - 1].append((markets[value[i]],markets[0]))
                else:
                    trucks_path_dicts[int(k) - 1][markets[value[i]]] = markets[value[i + 1]]
                    trucks_active_arcs[int(k) - 1].append((markets[value[i]],markets[value[i + 1]]))
           

    return refurbishing_cost, t, trucks_active_arcs, trucks_path_dicts
    
def plot_result(instance, cx_original, cy_original, markets, cx_markets, cy_markets, trucks_active_arcs, t):

    # Market Locations with routes plotting
        plt.figure(figsize=(8,6))
        plt.scatter(cx_markets[1:], cy_markets[1:], c="b")
        plt.scatter(cx_markets[0], cy_markets[0], c="r")
        for i in range(len(markets)):
            plt.annotate(str(markets[i]), (cx_markets[i], cy_markets[i]))
        color = iter(cm.rainbow(np.linspace(0, 1, t)))
        for active_arcs in trucks_active_arcs:
            c = next(color)
            for (i,j) in active_arcs:
                plt.plot([cx_original[i - 1], cx_original[j - 1]], [cy_original[i - 1], cy_original[j - 1]], c=c, alpha=0.3)
        plt.title(instance + " market locations and refurbish routes")
        plt.show()

def main():
    for i in range(2):
        
        if i == 0:
            instance = INSTANCE_1
        else:
            instance = INSTANCE_2

        print("\nINSTANCE: " + instance + '\n')

        timer_start = timeit.default_timer()

        opening_cost, cx_original, cy_original, n, x, vc, fc, capacity = solve_opening_problem(instance)

        # Creating 3 arrays to store: markets locations, markets x coordinates, markets y coordinates
        markets = []
        cx_markets = []
        cy_markets = []

        for i in range(n):
            if(x[i] == 1):
                markets.append(i + 1)
                cx_markets.append(cx_original[i])
                cy_markets.append(cy_original[i])

        #refurbishing_cost, t, trucks_active_arcs, trucks_path_dicts = solve_routing_problem_optimal(markets, cx_markets, cy_markets, vc, fc, capacity)
        refurbishing_cost, t, trucks_active_arcs, trucks_path_dicts = solve_routing_problem_vrpy(markets, cx_markets, cy_markets, vc, fc, capacity)
        
        timer_stop = timeit.default_timer()
        
        print("Elapsed time to solve the problem: ", timer_stop - timer_start)

        plot_result(instance, cx_original, cy_original, markets, cx_markets, cy_markets, trucks_active_arcs, t)

        # Total solution

        total_cost = opening_cost + refurbishing_cost

        print("Total Optimized Objective Function value: ", total_cost)

        # Output file writing
        
        f = open("optimal_solution-" + instance + ".txt","w")

        f.write(str(total_cost) + "\n")
        f.write(str(opening_cost) + "\n")
        f.write(str(refurbishing_cost) + "\n")
        for l in markets:
            if(l > 1):
                f.write("," + str(l))
            else:
                f.write(str(l))
        f.write("\n")

        
        for i in range(t):
            elem = 1
            f.write(str(1))
            while(trucks_path_dicts[i][elem] != 1):
                f.write("," + str(trucks_path_dicts[i][elem]))
                elem = trucks_path_dicts[i][elem]
            f.write("," + str(1))
                
            f.write("\n")
        
    
    
if __name__ == '__main__':
    main()