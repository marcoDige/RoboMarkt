from amplpy import AMPL, Environment
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from scipy.spatial import distance
import statistics
import networkx as nx
import timeit
import sys

INSTANCE_1 = "minimart-I-50"
INSTANCE_2 = "minimart-I-100"
AMPL_INSTALLATION_PATH = "C:\Program Files\\ampl"

# This function elaborate an optimal solution of the first half of the problem (opening problem).
# It uses a cplex solver in an AMPL environment (using the amplpy library) 
def solve_opening_problem(instance):

    # Environment('full path to the AMPL installation directory')
    ampl = AMPL(Environment(AMPL_INSTALLATION_PATH))
    ampl.set_option('solver', 'cplex')
    # Load the AMPL model from file
    ampl.read("opening_model.mod")
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
    print("\nRouting Problem")
    n = len(markets) # n is the number of markets
    n_to_supply = n - 1 # n_to_supply is the number of market that trucks has to refurbish
    d_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d_matrix[i][j] = distance.euclidean((cx[i], cy[i]), (cx[j], cy[j])) * vc

    G = nx.from_numpy_matrix(np.matrix(d_matrix), create_using=nx.DiGraph)
    G = nx.relabel_nodes(G, {i:markets[i] for i in range(n)})

    ZONE_UP = {markets[i]:'up' for i in range(1,n) if(cy[i] >= cy[0])} 
    ZONE_DOWN = {markets[i]:'down' for i in range(1,n) if(cy[i] < cy[0])}

    nx.set_node_attributes(G, False, "supply")
    nx.set_node_attributes(G, values=ZONE_UP, name="zone")
    nx.set_node_attributes(G, values=ZONE_DOWN, name="zone")
    nx.set_node_attributes(G, values={1:"root"}, name="zone")

    # Number of truck is setted at the minimum number of truck necessary to supply all markets
    if(n_to_supply % capacity == 0):
        trucks_number = int(n_to_supply / capacity)
    else:
        trucks_number = int(n_to_supply / capacity + 1)

    best_obj = 0
    pz_ub = 50
    pz_lb = 0
    best_penalty = -1
    best_trucks_path = []

    if(n_to_supply % capacity == 0 and trucks_number == int(n_to_supply / capacity)):
        possible_capacity_reduct = 0
    else:
        possible_capacity_reduct = int((trucks_number * capacity - n_to_supply)/trucks_number)
    
    for first_step in range(n_to_supply):
        for capacity_reducer in range(possible_capacity_reduct + 1):
            for magnet in range(int(capacity/2)):
                penalty_changed = 0
                for penalty in range(pz_lb,pz_ub + 1):
                    # Reset all supply attributes for a new execution
                    for i in G.nodes():
                        G.nodes[i]['supply'] = False
                    
                    # Reset cost to elaborate a new one
                    routing_cost = trucks_number * fc

                    trucks_path = []

                    for i in range(trucks_number):
                        trucks_path.append([])
                        node = 1
                        trucks_path[i].append(node)
                        G.nodes[node]['supply'] = True

                        for j in range(capacity - capacity_reducer):
                            next_node = 0
                            priority = {i:G.edges[node, i]['weight'] for i in G.neighbors(node) if(not G.nodes[i]['supply'])}
                            
                            # Last x element has to go closer to the node 1
                            if(j >= capacity - magnet):
                                root_closeness = {i:G.edges[1, i]['weight'] for i in G.neighbors(1) if(not G.nodes[i]['supply'])}
                                
                                for key in priority.keys():
                                    priority[key] = statistics.mean([priority[key], root_closeness[key]])
                            
                            # Normalization of the priority score
                            priority = {key: value/max(priority.values()) for key,value in priority.items()}

                            # Adjust the priority score with +x if the nodes aren'first_step in the same zone
                            for key in priority.keys():
                                if(G.nodes[node]['zone'] != G.nodes[key]['zone']):
                                    priority[key] += penalty/100 

                            # Sort the priority dict by ascending score
                            priority = {k: v for k, v in sorted(priority.items(), key=lambda item: item[1])}

                            if(i == 0 and node == 1):
                                next_node = list(priority.keys())[first_step]
                            elif(bool(priority)):
                                next_node = list(priority.keys())[0]

                            node = next_node
                            if(node == 0):
                                break
                            
                            trucks_path[i].append(node)
                            G.nodes[node]['supply'] = True
                            routing_cost += G.edges[trucks_path[i][-2], trucks_path[i][-1]]['weight']

                            # Discard the solution if the routing cost overtake the best obj result
                            if(routing_cost > best_obj and best_obj != 0):
                                break

                        trucks_path[i].append(1)
                        routing_cost += G.edges[trucks_path[i][-2], trucks_path[i][-1]]['weight']

                        # Discard the solution if the routing cost overtake the best obj result
                        if(routing_cost > best_obj and best_obj != 0):
                            break

                    if(best_obj == 0):
                        best_obj = routing_cost
                        best_trucks_path = trucks_path
                        best_penalty = penalty
                        penalty_changed = 1
                    elif(routing_cost < best_obj):
                        best_obj = routing_cost
                        best_trucks_path = trucks_path
                        best_penalty = penalty
                        penalty_changed = 1

                if(best_penalty != 0 and penalty_changed):
                    pz_ub = best_penalty + 5
                    pz_lb = best_penalty - 5
                    if(pz_lb < 0):
                        pz_lb = 0
        
        # If the best penality zone remain unchanged for a lot of first_steps, value upper bound an lower bound will converge
        if(first_step % trucks_number == 0):
            pz_ub -= 1
            pz_lb += 1
            if(pz_lb > pz_ub):
                pz_ub = best_penalty
                pz_lb = pz_ub
        
        print("Current best solution: "  + str(best_obj))

    return best_obj, best_trucks_path

# This function plots the market in their location (using the cx and cy coordinates) and the routes for each truck
def plot_result(instance, cx_original, cy_original, markets, cx_markets, cy_markets, trucks_path):

    # Market Locations with routes plotting
        plt.figure(figsize=(8,6))
        plt.scatter(cx_original[0:], cy_original[0:], c="#DDDEE8", label = "Village")
        plt.scatter(cx_markets[1:], cy_markets[1:], c="#021ACB", label = "Market")
        plt.scatter(cx_markets[0], cy_markets[0], c="#FF2D00", label = "Depot")
        for i in range(len(markets)):
            plt.annotate(str(markets[i]), (cx_markets[i], cy_markets[i]))
        color = iter(cm.rainbow(np.linspace(0, 1, len(trucks_path))))
        for path in trucks_path:
            c = next(color)
            for i in range(len(path) - 1):
                plt.plot([cx_original[path[i] - 1], cx_original[path[i+1] - 1]], [cy_original[path[i] - 1], cy_original[path[i+1] - 1]], c=c, alpha=0.3)
        plt.title(instance + " Village and Market locations with Vehicle Routing")
        plt.xlabel("Cx")
        plt.ylabel("Cy")
        plt.legend(loc="upper left")
        plt.show()

# This function write the problem solution in a txt file
def solution_writer(instance, total_cost, opening_cost, routing_cost, markets, trucks_path):
    
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
    for i in range(len(trucks_path)):
        for j in range(len(trucks_path[i])):
            if(trucks_path[i][j] == 1 and j == 0):
                f.write(str(trucks_path[i][j]))
            else:
                f.write("," + str(trucks_path[i][j]))
        if(i < len(trucks_path) - 1):
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
        routing_cost, trucks_path = solve_routing_problem(markets, cx_markets, cy_markets, vc, fc, capacity)
        
        timer_stop = timeit.default_timer()
        
        print("Elapsed time to solve the problem: ", timer_stop - timer_start)

        # Total solution

        total_cost = opening_cost + routing_cost

        print("Total Optimized Objective Function value: ", total_cost)

        # Write solution in a txt file
        solution_writer(instance, total_cost, opening_cost, routing_cost, markets, trucks_path)

        if(len(sys.argv) == 2 and sys.argv[1] == '--plot'):
            plot_result(instance, cx_original, cy_original, markets, cx_markets, cy_markets, trucks_path)
        
    
    
if __name__ == '__main__':
    main()