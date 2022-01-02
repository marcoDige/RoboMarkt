# This model is formalized as a CVRP (Capacitated Vehicle Problem)

param n; # Number of markets
param t; # Number of trucks

set M := 1..n; # Markets with first market
set T := 1..t; # Trucks

param Cx{M}; # x coordinate of market i
param Cy{M}; # y coordinate of market i
param Vc; #Cost of the truck driver for 1 km
param Fc; # Fixed Cost for one truck driver
param capacity; # Max number of visitable market for one driver
param c{i in M, j in M} := sqrt((Cx[i]-Cx[j]) * (Cx[i]-Cx[j]) + (Cy[i]-Cy[j]) * (Cy[i]-Cy[j])) * Vc; # Cost of moving from the node i to the node j


var y{M,M,T} binary; # Binary variable with value 1 if the arc from the node i to the node j is in the optimal route and is driven by vehicle k
var u{i in M: i != 1} integer;

minimize obj:
    Fc * t + sum{i in M, j in M, k in T} (y[i,j,k] * c[i,j]);

s.t. no_autoarc{i in M, k in T}:
    y[i,i,k] = 0;

s.t. node_is_entered_once{j in M: j != 1}:
    sum{k in T, i in M} (y[i,j,k]) = 1;

s.t. vehicles_leaves_nodes_entered{j in M, k in T}:
    sum{i in M} (y[i,j,k]) = sum{i in M} (y[j,i,k]);

s.t. every_vheicles_leaves_first_market{k in T}:
    sum{i in M: i != 1} (y[1,i,k]) = 1;

s.t. capacity_constraint{k in T}:
    sum{i in M, j in M: j != 1} (y[i,j,k]) <= capacity;

s.t. u_capacity_constraint1{i in M, j in M, k in T : i != 1 and j != 1 and i != j}:
    u[i] - u[j]  >= 1 - capacity * (1 - y[i,j,k]);

s.t. u_capacity_constraint2{i in M: i != 1}:
    u[i] <= capacity



    
