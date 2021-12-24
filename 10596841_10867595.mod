
param n; # Number of locations

param range; # Range in km of maximum distance accetable
param Vc; # Cost of the truck driver for 1 km
param Fc; # Fixed Cost for one truck driver
param capacity; # Max number of visitable store for one driver

set L := 1..n; # Locations

param Cx{L}; # x coordinate of village i
param Cy{L}; # y coordinate of village i
param Dc{L}; # Cost of installation in village i
param usable{L}; # Constructing permission of village i
param d{i in L, j in L} := sqrt((Cx[i]-Cx[j]) * (Cx[i]-Cx[j]) + (Cy[i]-Cy[j]) * (Cy[i]-Cy[j])); # Distance from the village i to the village j
param r{i in L, j in L} := if d[i,j] <= range then 1 else 0; # Village i is in the range of the house j or not, boolean parameter

var x{L} binary; # Boolean variable that indicates if the village site has been choosed as store site

minimize obj:
	sum{i in L} (Dc[i] * x[i]);
	
s.t. first_house_market:
	x[1] = 1;
	
s.t. activation_constraint {i in L}:
	x[i] <= usable[i];
	
s.t. range_constraint {i in L}: # This constraint check, for each village, if there's at least one village choosed as store site in the km range
	sum{j in L} (x[j] * r[i,j]) >= 1;