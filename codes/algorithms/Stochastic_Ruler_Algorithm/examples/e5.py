#facility location problem 

import numpy as np
import dissim as ds


def facility_loc(x):
  #normal demand
  X1,Y1 = x['x1'],x['y1'] 
  X2,Y2 = x['x2'],x['y2'] 
  X3,Y3 = x['x3'],x['y3'] 
  avg_dist_daywise = []
  T0 = 30
  n = 6
  for t in range(T0):
      total_day = 0                  ##### total distance travelled by people
      ###### now finding nearest facility and saving total distance 
      #travelled in each entry of data
      for i in range(n):
          for j in range(n):
              demand=-1
              while(demand<0):    
                  demand = np.random.normal(180, 30, size=1)[0]
              total_day += demand*min(abs(X1-i)+abs(Y1-j) ,
                                      abs(X2-i)+abs(Y2-j),abs(X3-i)+abs(Y3-j) ) 
              ### total distance from i,j th location to nearest facility
      avg_dist_daywise.append(total_day/(n*n))    
  return sum(avg_dist_daywise)/T0

dom5 = {'x1' : [i for i in range(1,7)], 'y1':[i for i in range(1,7)],
       'x2' : [i for i in range(1,7)], 'y2':[i for i in range(1,7)],
       'x3' : [i for i in range(1,7)], 'y3':[i for i in range(1,7)]}
sr_userdef5 = ds.stochastic_ruler(space = dom5, maxevals = 10, prob_type = 'opt_sol',func = facility_loc, neigh_structure=2)
print(sr_userdef5.optsol())
sr_userdef5.plot_minh_of_z()