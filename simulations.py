#import numpy as np
#import pandas as pd
from rev_lib import *
import time



gl = params(10,5,3,'c') #cynical model
#gl = params(10,7,3,'c') #cynical with 7 friends
#gl = params(10,9,3,'c') #cynical with 9 friends

#gl = params(10,5,3,'q') #quality model
#gl = params(10,7,3,'q') #quality with 7 friends
#gl = params(10,9,3,'q') #quality with 9 friends


#gl = params(10,5,3,'q',2.0625,.6875)#expected quality of .75 #.05 variance
#gl = params(10,5,3,'q',13.3125,4.4375) #.01 variance
#gl = params(10,5,3,'q',27.375,9.125) #.005 variance
#gl = params(10,5,3,'q',3,1,delta=True) #0 variance

#gl = params(10,5,3,'q',2,2)#here starts expected quality of .5 #.05 variance
#gl = params(10,5,3,'q',12,12) #.01 variance
#gl = params(10,5,3,'q',24.5,24.5) #.005 variance
#gl = params(10,5,3,'q',1,1,delta=True)

#gl = params(10,5,3,'q',.6875,2.0625)#here starts expected quality of .25 #.05 variance
#gl = params(10,5,3,'q',4.4375,13.3125) #.01 variance
#gl = params(10,5,3,'q',9.125,27.375) #.005 variance
#gl = params(10,5,3,'q',1,3,delta=True) #0 variance

         
print('Simulating {}'.format(gl.suffix))
suggested,positives = simulate(gl)
 
start = time.time()
log_L = log_like(suggested,positives,gl)
end = time.time()
print('        Calculating likelihood took {} mins'.format((end-start)/60))
    
start = time.time()
get_entropies(log_L,gl)
get_map(log_L,gl)
end = time.time()
print('        Calculating entropy and map took {} mins'.format((end-start)/60))
    
start = time.time()
get_marginalized(log_L,suggested,gl)
end = time.time()
print('        Calculating marginalized took {} mins'.format((end-start)/60))
