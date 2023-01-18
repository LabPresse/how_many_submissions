#import numpy as np
#import pandas as pd
from rev_lib import *
import time



#This code performs the simulation and inference. 
# this is done by the object gl of the class rev_lib.params, which is called as:
# params(reviewers,friends,suggestions,model,alpha,beta)
#
# with parameters:
# reviewers   - #of reviewers to be classified
# friends     - #of friendly reviewers in the dataset
# suggestions - #of suggested reviewers per submission
# model       - 'c' for cynical, 'q' for quality
# alpha = 12  - parameter alpha (in the quality model)
# beta  = 12  - parameter beta  (in the quality model)
#
# below we write all of the parameters used in the manuscript, they are run one at a time. In order to reproduce our results, uncomment the desired one and run the script individually. 


gl = params(10,5,3,'c') #cynical model
#gl = params(10,7,3,'c') #cynical with 7 friends
#gl = params(10,9,3,'c') #cynical with 9 friends

#gl = params(10,5,3,'q') #quality model
#gl = params(10,7,3,'q') #quality with 7 friends
#gl = params(10,9,3,'q') #quality with 9 friends


gl = params(10,5,3,'q',2.0625,.6875)#expected quality of .75 #.05 variance
gl = params(10,5,3,'q',13.3125,4.4375) #.01 variance
gl = params(10,5,3,'q',27.375,9.125) #.005 variance
gl = params(10,5,3,'q',3,1,delta=True) #0 variance

gl = params(10,5,3,'q',2,2)#here starts expected quality of .5 #.05 variance
#gl = params(10,5,3,'q',12,12) #.01 variance
gl = params(10,5,3,'q',24.5,24.5) #.005 variance
#gl = params(10,5,3,'q',1,1,delta=True)

gl = params(10,5,3,'q',.6875,2.0625)#here starts expected quality of .25 #.05 variance
gl = params(10,5,3,'q',4.4375,13.3125) #.01 variance
gl = params(10,5,3,'q',9.125,27.375) #.005 variance
#gl = params(10,5,3,'q',1,3,delta=True) #0 variance

##BIASED PRIORS
#gl = params(10,5,3,'c',prior_name = 'biased') #cynical model
#gl = params(10,9,3,'c',prior_name = 'biased') #cynical with 9 friends

#gl = params(10,5,3,'q',prior_name = 'biased') #quality model
#gl = params(10,9,3,'q',prior_name = 'biased') #quality with 9 friends

print(gl.suffix)
to_simulate = True

try:
    df= pd.read_csv('sim_data_{}.csv'.format(gl.suffix))
    suggested = df.to_numpy()[:,:3]
    positives = df.to_numpy()[:,3]
    
    print("We already have a simulation for these parameters. Do you wish do run another simulation? if so press 'y'")
    x = input()
    to_simulate = (x=='y')
    
except:
    pass

if to_simulate:
    print('Simulating {}'.format(gl.suffix))
    suggested,positives = simulate(gl)
    
has_S_map,has_marg,has_3rd = False,False,False
    
    
    
if not to_simulate:
    try:
        S = pd.read_csv('entropy_{}.csv'.format(gl.suffix))
        mp = pd.read_csv('map_{}.csv'.format(gl.suffix))
        has_S_map = True
    except:
        pass
    
    try:
        f = pd.read_csv('rho_friend_{}.csv'.format(gl.suffix))
        r = pd.read_csv('rho_rival_{}.csv'.format(gl.suffix))
        has_marg = True
    except:
        pass
    
    try:
        p3 = pd.read_csv('prob3_{}.csv'.format(gl.suffix))
        has_3rd = True
    except:
        pass
    
    
    
if all([has_S_map,has_marg,has_3rd]):
    print("We already have a trajectories for all metrics of these parameters. Do you wish do run another simulation? if so press 'y'.")
    x = input()
    to_redo = (x=='y')    
elif any([has_S_map,has_marg,has_3rd]):
    print("We already have a trajectories for some metrics of these parameters. Do you wish do run another simulation? if so press 'y'.")
    print("Otherwise it will only run trajectories for metrics we do not found.")
    x = input()
    to_redo = (x=='y')
else:
    to_redo=True

    
    
if to_redo or (not all([has_S_map,has_marg,has_3rd])):
    start = time.time()
    log_L = log_like(suggested,positives,gl)
    end = time.time()
    print('        Calculating likelihood took {} mins'.format((end-start)/60))
    
if to_redo or (not has_S_map):
    start = time.time()
    get_entropies(log_L,gl)
    get_map(log_L,gl)
    end = time.time()
    print('        Calculating entropy and map took {} mins'.format((end-start)/60))
    
if to_redo or (not has_marg):
    start = time.time()
    get_marginalized(log_L,suggested,gl)
    end = time.time()
    print('        Calculating marginalized took {} mins'.format((end-start)/60))

if to_redo or (not has_3rd):
    start = time.time()
    get_third(log_L,suggested,gl)
    end = time.time()
    print('        Calculating 3rd took {} mins'.format((end-start)/60))
