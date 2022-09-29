import numpy as np
import pandas as pd
from numba import jit,njit,float64
from itertools import combinations

@njit(float64[:](float64[:]))
def normalize(x):
    return x/np.sum(x)

binary_list = lambda n: np.vstack([[int(j) for j in'{:0{}b}'.format(i, n)] for i in range(2**n)])
        
def marginalized_like(alpha,beta=1,delta=False):
    coefs = np.array([[[1,-1,-1,1],[0,1,1,-2],[0,0,0,1]],[[1,-3,3,-1],[0,3,-5,2],[0,0,2,-1]]])
    if delta:
        beta_moments = (alpha/(alpha+beta))**np.arange(4)
        return (coefs*beta_moments).sum(axis=2)
    r=np.arange(3)
    beta_moments=np.append(1,np.cumprod((alpha+r)/(alpha+beta+r)))
    return (coefs*beta_moments).sum(axis=2)

#parameter's class
class params:
    def __init__(self,N_revs,N_friends,N_suggested,model,alpha=12,beta=12,delta=False):
        self.N_revs, self.N_suggested, self.model = N_revs, N_suggested, model
        
        self.ground = np.arange(N_revs)<N_friends
        self.configs = binary_list(N_revs) 
        self.errors = np.abs(self.configs-self.ground).sum(axis=1) 
        self.ground_ind = self.errors.argmin()
        
        self.suffix = 'cynical'
        self.model_name = self.suffix
        self.p_beta = np.zeros((2,3)) #will be overrun if quality
        if model == 'q':
            self.suffix = 'quality'
            self.model_name = 'quality'
            if delta:
                self.suffix+='_delta_{}'.format(alpha/(alpha+beta))
                self.run_qualities = lambda size: np.ones(size)*(alpha/(alpha+beta))
            else:
                self.suffix+='_{},{}'.format(alpha,beta)
                self.run_qualities = lambda size: np.random.beta(alpha,beta,size)
            self.p_beta = marginalized_like(alpha,beta,delta)
        if 2*N_friends!=N_revs:
            self.suffix+='_{}friends'.format(N_friends)   
            
        if self.model == 'c':
            self.S_slice,self.mp_slice,self.rho_slice = (300,1001),(300,1001),(200,500)
        elif self.model == 'q':
            self.S_slice,self.mp_slice,self.rho_slice = (3000,1001),(2500,1001),(1000,201)
        #self.model_name = self.suffix
        
    def run_suggested(self): 
        comb_list= np.stack(list(combinations(np.arange(self.N_revs),self.N_suggested)))
        suggested= comb_list[np.random.randint(comb_list.shape[0],size=400000)]
        return suggested


#Simulation
def sel_n(arr,n=1):
    np.random.shuffle(arr)
    return arr[:n]

def run_simulation_cynical(suggested,ground):
    R= [sel_n(ground[si],1)[0]+np.random.binomial(1,.5) for si in suggested]#.astype(int)
    return np.array(R).astype(int)

def run_simulation_quality(suggested,Q,ground):
    if isinstance(Q, np.ndarray):
        return np.array([run_simulation_quality(s,q,ground) for (s,q) in zip(suggested,Q)])
    r1=sel_n(ground[suggested],1)[0]*1
    r1_ac_rate = [Q**2,Q*(2-Q)][r1]
    return np.random.binomial(1,Q)+np.random.binomial(1,r1_ac_rate)

def run_simulation(gl,suggested):
    if gl.model == 'c':
        return run_simulation_cynical(suggested,gl.ground)
    elif gl.model == 'q':
        Q = gl.run_qualities(suggested.shape[0])
        return run_simulation_quality(suggested,Q,gl.ground)

def simulate(gl):
    suggested = gl.run_suggested()
    positives = run_simulation(gl,suggested)

    cols = ['suggestion_{}'.format(i) for i in range(1,suggested[0].size+1)]
    df = pd.DataFrame(suggested,columns=cols)
    df['positives'] = positives
    df.to_csv('sim_data_{}.csv'.format(gl.suffix),index=False)
    
    return suggested,positives


#likelihood calculation
@njit
def pr_c(c):
    nf,n=c.sum(),c.size
    return (1-nf/n)*np.array([.5,.5,0])+nf/n*np.array([0,.5,.5])

@njit
def pr_q(c,p_beta):
    nf,n=c.sum(),c.size
    return(np.array([[1-nf/n],[nf/n]])*p_beta).sum(axis=0)

@njit
def config_log_like(suggested,positives,model,c,p_beta): #Find log_like for a single data point and configuration
    if model=='c':
        return np.log(pr_c(c[suggested])[positives])
    if model =='q':
        return np.log(pr_q(c[suggested],p_beta)[positives])

def point_log_like(suggested,positives,model,configs,p_beta=np.ones((2,3))): #Find log_like for a single data point
    return np.array([config_log_like(suggested,positives,model,cc,p_beta) for cc in configs])  

#routines
@njit
def comp_routine(log_L,ind):
    log_post= np.log(normalize(np.ones(log_L.shape[1])))
    V = [normalize(np.exp(log_post))]
    for li in log_L[ind]:
        log_post +=li
        log_post -=log_post.max()
        pi = normalize(np.exp(log_post))
        V.append(pi)
    return V

def routine(fun,log_L,ind):
    return np.array([fun(p) for p in comp_routine(log_L,ind)])

def log_like(suggested,positives,gl):
    return np.stack([point_log_like(s,p,gl.model,gl.configs,gl.p_beta) for (s,p) in zip(suggested,positives)])

#Checking 
@njit
def entropy(p):
    p_pos=p[p!=0]
    return (-p_pos*np.log2(p_pos)).sum()

def posterior_entropy(log_L,ind):
    return routine(entropy,log_L,ind)

def reviewer_targeted(log_L,suggested,target,ind,gl):
    tar_inds = [target in s for s in suggested]
    log_L_i = log_L[tar_inds] #remove submissions where target was not suggested.
    i_friend = gl.configs[:,target]
    mar_i = lambda pi: (pi*i_friend).sum()
    return routine(mar_i,log_L_i,ind)

def max_post(log_L,ind,gl):
    maximum_posterior = lambda pi: gl.errors[pi.argmax()]
    return routine(maximum_posterior,log_L,ind)

def exp_error(log_L,ind,gl):
    expected_error = lambda pi: (pi*gl.errors).sum()
    return routine(expected_error,log_L,ind)

# Obtaining the data
def slices(i,j,lim):
    x = np.arange(lim)
    y = 1*x
    while y.size<=i*j:
        np.random.shuffle(x)
        y = np.concatenate((y,x))
    return (y[:i*j]).reshape(j,i)

def get_entropies(log_L,gl):
    S = np.stack([posterior_entropy(log_L,ind) for ind in slices(*gl.S_slice,log_L.shape[0])])
    df = pd.DataFrame(S)
    df.to_csv('entropy_{}.csv'.format(gl.suffix),index=False)
    return S

def get_map(log_L,gl):
    mp =  np.stack([max_post(log_L,ind,gl) for ind in slices(*gl.mp_slice,log_L.shape[0])])
    df = pd.DataFrame(mp)
    df.to_csv('map_{}.csv'.format(gl.suffix),index=False)
    return mp

def get_marginalized(log_L,suggested,gl):
    lim = min([sum([i in s for s in suggested]) for i in range(gl.N_revs)])

    rho_cf =[]
    for i in np.arange(gl.N_revs)[gl.ground]:
        rho_cf.append([reviewer_targeted(log_L,suggested,i,inds,gl) for inds in slices(*gl.rho_slice,lim)])

    rho_cr = []
    for i in np.arange(gl.N_revs)[np.logical_not(gl.ground)]:
        rho_cr.append([reviewer_targeted(log_L,suggested,i,inds,gl) for inds in slices(*gl.rho_slice,lim)])
        
    rho_cf,rho_cr = np.vstack(rho_cf), np.vstack(rho_cr)
    
    dff = pd.DataFrame(rho_cf)
    dff.to_csv('rho_friend_{}.csv'.format(gl.suffix),index=False)
    
    dfr = pd.DataFrame(rho_cr)
    dfr.to_csv('rho_rival_{}.csv'.format(gl.suffix),index=False)
    return np.vstack(rho_cf), np.vstack(rho_cr)
