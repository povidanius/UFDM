import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import dcor
import sys
sys.path.insert(0, "../")
from MI import calculate_MI, get_sigma, HSIC
from ufdm import UFDM
import pandas as pd
from uuid import uuid4
from scipy.stats import norm, uniform, cauchy, t as student_t
import uuid
from IPython import embed
import json
from sklearn.datasets import make_moons


#np.random.seed(42)
#torch.manual_seed(42)

#
distributions0 = [
        #"independent_student_t", "independent_gaussian", "independent_uniform", 
        "linear_strong",
        "linear_weak", "logarithmic",
        "quadratic", "polynomial", "contaminated_sine", "conditional_variance"
    ]

distributions1 = ['mixture_bimodal_marginal','mixture_bimodal','circular','gaussian_copula','clayton_copula','interleaved_moons'] 
#distributions1 = ['gaussian_copula','clayton_copula']

permisable_x_distributions = ['uniform','gaussian','student_t']
x_dist_type = 'complex_patterns'

if len(sys.argv) == 3:
    n_samples, d = int(sys.argv[1]), int(sys.argv[2])
    distributions = distributions1
elif len(sys.argv) == 4:
    n_samples, d, x_dist_type = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    assert x_dist_type in ['uniform','gaussian','student_t'], f"distribution should be in {permisable_x_distributions}"    
    distributions = distributions0
else:
    print("Usage:")
    print(f"{sys.argv[0]} n_samples, dimension -- for experiment with complext patterns")
    print("or")
    print(f"{sys.argv[0]} n_samples, dimension, x_distribution -- for experiment with simple patterns")
    sys.exit(0)

print(f'{x_dist_type} {n_samples} {d}')

freq = 6
n_permutations = 500  
num_experiments = 20   # Number of trials per distribution
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dim_x = d
dim_y = d
num_iter = 100 #150
lr = 0.025
alpha = 0.05  # Significance level

def sample_clayton_copula(n_samples: int, dim_x: int, dim_y: int, theta: float):
    """
    Samples from a Clayton copula using the Marshall-Olkin algorithm
    and transforms the marginals to be standard normal.

    Args:
        n_samples (int): The number of samples to generate.
        dim_x (int): The dimension of the first variable set (x).
        dim_y (int): The dimension of the second variable set (y).
        theta (float): The Clayton copula dependence parameter (theta > 0).
                         Higher theta means stronger lower-tail dependence.

    Returns:
        (torch.Tensor, torch.Tensor): Tensors x and y with the specified dimensions
                                      and a Clayton dependence structure.
    """
    if theta <= 0:
        raise ValueError("Clayton copula parameter theta must be positive.")

    d = dim_x + dim_y
    epsilon = 1e-10  # For numerical stability

    # Step 1: Sample from a Gamma distribution
    # The shape is 1/theta. We need one gamma sample for each row in our output.
    g = np.random.gamma(shape=1.0/theta, scale=1, size=(n_samples, 1))

    # Step 2: Sample from independent standard Exponential distributions
    e = np.random.exponential(scale=1, size=(n_samples, d))

    # Step 3: Combine to get the uniform variables with Clayton dependence
    # This is the core transformation: v_i = (1 + e_i / g)^(-1/theta)
    v = (1 + e / g)**(-1.0 / theta)

    # Step 4: Clip for numerical stability before transforming to normal
    v = np.clip(v, epsilon, 1 - epsilon)

    # Step 5: Transform to standard normal marginals using the inverse CDF (ppf)
    g_normal = norm.ppf(v)  # Shape: (n_samples, d)

    # Step 6: Split into x and y and convert to PyTorch tensors
    x = torch.tensor(g_normal[:, :dim_x], dtype=torch.float32)
    y = torch.tensor(g_normal[:, dim_x:], dtype=torch.float32)

    return x, y


def contaminated_sine(x_clean, d, freq=6, p=0.05, scale=25.0):
    n = x_clean.shape[0]
    n_clean = int((1-p)*n)
    # core dependence ---------------------------------------------------------
    #X0 = torch.randn(n_clean, d)
    X0 = x_clean[:n_clean,:]
    
    Y0 = torch.sin(freq * (X0 @ torch.randn(d,1))).repeat(1, d)    # strong link
    Y0 += 0.1*torch.randn_like(Y0)
    # gross outliers ----------------------------------------------------------
    X1 = scale*torch.randn(n-n_clean, d)           # very far away
    Y1 = scale*torch.randn(n-n_clean, d)           # independent noise
    #print(f'{n_clean},{n-n_clean}')    
    #print(f'inliers{X0.shape[0]}, outliers {X1.shape[0]}')
    # combine & shuffle -------------------------------------------------------
    X  = torch.cat([X0, X1])
    Y  = torch.cat([Y0, Y1])
    idx = torch.randperm(n)
    return X[idx], Y[idx]

# Define a wide range of multivariate distributions
def generate_data(dist_type, n_samples, dim_x, dim_y,  x_type ='uniform'):

    if x_type == 'gaussian':
        x = torch.randn(n_samples, dim_x) 
    elif x_type == 'uniform':        
        x = torch.rand(n_samples, dim_x)
    elif x_type == 'student_t':
        x = torch.tensor(student_t.rvs(3, size=(n_samples, dim_x)), dtype=torch.float32)        
    
    noise = 0.1 * torch.randn(n_samples, dim_y)

    
    if dist_type == "independent_gaussian":
        y = torch.randn(n_samples, dim_y)
    elif dist_type == "independent_uniform":
        y = torch.tensor(uniform.rvs(size=(n_samples, dim_y)), dtype=torch.float32)        
    elif dist_type == "independent_student_t":
        df = 3  # > 2  ⇒ finite variance
        y = torch.tensor(student_t.rvs(df, size=(n_samples, dim_y)), dtype=torch.float32)        
    elif dist_type == "linear_strong":
        proj = nn.Linear(dim_x, dim_y)
        y = proj(x) + noise
    elif dist_type == "linear_weak":
        proj = nn.Linear(dim_x, dim_y)
        y = 0.3 * proj(x) + noise
    elif dist_type == 'logarithmic':
        proj = nn.Linear(dim_x, dim_y)
        px = proj(x)
        y = torch.log(1.0 + px*px) + noise
    elif dist_type == "quadratic":
        proj = nn.Linear(dim_x, dim_y)
        y = torch.mul(proj(x), proj(x)) + noise
    elif dist_type == "polynomial":
        proj = nn.Linear(dim_x, dim_y)
        px = proj(x)
        y = 0.5 * px**3 - px**2 + noise        
    elif dist_type == "high_freq_sine":
        proj = nn.Linear(dim_x, 1, bias=False) # Project to 1D
        high_frequency = 20.0
        px = proj(x)
        y = torch.sin(high_frequency * px).repeat(1, dim_y) + noise
    elif dist_type == "rotational_sine":
        if dim_x < 2:
            # This dependency requires at least 2 dimensions for x
            x = torch.cat([x, torch.randn(n_samples, 1)], dim=1)
            dim_x = x.shape[1] # Update dim_x if we added a dimension

        angle = torch.atan2(x[:, 1], x[:, 0]).reshape(-1, 1)
        frequency = 8.0
        # The y value depends on the angle of the x vector
        y_signal = torch.sin(frequency * angle)
        # Replicate the signal across all y dimensions and add some noise
        y = y_signal.repeat(1, dim_y) + noise        
    elif dist_type == "radial_sine":    
        r = torch.norm(x, dim=1, keepdim=True)              # (n,1)
        omega = 11.0
        y = torch.sin(omega * r).repeat(1, dim_x) + noise   # broadcast to (n, dim_x)        
    elif dist_type == "circular":
        t = torch.linspace(0, 2 * np.pi, n_samples).reshape(-1, 1)
        r = torch.randn(n_samples, 1) * 0.2 + 1.0
        x = torch.cat([r * torch.cos(t), r * torch.sin(t), torch.randn(n_samples, dim_x - 2)], dim=1)
        y = torch.cat([r * torch.cos(t + torch.randn(n_samples, 1)), r * torch.sin(t + torch.randn(n_samples, 1)), torch.randn(n_samples, dim_y - 2)], dim=1) + noise
    elif dist_type == "checkerboard":
        x = torch.randn(n_samples, dim_x)
        proj = nn.Linear(dim_x, 1)
        px = proj(x).reshape(-1, 1)
        y = torch.cat([torch.sin(3 * px), torch.randn(n_samples, dim_y - 1)], dim=1) + noise
    elif dist_type == "gaussian_copula":
        rho = 0.85
        d = dim_x + dim_y  # total dimension of joint Gaussian

        # Construct block covariance matrix with rho across X-Y blocks
        Sigma = rho * torch.ones((d, d)) + (1 - rho) * torch.eye(d)
        z = np.random.multivariate_normal(mean=np.zeros(d), cov=Sigma.numpy(), size=n_samples)

        # Apply CDF and inverse CDF to impose standard marginals
        u = norm.cdf(z)  # shape: (n_samples, d)
        g = norm.ppf(u)  # fully Gaussian marginals, copula dependence

        # Split into x and y
        x = torch.tensor(g[:, :dim_x], dtype=torch.float32)
        y = torch.tensor(g[:, dim_x:], dtype=torch.float32)        
    elif dist_type == "clayton_copula":
        theta = 5.0
        x,y = sample_clayton_copula(n_samples, dim_x, dim_y, theta)
    elif dist_type == "mixture_bimodal_marginal":
        cluster = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        x = torch.zeros(n_samples, dim_x)
        y = torch.zeros(n_samples, dim_y)
        for i in range(n_samples):
            if cluster[i] == 0:
                x[i] = torch.randn(1, dim_x) + torch.tensor([2.0] + [0.0] * (dim_x - 1))
                y[i] = torch.randn(1, dim_y) + torch.tensor([2.0] + [0.0] * (dim_y - 1))
            else:
                x[i] = torch.randn(1, dim_x) - torch.tensor([2.0] + [0.0] * (dim_x - 1))
                y[i] = torch.randn(1, dim_y) - torch.tensor([2.0] + [0.0] * (dim_y - 1))
        y += noise
    elif dist_type == "mixture_bimodal":
        cluster = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
        x = torch.zeros(n_samples, dim_x)
        y = torch.zeros(n_samples, dim_y)
        for i in range(n_samples):
            if cluster[i] == 0:
                x[i] = torch.randn(1, dim_x) + torch.tensor([2.0] * dim_x)
                y[i] = torch.randn(1, dim_y) + torch.tensor([2.0] * dim_y)
            else:
                x[i] = torch.randn(1, dim_x) - torch.tensor([2.0] * dim_x)
                y[i] = torch.randn(1, dim_y) - torch.tensor([2.0] * dim_y)
        y += noise        
    elif dist_type == "interleaved_moons":       
        # 1. Generate two separate "two moons" datasets.
        # The labels (0 or 1) indicate which moon a point belongs to.
        X_moons, x_labels = make_moons(n_samples=n_samples, noise=0.05)
        Y_moons, y_labels = make_moons(n_samples=n_samples, noise=0.05)

        # 2. Create the interleaved dependency.
        x = torch.zeros(n_samples, 2)
        y = torch.zeros(n_samples, 2)
        
        # Get points from the top and bottom moons of Y
        y_moon_0 = torch.from_numpy(Y_moons[y_labels == 0])
        y_moon_1 = torch.from_numpy(Y_moons[y_labels == 1])

        for i in range(n_samples):
            x[i] = torch.from_numpy(X_moons[i])
            if x_labels[i] == 0: # If x is on the bottom moon...
                # ...pick a random y from the top moon.
                rand_idx = torch.randint(0, y_moon_1.shape[0], (1,))
                y[i] = y_moon_1[rand_idx]
            else: # If x is on the top moon...
                # ...pick a random y from the bottom moon.
                rand_idx = torch.randint(0, y_moon_0.shape[0], (1,))
                y[i] = y_moon_0[rand_idx]

        # 3. Add final noise and pad with extra dimensions
        x = torch.cat([x, torch.randn(n_samples, dim_x - 2)], dim=1)
        y = torch.cat([y, torch.randn(n_samples, dim_y - 2)], dim=1)
        
        # Ensure dtype is correct
        x = x.to(torch.float32)
        y = y.to(torch.float32) #+ noise      
    elif dist_type == "conditional_variance":
        proj = nn.Linear(dim_x, dim_y)
        px = proj(x)
        y = 1.0*px + torch.randn(n_samples, dim_y) * px + noise    
    elif dist_type == 'contaminated_sine':
        x,y = contaminated_sine(x, dim_x, p=0.05, freq=freq, scale=25.0)
    elif dist_type == "high_freq_rotational":
        # 1. Define parameters for this specific case
        high_frequency = 10.0  # A high k-value
        
        # 2. Generate X from a standard normal distribution (a spherical cloud)
        # We override the x_type here because this example requires a Gaussian X
        x = torch.randn(n_samples, dim_x)
        
        # 3. Create the secret, random projection vector (the hidden direction)
        # This vector is unknown to the dependence measures.
        v = torch.randn(dim_x, 1)
        w = v / torch.norm(v)  # Normalize to a unit vector
        
        # 4. Project the high-dimensional X onto this 1D direction
        p = x @ w  # Shape: (n_samples, 1)
        
        # 5. Generate Y based on a sine wave of the projection
        # The signal is replicated across all dimensions of Y.
        y_signal = torch.sin(high_frequency * p)
        y = y_signal.repeat(1, dim_y) + noise # Add a small amount of noise
        
        return x, y
           
    else:        
        raise ValueError(f"Unknown distribution type: {dist_type}")
    return x, y

# Permutation test function for UFDM
def permutation_test(x, y, measure_func, n_permutations, **kwargs):
    num_restarts = 0
    original_stat = measure_func(x, y, debug=False, **kwargs)
    if num_restarts > 0:
        for i in range(num_restarts):
            original_stat = max(original_stat,measure_func(x, y, debug=False, **kwargs))
            print(f'original stat ufdm {original_stat}')


    perm_stats = []
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y

    for _ in range(n_permutations):
        if _ % 100 == 0:
            print(f"permutation {_}/{n_permutations}")
        perm_y = torch.tensor(np.random.permutation(y_np), device=y.device, dtype=y.dtype)

        perm_stat = measure_func(x, perm_y, **kwargs)
        if num_restarts > 0:
            for i in range(num_restarts):
                perm_stat = max(perm_stat,measure_func(x, perm_y, debug=False, **kwargs))

        perm_stats.append(perm_stat)
      
    perm_stats = np.array(perm_stats)
    exceedances = np.sum(perm_stats >= original_stat)
    p_value     = (exceedances + 1) / (n_permutations + 1)    
 
    return original_stat, p_value

# Permutation test function
def permutation_test1(x, y, measure_func, n_permutations, **kwargs):

    original_stat =  measure_func(x, y, **kwargs)
    perm_stats = []
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    #x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    for _ in range(n_permutations):
        if _ % 100 == 0:
            print(f"permutation {_}/{n_permutations}")
        #perm_x = torch.tensor(np.random.permutation(x_np), device=y.device, dtype=y.dtype)
        perm_y = torch.tensor(np.random.permutation(y_np), device=y.device, dtype=y.dtype)

        #perm_stat, _, _ = measure_func(x, perm_y, a = alpha, b = beta, optimise=False, **kwargs)
        perm_stat = measure_func(x, perm_y, **kwargs)

        perm_stats.append(perm_stat)
      
    perm_stats = np.array(perm_stats)
    exceedances = np.sum(perm_stats >= original_stat)
    p_value     = (exceedances + 1) / (n_permutations + 1)    
   
    return original_stat, p_value

# Measure functions
def compute_ufdm(x, y, num_iter=num_iter, lr=lr, a = False, b = False, optimise=True, debug=False):
    model = UFDM(dim_x, dim_y, lr=lr, weight_decay=2.5, device=device, init_scale_shift=[1.0, 1.0]) 
    model.reset()
    history_dep = []

    model.svd_initialise0(x.to(device), y.to(device))   


    for i in range(num_iter):
        dep = model.forward(x.to(device), y.to(device), normalize=True)
        dep_val = dep.item()            # scalar float; avoids extra .clone/.detach later
        history_dep.append(dep_val)


    if debug:
            plt.plot(history_dep)
            filename = f"ufdm_plot_{uuid.uuid4().hex[:8]}.png"
            plt.savefig(filename)
            plt.close()                        

    return np.max(history_dep) 
    
def compute_dcor(x, y):
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    return dcor.u_distance_correlation_sqr(x_np, y_np)

def compute_hsic(x, y, sx=None, sy=None):
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    sx = get_sigma(x).detach() if sx is None else sx
    sy = get_sigma(y).detach() if sy is None else sy
    return HSIC(x.to(device), y.to(device), sx, sy).detach().cpu().numpy()

def compute_mef(x, y, sx=None, sy=None):
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    y_np = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else y
    sx = get_sigma(x).detach() if sx is None else sx
    sy = get_sigma(y).detach() if sy is None else sy
    return calculate_MI(x.to(device), y.to(device), sx, sy, 1.01, normalize=False).detach().cpu().numpy()




measures = ["UFDM","DCOR", "HSIC", "MEF"]
results = {dist: {m: [] for m in measures} for dist in distributions}

for dist_type in distributions:
    print(f"Processing {dist_type}")
    for exp in range(num_experiments):
        print(f'Experiment {exp}/{num_experiments}')
        # Generate data
        x, y = generate_data(dist_type, n_samples, dim_x, dim_y, x_type=x_dist_type)
        x = x.detach()
        y = y.detach()
               
        # Compute sigmas for HSIC and MEF
        sx = get_sigma(x).detach()        
        sy = get_sigma(y).detach()
        
        # Perform permutation tests            
        if "UFDM" in measures:
                _, p_ufdm = permutation_test(x, y, compute_ufdm, n_permutations, num_iter=num_iter) 
                results[dist_type]["UFDM"].append(p_ufdm)
                print(f'original_stat {_}, p_ufdm {p_ufdm}')              

        if "DCOR" in measures:       
                _, p_dcor = permutation_test1(x, y, compute_dcor, n_permutations)
                results[dist_type]["DCOR"].append(p_dcor)
                print(f'original_stat {_}, p_dcor {p_dcor}')      

        if "HSIC" in measures:
                _, p_hsic = permutation_test1(x, y, compute_hsic, n_permutations, sx=sx, sy=sy)
                print(f'original_stat {_}, p_hsic {p_hsic}') 
                results[dist_type]["HSIC"].append(p_hsic)

        if "MEF" in measures:
                _, p_mef =  permutation_test1(x, y, compute_mef, n_permutations, sx=sx, sy=sy)             
                results[dist_type]["MEF"].append(p_mef)
                print(f'original_stat {_}, p_mef {p_mef}')      


        print(results)

        if not os.path.exists(f'./results3/{n_samples}/{x_dist_type}/'):
            os.makedirs(f'./results3/{n_samples}/{x_dist_type}/')        
        with open(f'./results3/{n_samples}/{x_dist_type}/data_{n_samples}_{d}_{freq}.json', 'w') as fp:
            json.dump(results, fp)
