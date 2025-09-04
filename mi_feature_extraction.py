import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import sys
from MI import calculate_MI
import torch.nn.functional as F


class MiFeatureExtraction(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005,  input_projection_dim = 0, output_projection_dim=0, weight_decay=0.01, orthogonality_enforcer = 1.0, device="cpu", init_scale_shift=[1,0]):
        super(MiFeatureExtraction, self).__init__()
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.lr = lr
        self.input_projection_dim = input_projection_dim
        self.output_projection_dim = output_projection_dim
        self.weight_decay = weight_decay
        self.orthogonality_enforcer = orthogonality_enforcer
        self.device = device
        self.init_scale_shift = init_scale_shift
        self.reset()

    def reset(self):


        param_list = []
        self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
        param_list = param_list + list(self.projection_x.parameters()) 
        
        self.bnx = nn.BatchNorm1d(self.dim_x, affine=True).to(self.device)
        self.bny = nn.BatchNorm1d(self.dim_y, affine=True).to(self.device)
        param_list +=  list(self.bnx.parameters()) + list(self.bny.parameters())

        self.optimizer = torch.optim.AdamW(param_list, lr=self.lr) 

    def pairwise_distances(self, x):
        #x should be two dimensional
        instances_norm = torch.sum(x**2,-1).reshape((-1,1))
        return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

    def GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ /sigma)


    def HSIC(self, x, y, s_x, s_y):
        """ calculate HSIC from https://github.com/danielgreenfeld3/XIC"""
        m,_ = x.shape #batch size
        K = self.GaussianKernelMatrix(x,s_x)
        L = self.GaussianKernelMatrix(y,s_y)
        H = torch.eye(m) - 1.0/m * torch.ones((m,m))
        H = H.float().cuda()
        HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
        return HSIC


    def DCOR(self, x, y, exponent=1.0, squared=True, eps=1e-12):
            """
            Unbiased distance correlation (Székely & Rizzo).
            If squared=True returns R^2, else returns R.
            """
            if x.shape[0] != y.shape[0]:
                raise ValueError("x and y must have same number of samples")
            if x.shape[0] < 4:
                raise ValueError("Need at least 4 samples for unbiased estimator")

            n = x.shape[0]

            def pairwise_dist(z):
                return torch.cdist(z, z, p=2).pow(exponent)

            def u_center(D):
                row_sums = D.sum(dim=1, keepdim=True)
                col_sums = D.sum(dim=0, keepdim=True)
                total_sum = D.sum()
                A = D - row_sums/(n-2) - col_sums/(n-2) + total_sum/((n-1)*(n-2))
                A = A.clone()
                A.fill_diagonal_(0.0)
                return A

            a = u_center(pairwise_dist(x))
            b = u_center(pairwise_dist(y))

            dcov2  = (a * b).sum() / (n * (n - 3))
            # unbiased estimator can be slightly negative in finite samples
            dcov2  = torch.clamp(dcov2, min=0.0)
            dvar_x = (a * a).sum() / (n * (n - 3))
            dvar_y = (b * b).sum() / (n * (n - 3))

            denom = torch.sqrt(torch.clamp(dvar_x * dvar_y, min=eps))
            r2 = dcov2 / denom  # this is distance correlation squared

            return r2 if squared else torch.sqrt(r2)

    def get_sigma(self, x):
        pd = self.pairwise_distances(x)
        return torch.median(pd)

    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = self.bnx(x)

        proj = self.projection_x(x)            
        return proj


    def forward(self, x, y, update = True, normalize=True, measure='mi'):
        x = x.to(self.device)
        y = y.to(self.device)
        if normalize:
            x = self.bnx(x)
            y = self.bny(y)
        
              
        x = self.projection_x(x)
        sx = max(0.000001 , self.get_sigma(x).detach())
        sy = max(0.000001 , self.get_sigma(y).detach())

        #print("sx sy {} {}".format(sx, sy))
        if measure == 'mi':
            mi = calculate_MI(x,y, sx, sy, 1.01, normalize=False, epsilon = 1e-5)
        elif measure == 'hsic':
            mi = self.HSIC(x,y, sx, sy)
        elif measure == 'dcor':
            mi = self.DCOR(x,y)

        
        if update:
            loss = -mi 
            if self.input_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_x.weight,self.projection_x.weight.T) - torch.eye(self.input_projection_dim).to(self.device)) # maximise => negative
            
            if self.output_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_y.weight,self.projection_y.weight.T) - torch.eye(self.output_projection_dim).to(self.device)) # maximise => negative

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

        return mi


if __name__ == "__main__":
    import numpy as np
    import dcor

    # For reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

    def compare_once(n=1000, d1=15, d2=15, dependent=True, noise=0.1, device="cpu"):
        """
        Build a synthetic dataset (independent or dependent),
        compute unbiased squared distance correlation with:
          (a) our torch implementation (MiFeatureExtraction.DCOR)
          (b) the reference dcor.u_distance_correlation_sqr
        and return both values.
        """
        if dependent:
            X = np.random.randn(n, d1)
            W = np.random.randn(d1, d2)
            Y = X @ W + noise * np.random.randn(n, d2)
        else:
            X = np.random.randn(n, d1)
            Y = np.random.randn(n, d2)

        xt = torch.from_numpy(X).float().to(device)
        yt = torch.from_numpy(Y).float().to(device)

        # Instantiate with a nonzero input_projection_dim to avoid init error,
        # but we don't use the projection or BN here.
        mfe = MiFeatureExtraction(
            dim_x=d1,
            dim_y=d2,
            input_projection_dim=1,
            device=device,
        )

        ours = mfe.DCOR(xt, yt, exponent=1.0, squared=True).detach().cpu().item()
        theirs = float(dcor.u_distance_correlation_sqr(X, Y))

        return ours, theirs

    # Run both an independent and a dependent check
    cases = [("independent", False), ("dependent", True)]
    for name, dep in cases:
        ours, ref = compare_once(dependent=dep)
        print(f"{name:>12} case -> ours = {ours:.8f} | dcor = {ref:.8f}")
        # Allow tiny numerical differences
        if not np.isclose(ours, ref, rtol=1e-5, atol=1e-7):
            raise AssertionError(
                f"Mismatch in {name} case: ours={ours}, dcor={ref}"
            )

    # Extra robustness: multiple random trials for the dependent case
    for trial in range(3):
        torch.manual_seed(trial + 1)
        np.random.seed(trial + 1)
        ours, ref = compare_once(dependent=True)
        print(f"trial {trial+1} (dependent) -> ours = {ours:.8f} | dcor = {ref:.8f}")
        assert np.isclose(ours, ref, rtol=1e-5, atol=1e-7)

    print("All DCOR unbiased (squared) tests passed ✅")