import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import itertools
import numpy as np


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()


class KacIndependenceMeasure(nn.Module):

    def __init__(self, dim_x, dim_y, lr = 0.005,  input_projection_dim = 0, output_projection_dim=0, weight_decay=0.01, orthogonality_enforcer = 1.0, device="cpu", init_scale_shift=[1,0]):
        super(KacIndependenceMeasure, self).__init__()
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
        if self.input_projection_dim > 0:
            self.a = Variable(self.init_scale_shift[0]*torch.rand(self.input_projection_dim,device=self.device)+self.init_scale_shift[1], requires_grad=True)
            self.projection_x = nn.Linear(self.dim_x, self.input_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_x.parameters()) 
        else:
            self.a = Variable(self.init_scale_shift[0]*torch.rand(self.dim_x, device=self.device)+self.init_scale_shift[1], requires_grad=True)

        if self.output_projection_dim > 0:
            self.b = Variable(self.init_scale_shift[0]*torch.rand(self.output_projection_dim,device=self.device)+self.init_scale_shift[1], requires_grad=True)
            self.projection_y = nn.Linear(self.dim_y, self.output_projection_dim).to(self.device)
            param_list = param_list + list(self.projection_y.parameters()) 
        else:            
            self.b = Variable(self.init_scale_shift[0]*torch.rand(self.dim_y,device=self.device)+self.init_scale_shift[1], requires_grad=True)
        
        self.bnx = nn.BatchNorm1d(self.dim_x, affine=True).to(self.device)
        self.bny = nn.BatchNorm1d(self.dim_y, affine=True).to(self.device)


        self.trainable_parameters = param_list  + [self.a, self.b] + list(self.bnx.parameters()) + list(self.bny.parameters())
        
        self.optimizer = torch.optim.AdamW(param_list  + [self.a, self.b] + list(self.bnx.parameters()) + list(self.bny.parameters()), lr=self.lr, weight_decay=self.weight_decay) 

        #self.a = self.a / torch.norm(self.a)
        #self.b = self.b / torch.norm(self.b)
   

    def project(self, x, normalize=True):
        x = x.to(self.device)
        if normalize:
            x = self.bnx(x)

        proj = self.projection_x(x)            
        return proj

    def smooth(self, x):
        w = torch.linalg.norm(x, keepdim=True, dim=1)
        w = torch.exp(-0.5 * torch.square(w))
        #print(w)
        #aa = torch.norm(self.a)
        #bb = torch.norm(self.b)
        #hn = x.shape[0] #*x.shape[0]
        #C = cov(x)
        #breakpoint()
        #w = torch.exp(-0.05 * (aa*aa + bb*bb)/hn)
        #print(w)
        return w


    def forward(self, x, y, update = True, normalize=True):
        x = x.to(self.device)
        y = y.to(self.device)
        if normalize:
            x = self.bnx(x)
            y = self.bny(y)
        

        if self.input_projection_dim > 0:                
                x = self.projection_x(x)
        if self.output_projection_dim > 0:
                y = self.projection_y(y)


        
        #an = torch.Tensor([self.a.shape[0]])[0]
        #bn = torch.Tensor([self.b.shape[0]])[0]
        #xa = (x @ (self.a))
        #yb = (y @ (self.b))
        

        n = x.shape[0]
        #d = x.shape[]
        
        dimx = self.a.shape[0]
        dimy = self.b.shape[0]
        xa = x @ (self.a/(dimx*torch.norm(self.a)))
        yb = y @ (self.b/(dimy*torch.norm(self.b)))

        #xa = x @ (self.a + 0.001*torch.randn(self.a.shape).to(self.device) )
        #yb = y @ (self.b + 0.001*torch.randn(self.b.shape).to(self.device) )

        """
        tn = np.exp(np.power(1/n, 1.0)) #np.power(n / np.log(n), 0.5)/np.power(n, )
        #print(tn)
        xa = x @ (tn*self.a/torch.norm(self.a))
        yb = y @ (tn*self.b/torch.norm(self.b))
        """

        

        """
        xy1 = torch.cat((x,y), axis=1)
        x1 = self.smooth(x)
        y1 = self.smooth(y)
        xy1 = self.smooth(xy1)

        if True:
            xy1 = torch.ones_like(xy1)
            x1 = torch.ones_like(x1)
            y1 = torch.ones_like(y1)

        fxy = (torch.exp(1j*(xa + yb))*(xy1)).mean()
        fx = (torch.exp(1j*(xa))*(x1)).mean()
        fy = (torch.exp(1j*(yb))*(y1)).mean()
        f1 = fxy - fx*fy
        """
      
        f1 = torch.exp(1j*(xa + yb)).mean() - torch.exp(1j*xa).mean() * torch.exp(1j*yb).mean()

        #breakpoint()

        kim = torch.norm(f1)
        #print(kim - torch.abs(f1))

        if update:
            loss = -kim 
            if self.input_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_x.weight,self.projection_x.weight.T) - torch.eye(self.input_projection_dim).to(self.device)) # maximise => negative
            
            if self.output_projection_dim > 0.0:
                loss = loss + self.orthogonality_enforcer*torch.norm(torch.matmul(self.projection_y.weight,self.projection_y.weight.T) - torch.eye(self.output_projection_dim).to(self.device)) # maximise => negative

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()   

        return kim
        