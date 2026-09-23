import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, X, r0, r1, r2, k=3.0):
        super(MF, self).__init__()
        self.out_shape = X[0].shape
        if len(self.out_shape) > 1:
            self.Is_pic = True
            X = X.reshape(X.shape[0], -1)
        else:
            self.Is_pic = False

        self.X = X
        self.r0_init = r0
        self.r1_init = r1
        self.r2_init = r2
        self.r0 = nn.Parameter(torch.tensor(r0), requires_grad=True)
        self.r1 = nn.Parameter(torch.tensor(r1), requires_grad=True)
        self.r2 = nn.Parameter(torch.tensor(r2), requires_grad=True)
        self.k = k
    
    def weight1(self,dist,r):
        w = torch.zeros_like(dist)
        if r.isnan():
            w = torch.ones_like(dist)
        else:
            flag1 = dist < r
            w[flag1] = (1 - (dist[flag1]/r)**2)**self.k

        return w
    
    def weight2(self,dist,r):
        w = torch.zeros_like(dist) 

        flag1 = dist < r / 2
        flag2 = (dist >= r / 2) & (dist < r)

        w[flag1] = 1.0
        w[flag2] = (1 - ((2 * dist[flag2] - r) / r) ** 2) ** self.k

        return w
    
    def Normalize_weight(self, w):
        # +eps keeps the backward pass finite when a whole weight column is
        # zero (e.g. no reference within the r1/r2 windows for some query);
        # a plain 0/0 produces NaN gradients that poison the radius params.
        return w / (torch.sum(w, dim=0, keepdim=True) + 1e-12)

    def forward(self, Z, mask=None):
        """Z: (n_q, d) query points.  mask: optional binary mask on the ambient
        dimensions; when given, the reference set X is masked consistently so
        that FM operates on the masked (informative) subspace (BCGAN Eq.(6))."""
        if self.Is_pic == True:
            Z = Z.reshape(Z.shape[0], -1)

        if self.r0 > self.r0_init*10:
            self.r0 = nn.Parameter(torch.tensor(self.r0_init*10), requires_grad=True)
        if self.r1 > self.r1_init*10:
            self.r1 = nn.Parameter(torch.tensor(self.r1_init*10), requires_grad=True)
        if self.r2 > self.r2_init*10:
            self.r2 = nn.Parameter(torch.tensor(self.r2_init*10), requires_grad=True)

        if mask is not None:
            m = mask.reshape(-1).to(Z.device)
            X_ref = self.X * m.unsqueeze(0)
        else:
            X_ref = self.X

        dist_all = torch.cdist(X_ref, Z)                    # (n_ref, n_q)
        has_nb = torch.any(dist_all <= self.r0, dim=0)      # queries with >= 1 neighbour
        if not bool(has_nb.any()):
            # no query has a reference point within r0: identity fallback
            e_Z = Z
            if self.Is_pic:
                e_Z = e_Z.reshape(e_Z.shape[0], *self.out_shape)
            return e_Z
        qinds = has_nb.nonzero(as_tuple=False).squeeze(1)   # (k,), k >= 1
        Zs = Z[qinds]
        dist = dist_all[:, qinds]
        rinds = torch.any(dist <= self.r0, dim=1).nonzero(as_tuple=False).squeeze(1)
        dist = dist[rinds]
        X = X_ref[rinds]

        alpha = self.weight1(dist, self.r0)
        alpha = self.Normalize_weight(alpha)

        mu = torch.matmul(alpha.t(), X)
        U = Zs - mu

        # squared distance along the local tangent direction.  U is normalised
        # so that r2 acts on true distances even when the local deviation is
        # large (e.g. unmasked noisy dimensions); the raw (diff . U)^2 |U|^2
        # form of the original code made dist_v sqrt(negative) = NaN whenever
        # |U| > dist, silently collapsing FM to the identity map.
        U_len2 = (U ** 2).sum(dim=1, keepdim=True) + 1e-12           # (k, 1)
        diff_vectors = torch.unsqueeze(X, dim=1) - torch.unsqueeze(Zs, dim=0)  # (r, k, d)
        dU = (diff_vectors * U.unsqueeze(0)).sum(dim=2)              # (r, k)
        dist_u = (dU ** 2) / U_len2.t()                              # (r, k) squared
        dist_u = torch.sqrt(dist_u + 1e-12)                          # along-tangent distance
        dist_v = torch.clamp(dist ** 2 - dist_u ** 2, min=0).sqrt()  # off-tangent distance

        w_prod = self.weight2(dist_v, self.r1) * self.weight2(dist_u, self.r2)
        wsum = w_prod.sum(dim=0, keepdim=True)                       # (1, k)
        beta = w_prod / (wsum + 1e-12)

        e_Z = torch.matmul(beta.t(), X)
        # queries whose neighbourhood carries no tangent/on-tangent weight
        # (all neighbours further than r1/r2 windows) keep their input
        degenerate = (wsum.squeeze(0) <= 1e-12) | torch.isnan(e_Z).any(dim=1)
        e_Z[degenerate] = Zs[degenerate]

        # queries without neighbours keep their input (identity), the rest get
        # the local manifold fit; index_put keeps the autograd graph intact.
        out = Z.clone()
        out[qinds] = e_Z

        if self.Is_pic:
            out = out.reshape(out.shape[0], *self.out_shape)

        return out