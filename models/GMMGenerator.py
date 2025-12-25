import torch



class GMMGenerator(torch.nn.Module):
    def __init__(self, input_W=64, input_H=64,USE_GPU=False):
        super(GMMGenerator, self).__init__()
        self.input_W = input_W
        self.input_H = input_H
        self.USE_GPU = USE_GPU

    def forward(self, phi, theta, delta):
        B = theta.size(0)
        K = phi.size(0)

        R_mat = Rot(theta)
        Sig = phi[:,3]
        W  = phi[:,4] 
        Center= torch.transpose(phi[:,0:3],0,1)
        Center = torch.matmul(R_mat, Center)
        Center = torch.transpose(Center,1,2)
        Center = Center[:,:,0:2]

        delta = delta.reshape((B,1,2))
        delta = torch.transpose(delta,1,2)
        mat2 = torch.tensor([1.0]).repeat(1,K)
        if self.USE_GPU:
            mat2 = mat2.cuda()
        delta = torch.matmul(delta, mat2)
        delta = torch.transpose(delta,1,2)
        Center= torch.add(Center,delta) 
        
        
        xs = torch.linspace(-1, 1, steps=self.input_W)
        ys = torch.linspace(-1, 1, steps=self.input_H)
        x, y = torch.meshgrid(xs, ys, indexing='xy')
        if self.USE_GPU:
            x = x.cuda()
            y = y.cuda()
        Flattened = Center.flatten(end_dim=-2)
        
        
        cen_x = Flattened[:,0]
        cen_x = cen_x[:, None, None]
        cen_x = cen_x.repeat([1,self.input_W,self.input_H])
        
        cen_y = Flattened[:,1]
        cen_y = cen_y[:, None, None]
        cen_y = cen_y.repeat([1,self.input_W,self.input_H])
        
        temp = (cen_x - x)**2 + (cen_y - y)**2
        Sig = Sig[:,None,None]
        Sig = Sig.repeat([B,1,1])
        temp = 1.0/(torch.sqrt(torch.pi*(Sig**2))) * torch.exp(-0.5*1/(Sig**2)*temp)
        W = W[:,None,None]
        W = W.repeat([B,1,1])
        temp = W*temp 
        temp = temp.reshape([B,K,self.input_W,self.input_H])
        out = 1.0 - temp.sum(1)
        
        return out


def Rot(rotvec: torch.Tensor, epsilon=1e-6) -> torch.Tensor:
    rotvec, batch_shape = flatten_batch_dims(rotvec, end_dim=-2)
    batch_size, D = rotvec.shape
    assert(D == 3), "Input should be a Bx3 tensor."

    theta = torch.norm(rotvec, dim=-1)
    is_angle_small = theta < epsilon
    
    axis = rotvec / torch.clamp_min(theta[...,None], epsilon)
    kx, ky, kz = axis[:,0], axis[:,1], axis[:,2]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    one_minus_cos_theta = 1 - cos_theta
    xs = kx*sin_theta
    ys = ky*sin_theta
    zs = kz*sin_theta
    xyc = kx*ky*one_minus_cos_theta
    xzc = kx*kz*one_minus_cos_theta
    yzc = ky*kz*one_minus_cos_theta
    xxc = kx**2*one_minus_cos_theta
    yyc = ky**2*one_minus_cos_theta
    zzc = kz**2*one_minus_cos_theta
    R_rodrigues = torch.stack([1 - yyc - zzc, xyc - zs, xzc + ys,
                     xyc + zs, 1 - xxc - zzc, -xs + yzc,
                     xzc - ys, xs + yzc, 1 -xxc - yyc], dim=-1).reshape(-1, 3, 3)

    xs, ys, zs = rotvec[:,0], rotvec[:,1], rotvec[:,2]
    one = torch.ones_like(xs)
    R_first_order = torch.stack([one, -zs, ys,
                                 zs, one, -xs,
                                 -ys, xs, one], dim=-1).reshape(-1, 3, 3)

    
    R = torch.where(is_angle_small[:,None,None], R_first_order, R_rodrigues)
    return unflatten_batch_dims(R, batch_shape)

def flatten_batch_dims(tensor, end_dim):

    batch_shape = tensor.shape[:end_dim+1]
    flattened = tensor.flatten(end_dim=end_dim) if len(batch_shape) > 0 else tensor.unsqueeze(0)
    return flattened, batch_shape

def unflatten_batch_dims(tensor, batch_shape):

    return tensor.reshape(batch_shape + tensor.shape[1:]) if len(batch_shape) > 0 else tensor.squeeze(0)