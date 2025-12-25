from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import numpy as np
import copy



def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def solve_v_total(weight, subset):
    weight = weight.view(-1)
    k = subset
    a, b = 0, 0
    b = max(b, weight.max())
    
    def f(v):
        s = (weight - v).clamp(0, 1).sum()
        return s - k
        
    if f(0) < 0:
        return 0
        
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v

def train_demo():
    opt = TrainOptions().parse()   
    dataset = create_dataset(opt)  
    opt.input_dim, opt.output_dim = dataset.dataset.data_shape    
    model = create_model(opt)      
    model.setup(opt)               
    visualizer = Visualizer(opt)   
    total_iters = 300               
    for epoch in range(opt.epoch_count, opt.epoch_count + opt.n_epochs + opt.n_epochs_decay + 1):  
        epoch_iter = 0                  
        visualizer.reset()              
        model.update_learning_rate()    
        for i, data in enumerate(dataset):  
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         
            model.optimize_parameters()   
            if total_iters % opt.save_latest_freq == 0:   
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
        if epoch % opt.save_epoch_freq == 0:           
            model.save_networks('latest')
            model.save_networks(epoch)
            
def constrainScoreByWhole(scores, coreset_size):
    with torch.no_grad():
        v = solve_v_total(scores, coreset_size)
        scores.sub_(v).clamp_(0, 1)

def obtain_mask(scores):
    subnet = (torch.rand_like(scores) < scores).float()
    grad_subnet = (subnet - scores) / ((scores + 1e-8) * (1 - scores + 1e-8))
    return subnet, grad_subnet

def calculateGrad(scores, fn_list, grad_list):
    scores.grad = torch.zeros_like(scores)
    K = len(fn_list)
    for i in range(K):
        scores.grad += (1/K) * fn_list[i] * grad_list[i]

def calculateGrad_vr(scores, fn_list, grad_list, fn_avg):
    scores.grad = torch.zeros_like(scores)
    K = len(fn_list)
    for i in range(K):
        scores.grad += (1 / (K - 1)) * (fn_list[i] - fn_avg) * grad_list[i]

def hyper_train():
    opt = TrainOptions().parse()   
    
    if not hasattr(opt, 'coreset_size'): opt.coreset_size = 100
    if not hasattr(opt, 'outer_lr'): opt.outer_lr = 1e-2
    if not hasattr(opt, 'K'): opt.K = 5
    if not hasattr(opt, 'vr'): opt.vr = True
    if not hasattr(opt, 'clip_grad'): opt.clip_grad = True

    dataset = create_dataset(opt)  
    opt.input_dim, opt.output_dim = dataset.dataset.data_shape    
    model = create_model(opt)      
    model.setup(opt)               
    visualizer = Visualizer(opt)   
    total_iters = 0               
    
    dummy_data = next(iter(dataset))
    model.set_input(dummy_data)
    
    input_tensor = model.real_A if hasattr(model, 'real_A') else dummy_data['A']
    shape = input_tensor.shape 
    
    num_elements = int(np.prod(shape[1:])) 
    
    init_prob = opt.coreset_size / num_elements
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scores = torch.full((num_elements,), init_prob, dtype=torch.float, requires_grad=True, device=device)
    
    scores_opt = torch.optim.SGD([scores], lr=opt.outer_lr)
    
    mask_shape = shape[1:] 

    print(f"Initialized feature selection mask with size: {mask_shape}, Total features: {num_elements}, Target keep: {opt.coreset_size}")

    for epoch in range(opt.epoch_count, opt.epoch_count + opt.n_epochs + opt.n_epochs_decay + 1):  
        print(f"Epoch: {epoch}")
        epoch_iter = 0                  
        visualizer.reset()              
        model.update_learning_rate()    
        
        total_epochs = opt.n_epochs + opt.n_epochs_decay
        current_outer_lr = 0.5 * (1 + np.cos(np.pi * (epoch - opt.epoch_count) / total_epochs)) * opt.outer_lr
        assign_learning_rate(scores_opt, current_outer_lr)

        for i, data in enumerate(dataset):  
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            
            fn_list = []       
            grad_list = []     
            fn_avg = 0
            
            for k in range(opt.K):
                subnet, grad_mask = obtain_mask(scores)
                
                subnet_expanded = subnet.view((1,) + mask_shape)
                
                masked_data = {}
                for key in data:
                    masked_data[key] = data[key].clone()
                
                if 'A' in masked_data:
                    masked_data['A'] = masked_data['A'] * subnet_expanded
                
                model.set_input(masked_data)
                model.optimize_parameters() 
                
                current_loss = 0
                if hasattr(model, 'loss_total'):
                    current_loss = model.loss_total.item()
                else:
                    losses = model.get_current_losses()
                    current_loss = sum(losses.values())
                
                fn_list.append(current_loss)
                grad_list.append(grad_mask)
            
            fn_avg = sum(fn_list) / opt.K
            if opt.vr:
                calculateGrad_vr(scores, fn_list, grad_list, fn_avg)
            else:
                calculateGrad(scores, fn_list, grad_list)
                
            if opt.clip_grad:
                torch.nn.utils.clip_grad_norm_(scores, 3.0)
            
            scores_opt.step()
            
            constrainScoreByWhole(scores, opt.coreset_size)
            
            if total_iters % opt.save_latest_freq == 0:   
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
                
            if total_iters % 50 == 0:
                active_count = scores.sum().item()
                demo_mask = (scores > 0.5).float().view(mask_shape)
                print(f"Iter {total_iters}: Loss {fn_avg:.4f}, Mask Active Sum {active_count:.1f}/{opt.coreset_size}")

        if epoch % opt.save_epoch_freq == 0:           
            model.save_networks('latest')
            model.save_networks(epoch)