import torch

def squeeze_cpu(x):
    return x.detach().squeeze().cpu().numpy() if torch.is_tensor(x) else x
  
def dict_to_cpu(d):
    return {name: squeeze_cpu(val).tolist() for (name, val) in d.items()}
