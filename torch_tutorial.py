import torch

x = torch.rand(5,6,3)
b,c, _ = x.size()
print(x.view(15, -1))