import torch

s1 = torch.Tensor(1,1,256)
s2 = torch.Tensor(1,1024,256)

c = torch.mul(s1,s2)
x = 1