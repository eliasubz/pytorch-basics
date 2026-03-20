import torch
torch.manual_seed(0)

a = torch.rand((7,7))

b = torch.rand((1,7))

c = a @ b.T

print(c)


