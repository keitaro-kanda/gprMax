from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)

print(torch.backends.mps.is_available())