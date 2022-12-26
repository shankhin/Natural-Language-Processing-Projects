import numpy as np
import torch.nn as nn
import torch
t = torch.randn((64,10))
x = torch.randn((64,10))
c = torch.cat((t,x))
print(c.size())

