from _imports import *

class nop(nn.Module):
    def __init__(self): super().__init__()
    def forward(self,x): return x

def linear(ni, no, act=nn.ReLU()): return nn.Sequential(nn.Linear(ni,no),act)
