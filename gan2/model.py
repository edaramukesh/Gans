import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,z_dim,out_dim):
        super().__init__()
        self.gen = nn.Sequential(nn.Linear(z_dim,256),nn.LeakyReLU(0.01),nn.Linear(256,out_dim),nn.Tanh())
    def forward(self,x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self,img_dim):
        super().__init__()
        self.dis = nn.Sequential(nn.Linear(img_dim,128),nn.LeakyReLU(0.01),nn.Linear(128,1),nn.Sigmoid())
    def forward(self,x):
        return self.dis(x)
    