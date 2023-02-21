import torch
from torchvision.datasets import MNIST
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Generator,Discriminator
from torch.optim import Adam
import torchvision.transforms as T
import torch.nn as nn


lr = 3e-4
batchsize = 64
img_size = 64
transforms = T.Compose([T.Resize(img_size),T.ToTensor(),T.Normalize(mean=(0.5),std=(0.5))])
dataset = MNIST("/home/mukesh/Desktop/4-2/cv_projects/gans/Mnist",train=True,transform=transforms,download=True)
dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True,drop_last=True)
gen = Generator().cuda()
disc = Discriminator().cuda()
g_opt = Adam(gen.parameters(),lr)
d_opt = Adam(disc.parameters(),lr)
loss_fn = nn.BCELoss()
num_epochs = 50

fixed_noise = torch.randn((batchsize,64,1,1)).cuda()

loop = tqdm(range(num_epochs))
for epoch in loop:
    for idx,(real,_) in enumerate(dataloader):
        noise = torch.randn((batchsize,64,1,1)).cuda()
        real = real.cuda()
        fake = gen(noise)
        d_real = disc(real)
        d_fake = disc(fake)
        d_real_loss = loss_fn(d_real,torch.ones_like(d_real))
        d_fake_loss = loss_fn(d_fake,torch.zeros_like(d_fake))
        d_loss = (d_fake_loss+d_real_loss)/2
        d_opt.zero_grad()
        d_loss.backward(retain_graph=True)
        d_opt.step()

        out = disc(fake)
        g_loss = loss_fn(out,torch.ones_like(out))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        with torch.no_grad():
            if idx == 0:
                pass
            if idx%100 == 0:
                loop.set_postfix({"":"","":""})

