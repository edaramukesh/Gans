from model import Generator,Discriminator
import torch.nn as nn
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import MNIST
import torch

transforms = T.Compose([T.ToTensor(),T.Normalize(mean=(0.5),std=(0.5))])

z_dim = 64
img_dim = 784
lr = 3e-4
num_epochs = 50
batchsize = 32
dataset = MNIST(root="/home/mukesh/Desktop/4-2/cv_projects/gans/Mnist",train=True,transform=transforms,download=True)
dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True)
gen = Generator(z_dim=z_dim,out_dim=img_dim).cuda()
disc = Discriminator(img_dim).cuda()
g_opt = optim.Adam(gen.parameters(),lr)
d_opt = optim.Adam(disc.parameters(),lr)
loss_fn = nn.BCELoss()

fixed_noise = torch.randn((batchsize,64)).cuda()

loop = tqdm(range(num_epochs))

for epoch in loop:
    for idx,(real,_) in enumerate(dataloader):
        noise = torch.randn((batchsize,64)).cuda()
        real = real.view(-1,784).cuda()
        fake = gen(noise).cuda()
        d_real = disc(real).view(-1)
        d_fake = disc(fake).view(-1)
        d_real_loss = loss_fn(d_real,torch.ones_like(d_real))
        d_fake_loss = loss_fn(d_fake,torch.zeros_like(d_fake))
        d_loss = (d_fake_loss+d_real_loss)/2
        d_opt.zero_grad()
        d_loss.backward(retain_graph=True)
        d_opt.step()

        out = disc(fake).view(-1)
        g_loss = loss_fn(out,torch.ones_like(out))
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

        with torch.no_grad():
            if idx == 0:
                imgs = gen(fixed_noise).view(-1,1,28,28)
                for i in range(imgs.shape[0]):
                    img = T.ToPILImage()(imgs[i])
                    img.save(f"gan2/images/{epoch}_{i}.png")

        if idx%10 == 0:
            loop.set_postfix({"g_loss":g_loss.item(),"d_loss":d_loss.item()})


