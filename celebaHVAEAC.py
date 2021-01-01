import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from HVAEAC_model import HVAEAC
from utils import Flatten, UnFlatten, ResBlock, MemoryLayer, RectangleGenerator


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

latent_dim = 128

L = 2
proposal_net = nn.Sequential(nn.Conv2d(6,8,1),
                        ResBlock(8,8), ResBlock(8,8),ResBlock(8,8),
                        nn.Conv2d(8,16,2,2),
                        ResBlock(16,16), ResBlock(16,16),ResBlock(16,16),
                        nn.Conv2d(16,32,2,2),
                        ResBlock(32,32), ResBlock(32,32), ResBlock(32,32), 
                        nn.Conv2d(32,64,2,2),
                        ResBlock(64,64), ResBlock(64,64),  ResBlock(64,64),
                        nn.Conv2d(64,128,2,2),
                        ResBlock(128,128), ResBlock(128,128), ResBlock(128,128),
                        nn.Conv2d(128,256,2,2),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(),
                        Flatten(),
                        nn.Linear(1024,2*latent_dim))

prior_net = nn.Sequential(MemoryLayer('#0'),
                        nn.Conv2d(6,8,1),
                        ResBlock(8,8), ResBlock(8,8),ResBlock(8,8),
                        MemoryLayer('#1'),
                        nn.Conv2d(8,16,2,2),
                        ResBlock(16,16), ResBlock(16,16),ResBlock(16,16),
                        MemoryLayer('#2'),
                        nn.Conv2d(16,32,2,2),
                        ResBlock(32,32), ResBlock(32,32), ResBlock(32,32),
                        MemoryLayer('#3'),
                        nn.Conv2d(32,64,2,2),
                        ResBlock(64,64), ResBlock(64,64),  ResBlock(64,64),
                        MemoryLayer('#4'),
                        nn.Conv2d(64,128,2,2),
                        ResBlock(128,128), ResBlock(128,128),  ResBlock(128,128),
                        MemoryLayer('#5'),
                        nn.Conv2d(128,256,2,2),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(),
                        Flatten(),
                        nn.Linear(1024,2*latent_dim))

gen_net = nn.Sequential(nn.Linear(latent_dim//L,1024),
                        UnFlatten(256,2,2),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(),
                        nn.ConvTranspose2d(256,256,2,2),
                        MemoryLayer('#5',True),
                        nn.Conv2d(384,128,1),
                        ResBlock(128,128), ResBlock(128,128), ResBlock(128,128),
                        nn.ConvTranspose2d(128,128,2,2),
                        MemoryLayer('#4',True),
                        nn.Conv2d(192,64,1),
                        ResBlock(64,64), ResBlock(64,64), ResBlock(64,64),
                        nn.ConvTranspose2d(64,64,2,2),
                        MemoryLayer('#3',True),
                        nn.Conv2d(96,32,1),
                        ResBlock(32,32), ResBlock(32,32), ResBlock(32,32),
                        nn.ConvTranspose2d(32,32,2,2),
                        MemoryLayer('#2',True),
                        nn.Conv2d(48,16,1),
                        ResBlock(16,16), ResBlock(16,16), ResBlock(16,16),
                        nn.ConvTranspose2d(16,16,2,2),
                        MemoryLayer('#1',True),
                        nn.Conv2d(24,8,1),
                        ResBlock(8,8), ResBlock(8,8),ResBlock(8,8),
                        MemoryLayer('#0',True),
                        nn.Conv2d(14,8,1),
                        ResBlock(8,8), ResBlock(8,8),ResBlock(8,8),
                        nn.Conv2d(8,3,1),
                        nn.Sigmoid())

l_net = lambda : nn.Sequential(nn.Linear(3*latent_dim//L,6*latent_dim//L),
                           nn.LeakyReLU(),
                           nn.Linear(6*latent_dim//L,6*latent_dim//L),
                           nn.LeakyReLU(),
                           nn.Linear(6*latent_dim//L,2*latent_dim//L)) 

latent_nets = torch.nn.ModuleList([l_net() for _ in range(L-1)])

transform = transforms.Compose([
            transforms.CenterCrop(148),
            transforms.Resize(64),
            transforms.ToTensor()])

dataset= datasets.ImageFolder('..\Data\celeba', transform)

torch.manual_seed(3)
train_data, test_data = torch.utils.data.random_split(dataset, [162079, 40520])
train_loader = DataLoader(train_data, batch_size=128, shuffle=False, pin_memory=True,num_workers=4)


celeba_HVAEAC= HVAEAC(proposal_net,prior_net,gen_net,latent_nets).to(device)
l_rate = 1e-4
opt = torch.optim.Adam(celeba_HVAEAC.parameters(), lr=l_rate)

beta=1
epoch = 10
mask_gen = RectangleGenerator()

#Training loop
for i in range(epoch):
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(torch.device('cuda'), non_blocking=True)
        masks = mask_gen(data)
        opt.zero_grad()

        loss, RE, KLD = celeba_HVAEAC.ELBO(data,masks)

        loss.backward()
        opt.step()

        if batch_idx % 100 == 0:
               print(f'Epoch: {i}, Loss: {loss}, RE: {RE}, KLD: {KLD}')
    print(f'Epoch: {i}, Loss: {loss}, RE: {RE}, KLD: {KLD}')

torch.save(celeba_HVAEAC.state_dict(), 'celeba_HVAEAC_L2.pt')
    