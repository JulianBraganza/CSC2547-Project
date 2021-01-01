import torch
import torch.nn as nn
import numpy as np

class Flatten(nn.Module):
    
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    
    def __init__(self,n_channels,size1,size2):
        super().__init__()
        self.n_channels = n_channels
        self.size1 = size1
        self.size2 = size2
        
    def forward(self, input):
        return input.view(input.size(0),self.n_channels,self.size1,self.size2)
    

class ResBlock(nn.Module):
    "Residual block with two convolutions"
    def __init__(self, outer_dim, inner_dim):
        super().__init__()
        self.net = nn.Sequential(    
            nn.BatchNorm2d(outer_dim),
            nn.LeakyReLU(),
            nn.Conv2d(outer_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.LeakyReLU(),
            nn.Conv2d(inner_dim, outer_dim, 3, 1, 1))
    def forward(self, input):
        return input + self.net(input)

class MemoryLayer(nn.Module):
    """
    Memory layer to implement skip connections. Code sourced from 
    Ivanov et, al. "Variational Autoencoder with Arbitrary Conditioning"
    """

    storage = {}

    def __init__(self, id, output=False, add=False):
        super().__init__()
        self.id = id
        self.output = output
        self.add = add

    def forward(self, input):
        if not self.output:
            self.storage[self.id] = input
            return input
        else:
            if self.id not in self.storage:
                err = 'MemoryLayer: id \'%s\' is not initialized. '
                err += 'You must execute MemoryLayer with the same id '
                err += 'and output=False before this layer.'
                raise ValueError(err)
            stored = self.storage[self.id]
            if not self.add:
                data = torch.cat([input, stored], 1)
            else:
                data = input + stored
            return data

def diag_gaussian_KLD(params1,params2,dim=1):
    mu1, logvar1 = params1
    mu2, logvar2 = params2
    
    return 0.5*torch.sum(logvar2 - logvar1 - 1 +
                         (logvar1-logvar2).exp() +
                         (mu2-mu1)**2/logvar2.exp(),dim=dim)

class FixedRectangleGenerator:
    """
    Always return an inpainting mask where unobserved region is
    a rectangle with corners in (x1, y1) and (x2, y2).
    """
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, batch):
        mask = torch.zeros_like(batch)
        mask[:, :, self.x1: self.x2, self.y1: self.y2] = 1
        return mask 

class RandomSplitGenerator:
    """
    Randomly generates a mask that covers the top or bottom half of the image
    """
    def __init__(self):
        pass
    def __call__(self, batch):
        batch_size, num_channels, width, height = batch.shape
        mask = torch.zeros_like(batch)
        y1, y2 = 0, width
        
        for i in range(batch_size):
            n = np.random.randint(0,2)
            if n == 0:
                x1, x2 = 0, height//2
            else:
                x1, x2 = height//2, height
                
            mask[i, :, x1: x2 , y1: y2 ] = 1
        return mask
    
    
class RectangleGenerator:
    """
    Generates for each object a mask where unobserved region is
    a rectangle which square divided by the image square is in
    interval [min_rect_rel_square, max_rect_rel_square].
    Code sourced from: 
    Ivanov et, al. "Variational Autoencoder with Arbitrary Conditioning"
    """
    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1):
        self.min_rect_rel_square = min_rect_rel_square
        self.max_rect_rel_square = max_rect_rel_square

    def gen_coordinates(self, width, height):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def __call__(self, batch):
        batch_size, num_channels, width, height = batch.shape
        mask = torch.zeros_like(batch)
        for i in range(batch_size):
            x1, y1, x2, y2 = self.gen_coordinates(width, height)
            sqr = width * height
            while not (self.min_rect_rel_square * sqr <=
                       (x2 - x1 + 1) * (y2 - y1 + 1) <=
                       self.max_rect_rel_square * sqr):
                x1, y1, x2, y2 = self.gen_coordinates(width, height)
            mask[i, :, x1: x2 + 1, y1: y2 + 1] = 1
        return mask