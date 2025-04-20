import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=3, downsamplings=5, hidden_size=256, down_channels=32):
        super().__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.start_channels = start_channels
        self.downsamplings = downsamplings
        self.c = down_channels
        self.cnn = nn.Conv2d(in_channels=3, out_channels=self.c, kernel_size=1, stride=1, padding=0)
        self.downsamplings_channels = [self.c * (2 ** i) for i in range(self.downsamplings)]
        self.downsamplings_blocks = nn.Sequential(*[nn.Sequential(nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
                                                    nn.BatchNorm2d(c * 2),
                                                    nn.ReLU()) for c in self.downsamplings_channels])
        self.hidden_size = hidden_size
        self.linear_model = nn.Sequential(nn.Linear(self.c * (self.img_size ** 2) // (2 ** self.downsamplings), self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, 2 * latent_size))
        self.flat = nn.Flatten()
    
    def forward(self, x):
        x = self.cnn(x)
        x = self.downsamplings_blocks(x)
        x = self.flat(x)
        x = self.linear_model(x)
        mu, sigma = x.split(self.latent_size, dim=1)
        sigma = torch.exp(sigma)
        return mu + torch.randn_like(mu) * sigma, (mu, sigma)
    
    
# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=3, upsamplings=5, hidden_size=256, up_channels=32):
        super().__init__()
        self.img_size = img_size
        self.latent_size = latent_size
        self.end_channels = end_channels
        self.upsamplings = upsamplings
        self.c = up_channels * (2 ** self.upsamplings)
        self.cnn = nn.Conv2d(in_channels=self.c // (2 ** self.upsamplings), out_channels=3, kernel_size=1, stride=1, padding=0)
        self.upsamplings_channels = [self.c // (2 ** i) for i in range(self.upsamplings)]
        self.upsamplings_blocks = nn.Sequential(*[nn.Sequential(nn.ConvTranspose2d(c, c // 2, kernel_size=4, stride=2, padding=1),
                                                                nn.BatchNorm2d(c // 2),
                                                                nn.ReLU()) for c in self.upsamplings_channels])
        self.hidden_size = hidden_size
        self.linear_model = nn.Sequential(nn.Linear(self.latent_size, self.hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(self.hidden_size, self.c * (self.img_size ** 2) // ((2 ** self.upsamplings) ** 2)))
        self.unflat = nn.Unflatten(1, (self.c, self.img_size // (2 ** self.upsamplings), self.img_size // (2 ** self.upsamplings)))
        self.tanh = nn.Tanh()
        
    def forward(self, z):
        x = self.linear_model(z)
        x = self.unflat(x)
        x = self.upsamplings_blocks(x)
        x = self.cnn(x)
        x = self.tanh(x)
        return x
    
# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=3, linear_hidden_size=256, latent_size=256, down_channels=6, up_channels=12):
        super().__init__()
        self.img_size = img_size
        self.downsamplings = downsamplings
        self.latent_size = latent_size
        self.down_channels = down_channels
        self.up_channels = up_channels
        self.linear_hidden_size = linear_hidden_size
        self.encoder = Encoder(img_size=self.img_size, latent_size=self.latent_size, start_channels=3, 
                               downsamplings=self.downsamplings, hidden_size=self.linear_hidden_size, down_channels=self.down_channels)
        self.decoder = Decoder(img_size=self.img_size, latent_size=self.latent_size, end_channels=3, 
                               upsamplings=self.downsamplings, hidden_size=self.linear_hidden_size * 2, up_channels=self.up_channels)
        
    def forward(self, x):
        res = self.encoder.forward(x)
        z = res[0]
        x_pred = self.decode(z)
        mu, sigma = res[1]
        kld = 0.5 * (sigma ** 2 + mu ** 2 - torch.log(sigma ** 2) - 1)
        return x_pred, kld
    
    def encode(self, x):
        z = self.encoder.forward(x)[0]
        return z
    
    def decode(self, z):
        x_pred = self.decoder.forward(z)
        return x_pred
    
    def save(self, path='model.pth'):
        model_path = __file__[:-7] + path
        torch.save(self.state_dict(), model_path)
    
    def load(self):
        model_path = __file__[:-7] + "model.pth"
        self.load_state_dict(torch.load(model_path))