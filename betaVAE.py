import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):

    def __init__(self, z_dim, beta, nc, device):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        self.nc = nc
        self.device = device

        self.conv1 = nn.Conv2d(self.nc, 32, 4, 2, 1) # B, 32, 32, 32
        self.conv2 = nn.Conv2d(32, 32, 4, 2, 1) # B, 32, 16, 16
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1) # B, 64, 8, 8
        self.conv4 = nn.Conv2d(64, 64, 4, 2, 1) # B, 64, 4, 4
        self.conv5 = nn.Conv2d(64, 256, 4, 1) # B, 256, 1, 1
        self.linear1 = nn.Linear(256, 128) # B, 128
        self.linear2 = nn.Linear(128, self.z_dim) # B, z_dim
        self.linear3 = nn.Linear(128, self.z_dim) # B, z_dim

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):

        x = x.to(self.device) # B, 1, 64, 64
        x = F.relu(self.conv1(x)) # B, 32, 32, 32
        x = F.relu(self.conv2(x)) # B, 32, 16, 16
        x = F.relu(self.conv3(x)) # B, 64, 8, 8
        x = F.relu(self.conv4(x)) # B, 64, 4, 4
        x = F.relu(self.conv5(x)) # B, 256, 1, 1
        x = torch.flatten(x, start_dim = 1) # B, 256
        x = F.relu(self.linear1(x)) # B, 128
        mu = self.linear2(x) # B, z_dim
        sigma = torch.exp(self.linear3(x)) # B, z_dim

        # reparameterize mu and sigma vectors
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = -0.5 * (1 + torch.log(sigma) - mu**2 - sigma**2).sum()
        self.beta_kl = self.beta * self.kl

        return z


class Decoder(nn.Module):

    def __init__(self, z_dim, nc, device):
        super().__init__()

        self.device = device
        self.z_dim = z_dim
        self.nc = nc

        self.linear1 = nn.Linear(self.z_dim, 128) # B, 128
        self.linear2 = nn.Linear(128, 256) # B, 256

        self.unflatten = nn.Unflatten(dim = 1, unflattened_size = (256,1,1)) # B, 256, 1, 1
        self.convT1 = nn.ConvTranspose2d(256, 64, 4) # B, 64, 4, 4
        self.convT2 = nn.ConvTranspose2d(64, 64, 4, 2, 1) # B, 64, 8, 8
        self.convT3 = nn.ConvTranspose2d(64, 32, 4, 2, 1) # B, 32, 16, 16
        self.convT4 = nn.ConvTranspose2d(32, 32, 4, 2, 1) # B, 32, 64, 64
        self.convT5 = nn.ConvTranspose2d(32, self.nc, 4, 2, 1) # B, nc, 64, 64


    def forward(self, x): # x here is z obtained from encoder

        x = x.to(self.device) # B, z_dim
        x = F.relu(self.linear1(x)) # B, 128
        x = F.relu(self.linear2(x)) # B, 256
        x = self.unflatten(x) # B, 256, 1, 1
        x = F.relu(self.convT1(x)) # B, 64, 4, 4
        x = F.relu(self.convT2(x)) # B, 64, 8, 8
        x = F.relu(self.convT3(x)) # B, 32, 16, 16
        x = F.relu(self.convT4(x)) # B, 32, 64, 64
        x = F.relu(self.convT5(x)) # B, nc, 64 ,64

        return x


class betaVAE(nn.Module):

    def __init__(self, z_dim, nc, beta, device):
        super(betaVAE, self).__init__()

        self.z_dim = z_dim
        self.nc = nc
        self.beta = beta
        self.device = device
        self.encoder = Encoder(self.z_dim, self.beta, self.nc, self.device)
        self.decoder = Decoder(self.z_dim, self.nc, self.device)
        self.weight_init()

    def weight_init(self):

        init_modules = [self.encoder, self.decoder]
        for m in init_modules :
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                torcg.nn.init.constant_(m.bias.data, 0)

    def forward(self, x):

        z = self.encoder(x)
        x = self.decoder(z)
        return x

    



















# make sure model runs

# device = torch.device('cuda')
# model = betaVAE(10,1, 4, device)
# model.to(device)
# inp = torch.rand(1,1,64,64)
# out = model(inp)
# print(out.size())
