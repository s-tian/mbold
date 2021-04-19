import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from classifier_control.classifier.utils.subnetworks import ConvEncoder, ConvDecoder
from classifier_control.classifier.utils.layers import Linear


class VAE(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp

        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        out_flat_size = out_size[0] * out_size[1] * out_size[2]
        self.linear1 = Linear(in_dim=out_flat_size, out_dim=128, builder=self._hp.builder)
        self.linear2 = Linear(in_dim=128, out_dim=self._hp.hidden_size * 2, builder=self._hp.builder)
        self.linear3 = Linear(in_dim=self._hp.hidden_size, out_dim=128, builder=self._hp.builder)
        self.linear4 = Linear(in_dim=128, out_dim=out_flat_size, builder=self._hp.builder)
        self.decoder = ConvDecoder(self._hp)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
      
    def encode(self, image):
        embeddings = self.encoder(image).reshape(image.size(0), -1)
        e = F.relu(self.linear1(embeddings))
        z = self.linear2(e)
        mu, logvar = z[:, :self._hp.hidden_size], z[:, self._hp.hidden_size:]
        return mu, logvar
        
    def decode(self, z):
        e = F.relu(self.linear3(z))
        e = F.relu(self.linear4(e))
        e = e.view(*([e.size(0)] + list(self.encoder.get_output_size())))
        im = F.sigmoid(self.decoder(e))
        return im
        
    def forward(self, image):
        mu, logvar = self.encode(image)
        z = self.reparameterize(mu, logvar)
        im = self.decode(z)
        return mu, logvar, z, im


class Dynamics(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        self.linear1 = Linear(in_dim=self._hp.hidden_size, out_dim=self._hp.hidden_size, builder=self._hp.builder)

    def forward(self, z):
        return self.linear1(z)
