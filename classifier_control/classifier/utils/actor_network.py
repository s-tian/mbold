import torch
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder


class ActorNetwork(torch.nn.Module):
    def __init__(self, hp):
        super().__init__()
        self._hp = hp
        if self._hp.low_dim:
            self.encoder = Linear(in_dim=self._hp.state_size * 2, out_dim=128, builder=self._hp.builder)
            out_size = 128
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()[0] * 5 * 5
        self.mlp = torch.nn.Sequential()

        self.mlp.add_module('linear_1', Linear(in_dim=out_size, out_dim=128, builder=self._hp.builder))
        for i in range(10):
            self.mlp.add_module(f'linear_{i+2}', Linear(in_dim=128, out_dim=128, builder=self._hp.builder))
            self.mlp.add_module(f'relu_{i+2}', torch.nn.ReLU())
        self.mlp.add_module('linear_final', Linear(in_dim=128, out_dim=self._hp.action_size, builder=self._hp.builder))
        self.mlp.add_module('tanh', torch.nn.Tanh())

    def forward(self, image_pairs):
        embeddings = self.encoder(image_pairs).reshape(image_pairs.size(0), -1)
        return self.mlp(embeddings)
