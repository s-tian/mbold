import torch
import torch.nn.functional as F
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder


class QNetwork(torch.nn.Module):
    def __init__(self, hp, num_outputs):
        super().__init__()
        self._hp = hp
        self.num_outputs = num_outputs
        self.ll_size = ll_size = self._hp.linear_layer_size
        self.fc_layers = torch.nn.ModuleList()
        if self._hp.low_dim:
            self.linear1 = Linear(in_dim=2*self._hp.state_size, out_dim=ll_size, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=ll_size + self._hp.action_size, out_dim=ll_size, builder=self._hp.builder)
        else:
            self.encoder = ConvEncoder(self._hp)
            out_size = self.encoder.get_output_size()
            self.linear1 = Linear(in_dim=out_size[0]*out_size[1]*out_size[2], out_dim=ll_size, builder=self._hp.builder)
            self.linear2 = Linear(in_dim=ll_size + self._hp.action_size, out_dim=ll_size, builder=self._hp.builder)
        for i in range(3):
            self.fc_layers.append(Linear(in_dim=ll_size, out_dim=ll_size, builder=self._hp.builder))
        self.fc_layers.append(Linear(in_dim=ll_size, out_dim=self.num_outputs, builder=self._hp.builder))

    def forward(self, image_pairs, actions):
        if self._hp.low_dim:
            embeddings = image_pairs
        else:
            embeddings = self.encoder(image_pairs).reshape(image_pairs.size(0), -1)
        x = F.relu(self.linear1(embeddings))
        x = torch.cat([x, actions], dim=1)
        x = F.relu(self.linear2(x))
        for layer in self.fc_layers:
            x = F.relu(layer(x))
        return x

