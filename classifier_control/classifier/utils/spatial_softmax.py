import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width)
        )
        self.pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        self.pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer('_pos_x', self.pos_x)
        self.register_buffer('_pos_y', self.pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height * self.width)
        else:
            feature = feature.view(-1, self.height * self.width)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self._pos_x) * softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self._pos_y) * softmax_attention, dim=1, keepdim=True)
        # expected_x = torch.sum(self.pos_x * softmax_attention, dim=1, keepdim=True)
        # expected_y = torch.sum(self.pos_y * softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel * 2)

        return feature_keypoints


if __name__ == '__main__':
    from torch.autograd import Variable

    data = Variable(torch.zeros([1, 3, 3, 3]))
    data[0, 0, 0, 1] = 10
    data[0, 1, 1, 1] = 10
    data[0, 2, 1, 2] = 10
    layer = SpatialSoftmax(3, 3, 3, temperature=1)
    print(layer(data))