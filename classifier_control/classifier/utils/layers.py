import torch.nn as nn
from .general_utils import AttrDict
from functools import partial
from classifier_control.classifier.utils.general_utils import HasParameters
import math


def init_weights_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    if isinstance(m, nn.Conv2d):
        pass    # by default PyTorch uses Kaiming_Normal initializer

class Block(nn.Sequential, HasParameters):
    def __init__(self, **kwargs):
        self.builder = kwargs.pop('builder')
        nn.Sequential.__init__(self)
        HasParameters.__init__(self, **kwargs)
        if self.params.normalization is not None and self.params.normalization == 'none':
            self.params.normalize = False
        if not self.params.normalize:
            self.params.normalization = None

        self.build_block()
        self.complete_block()

    def get_default_params(self):
        params = AttrDict(
            normalize=True,
            activation=nn.LeakyReLU(0.2, inplace=True),
            normalization=self.builder.normalization,
            normalization_params=AttrDict(),
            spectral_norm=self.builder.spectral_norm,
        )
        return params

    def build_block(self):
        raise NotImplementedError

    def complete_block(self):
        if self.params.normalization is not None:
            self.params.normalization_params.affine = True
            # TODO add a warning if the normalization is over 1 element
            if self.params.normalization == 'batch':
                normalization = nn.BatchNorm1d if self.params.d == 1 else nn.BatchNorm2d
                self.params.normalization_params.track_running_stats = True

            elif self.params.normalization == 'instance':
                normalization = nn.InstanceNorm1d if self.params.d == 1 else nn.InstanceNorm2d
                self.params.normalization_params.track_running_stats = True
                # TODO if affine is false, the biases will not be learned

            elif self.params.normalization == 'group':
                normalization = partial(nn.GroupNorm, 8)
                if self.params.out_dim < 32:
                    raise NotImplementedError("note that group norm is likely to not work with this small groups")

            else:
                raise ValueError("Normalization type {} unknown".format(self.params.normalization))
            self.add_module('norm', normalization(self.params.out_dim, **self.params.normalization_params))

        if self.params.activation is not None:
            self.add_module('activation', self.params.activation)


class ConvBlock(Block):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            d=2,
            kernel_size=3,
            stride=1,
            padding=1,
        ))
        return params

    def build_block(self):
        if self.params.d == 2:
            cls = nn.Conv2d
        elif self.params.d == 1:
            cls = nn.Conv1d
        elif self.params.d == -2:
            cls = nn.ConvTranspose2d

        self.add_module('conv', cls(
            self.params.in_dim, self.params.out_dim, self.params.kernel_size, self.params.stride, self.params.padding,
            bias=not self.params.normalize))


class FCBlock(Block):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            d=1,
        ))
        return params

    def build_block(self):
        mod = nn.Linear(self.params.in_dim, self.params.out_dim, bias=not self.params.normalize)
        if self.params.spectral_norm:
            self.add_module('linear', nn.utils.spectral_norm(mod))
        else:
            self.add_module('linear', mod)


class Linear(FCBlock):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            activation=None
        ))
        return params


class ConvBlockEnc(ConvBlock):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            kernel_size=4,
            stride=2,
        ))
        return params


class ConvBlockDec(ConvBlock):
    def get_default_params(self):
        params = super().get_default_params()
        params.update(AttrDict(
            d = -2, 
            kernel_size=4,
            stride=2,
        ))
        return params


def get_num_conv_layers(img_sz):
    n = math.log2(img_sz[0])
    assert n >= 3, 'imageSize must be at least 8'
    return int(n)


class LayerBuilderParams:
    """ This class holds general parameters for all subnetworks, such as whether to use convolutional networks, etc """

    def __init__(self, use_convs, normalize=True, normalization='batch', predictor_normalization=None, spectral_norm=False):
        self.use_convs = use_convs
        self.init_fn = init_weights_xavier
        self.normalize = normalize
        self.normalization = normalization
        self.predictor_normalization = predictor_normalization
        self.spectral_norm = spectral_norm

    @property
    def get_num_layers(self):
        if self.use_convs:
            return get_num_conv_layers
        else:
            return lambda: 2

    @property
    def def_block(self):
        """ Fetches the default block to use"""
        if self.use_convs:
            return ConvBlock
        else:
            return FCBlock

    def wrap_block(self, block):
        """ Wraps a block with the builder defaults. This function needs to be used every time a block is created. """
        # TODO fix this up. The blocks can do this.
        return partial(block, builder=self, normalization=self.predictor_normalization)