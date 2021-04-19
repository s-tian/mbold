import numpy as np
import torch
import torch.nn.functional as F

from classifier_control.classifier.utils.spatial_softmax import SpatialSoftmax
from classifier_control.classifier.utils.layers import Linear
from classifier_control.classifier.utils.subnetworks import ConvEncoder
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.models.utils.utils import select_indices


class TempdistRegressor(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.tdist_classifiers = []
        self.build_network()
        self.proxy_ctrl_counter = 0

    def _default_hparams(self):
        default_dict = AttrDict({
            'tmax_label':10,  # the highest label for temporal distance, values are clamped after that
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'nz_enc': 64,
            'classifier_restore_path':None, # not really needed here.
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self, build_encoder=True):
        self.encoder = ConvEncoder(self._hp)
        out_size = self.encoder.get_output_size()
        if self._hp.spatial_softmax:
            self.spatial_softmax = SpatialSoftmax(out_size[1], out_size[2], out_size[0])  # height, width, channel
            self.linear = Linear(in_dim=out_size[0]*2, out_dim=1, builder=self._hp.builder)
        else:
            self.linear = Linear(in_dim=256, out_dim=1, builder=self._hp.builder)
            self.fc_layers = []
            self.fc_layers.append(Linear(in_dim=out_size[0]*out_size[1]*out_size[2], out_dim=256, builder=self._hp.builder))
            for i in range(3):
                self.fc_layers.append(Linear(in_dim=256, out_dim=256, builder=self._hp.builder))

    def sample_image_pair(self, images):
        tlen = images.shape[1]

        # get positives:
        t0 = np.random.randint(0, tlen, self._hp.batch_size)
        t1 = np.array([np.random.randint(t0[b], tlen, 1) for b in range(images.shape[0])]).squeeze()

        t0, t1 = torch.from_numpy(t0), torch.from_numpy(t1)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)

        self.labels = torch.clamp_max(t1 - t0, self._hp.tmax_label-1).float()

        img_pair_stack = torch.stack([im_t0, im_t1], dim=1)
        return img_pair_stack

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        image_pairs = self.sample_image_pair(inputs.demo_seq_images)
        self.img_pair = image_pairs
        model_output = self.make_prediction(image_pairs)
        return model_output

    def make_prediction(self, image_pairs_stacked):
        im_t0, im_t1 = image_pairs_stacked[:,0], image_pairs_stacked[:,1]
        embeddings = self.encoder(torch.cat([im_t0, im_t1], dim=1))
        if self._hp.spatial_softmax:
            embeddings = self.spatial_softmax(embeddings)
        else:
            embeddings = torch.flatten(embeddings, start_dim=1)
            for fc_layer in self.fc_layers:
                embeddings = F.relu(fc_layer(embeddings))
        self.tdist_estimates = self.linear(embeddings)
        model_output = AttrDict(tdist_estimates=self.tdist_estimates, img_pair=image_pairs_stacked)
        return model_output

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        if log_images:
            self._logger.log_pair_predictions(self.img_pair, self.tdist_estimates, self.labels,'tdist_regression', step, phase)

    def loss(self, model_output):
        losses = AttrDict()
        setattr(losses, 'mse', torch.nn.MSELoss()(model_output.tdist_estimates.squeeze(), self.labels.to(self._hp.device)))

        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def get_device(self):
        return self._hp.device


class TempdistRegressorTestTime(TempdistRegressor):
    def __init__(self, overrideparams, logger=None, restore_from_disk=True):
        super(TempdistRegressorTestTime, self).__init__(overrideparams, logger)
        if self._hp.classifier_restore_path is not None:
            checkpoint = torch.load(self._hp.classifier_restore_path, map_location=self._hp.device)
            self.load_state_dict(checkpoint['state_dict'])
        else:
            print('#########################')
            print("Warning Classifier weights not restored during init!!")
            print('#########################')

    def _default_hparams(self):
        parent_params = super()._default_hparams()
        parent_params.add_hparam('classifier_restore_path', None)
        return parent_params

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        image_pairs = torch.stack([inputs['current_img'], inputs['goal_img']], dim=1)
        expected_distance = self.make_prediction(image_pairs).tdist_estimates.data.cpu().numpy().squeeze()
        return expected_distance

    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass


def ptrch2uint8(img):
    return ((img + 1)/2*255.).astype(np.uint8)
