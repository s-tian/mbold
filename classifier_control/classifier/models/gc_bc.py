import numpy as np
import torch
from classifier_control.classifier.utils.general_utils import AttrDict
from classifier_control.classifier.models.base_model import BaseModel
from classifier_control.classifier.models.utils.utils import select_indices
from classifier_control.classifier.utils.actor_network import ActorNetwork


class GCBC(BaseModel):
    def __init__(self, overrideparams, logger=None):
        super().__init__(logger)
        self._hp = self._default_hparams()
        self.overrideparams = overrideparams
        self.override_defaults(overrideparams)  # override defaults with config file
        self.postprocess_params()

        assert self._hp.batch_size != -1   # make sure that batch size was overridden

        self.tdist_classifiers = []
        self.build_network()

    def _default_hparams(self):
        default_dict = AttrDict({
            'use_skips':False, #todo try resnet architecture!
            'ngf': 8,
            'nz_enc': 64,
            'classifier_restore_path':None,  # not really needed here.
            'skips_stride': None,
            'low_dim': False,
            'action_size': 4,
        })

        # add new params to parent params
        parent_params = super()._default_hparams()
        for k in default_dict.keys():
            parent_params.add_hparam(k, default_dict[k])
        return parent_params

    def build_network(self):
        self.actor_network = ActorNetwork(self._hp)

    def sample_image_pair(self, images, actions):

        tlen = images.shape[1]

        # get positives:
        t0 = np.random.randint(0, tlen-1, self._hp.batch_size)
        t1 = np.array([np.random.randint(t0[b]+1, tlen, 1) for b in range(images.shape[0])]).squeeze()

        t0, t1 = torch.from_numpy(t0), torch.from_numpy(t1)

        im_t0 = select_indices(images, t0)
        im_t1 = select_indices(images, t1)
        act = select_indices(actions, t0)

        self.labels = act

        img_pair_stack = torch.cat([im_t0, im_t1], dim=1)
        return img_pair_stack

    def forward(self, inputs):
        """
        forward pass at training time
        :param
            images shape = batch x time x channel x height x width
        :return: model_output
        """
        if 'demo_seq_images' in inputs:
            image_pairs = self.sample_image_pair(inputs.demo_seq_images, inputs.actions)
            self.img_pair = image_pairs
            model_output = self.make_prediction(self.img_pair)
            return model_output
        else:
            img_pair = torch.cat((inputs['current_img'], inputs['goal_img']), dim=1)
            output = self.make_prediction(img_pair)
            return output

    def make_prediction(self, images):
        self.action = self.actor_network(images)
        model_output = AttrDict(action=self.action)
        return model_output

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase):
        max_act = torch.max(model_output.action, dim=1)[0].mean()
        min_act = torch.min(model_output.action, dim=1)[0].mean()
        mean_mag = torch.abs(model_output.action).mean()
        self._logger.log_scalar(max_act, 'max_act', step, phase)
        self._logger.log_scalar(min_act, 'min_act', step, phase)
        self._logger.log_scalar(mean_mag, 'mean_mag', step, phase)
        pass

    def loss(self, model_output):
        losses = AttrDict()
        setattr(losses, 'mse', torch.nn.MSELoss()(model_output.action.squeeze(), self.labels.to(self._hp.device)))
        # compute total loss
        losses.total_loss = torch.stack(list(losses.values())).sum()
        return losses

    def get_device(self):
        return self._hp.device


def ptrch2uint8(img):
    return ((img + 1)/2*255.).astype(np.uint8)


class GCBCTestTime(GCBC):
    def __init__(self, overrideparams, logger=None, restore_from_disk=True):
        super(GCBCTestTime, self).__init__(overrideparams, logger)
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
        return super().forward(inputs)


def ptrch2uint8(img):
    return ((img + 1)/2*255.).astype(np.uint8)
