import yaml
import torch
from classifier_control.baseline_costs.image_mse_cost import ImageMseCost


class DistFuncEvaluation:

    def __init__(self, testmodel, testparams):
        if testmodel is ImageMseCost:
            self.model = ImageMseCost()
        else:
            model_path = testparams['classifier_restore_path']
            testparams['classifier_restore_path'] = model_path
            if model_path is not None:
                config_path = '/'.join(str.split(model_path, '/')[:-2]) + '/params.yaml'
                with open(config_path) as config:
                    overrideparams = yaml.load(config)
            else:
                overrideparams = dict()
            if 'builder' in overrideparams:
                overrideparams.pop('builder')
            overrideparams.update(testparams)
            overrideparams['ignore_same_as_default'] = ''  # adding this flag prevents error because of hparam values being equal to default
            model = testmodel(overrideparams).eval()
            if torch.cuda.is_available():
                model.to(torch.device('cuda'))
            self.model = model

    def predict(self, inputs):
        scores = self.model(inputs)
        return scores



