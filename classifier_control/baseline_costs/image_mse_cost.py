import torch


class ImageMseCost():
    def __init__(self):
        pass

    def __call__(self, inputs):
        return torch.mean((inputs['goal_img'] - inputs['current_img'])**2, dim=[1, 2, 3])

    def get_device(self):
        return torch.device('cpu')

    def visualize_test_time(self, content_dict, visualize_indices, verbose_folder):
        pass
