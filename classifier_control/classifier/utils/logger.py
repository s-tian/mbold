import os
import pdb
import torchvision
import torch
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from classifier_control.classifier.utils.vis_utils import draw_text_image
import cv2
from classifier_control.classifier.utils.vis_utils import plot_graph

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        self._n_logged_samples = n_logged_samples
        if summary_writer is not None:
            self._summ_writer = summary_writer
        else:
            self._summ_writer = SummaryWriter(log_dir)

    def _loop_batch(self, fn, name, val, *argv, **kwargs):
        """Loops the logging function n times."""
        for log_idx in range(min(self._n_logged_samples, len(val))):
            name_i = os.path.join(name, "_%d" % log_idx)
            fn(name_i, val[log_idx], *argv, **kwargs)

    @staticmethod
    def _check_size(val, size):
        if isinstance(val, torch.Tensor) or isinstance(val, np.ndarray):
            assert len(val.shape) == size, "Size of tensor does not fit required size, {} vs {}".format(len(val.shape),
                                                                                                        size)
        elif isinstance(val, list):
            assert len(val[0].shape) == size - 1, "Size of list element does not fit required size, {} vs {}".format(
                len(val[0].shape), size - 1)
        else:
            raise NotImplementedError("Input type {} not supported for dimensionality check!".format(type(val)))
        if (val[0].shape[1] > 10000) or (val[0].shape[2] > 10000):
            raise ValueError("This might be a bit too much")

    def log_scalar(self, scalar, name, step, phase):
        self._summ_writer.add_scalar('{}_{}'.format(name, phase), scalar, step)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def log_images(self, image, name, step, phase):
        self._check_size(image, 4)  # [N, C, H, W]
        self._loop_batch(self._summ_writer.add_image, '{}_{}'.format(name, phase), image, step)

    def log_video(self, video_frames, name, step, phase):
        assert len(video_frames.shape) == 4, "Need [T, C, H, W] input tensor for single video logging!"
        if not isinstance(video_frames, torch.Tensor): video_frames = torch.tensor(video_frames)
        video_frames = torch.transpose(video_frames, 0, 1)  # tbX requires [C, T, H, W]
        video_frames = video_frames.unsqueeze(0)  # add an extra dimension to get grid of size 1
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step)

    def log_videos(self, video_frames, name, step, phase, fps=3):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        video_frames = video_frames.unsqueeze(1)  # add an extra dimension after batch to get grid of size 1
        self._loop_batch(self._summ_writer.add_video, '{}_{}'.format(name, phase), video_frames, step, fps=fps)

    def log_image(self, images, name, step, phase):
        self._summ_writer.add_image('{}_{}'.format(name, phase), images, step)

    def log_image_grid(self, images, name, step, phase, nrow=8):
        assert len(images.shape) == 4, "Image grid logging requires input shape [batch, C, H, W]!"
        img_grid = torchvision.utils.make_grid(images, nrow=nrow)
        self.log_images(img_grid, '{}_{}'.format(name, phase), step)

    def log_video_grid(self, video_frames, name, step, phase, fps=3):
        assert len(video_frames.shape) == 5, "Need [N, T, C, H, W] input tensor for video logging!"
        self._summ_writer.add_video('{}_{}'.format(name, phase), video_frames, step, fps=fps)

    def log_figures(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        assert figure.shape[0] > 0, "Figure logging requires input shape [batch x figures]!"
        self._loop_batch(self._summ_writer.add_figure, '{}_{}'.format(name, phase), figure, step)

    def log_figure(self, figure, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        self._summ_writer.add_figure('{}_{}'.format(name, phase), figure, step)

    def log_graph(self, array, name, step, phase):
        """figure: matplotlib.pyplot figure handle"""
        im = plot_graph(array)
        self._summ_writer.add_image('{}_{}'.format(name, phase), im, step)

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, "scalar_data.json") if log_path is None else log_path
        self._summ_writer.export_scalars_to_json(log_path)



def unstack(array, dim):
    arr = np.split(array, array.shape[dim], dim)
    arr = [a.squeeze() for a in arr]
    return arr


def get_text_row(pred_scores, _n_logged_samples, shape=(30, 64)):
    text_images = []
    for b in range(_n_logged_samples):
        text_images.append(draw_text_image('{}'.format(pred_scores[b]), image_size=shape).transpose(2, 0, 1))
    return np.concatenate(text_images, 2)


class TdistClassifierLogger(Logger):
    def log_one_ex(self, pair, out_sigmoid, name, step, phase, tag):

        pair = pair.data.cpu().numpy().squeeze()
        pred = out_sigmoid

        def image_row(image_pairs, scores, _n_logged_samples):

            first_row = image_pairs[:, 0]
            first_row = first_row[:_n_logged_samples]
            first_row = np.concatenate(unstack(first_row, 0), 2)

            second_row = image_pairs[:, 1]
            second_row = second_row[:_n_logged_samples]
            second_row = np.concatenate(unstack(second_row, 0), 2)

            numbers = get_text_row(scores, _n_logged_samples, shape=(30, image_pairs[0, 0].shape[1]))

            return (np.concatenate([first_row, second_row, numbers], 1) + 1.)/2.0

        image = image_row(pair, pred, self._n_logged_samples)
        # import pdb; pdb.set_trace()
        self._summ_writer.add_image('{}_{}_{}'.format(name, tag, phase), image, step)

    def log_single_tdist_classifier_image(self, pos_pair, neg_pair, out_sigmoid,
                                                  name, step, phase):
        self.log_one_ex(pos_pair, out_sigmoid[:out_sigmoid.shape[0]//2].data.cpu().numpy(), name, step, phase, 'positives')
        self.log_one_ex(neg_pair, out_sigmoid[out_sigmoid.shape[0]//2:].data.cpu().numpy(), name, step, phase, 'negatives')

    def log_heatmap_image(self, pos_pair, heatmap, out_sigmoid,
                                                  name, step, phase):

        pos_pair = pos_pair.data.cpu().numpy().squeeze()
        heatmap = heatmap.permute(2, 0, 1).unsqueeze(-1).repeat(1,1,1,3).data.cpu().numpy()
#         print(pos_pair.shape)
#         print(heatmap.shape)
        reshaped = []
        for i in range(heatmap.shape[0]):
            im = heatmap[i]
#             print(im.shape)
            im = cv2.resize(im, (64, 64))
#             print(im.min(), im.max())
            im -= im.mean()
            im /= im.std()
#             print(im.min(), im.max())
            cv2.imwrite('test'+str(i)+'.png', im*255)
#             assert(False)
            reshaped.append(im)
        reshaped = np.stack(reshaped)
#         print(reshaped.shape)
        reshaped = np.swapaxes(reshaped, 1,3)
#         print(reshaped.shape)
        reshaped = np.swapaxes(reshaped, 2,3)
#         print(reshaped.shape)
        pos_pair[:,0] = reshaped

        pos_pred = out_sigmoid[:out_sigmoid.shape[0]//2].data.cpu().numpy()

        def image_row(image_pairs, scores):

            first_row = image_pairs[:, 0]
            first_row = first_row[:self._n_logged_samples]
            first_row = np.concatenate(unstack(first_row, 0), 2)

            second_row = image_pairs[:, 1]
            second_row = second_row[:self._n_logged_samples]
            second_row = np.concatenate(unstack(second_row, 0), 2)

            numbers = get_sigmoid_annotations(scores)

            return (np.concatenate([first_row, second_row, numbers], 1) + 1.)/2.0

        def get_sigmoid_annotations(pred_scores):
            text_images = []
            for b in range(self._n_logged_samples):
                text_images.append(draw_text_image('{}'.format(pred_scores[b])).transpose(2, 0, 1))
            return np.concatenate(text_images, 2)

        positives_image = image_row(pos_pair, pos_pred)
        
        # import pdb; pdb.set_trace()
        self._summ_writer.add_image('{}_{}'.format(name + '_heatmaps', phase), positives_image, step)




from classifier_control.classifier.utils.vis_utils import visualize_barplot_array

class TdistMultiwayClassifierLogger(Logger):
    def log_pair_predictions(self, img_pair, softmax_prediction, label,
                             name, step, phase):

        image_pairs = img_pair.data.cpu().numpy().squeeze()
        softmax_prediction = softmax_prediction.data.cpu().numpy().squeeze()

        first_row = image_pairs[:, 0]
        first_row = first_row[:self._n_logged_samples]
        first_row = np.concatenate(unstack(first_row, 0), 2)

        second_row = image_pairs[:, 1]
        second_row = second_row[:self._n_logged_samples]
        second_row = np.concatenate(unstack(second_row, 0), 2)


        pred_score_images = visualize_barplot_array(softmax_prediction[:self._n_logged_samples])
        pred_score_images = [np.transpose((img.astype(np.float32))/255., [2,0,1]) for img in pred_score_images]
        pred_row = np.concatenate(pred_score_images, axis=2)

        label_row = get_text_row(label, self._n_logged_samples, shape=(30, pred_score_images[0].shape[1]))

        full_image = (np.concatenate([first_row, second_row, pred_row, label_row], 1) + 1.)/2.0

        self._summ_writer.add_image('{}_{}'.format(name, phase), full_image, step)


class TdistRegressorLogger(Logger):
    def log_pair_predictions(self, img_pair, prediction, label,
                                          name, step, phase):

        image_pairs = img_pair.data.cpu().numpy().squeeze()

        first_row = image_pairs[:, 0]
        first_row = first_row[:self._n_logged_samples]
        first_row = np.concatenate(unstack(first_row, 0), 2)

        second_row = image_pairs[:, 1]
        second_row = second_row[:self._n_logged_samples]
        second_row = np.concatenate(unstack(second_row, 0), 2)

        pred_row = get_text_row(prediction.data.cpu().numpy().squeeze(), self._n_logged_samples)
        label_row = get_text_row(label, self._n_logged_samples)

        full_image = (np.concatenate([first_row, second_row, pred_row, label_row], 1) + 1.)/2.0

        # import pdb; pdb.set_trace()
        self._summ_writer.add_image('{}_{}'.format(name, phase), full_image, step)
