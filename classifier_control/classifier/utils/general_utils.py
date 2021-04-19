import numpy as np
from PIL import Image
from torchvision.transforms import Resize
import torch
from functools import partial, reduce

def str2int(str):
    try:
        return int(str)
    except ValueError:
        return None


class HasParameters:
    def __init__(self, **kwargs):
        self.build_params(kwargs)

    def build_params(self, inputs):
        # If params undefined define params
        try:
            self.params
        except AttributeError:
            self.params = self.get_default_params()
            self.params.update(inputs)

    # TODO allow to access parameters by self.<param>


class AttrDict(dict):
    __setattr__ = dict.__setitem__

    def __getattr__(self, attr):
        # Take care that getattr() raises AttributeError, not KeyError.
        # Required e.g. for hasattr(), deepcopy and OrderedDict.
        try:
            return self.__getitem__(attr)
        except KeyError:
            raise AttributeError("Attribute %r not found" % attr)

    def __getstate__(self): return self
    def __setstate__(self, d): self = d


def map_dict(fn, d):
    """takes a dictionary and applies the function to every element"""
    return type(d)(map(lambda kv: (kv[0], fn(kv[1])), d.items()))


def get_clipped_optimizer(*args, optimizer_type=None, **kwargs):
    assert optimizer_type is not None   # need to set optimizer type!

    class ClipGradOptimizer(optimizer_type):
        def __init__(self, *args, gradient_clip=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.gradient_clip = gradient_clip

        def step(self, *args, **kwargs):
            if self.gradient_clip is not None:
                params = np.concatenate([group['params'] for group in self.param_groups])
                torch.nn.utils.clip_grad_norm_(params, self.gradient_clip)

            super().step(*args, **kwargs)

    return ClipGradOptimizer(*args, **kwargs)


def resize_video(video, size):
    # return video
    # video = ch_first2last(video).astype(np.uint8)
    # lazy transform ;)
    # transformed_video = video

    # moviepy is memory inefficient
    # clip = mpy.ImageSequenceClip(np.split(images, images.shape[0]), fps=1)
    # clip = clip.resize((self.img_sz, self.img_sz))
    # images = np.array([frame for frame in clip.iter_frames()])

    # looping over time is too slow for long videos
    transformed_video = np.stack([np.asarray(Resize(size)(Image.fromarray(im))) for im in video], axis=0)

    # Using pytorch is also slow
    # x = torch.from_numpy(video.transpose((0,3,1,2))).float()
    # x = F.interpolate(x, size)
    # transformed_video = x.data.numpy().transpose((0,2,3,1))

    # It seems that cv2 can resize images with up to 512 channels, which is what this implementation uses
    # It is as fast as the looping implementation above
    # sh = list(video.shape)
    # x = video.transpose((1, 2, 3, 0)).reshape(sh[1:3]+[-1])
    # n_split = math.ceil(x.shape[2] / 512.0)
    # x = np.array_split(x, n_split, 2)
    # x = np.concatenate([cv2.resize(im, size) for im in x], 2)
    # transformed_video = x.reshape(list(size) + [sh[3], sh[0]]).transpose((3, 0, 1, 2))

    # scipy.ndimage.zoom seems to be even slower than looping. Wow.
    # sh = [video.shape[1] / size[0], video.shape[2] / size[1]]
    # transformed_video = ndimage.zoom(video, [1] + sh + [1])

    # transformed_video = ch_last2first(transformed_video).astype(np.float32)
    return transformed_video


def make_recursive(fn, *argv, **kwargs):
    """ Takes a fn and returns a function that can apply fn on tensor structure
     which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors, list) or isinstance(tensors, tuple):
            return type(tensors)(map(recursive_map, tensors))
        elif isinstance(tensors, dict):
            return type(tensors)(map_dict(recursive_map, tensors))
        elif isinstance(tensors, torch.Tensor):
            return fn(tensors, *argv, **kwargs)
        else:
            try:
                return fn(tensors, *argv, **kwargs)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


def listdict2dictlist(LD):
    """ Converts a list of dicts to a dict of lists """

    # Take intersection of keys
    keys = reduce(lambda x, y: x & y, (map(lambda d: d.keys(), LD)))
    return AttrDict({k: [dic[k] for dic in LD] for k in keys})

def make_recursive_list(fn):
    """ Takes a fn and returns a function that can apply fn across tuples of tensor structures,
     each of which can be a single tensor, tuple or a list. """

    def recursive_map(tensors):
        if tensors is None:
            return tensors
        elif isinstance(tensors[0], list) or isinstance(tensors[0], tuple):
            return type(tensors[0])(map(recursive_map, zip(*tensors)))
        elif isinstance(tensors[0], dict):
            return map_dict(recursive_map, listdict2dictlist(tensors))
        elif isinstance(tensors[0], torch.Tensor):
            return fn(*tensors)
        else:
            try:
                return fn(*tensors)
            except Exception as e:
                print("The following error was raised when recursively applying a function:")
                print(e)
                raise ValueError("Type {} not supported for recursive map".format(type(tensors)))

    return recursive_map


recursively = make_recursive


def map_recursive(fn, tensors):
    return make_recursive(fn)(tensors)


def map_recursive_list(fn, tensors):
    return make_recursive_list(fn)(tensors)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class RecursiveAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = 0

    def update(self, val):
        self.val = val
        if self.sum is None:
            self.sum = val
        else:
            self.sum = map_recursive_list(lambda x, y: x + y, [self.sum, val])
        self.count += 1
        self.avg = map_recursive(lambda x: x / self.count, self.sum)