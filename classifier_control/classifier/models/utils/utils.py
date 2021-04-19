import torch


def select_indices(tensor, indices):
    new_images = []
    for b in range(tensor.shape[0]):
        new_images.append(tensor[b, indices[b]])
    tensor = torch.stack(new_images, dim=0)
    return tensor