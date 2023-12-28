import numpy as np
import torch
from PIL import Image


# visual images
def plot_image(*xs, normalize=False, fp="out.png"):
    # x: [B, 3, H, W], [3, H, W], [1, H, W] or [H, W] torch.Tensor
    #    [B, H, W, 3], [H, W, 3], [H, W, 1] or [H, W] numpy.ndarray

    def _plot_image(image):
        if isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                if image.shape[0] == 1 or image.shape[0] == 3 or image.shape[0] == 4:
                    image = image.permute(1, 2, 0).squeeze()
            image = image.detach().cpu().numpy()

        image = image.astype(np.float32)

        # normalize
        if normalize:
            image = (image - image.min(axis=0, keepdims=True)) / (
                image.max(axis=0, keepdims=True)
                - image.min(axis=0, keepdims=True)
                + 1e-8
            )

        if image.max() <= 1:
            image *= 255
        Image.fromarray(image.astype(np.uint8)).save(fp)

    for x in xs:
        if len(x.shape) == 4:
            for i in range(x.shape[0]):
                _plot_image(x[i])
        else:
            _plot_image(x)