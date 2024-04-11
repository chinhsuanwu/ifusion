import importlib
import random
from inspect import isfunction
import cv2

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

str2list = lambda x: list(map(int, x.split(",")))


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    return conf


def parse_optimizer(config, params):
    optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler(config, optim):
    scheduler = getattr(torch.optim.lr_scheduler, config.name)(optim, **config.args)
    return scheduler


def parse_model(config):
    models = importlib.import_module("model." + config.name)
    model = getattr(models, config.name[0].upper() + config.name[1:])(**config.args)
    return model


def load_image(fp, resize=True, to_clip=True, verbose=True, device="cuda"):
    if verbose:
        print(f"[INFO] Loading image {fp}")

    image = np.array(Image.open(fp))
    if image.shape[-1] == 4:
        image[image[..., -1] < 128] = [255] * 4
        image = image[..., :3]

    if resize:
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).contiguous().to(device)
    image = image.permute(2, 0, 1).unsqueeze(0)

    if to_clip:
        image = image * 2 - 1
    return image


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d
