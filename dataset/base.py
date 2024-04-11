import itertools
import json
import os

import numpy as np
import torch

from util.pose import make_T, mat2latlon
from util.typing import *
from util.util import load_image


def load_frames(
    image_dir: str = None,
    transform_fp: str = None,
    return_images=True,
    to_clip=True,
    verbose=True,
):
    """Load images from disk."""
    if not transform_fp.startswith("/"):
        # allow relative path
        transform_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            transform_fp,
        )

    with open(transform_fp, "r") as fp:
        meta = json.load(fp)

    all_images = []
    all_camtoworlds = []
    all_latlons = []
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]

        if return_images:
            fp = os.path.join(image_dir, frame["file_path"])
            all_images.append(
                load_image(fp, device="cpu", to_clip=to_clip, verbose=verbose)
            )

        all_camtoworlds.append(torch.tensor([frame["transform_matrix"]]))
        all_latlons.append(torch.tensor([frame["latlon"]]))

    if return_images:
        all_images = torch.cat(all_images)

    all_camtoworlds = torch.cat(all_camtoworlds)
    all_latlons = torch.cat(all_latlons)

    return all_images, all_camtoworlds, all_latlons


class BaseDataset:
    def setup(self, image_dir, transform_fp):
        self.all_images, self.all_camtoworlds, _ = load_frames(image_dir, transform_fp)

        assert len(self.all_camtoworlds) == len(self.all_images)
        assert self.all_images.shape[2:] == (256, 256)

        self.perm = list(itertools.permutations(range(len(self.all_images)), 2))
        self.perm = torch.from_numpy(np.array(self.perm))

    def get_trans(self, target_camtoworld, cond_camtoworld, in_T=True, verbose=False):
        """Returns the relative transformation from cond to target"""
        rel_latlon = mat2latlon(target_camtoworld, return_radius=True) - mat2latlon(
            cond_camtoworld, return_radius=True
        )
        rel_latlon = rel_latlon.squeeze(0)
        rel_latlon[1] = (rel_latlon[1] + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi]

        if verbose:
            print(
                "theta, azimuth, distance:",
                torch.rad2deg(rel_latlon[0]).item(),
                torch.rad2deg(rel_latlon[1]).item(),
                rel_latlon[2].item(),
            )

        if in_T:
            return make_T(rel_latlon[0], rel_latlon[1], rel_latlon[2])
        return rel_latlon

    def get_distance(self, target_camtoworld, cond_camtoworld):
        R, t = target_camtoworld[:3, :3], target_camtoworld[:3, [-1]]
        T_target = -R.T @ t

        R, t = cond_camtoworld[:3, :3], cond_camtoworld[:3, [-1]]
        T_cond = -R.T @ t

        distance = torch.linalg.norm(T_target - T_cond, ord=2)
        return distance

    def get_nearest(self, target_camtoworld):
        """Returns the nearest frame to target"""
        distances = [
            self.get_distance(target_camtoworld, cond_camtoworld)
            for cond_camtoworld in self.all_camtoworlds
        ]
        nearest_idx = torch.argmin(torch.tensor(distances))
        return nearest_idx
