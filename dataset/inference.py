import torch
from torch.utils.data import DataLoader, Dataset

from dataset.base import BaseDataset, load_frames
from util.pose import latlon2mat
from util.typing import *
from util.util import load_image


def make_circular_poses(
    n_views: int = 8, theta: int = -20, default_radius: float = 1.0
):
    """Generate a list of poses on a circle."""
    azimuths = torch.arange(0, 360, 360 / n_views)
    latlon = torch.stack(
        (
            torch.ones_like(azimuths) * theta,
            azimuths,
            torch.ones_like(azimuths) * default_radius,
        )
    ).T
    return latlon2mat(latlon)


class SingleImageInferenceDataset(Dataset, BaseDataset):
    def __init__(
        self,
        image_fp: str = None,
        image_dir: str = None,
        transform_fp: str = None,
        test_transform_fp: str = None,
        n_views: int = 8,
        theta: int = -20,
        radius: float = 1.0,
        default_latlon: List[float] = [0, 0, 1],
    ):
        if image_fp:
            self.image = load_image(image_fp, device="cpu").squeeze(0)
            self.camtoworld = latlon2mat(torch.tensor([default_latlon]))
        elif transform_fp:
            self.setup(image_dir, transform_fp)
            self.image, self.camtoworld = self.all_images[0], self.all_camtoworlds[0]
        else:
            raise ValueError("Either image_fp or transform_fp must be provided.")

        if test_transform_fp:
            self.infer_camtoworlds = load_frames(None, test_transform_fp, return_images=False)[1]
        else:
            self.infer_camtoworlds = make_circular_poses(n_views, theta, radius)

    def __len__(self):
        return len(self.infer_camtoworlds)

    def __getitem__(self, index):
        target_camtoworld = self.infer_camtoworlds[index]
        latlon = self.get_trans(target_camtoworld, self.camtoworld, in_T=False)
        return {
            "image_cond": self.image,
            "theta": latlon[0],
            "azimuth": latlon[1],
            "distance": latlon[2],
        }

    def loader(self, batch_size=1, num_workers=8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            **kwargs,
        )


class MultiImageInferenceDataset(Dataset, BaseDataset):
    def __init__(
        self,
        image_dir: str = None,
        transform_fp: str = None,
        test_transform_fp: str = None,
        n_views: int = 8,
        theta: int = -20,
        radius: float = 1.0,
    ):
        self.setup(image_dir, transform_fp)
        if test_transform_fp:
            self.infer_camtoworlds = load_frames(image_dir, test_transform_fp, return_images=False)[1]
        else:
            self.infer_camtoworlds = make_circular_poses(n_views, theta, radius)

    def __len__(self):
        return len(self.infer_camtoworlds)

    def __getitem__(self, index):
        target_camtoworld = self.infer_camtoworlds[index]
        latlon = torch.stack(
            [
                self.get_trans(target_camtoworld, self.all_camtoworlds[i], in_T=False)
                for i in range(len(self.all_camtoworlds))
            ]
        )
        return {
            "image_cond": self.all_images,
            "theta": latlon[:, 0],
            "azimuth": latlon[:, 1],
            "distance": latlon[:, 2],
        }

    def loader(self, batch_size=1, num_workers=8, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            sampler=None,
            **kwargs,
        )
