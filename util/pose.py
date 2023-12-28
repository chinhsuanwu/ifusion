import torch
import numpy as np


def translate(x, y, z):
    return torch.tensor(
        [
            [1,  0,  0,  x], 
            [0,  1,  0,  y], 
            [0,  0,  1,  z], 
            [0,  0,  0,  1]
        ],
        dtype=torch.float32,
    )


def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [
            [1,  0,  0,  0], 
            [0,  c,  s,  0], 
            [0, -s,  c,  0], 
            [0,  0,  0,  1]
        ],
        dtype=torch.float32,
    )


def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor(
        [
            [c,  0,  s,  0],
            [0,  1,  0,  0], 
            [-s, 0,  c,  0], 
            [0,  0,  0,  1]
        ],
        dtype=torch.float32,
    )


def scale(s):
    return torch.tensor(
        [
            [s,  0,  0,  0],
            [0,  s,  0,  0], 
            [0,  0,  s,  0], 
            [0,  0,  0,  1]
        ],
        dtype=torch.float32,
    )


def latlon2mat(latlon, in_deg=True, default_radius=1.0):
    if latlon.shape[-1] == 2:
        radius = torch.ones_like(latlon[:, 0]) * default_radius
        latlon = torch.cat((latlon, radius.unsqueeze(1)), dim=1)
    
    if in_deg:
        latlon[:, :2] = latlon[:, :2].deg2rad()
    mv = [
        translate(0, 0, -radius) @ rotate_x(theta) @ rotate_y(-azimuth)
        for theta, azimuth, radius in latlon
    ]
    c2w = torch.linalg.inv(torch.stack(mv))
    return c2w


def mat2latlon(T, in_deg=False, return_radius=False):
    if len(T.shape) == 2:
        T = T.unsqueeze(0)
    xyz = T[:, :3, 3]
    radius = torch.norm(xyz, dim=1, keepdim=True)
    xyz = xyz / radius
    theta = -torch.asin(xyz[:, 1])
    azimuth = torch.atan2(xyz[:, 0], xyz[:, 2])

    if in_deg:
        theta, azimuth = theta.rad2deg(), azimuth.rad2deg()
    if return_radius:
        return torch.stack((theta, azimuth, radius.squeeze(0))).T
    return torch.stack((theta, azimuth)).T


def make_T(theta, azimuth, distance, in_deg=False):
    if in_deg:
        theta, azimuth = theta.deg2rad(), azimuth.deg2rad()
    return torch.stack(
        (
            theta,
            torch.sin(azimuth),
            torch.cos(azimuth),
            distance,
        )
    )
