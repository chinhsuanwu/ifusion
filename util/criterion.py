import numpy as np
import torch

from lpips import LPIPS
from util.ssim import SSIM


lpips_net = LPIPS(net="vgg").to("cuda")


def psnr_fn(pred, gt):
    mse = torch.mean((pred - gt) ** 2, dim=(1, 2, 3))
    psnr = 10.0 * torch.log10(1.0 / mse).mean()
    return psnr


def ssim_fn(pred, gt):
    ssim_loss = SSIM(window_size = 11)
    return ssim_loss(pred, gt)


def lpips_fn(pred, gt):
    assert pred.min() >= 0 and pred.max() <= 1 and gt.min() >= 0 and gt.max() <= 1
    return lpips_net(pred * 2 - 1, gt * 2 - 1).mean()


def pose_err_fn(pred_T, gt_T, in_deg=True):
    assert pred_T.shape == gt_T.shape == (4, 4)

    if isinstance(pred_T, torch.Tensor):
        pred_T = pred_T.detach().cpu().numpy()
    if isinstance(gt_T, torch.Tensor):
        gt_T = gt_T.detach().cpu().numpy()

    rot_error = np.arccos((np.trace(np.dot(gt_T[:3, :3].T, pred_T[:3, :3])) - 1) / 2.0)
    trans_error = np.linalg.norm(gt_T[:3, 3] - pred_T[:3, 3])

    if in_deg:
        rot_error = np.degrees(rot_error)
    return rot_error, trans_error
