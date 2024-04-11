import argparse
import itertools
import os

import numpy as np
import torch

from dataset.base import load_frames

from util.criterion import lpips_fn, pose_err_fn, psnr_fn, ssim_fn
from util.util import load_config, load_image, set_random_seed, str2list


def eval_pose(transform_fp, gt_transform_fp, image_dir, id, **kwargs):
    camtoworlds = load_frames(image_dir, transform_fp, verbose=False)[1]
    gt_camtoworlds = load_frames(image_dir, gt_transform_fp, verbose=False)[1]
    gt_camtoworlds = gt_camtoworlds[str2list(id)]

    pose_err = [pose_err_fn(pred, gt) for pred, gt in zip(camtoworlds[1:], gt_camtoworlds[1:])]
    pose_err = np.array(pose_err).mean(axis=0)
    return pose_err


def eval_nvs(demo_fp, test_image_dir, test_transform_fp, **kwargs):
    pred = load_image(demo_fp, resize=False, to_clip=False)
    pred = torch.cat(torch.chunk(pred, 8, dim=-1))
    gt = load_frames(test_image_dir, test_transform_fp, to_clip=False)[0]
    gt = gt.to(pred.device)

    psnr = psnr_fn(pred, gt).item()
    ssim = ssim_fn(pred, gt).item()
    lpips = lpips_fn(pred, gt).item()
    return psnr, ssim, lpips


def eval_pose_all(config, scenes, ids):
    metric = []
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating pose {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            metric.append(eval_pose(**config.data))
    metric = np.array(metric)
    np.savez(f"{config.data.exp_root_dir}/pose_{config.data.name}.npz", metric)

    # NOTE: report the median error and recall < 5 degree
    print(f"Rot. error: {np.median(metric[:, 0])}, Trans. error: {np.median(metric[:, 1])}, Recall: {sum(metric[:, 0] <= 5) / len(metric)}")


def eval_nvs_all(config, scenes, ids):
    metric = []
    for scene in scenes:
        for id in ids:
            print(f"[INFO] Evaluating nvs {scene}:{id}")
            config.data.scene = scene
            config.data.id = id
            metric.append(eval_nvs(**config.data))
    metric = np.array(metric)
    np.savez(f"{config.data.exp_root_dir}/nvs_{config.data.name}.npz", metric)
    print(
        f"PSNR: {metric[:, 0].mean()}, SSIM: {metric[:, 1].mean()}, LPIPS: {metric[:, 2].mean()}"
    )


def main(config, mode):
    perm = list(itertools.permutations(range(5), 2))
    ids = [",".join(map(str, p)) for p in perm]
    scenes = sorted(os.listdir(f"{config.data.root_dir}/{config.data.name}"))

    if mode[0]:
        eval_pose_all(config, scenes, ids)
    if mode[1]:
        eval_nvs_all(config, scenes, ids=["0,1"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/main.yaml")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--nvs", action="store_true")
    args, extras = parser.parse_known_args()
    config = load_config(args.config, cli_args=extras)

    set_random_seed(config.seed)
    main(config, [args.pose, args.nvs])
