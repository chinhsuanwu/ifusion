import json
from glob import glob

import numpy as np
import torch
from einops import rearrange
from liegroups.torch import SE3
from tqdm import trange

from dataset.finetune import FinetuneIterableDataset
from dataset.inference import MultiImageInferenceDataset, SingleImageInferenceDataset
from util.pose import latlon2mat, make_T, mat2latlon
from util.typing import *
from util.util import load_image, parse_optimizer, parse_scheduler
from util.viz import plot_image


def optimize_pose_loop(
    model,
    image_cond: Float[Tensor, "2 3 256 256"],
    image_target: Float[Tensor, "2 3 256 256"],
    T: Float[Tensor, "4 4"],
    default_radius: float,
    search_radius_range: float,
    use_step_ratio: bool,
    args,
    **kwargs,
):
    # init xi in se(3)
    xi = torch.randn(6) * 1e-6
    xi.requires_grad_()
    optimizer = parse_optimizer(args.optimizer, [xi])
    scheduler = parse_scheduler(args.scheduler, optimizer)

    total_loss = 0.0
    with trange(args.max_step) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            # se(3) -> SE(3)
            T_delta = SE3.exp(xi).as_matrix()
            T_ = T @ T_delta

            latlon = mat2latlon(T_).squeeze()
            theta, azimuth = latlon[0], latlon[1]
            distance = (
                torch.sin(torch.norm(T_[:3, 3]) - default_radius) * search_radius_range
            )

            idx = [0, 1] if torch.rand(1) < 0.5 else [1, 0]
            batch = {
                "image_cond": image_cond[idx],
                "image_target": image_target[idx],
                "T": torch.stack(
                    (
                        make_T(theta, azimuth, distance),
                        make_T(-theta, -azimuth, -distance),
                    )
                )[idx].to(model.device),
            }

            if use_step_ratio:
                loss = model(batch, step_ratio=step / args.max_step)
            else:
                loss = model(batch)

            total_loss += loss

            pbar.set_description(
                f"step: {step}, total_loss: {total_loss:.4f}, loss: {loss.item():.2f}, theta: {theta.rad2deg().item():.2f}, azimuth: {azimuth.rad2deg().item():.2f}, distance: {distance.item():.2f}"
            )

            loss.backward()
            optimizer.step()
            scheduler.step(total_loss)

    return total_loss, theta, azimuth, distance


def optimize_pose_pair(
    model,
    ref_image: Float[Tensor, "1 3 256 256"],
    qry_image: Float[Tensor, "1 3 256 256"],
    init_latlon: List[List],
    **kwargs,
):
    image_cond = torch.cat((ref_image, qry_image)).to(model.device)
    image_target = torch.cat((qry_image, ref_image)).to(model.device)
    init_T = latlon2mat(torch.tensor(init_latlon))
    results = []

    for T in init_T:
        total_loss, theta, azimuth, distance = optimize_pose_loop(
            model,
            image_cond=image_cond,
            image_target=image_target,
            T=T,
            **kwargs,
        )

        results.append(
            (
                total_loss.item(),
                theta.rad2deg().item(),
                azimuth.rad2deg().item(),
                distance.item(),
            )
        )

    results = torch.tensor(results)
    best_idx = torch.argmin(results[:, 0])
    pred_pose = results[best_idx][1:]
    print(
        f"[INFO] Best pose: theta: {pred_pose[0]:.2f}, azimuth: {pred_pose[1]:.2f}, distance: {pred_pose[2]:.2f}"
    )

    return pred_pose


def optimize_pose(
    model,
    image_dir: str,
    transform_fp: str,
    demo_fp: str,
    default_latlon: List[float] = [0, 0, 1],
    **kwargs,
):
    image_fps = sorted(glob(image_dir + "/*.png") + glob(image_dir + "/*.jpg"))
    image_fps = [fp for fp in image_fps if fp != demo_fp]

    # FIXME: always pick the first image as reference
    ref_image = load_image(image_fps[0])
    qry_images = [load_image(image_fps[i]) for i in range(1, len(image_fps))]

    out_dict = {"camera_angle_x": np.deg2rad(49.1), "frames": []}
    out_dict["frames"].append(
        {
            "file_path": image_fps[0].replace(image_dir + "/", ""),
            "transform_matrix": latlon2mat(torch.tensor([default_latlon])).tolist(),
            "latlon": list(default_latlon),
        }
    )
    for qry_fp, qry_image in zip(image_fps[1:], qry_images):
        assert ref_image.shape == qry_image.shape
        pose = optimize_pose_pair(
            model=model, ref_image=ref_image, qry_image=qry_image, **kwargs
        )
        pose = np.add(default_latlon, pose.unsqueeze(0))
        out_dict["frames"].append(
            {
                "file_path": qry_fp.replace(image_dir + "/", ""),
                "transform_matrix": latlon2mat(pose.clone()).tolist(),
                "latlon": pose.squeeze().tolist(),
            }
        )

    # save poses to json
    with open(transform_fp, "w") as f:
        json.dump(out_dict, f, indent=4)


def finetune(
    model,
    transform_fp: str,
    lora_ckpt_fp: str,
    lora_rank: int,
    lora_target_replace_module: List[str],
    args,
):
    model.inject_lora(
        rank=lora_rank,
        target_replace_module=lora_target_replace_module,
    )

    train_dataset = FinetuneIterableDataset(transform_fp)
    train_loader = train_dataset.loader(args.batch_size)
    optimizer = parse_optimizer(args.optimizer, model.require_grad_params)
    scheduler = parse_scheduler(args.scheduler, optimizer)

    train_loader = iter(train_loader)
    with trange(args.max_step) as pbar:
        for step in pbar:
            optimizer.zero_grad()

            batch = next(train_loader)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(batch)

            pbar.set_description(f"step: {step}, loss: {loss.item():.4f}")
            loss.backward()

            optimizer.step()
            scheduler.step()

    model.save_lora(lora_ckpt_fp)
    model.remove_lora()


def inference(
    model,
    transform_fp: str,
    lora_ckpt_fp: str,
    demo_fp: str,
    lora_rank: int,
    lora_target_replace_module: List[str],
    use_multi_view_condition: bool,
    n_views: int,
    theta: float,
    radius: float,
    args,
):
    if lora_ckpt_fp:
        model.inject_lora(
            ckpt_fp=lora_ckpt_fp,
            rank=lora_rank,
            target_replace_module=lora_target_replace_module,
        )

    if use_multi_view_condition:
        test_dataset = MultiImageInferenceDataset
        generate_fn = model.generate_from_tensor_multi_cond
    else:
        test_dataset = SingleImageInferenceDataset
        generate_fn = model.generate_from_tensor

    test_dataset = test_dataset(
        transform_fp=transform_fp, n_views=n_views, theta=theta, radius=radius
    )
    test_loader = test_dataset.loader(args.batch_size)
    for batch in test_loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        out = generate_fn(
            image=batch["image_cond"],
            theta=batch["theta"],
            azimuth=batch["azimuth"],
            distance=batch["distance"],
        )

    if lora_ckpt_fp:
        model.remove_lora()

    out = rearrange(out, "b c h w -> 1 c h (b w)")
    plot_image(out, fp=demo_fp)

    return out
