import itertools
from dataclasses import dataclass

import torch
import torch.nn as nn
from diffusers import DDIMScheduler
from einops import rearrange
from omegaconf import OmegaConf

from ldm.lora import (
    inject_trainable_lora_extended,
    monkeypatch_remove_lora,
    save_lora_weight,
)
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import load_model_from_config
from util.pose import make_T
from util.typing import *
from util.util import default


class Zero123(nn.Module):
    @dataclass
    class Config:
        pretrained_model_name_or_path: str = "ldm/ckpt/zero123-xl.ckpt"
        pretrained_config: str = "ldm/ckpt/sd-objaverse-finetune-c_concat-256.yaml"
        vram_O: bool = False

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

    config: Config

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.config = OmegaConf.structured(self.Config(**kwargs))
        self.device = "cuda"
        self.require_grad_params = []
        self.configure()

    def configure(self) -> None:
        print("[INFO] Loading Zero123...")

        self.pretrained_config = OmegaConf.load(self.config.pretrained_config)
        self.weights_dtype = torch.float32
        self.model: LatentDiffusion = load_model_from_config(
            self.pretrained_config,
            self.config.pretrained_model_name_or_path,
            device=self.device,
            vram_O=self.config.vram_O,
        )

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.pretrained_config.model.params.timesteps
        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.pretrained_config.model.params.linear_start,
            self.pretrained_config.model.params.linear_end,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps(
            min_step_percent=self.config.min_step_percent,
            max_step_percent=self.config.max_step_percent,
        )

        print("[INFO] Loaded Zero123")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(
        self, min_step_percent: float = 0.02, max_step_percent: float = 0.98
    ):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def get_image_embeds(
        self, image: Float[Tensor, "B 3 256 256"]
    ) -> Tuple[Float[Tensor, "B 1 768"], Float[Tensor, "B 4 32 32"]]:
        c_crossattn = self.model.get_learned_conditioning(image.to(self.weights_dtype))
        c_concat = self.model.encode_first_stage(image.to(self.weights_dtype)).mode()
        return c_crossattn, c_concat

    @torch.cuda.amp.autocast(enabled=False)
    def encode_image(
        self, image: Float[Tensor, "B 3 256 256"]
    ) -> Float[Tensor, "B 4 32 32"]:
        input_dtype = image.dtype
        latent = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(image.to(self.weights_dtype))
        )
        return latent.to(input_dtype)  # [B, 4, 32, 32] Latent space image

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latent(
        self,
        latent: Float[Tensor, "B 4 H W"],
    ) -> Float[Tensor, "B 3 512 512"]:
        input_dtype = latent.dtype
        image = self.model.decode_first_stage(latent)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @staticmethod
    @torch.no_grad()
    def make_cond(cond):
        """Add zeros to the beginning of cond"""
        return {k: [torch.cat([torch.zeros_like(v), v])] for k, v in cond.items()}

    @torch.cuda.amp.autocast(enabled=False)
    @torch.no_grad()
    def clip_camera_projection(
        self,
        theta: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        distance: Float[Tensor, "B"],
        c_crossattn: Float[Tensor, "B 1 768"],
        in_deg: bool = False,
    ):
        T = make_T(theta, azimuth, distance, in_deg=in_deg).T[:, None, :]
        clip_emb = self.model.cc_projection(torch.cat([c_crossattn, T], dim=-1))
        return clip_emb

    def inject_lora(
        self,
        ckpt_fp: str = None,
        rank: int = 12,
        target_replace_module: List[str] = ["CrossAttention", "GEGLU"],
        eval: bool = False,
    ):
        print(
            f"[INFO] Injecting LoRA from "
            + (str(ckpt_fp) if ckpt_fp is not None else "scratch"),
        )

        lora_params, _ = inject_trainable_lora_extended(
            self.model.model,
            target_replace_module=set(target_replace_module),
            r=rank,
            loras=ckpt_fp,
            eval=eval,
        )
        if not eval:
            self.require_grad_params += itertools.chain(*lora_params)
        return self

    def save_lora(
        self,
        ckpt_fp: str,
        target_replace_module: List[str] = ["CrossAttention", "GEGLU"],
    ):
        save_lora_weight(
            self.model.model,
            ckpt_fp,
            target_replace_module=set(target_replace_module),
        )
        print(f"[INFO] Saved LoRA to {ckpt_fp}")

    def remove_lora(self):
        print("[INFO] Removing LoRA")
        monkeypatch_remove_lora(self.model.model)
        self.require_grad_params = []
        return self

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, batch, step_ratio=None):
        batch["image_cond"] = rearrange(batch["image_cond"], "b c h w -> b h w c")
        batch["image_target"] = rearrange(batch["image_target"], "b c h w -> b h w c")
        loss, _ = self.model.shared_step(batch, step_ratio=step_ratio)
        return loss

    def generate_from_tensor(
        self,
        image: Float[Tensor, "B 3 256 256"],
        theta: Float[Tensor, "B"],
        azimuth: Float[Tensor, "B"],
        distance: Float[Tensor, "B"],
        scale=3,
        ddim_steps=50,
        ddim_eta=1,
        in_deg: bool = False,
    ):
        if len(image) != len(theta):
            image = image.repeat(len(theta), 1, 1, 1)
        c_crossattn, c_concat = self.get_image_embeds(image)
        c_crossattn = self.clip_camera_projection(
            theta, azimuth, distance, c_crossattn, in_deg
        )
        out = self.gen_from_cond(
            cond={"c_crossattn": c_crossattn, "c_concat": c_concat},
            scale=scale,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        return out

    def generate_from_tensor_multi_cond(
        self,
        image: Float[Tensor, "B N 3 256 256"],
        theta: Float[Tensor, "B N"],
        azimuth: Float[Tensor, "B N"],
        distance: Float[Tensor, "B N"],
        scale=3,
        ddim_steps=50,
        ddim_eta=1,
        in_deg: bool = False,
    ):
        c_crossattn, c_concat = zip(*[self.get_image_embeds(x) for x in image])
        c_crossattn, c_concat = torch.stack(c_crossattn), torch.stack(c_concat)
        c_crossattn = torch.stack(
            [
                self.clip_camera_projection(t, a, d, c, in_deg)
                for t, a, d, c in zip(theta, azimuth, distance, c_crossattn)
            ]
        )
        out = self.gen_from_cond(
            cond={"c_crossattn": c_crossattn, "c_concat": c_concat},
            scale=scale,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
            use_multi_view_condition=True,
        )
        return out

    def generate(
        self,
        image: Float[Tensor, "B 3 256 256"],
        theta: float,
        azimuth: float,
        distance: float,
        in_deg: bool = True,
        **kwargs,
    ):
        theta = torch.tensor([theta], device=self.device)
        azimuth = torch.tensor([azimuth], device=self.device)
        distance = torch.tensor([distance], device=self.device)
        out = self.generate_from_tensor(image, theta, azimuth, distance, in_deg=in_deg, **kwargs)
        return out

    @torch.no_grad()
    def gen_from_cond(
        self, cond, scale=3, ddim_steps=50, ddim_eta=1, use_multi_view_condition=False
    ):
        B = len(cond["c_crossattn"])
        if use_multi_view_condition:
            N = len(cond["c_crossattn"][0])

        latent = torch.randn((B, 4, 32, 32), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)

        cond_ = None  # temporary condition
        for t in self.scheduler.timesteps:
            x_in = torch.cat([latent] * 2)
            t_in = torch.cat([t.reshape(1).repeat(B)] * 2).to(self.device)

            if use_multi_view_condition:
                # multi-view stochastic condition
                index = torch.randint(0, N, (B,))
                cond_ = {
                    "c_crossattn": cond["c_crossattn"][torch.arange(B), index],
                    "c_concat": cond["c_concat"][torch.arange(B), index],
                }
                cond_ = self.make_cond(cond_)
            else:
                cond_ = default(cond_, self.make_cond(cond))

            noise_pred = self.model.apply_model(x_in, t_in, cond_)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (
                noise_pred_cond - noise_pred_uncond
            )

            latent = self.scheduler.step(noise_pred, t, latent, eta=ddim_eta)[
                "prev_sample"
            ]
        images = self.decode_latent(latent)

        return images
