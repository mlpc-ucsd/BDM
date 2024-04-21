import inspect
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_pndm import PNDMScheduler
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Pointclouds
from torch import Tensor
from tqdm import tqdm

from .model_utils import get_num_points, get_custom_betas
from .point_cloud_model import PointCloudModel, PC2_PVDFusionModel
from .projection_model import PointCloudProjectionModel
from config.structured import ProjectConfig


class ConditionalPointCloudDiffusionModel(PointCloudProjectionModel):
    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model: str,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        # Checks
        if not self.predict_shape:
            raise NotImplementedError(
                "Must predict shape if performing diffusion.")

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == "custom":
            scheduler_kwargs.update(
                dict(
                    trained_betas=get_custom_betas(
                        beta_start=beta_start, beta_end=beta_end
                    )
                )
            )
        else:
            scheduler_kwargs.update(
                dict(
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_schedule=beta_schedule,
                )
            )
        self.schedulers_map = {
            "ddpm": DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            "ddim": DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            "pndm": PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map[
            "ddpm"
        ]  # this can be changed for inference

        # Create point cloud model for processing point cloud at each diffusion step
        self.point_cloud_model = PointCloudModel(
            model_type=point_cloud_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def forward_train(
        self,
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
    ):
        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=True)
        B, N, D = x_0.shape

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (B,),
            device=self.device,
            dtype=torch.long,
        )

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # Conditioning
        x_t_input = self.get_input_with_conditioning(
            x_t, camera=camera, image_rgb=image_rgb, mask=mask, t=timestep
        )

        # Forward
        noise_pred = self.point_cloud_model(x_t_input, timestep)

        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f"{noise_pred.shape=} and {noise.shape=}")

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss

    @torch.no_grad()
    def forward_sample(
        self,
        num_points: int,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        # Optional overrides
        scheduler: Optional[str] = "ddpm",
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        # TODO: add arguments for start_time(scheduler.timesteps), end_time(0) and x_t(None)
    ):
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = (
            self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        )

        # Get the size of the noise
        N = num_points
        B = 1 if image_rgb is None else image_rgb.shape[0]
        D = 3 + (self.color_channels if self.predict_color else 0)
        device = self.device if image_rgb is None else image_rgb.device

        # Sample noise
        # TODO: if start_time != scheduler.timesteps, assert we have x_t argument
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(
            scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = return_sample_every_n_steps > 0

        # TODO: use start_time, end_time to control this loop
        progress_bar = tqdm(
            scheduler.timesteps.to(device),
            desc=f"Sampling ({x_t.shape})",
            disable=disable_tqdm,
        )

        for i, t in enumerate(progress_bar):
            # Conditioning
            x_t_input = self.get_input_with_conditioning(
                x_t, camera=camera, image_rgb=image_rgb, mask=mask, t=t
            )

            # Forward
            noise_pred = self.point_cloud_model(
                x_t_input, t.reshape(1).expand(B))

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **
                                 extra_step_kwargs).prev_sample

            # Append to output list if desired
            if return_all_outputs and (
                i % return_sample_every_n_steps == 0
                or i == len(scheduler.timesteps) - 1
            ):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(
            x_t, denormalize=True, unscale=True)
        if return_all_outputs:
            # (B, sample_steps, N, D)
            all_outputs = torch.stack(all_outputs, dim=1)
            all_outputs = [
                self.tensor_to_point_cloud(o, denormalize=True, unscale=True)
                for o in all_outputs
            ]

        return (output, all_outputs) if return_all_outputs else output

    @torch.no_grad()
    def interaction_sample(
        self,
        # num_points: int,
        point_cloud: Optional[Tensor],
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],  # 之后换数据集的时候需要把这个mask的输入去掉
        # Optional overrides
        scheduler: Optional[str] = "ddpm",
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
        # TODO: add arguments for start_time(scheduler.timesteps), end_time(0) and x_t(None)
        start_time: int = 1000,
        end_time: int = 0,
    ):
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = (
            self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        )

        pred_pc = point_cloud
        # Get the size of the noise
        # N = num_points
        # B = 1 if image_rgb is None else image_rgb.shape[0]
        B = image_rgb.shape[0]
        # D = 3 + (self.color_channels if self.predict_color else 0)
        device = self.device if image_rgb is None else image_rgb.device

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(
            scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        progress_bar = tqdm(
            scheduler.timesteps[
                num_inference_steps - start_time: num_inference_steps - end_time
            ].to(device),
            desc=f"Sampling ({pred_pc.shape})",
            disable=disable_tqdm,
        )

        x_t = pred_pc

        for i, t in enumerate(progress_bar):
            # Conditioning
            x_t_input = self.get_input_with_conditioning(
                x_t, camera=camera, image_rgb=image_rgb, mask=mask, t=t
            )

            # Forward
            noise_pred = self.point_cloud_model(
                x_t_input, t.reshape(1).expand(B))

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **
                                 extra_step_kwargs).prev_sample

        pred_pc = x_t

        return pred_pc

    def forward(self, batch: FrameData, mode: str = "train", **kwargs):
        """A wrapper around the forward method for training and inference"""
        if isinstance(batch, dict):
            batch = FrameData(**batch)
        if mode == "train":
            return self.forward_train(
                pc=batch.sequence_point_cloud,
                camera=batch.camera,
                image_rgb=batch.image_rgb,
                mask=batch.fg_probability,
                **kwargs,
            )
        elif mode == "sample":
            num_points = kwargs.pop(
                "num_points", get_num_points(batch.sequence_point_cloud)
            )
            return self.forward_sample(
                num_points=num_points,
                camera=batch.camera,
                image_rgb=batch.image_rgb,
                mask=batch.fg_probability,
                **kwargs,
            )
        else:
            raise NotImplementedError()


class PointCloudFusionModel(PointCloudProjectionModel):
    def __init__(
        self,
        pvd_model: torch.nn.Module,
        pc2_model: ConditionalPointCloudDiffusionModel,
        beta_start: float,
        beta_end: float,
        beta_schedule: str,
        point_cloud_model: str,
        point_cloud_model_embed_dim: int,
        **kwargs,  # projection arguments
    ):
        super().__init__(**kwargs)

        self.p_forget = 0.2

        # Create diffusion model schedulers which define the sampling timesteps
        scheduler_kwargs = {}
        if beta_schedule == "custom":
            scheduler_kwargs.update(
                dict(
                    trained_betas=get_custom_betas(
                        beta_start=beta_start, beta_end=beta_end
                    )
                )
            )
        else:
            scheduler_kwargs.update(
                dict(
                    beta_start=beta_start,
                    beta_end=beta_end,
                    beta_schedule=beta_schedule,
                )
            )
        self.schedulers_map = {
            "ddpm": DDPMScheduler(**scheduler_kwargs, clip_sample=False),
            "ddim": DDIMScheduler(**scheduler_kwargs, clip_sample=False),
            "pndm": PNDMScheduler(**scheduler_kwargs),
        }
        self.scheduler = self.schedulers_map[
            "ddpm"
        ]  # this can be changed for inference

        # NOTE: Create point clouds fuse model to process
        self.fusion_model = PC2_PVDFusionModel(
            pc2_model=pc2_model,
            pvd_model=pvd_model,
            embed_dim=point_cloud_model_embed_dim,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
        )

    def forward_train(
        self,
        pc: Pointclouds,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        return_intermediate_steps: bool = False,
    ):
        # Normalize colors and convert to tensor
        x_0 = self.point_cloud_to_tensor(pc, normalize=True, scale=True)

        B, N, D = x_0.shape

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Sample random timesteps for each point_cloud
        timestep = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (B,),
            device=self.device,
            dtype=torch.long,
        )

        # Add noise to points
        x_t = self.scheduler.add_noise(x_0, noise, timestep)

        # Conditioning
        x_t_input = self.get_input_with_conditioning(
            x_t, camera=camera, image_rgb=image_rgb, mask=mask, t=timestep
        )
        # Forward
        noise_pred = self.fusion_model(
            x_t_input, x_t, timestep, mode='fusion_1step')

        # Check
        if not noise_pred.shape == noise.shape:
            raise ValueError(f"{noise_pred.shape=} and {noise.shape=}")

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        # Whether to return intermediate steps
        if return_intermediate_steps:
            return loss, (x_0, x_t, noise, noise_pred)

        return loss

    @torch.no_grad()
    def forward_sample(
        self,
        num_points: int,
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        scheduler: Optional[str] = "ddpm",
        # Inference parameters
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,  # for DDIM
        # Whether to return all the intermediate steps in generation
        return_sample_every_n_steps: int = -1,
        # Whether to disable tqdm
        disable_tqdm: bool = False,
    ):
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = (
            self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        )

        # Get the size of the noise
        N = num_points
        B = 1 if image_rgb is None else image_rgb.shape[0]
        D = 3 + (self.color_channels if self.predict_color else 0)
        device = self.device if image_rgb is None else image_rgb.device

        # Sample noise
        # TODO: if start_time != scheduler.timesteps, assert we have x_t argument
        x_t = torch.randn(B, N, D, device=device)

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(
            scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        # Loop over timesteps
        all_outputs = []
        return_all_outputs = return_sample_every_n_steps > 0

        progress_bar = tqdm(
            scheduler.timesteps.to(device),
            desc=f"Sampling ({x_t.shape})",
            disable=disable_tqdm,
        )
        for i, t in enumerate(progress_bar):
            # Conditioning
            x_t_input = self.get_input_with_conditioning(
                x_t, camera=camera, image_rgb=image_rgb, mask=mask, t=t
            )

            # Forward
            noise_pred = self.fusion_model(
                x_t_input, x_t, t.reshape(1).expand(B), mode='fusion_1step')

            # Step
            x_t = scheduler.step(noise_pred, t, x_t, **
                                 extra_step_kwargs).prev_sample

            # Append to output list if desired
            if return_all_outputs and (
                i % return_sample_every_n_steps == 0
                or i == len(scheduler.timesteps) - 1
            ):
                all_outputs.append(x_t)

        # Convert output back into a point cloud, undoing normalization and scaling
        output = self.tensor_to_point_cloud(
            x_t, denormalize=True, unscale=True)
        if return_all_outputs:
            # (B, sample_steps, N, D)
            all_outputs = torch.stack(all_outputs, dim=1)
            all_outputs = [
                self.tensor_to_point_cloud(o, denormalize=True, unscale=True)
                for o in all_outputs
            ]

        return (output, all_outputs) if return_all_outputs else output

    @torch.no_grad()
    def nstep_fuse(
        self,
        pred_from_prior: Optional[Tensor],
        pred_from_recon: Optional[Tensor],
        camera: Optional[CamerasBase],
        image_rgb: Optional[Tensor],
        mask: Optional[Tensor],
        scheduler: Optional[str] = "ddpm",
        num_inference_steps: Optional[int] = 1000,
        eta: Optional[float] = 0.0,
        return_sample_every_n_steps: int = -1,
        disable_tqdm: bool = False,
        timestep: int = 0,
    ):
        # Get scheduler from mapping, or use self.scheduler if None
        scheduler = (
            self.scheduler if scheduler is None else self.schedulers_map[scheduler]
        )
        device = self.device if image_rgb is None else image_rgb.device
        pred_from_prior -= pred_from_prior.mean(dim=1, keepdim=True)
        pred_from_recon -= pred_from_recon.mean(dim=1, keepdim=True)

        t = torch.from_numpy(np.array(timestep)).to(device)
        B = 1 if image_rgb is None else image_rgb.shape[0]

        # Set timesteps
        accepts_offset = "offset" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        extra_set_kwargs = {"offset": 1} if accepts_offset else {}
        scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(
            scheduler.step).parameters.keys())
        extra_step_kwargs = {"eta": eta} if accepts_eta else {}

        scheduler.timesteps.to(device)
        pred_from_recon_with_input = self.get_input_with_conditioning(
            pred_from_recon, camera=camera, image_rgb=image_rgb, mask=mask, t=t
        )

        noise_pred = self.fusion_model(
            pred_from_recon_with_input,
            pred_from_prior,
            t.reshape(1).expand(B),
            mode="fusion_nstep",
        )
        # Step
        pred_fuse = scheduler.step(
            noise_pred, t, pred_from_recon, **extra_step_kwargs
        ).prev_sample

        # Convert output back into a point cloud, undoing normalization and scaling
        # output = self.tensor_to_point_cloud(pred_fuse, denormalize=True, unscale=True)

        return pred_fuse

    def forward(self, batch: FrameData, mode: str = "train", **kwargs):
        """A wrapper around the forward method for training and inference"""
        if isinstance(
            batch, dict
        ):  # fixes a bug with multiprocessing where batch becomes a dict
            batch = FrameData(
                **batch
            )  # it really makes no sense, I do not understand it
        if mode == "train":
            return self.forward_train(
                pc=batch.sequence_point_cloud,
                camera=batch.camera,
                image_rgb=batch.image_rgb,
                mask=batch.fg_probability,
                **kwargs,
            )
        elif mode == "sample":
            num_points = kwargs.pop(
                "num_points", get_num_points(batch.sequence_point_cloud)
            )
            return self.forward_sample(
                num_points=num_points,
                camera=batch.camera,
                image_rgb=batch.image_rgb,
                mask=batch.fg_probability,
                **kwargs,
            )
        else:
            raise NotImplementedError()
