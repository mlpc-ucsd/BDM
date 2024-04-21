import datetime
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, List, Optional
from PIL import Image

import random
import numpy as np
from pvd import generate_pvd_xyz, prepare_pvd_model
import trimesh
import warnings
from tqdm.auto import tqdm

import hydra
import torch
import wandb
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from torchvision.transforms import functional as TVF
from pytorch3d.structures import Pointclouds
from pytorch3d.io import IO
from pytorch3d.implicitron.dataset.data_loader_map_provider import FrameData

import training_utils
import diffusion_utils
from dataset import get_dataset
from model import get_model, ConditionalPointCloudDiffusionModel
from model import get_fusion_model, PointCloudFusionModel
from config.structured import ProjectConfig

torch.multiprocessing.set_sharing_strategy("file_system")


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: ProjectConfig):
    # Accelerator
    accelerator = Accelerator(
        mixed_precision=cfg.run.mixed_precision,
        cpu=cfg.run.cpu,
        gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
    )

    # Logging
    training_utils.setup_distributed_print(accelerator.is_main_process)
    if cfg.logging.wandb and accelerator.is_main_process:
        wandb.init(
            project=cfg.logging.wandb_project,
            name=cfg.run.name,
            job_type=cfg.run.job,
            config=OmegaConf.to_container(cfg),
        )
        wandb.run.log_code(
            root=hydra.utils.get_original_cwd(),
            include_fn=lambda p: any(
                p.endswith(ext)
                for ext in (".py", ".json", ".yaml", ".md", ".txt.", ".gin")
            ),
            exclude_fn=lambda p: any(
                s in p for s in ("output", "tmp", "wandb", ".git", ".vscode")
            ),
        )
        cfg: ProjectConfig = DictConfig(
            wandb.config.as_dict()
        )  # get the config back from wandb for hyperparameter sweeps

    # Configuration
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory: {os.getcwd()}")

    # Set random seed
    training_utils.set_seed(cfg.run.seed)

    # Build two diffusion model: PVD and PC2 and set them both into the mode of eval
    print("Building prior model...")
    pvd_path = cfg.aux_run.prior_ckpt
    opt = {
        "model": pvd_path,
        "nc": 3,
        "embed_dim": 64,
        "attention": True,
        "dropout": 0.1,
    }
    prior_model = prepare_pvd_model(opt, accelerator.device)
    prior_model.eval()
    print("Built prior model!")

    print("Building recon model...")
    model = get_model(cfg)
    model.eval()

    # Exponential moving average of model parameters
    if cfg.ema.use_ema:
        from torch_ema import ExponentialMovingAverage

        model_ema = ExponentialMovingAverage(
            model.parameters(), decay=cfg.ema.decay)
        model_ema.to(accelerator.device)
        print("Initialized model EMA")
    else:
        model_ema = None
        print("Not using model EMA")

    print(f"Loading recon checkpoint ({datetime.datetime.now()})")
    checkpoint = torch.load(cfg.aux_run.recon_ckpt, map_location="cpu")
    if "model" in checkpoint:
        state_dict, key = checkpoint["model"], "model"
    else:
        state_dict, key = checkpoint, "N/A"
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k,
                      v in state_dict.items()}
        print('Removed "module." from checkpoint state dict')
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False)
    print(f"Loaded model checkpoint key {key} from {cfg.aux_run.recon_ckpt}")
    if len(missing_keys):
        print(f" - Missing_keys: {missing_keys}")
    if len(unexpected_keys):
        print(f" - Unexpected_keys: {unexpected_keys}")
    print("Built recon model!")

    # Fusion model
    print("Building fusion model...")
    fusion_model = get_fusion_model(
        cfg, prior_model, model).to(accelerator.device)
    print("Built fusion model!")

    print(
        f"Parameters of fusion model (total): {sum(p.numel() for p in fusion_model.parameters()):_d}")
    print(
        f"Parameters of fusion model (train): {sum(p.numel() for p in fusion_model.parameters() if p.requires_grad):_d}")

    # Optimizer and scheduler
    optimizer = training_utils.get_optimizer(cfg, fusion_model, accelerator)
    scheduler = training_utils.get_scheduler(cfg, optimizer)

    if cfg.run.job == "training_bdm_merging":
        print(
            "Loading fusion model from recon model and prior model's encoder, and resume finetuning on the decoder..."
        )
        cfg.checkpoint.resume = (
            None  # NOTE: to set for our fusion model, only aim at training_fusion_state
        )
        training_fusion_state: training_utils.TrainState = (
            training_utils.resume_from_checkpoint(
                cfg, fusion_model, optimizer, scheduler, model_ema=None
            )
        )
        print("Built fusion model (Initialize)!")

    else:
        print(f"Loading fusion checkpoint ({datetime.datetime.now()})")
        checkpoint = torch.load(cfg.aux_run.fusion_ckpt, map_location="cpu")
        if "model" in checkpoint:
            state_dict, key = checkpoint["model"], "model"
        else:
            state_dict, key = checkpoint, "N/A"
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k,
                          v in state_dict.items()}
            print('Removed "module." from checkpoint state dict')
        missing_keys, unexpected_keys = fusion_model.load_state_dict(
            state_dict, strict=False
        )
        print(
            f"Loaded model checkpoint key {key} from {cfg.aux_run.fusion_ckpt}")
        if len(missing_keys):
            print(f" - Missing_keys: {missing_keys}")
        if len(unexpected_keys):
            print(f" - Unexpected_keys: {unexpected_keys}")
        print("Built fusion model!")

    # Datasets
    dataloader_train, dataloader_val, dataloader_vis = get_dataset(cfg)

    # Setup
    (
        model,  # NOTE: fusion_model will raise error!!!
        optimizer,
        scheduler,
        dataloader_train,
        dataloader_val,
        dataloader_vis,
    ) = accelerator.prepare(
        model, optimizer, scheduler, dataloader_train, dataloader_val, dataloader_vis
    )

    # Type hints
    model: ConditionalPointCloudDiffusionModel
    fusion_model: PointCloudFusionModel
    optimizer: torch.optim.Optimizer

    # Train Merging
    if cfg.run.job == "training_bdm_merging":
        training_bdm_merging(
            cfg=cfg,
            fusion_model=fusion_model,
            dataloader_train=dataloader_train,
            accelerator=accelerator,
            optimizer=optimizer,
            training_fusion_state=training_fusion_state,
            scheduler=scheduler,
        )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    elif cfg.run.job == "sample_bdm_merging":
        sample_bdm_merging(
            cfg=cfg,
            fusion_model=fusion_model,
            dataloader=dataloader_val,
            accelerator=accelerator,
            prior_model=prior_model,
            recon_model=model,
        )
        if cfg.logging.wandb and accelerator.is_main_process:
            wandb.finish()
        time.sleep(5)
        return

    else:
        raise ValueError(f"Invalid job: {cfg.run.job}")


# @torch.no_grad() NOTE: we need to remove this decorator for training
def pvd_prior(pvd_model, points, start_time, end_time):
    points = points.permute(0, 2, 1).float()
    pcd_tensor_after_prior = generate_pvd_xyz(
        pvd_model, points, start_time, end_time)
    pcd_tensor_after_prior = pcd_tensor_after_prior.permute(
        0, 2, 1)  # (bsz, 4096, 3)

    return pcd_tensor_after_prior


def training_bdm_merging(
    cfg: ProjectConfig,
    fusion_model: torch.nn.Module,
    dataloader_train: Iterable,
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    training_fusion_state: training_utils.TrainState,
    scheduler,
):
    print(f"***** Starting training fusion at {datetime.datetime.now()} *****")
    print(f"    Dataset train size: {len(dataloader_train.dataset):_}")
    print(f"    Dataloader train size: {len(dataloader_train):_}")
    print(f"    Batch size per device = {cfg.dataloader.batch_size}")
    print(
        f"    Gradient Accumulation steps = {cfg.optimizer.gradient_accumulation_steps}")
    print(f"    Max training fusion steps = {cfg.run.max_fusion_steps}")
    print(f"    Training fusion state = {training_fusion_state}")

    # Infinitely loop training
    while True:
        # Finetune progress bar
        log_header = f"Epoch: [{training_fusion_state.epoch}]"
        metric_logger = training_utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "step", training_utils.SmoothedValue(
                window_size=1, fmt="{value:.0f}")
        )
        metric_logger.add_meter(
            "lr", training_utils.SmoothedValue(
                window_size=1, fmt="{value:.6f}")
        )
        progress_bar: Iterable[Any] = metric_logger.log_every(
            dataloader_train, cfg.run.print_step_freq, header=log_header
        )

        # Finetune
        for i, batch in enumerate(progress_bar):
            if (cfg.run.limit_train_batches is not None) and (
                i >= cfg.run.limit_train_batches
            ):
                break
            fusion_model.train()
            # Gradient accumulation
            with accelerator.accumulate(fusion_model):
                # batch is type of Dict() here.
                # Forward
                loss = fusion_model(batch, mode="train")

                # Backward
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # grad_norm_unclipped = training_utils.compute_grad_norm(model.parameters())  # useless w/ mixed prec
                    if cfg.optimizer.clip_grad_norm is not None:
                        accelerator.clip_grad_norm_(
                            fusion_model.parameters(), cfg.optimizer.clip_grad_norm
                        )
                    grad_norm_clipped = training_utils.compute_grad_norm(
                        fusion_model.parameters()
                    )

                # Step optimizer
                optimizer.step()
                optimizer.zero_grad()
                if accelerator.sync_gradients:
                    scheduler.step()
                    training_fusion_state.step += 1

                # Exit if loss was NaN
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print("Loss is {}, stopping training".format(loss_value))
                    sys.exit(1)

            # Gradient accumulation
            if accelerator.sync_gradients:
                # Logging
                log_dict = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": training_fusion_state.step,
                    "train_loss": loss_value,
                    # 'grad_norm_unclipped': grad_norm_unclipped,  # useless w/ mixed prec
                    "grad_norm_clipped": grad_norm_clipped,
                }
                metric_logger.update(**log_dict)
                if (
                    cfg.logging.wandb
                    and accelerator.is_main_process
                    and training_fusion_state.step % cfg.run.log_step_freq == 0
                ):
                    wandb.log(log_dict, step=training_fusion_state.step)

                # Save a checkpoint
                if accelerator.is_main_process and (
                    training_fusion_state.step % cfg.run.checkpoint_freq == 0
                ):
                    checkpoint_dict = {
                        "model": accelerator.unwrap_model(fusion_model).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": training_fusion_state.epoch,
                        "step": training_fusion_state.step,
                        "best_val": training_fusion_state.best_val,
                        "model_ema": {},
                        "cfg": cfg,
                    }
                    checkpoint_path = f"checkpoint-{training_fusion_state.step}.pth"
                    accelerator.save(checkpoint_dict, checkpoint_path)
                    print(
                        f"Saved checkpoint to {Path(checkpoint_path).resolve()}")

                # End training after the desired number of steps/epochs
                if training_fusion_state.step >= cfg.run.max_fusion_steps:
                    print(f"Ending training at: {datetime.datetime.now()}")
                    print(f"Final train state: {training_fusion_state}")

                    wandb.finish()
                    time.sleep(5)
                    return

        # Epoch complete, log it and continue training
        training_fusion_state.epoch += 1

        # Gather stats from all processes
        metric_logger.synchronize_between_processes(device=accelerator.device)
        print(f"{log_header}  Average stats --", metric_logger)


@torch.no_grad()
def bdm_merging(
    accelerator: Accelerator,
    batch: FrameData or dict,
    cfg: ProjectConfig,
    prior_model: torch.nn.Module,
    recon_model: ConditionalPointCloudDiffusionModel,
    fusion_model: PointCloudFusionModel,
):
    """
    we fulfill our need to interact two diffusion model in this function
    the meaning of some variables should be clearified here

    1) status: this variable is set to be either 1 or 0, which means it is on the branch of prior or recon
    specially, 1 represents prior and 0 represents recon
    2) timelist: obviously
    3) timephase: a phase that specify the start and the end of interaction
    4) pred_pc: we recommend to use only one variable to transmit tensor of point cloud between two branches.
    """

    img = batch.image_rgb
    mask = batch.fg_probability
    camera = batch.camera

    roll_step = cfg.aux_run.roll_step  # 1
    milestones = cfg.aux_run.milestones  # [64,62,60,56,8,4,2,0]
    times = len(milestones) - 1

    if cfg.run.diffusion_scheduler == "ddim":
        prior_roll_step = int(roll_step * 16)  # 16
        prior_milestones = [
            int(i / 64 * 1000) for i in milestones
        ]  # [1000,968,936,872,128,64,32,0]
    else:
        assert cfg.run.diffusion_scheduler == "ddpm"
        prior_roll_step = roll_step
        prior_milestones = milestones

    B = img.shape[0]
    num_points = cfg.dataset.max_points
    device = recon_model.point_cloud_model.device

    pred_pc = torch.randn(B, num_points, 3).to(device)
    pred_pc -= pred_pc.mean(dim=1, keepdim=True)

    # start
    for i in range(times):  # (0, 1, ..., 19)
        if i == 0:  # only sample recon
            print(
                "Go through recon model from {} to {}".format(
                    milestones[i], milestones[i + 1] - roll_step
                )
            )
            pred_pc = recon_model.interaction_sample(
                pred_pc,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i],  # 64
                end_time=milestones[i + 1] - roll_step,  # 61
            )  # (B, N, 3)
        elif i == times - 1:  # only sample recon
            print(
                "Go through recon model from {} to {}".format(
                    milestones[i] - roll_step, milestones[i + 1]
                )
            )
            pred_pc = recon_model.interaction_sample(
                pred_pc,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i] - roll_step,  # 1
                end_time=milestones[i + 1],  # 0
            )  # (B, N, 3)
        else:
            print(
                "Go through recon model from {} to {}".format(
                    milestones[i] - roll_step, milestones[i + 1]
                )
            )
            pred_pc = recon_model.interaction_sample(
                pred_pc,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i] - roll_step,
                end_time=milestones[i + 1],
            )  # (B, N, 3)

            print("Begin to combine outputs of recon model and prior model")
            print(
                "Go through Branch1: recon model from {} to {}".format(
                    milestones[i + 1], milestones[i + 1] - roll_step + 1
                )
            )
            pred_pc_recon = pred_pc.clone()
            pred_out_recon = recon_model.interaction_sample(
                pred_pc_recon,
                camera,
                img,
                mask,
                return_sample_every_n_steps=10,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                disable_tqdm=(not accelerator.is_main_process),
                start_time=milestones[i + 1],
                end_time=milestones[i + 1] - roll_step + 1,
            )  # (B, N, 3)

            print(
                "Go through Branch2: prior from {} to {}".format(
                    prior_milestones[i + 1],
                    prior_milestones[i + 1] - prior_roll_step + 1
                )
            )
            pred_pc_prior = pred_pc.clone()
            pred_out_prior = pvd_prior(
                prior_model,
                pred_pc_prior,
                start_time=prior_milestones[i + 1],
                end_time=prior_milestones[i + 1] - prior_roll_step + 1,
            )  # (B, N, 3)

            # put these two inputs to fusion model
            print(
                "Fuse two point clouds at this step {}".format(
                    prior_milestones[i + 1] - prior_roll_step
                )
            )
            pred_pc = fusion_model.nstep_fuse(
                pred_out_prior,
                pred_out_recon,
                camera,
                img,
                mask,
                scheduler=cfg.run.diffusion_scheduler,
                num_inference_steps=cfg.run.num_inference_steps,
                timestep=milestones[i + 1] - roll_step,
            )

    output = Pointclouds(pred_pc)
    return output


@torch.no_grad()
def sample_bdm_merging(
    *,
    cfg: ProjectConfig,
    fusion_model: PointCloudFusionModel,
    dataloader: Iterable,
    accelerator: Accelerator,
    prior_model: torch.nn.Module,
    recon_model: ConditionalPointCloudDiffusionModel,
    output_dir: str = "sample_bdm_merging",
):
    # Eval mode
    fusion_model.eval()
    progress_bar: Iterable[FrameData] = tqdm(
        dataloader, disable=(not accelerator.is_main_process)
    )

    # Output dir
    output_dir: Path = Path(output_dir)

    # PyTorch3D IO
    io = IO()

    # Visualize
    for batch_idx, batch in enumerate(progress_bar):
        if isinstance(batch, dict):
            batch = FrameData(**batch)

        progress_bar.set_description(
            f"Processing batch {batch_idx:4d} / {len(dataloader):4d}"
        )

        if (
            cfg.run.num_sample_batches is not None
            and batch_idx >= cfg.run.num_sample_batches
        ):
            break

        # Optionally produce multiple samples for each point cloud
        for sample_idx in range(cfg.run.num_samples):
            # Filestring
            filename = (
                f"{{name}}-{sample_idx}.{{ext}}"
                if cfg.run.num_samples > 1
                else "{name}.{ext}"
            )
            filestr = str(output_dir / "{dir}" / "{category}" / filename)

            output = bdm_merging(
                accelerator=accelerator,
                batch=batch,
                cfg=cfg,
                prior_model=prior_model,
                recon_model=recon_model,
                fusion_model=fusion_model,
            )
            assert isinstance(output, Pointclouds)

            # Save individual samples
            for i in range(len(output)):
                frame_number = batch["frame_number"][i]
                sequence_name = batch["sequence_name"][i]
                sequence_category = batch["sequence_category"][i]
                (output_dir / "gt" / sequence_category).mkdir(
                    exist_ok=True, parents=True
                )
                (output_dir / "pred" / sequence_category).mkdir(
                    exist_ok=True, parents=True
                )
                (output_dir / "images" / sequence_category).mkdir(
                    exist_ok=True, parents=True
                )

                pcd = batch["sequence_point_cloud"][i]
                if not isinstance(pcd, Pointclouds):
                    pcd = Pointclouds(pcd[None])
                # Save ground truth
                io.save_pointcloud(
                    data=pcd,
                    path=filestr.format(
                        dir="gt",
                        category=sequence_category,
                        name=sequence_name,
                        ext="ply",
                    ),
                )

                # Save generation
                io.save_pointcloud(
                    data=output[i],
                    path=filestr.format(
                        dir="pred",
                        category=sequence_category,
                        name=sequence_name,
                        ext="ply",
                    ),
                )

                # Save input images
                filename = filestr.format(
                    dir="images",
                    category=sequence_category,
                    name=sequence_name,
                    ext="png",
                )
                TVF.to_pil_image(batch["image_rgb"][i]).save(filename)
    print("Saved samples to: ")
    print("{}".format(output_dir.absolute()))


if __name__ == "__main__":
    main()
