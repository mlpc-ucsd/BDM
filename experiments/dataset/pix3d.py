import os
import math
import random
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import functools
import json

import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
import pytorch3d.io as p3dio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.renderer import PointsRasterizer, PointsRasterizationSettings

import open3d as o3d
import trimesh
from PIL import Image
from collections import OrderedDict

from tqdm.auto import tqdm

from config.structured import Pix3DConfig, DataloaderConfig, ProjectConfig


class Pix3D(Dataset):
    """
    Pix3D dataset.
    """

    def __init__(
        self,
        root_dir="/mnt/hypercube/datasets/pix3d",
        split="train",
        sample_size=4096,
        img_size=224,
        pc_dict="pix3d.json",
        category="chair",
        subset_ratio=1.0,
        processed=True,
    ):
        json_file = json.load(open(os.path.join(root_dir, pc_dict), "r"))

        # split out the 'category', and random 4:1 split for train/test
        cat_json = [x for x in json_file if x["category"] == category]
        print(f"Found {len(cat_json)} samples for category {category}")
        if split == "train":
            json_file = cat_json[: int(len(cat_json) * 0.8)]
            if subset_ratio != 1.0:
                json_file = json_file[: int(len(json_file) * subset_ratio)]
            print(f"Using {len(json_file)} samples for training")
        elif split == "test":
            json_file = cat_json[int(len(cat_json) * 0.8) :]
            print(f"Using {len(json_file)} samples for testing")
        else:
            raise ValueError("split must be 'train' or 'test'")
        self.data = json_file

        self.root_dir = root_dir
        self.processed = processed
        print(f"Using {'processed' if self.processed else 'raw'} data")
        self.processed_root_dir = root_dir.replace("pix3d", "pix3d_processed")
        self.sample_size = sample_size
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # pts
        if self.processed:
            pointcloud = trimesh.load(
                os.path.join(self.processed_root_dir, sample["model"])
            )
            pts = np.array(pointcloud.vertices)
        else:
            mesh = p3dio.load_objs_as_meshes(
                [os.path.join(self.root_dir, sample["model"])]
            )
            pointcloud = sample_points_from_meshes(mesh, self.sample_size).squeeze()
            pts = np.array(pointcloud)
        m = pts.mean(axis=0)
        s = pts.reshape(1, -1).std(axis=1)
        pts_norm = (pts - m) / s

        # shapenet v2 -> v1
        v2_to_v1 = np.array(
            [
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0],
            ]
        )
        pts_v1 = (v2_to_v1 @ pts_norm.T).T

        # normalization
        R = np.array(sample["rot_mat"])
        t = np.array(sample["trans_mat"])
        R_norm = R * s
        t_norm = t + m @ R.T

        # opencv -> pytorch3d
        convert = np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0],
            ]
        )
        R_v1 = (R_norm @ convert).T
        t_v1 = t_norm

        # img params
        w, h = sample["img_size"]
        x0, y0, x1, y1 = sample["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = max(y1 - y0, x1 - x0) / 2
        new_w = 2 * half_w
        x0, y0, x1, y1 = cx - half_w, cy - half_w, cx + half_w, cy + half_w

        # ---------------------------calculate the intrinsic------------------------------#
        f = sample["focal_length"] * w / 32  # width of sensor is 32mm
        K = np.array(
            [
                [f, 0, w / 2],
                [0, f, h / 2],
                [0, 0, 1],
            ]
        )
        s = self.img_size / (2 * half_w)
        affine = np.array(
            [
                [s, 0, s * (-x0)],
                [0, s, s * (-y0)],
                [0, 0, 1],
            ]
        )
        proj_trans = affine @ K
        # ----------------------------------------------------------------------------------#
        fx, fy = proj_trans[0, 0], proj_trans[1, 1]
        tx, ty = proj_trans[0, 2], proj_trans[1, 2]

        camera = PerspectiveCameras(
            focal_length=torch.as_tensor([fx, fy], dtype=torch.float32)[None],
            principal_point=torch.as_tensor((tx, ty), dtype=torch.float32)[None],
            R=torch.as_tensor(R_v1, dtype=torch.float32)[None],
            T=torch.as_tensor(t_v1, dtype=torch.float32)[None],
            in_ndc=False,
            image_size=torch.as_tensor([self.img_size, self.img_size])[None],
        )

        # img
        if self.processed:
            img_cropped = Image.open(
                os.path.join(self.processed_root_dir, sample["img"])
            )
        else:
            img = Image.open(os.path.join(self.root_dir, sample["img"]))
            img_cropped = img.crop((x0, y0, x1, y1)).resize(
                (self.img_size, self.img_size)
            )
            # transform to RGB
            if img_cropped.mode != "RGB":
                img_cropped = img_cropped.convert("RGB")
        img_cropped_tensor = (
            (torch.from_numpy(np.array(img_cropped) / 255.0)[..., :3])
            .permute(2, 0, 1)
            .float()
        )

        rt = OrderedDict()
        # 'img': 'img/bed/0001.png'
        # 'model': 'model/bed/IKEA_MALM_2/model.obj'
        rt["frame_number"] = sample["img"].split("/")[-1].split(".")[0]
        rt["sequence_name"] = sample["model"].split("/")[-2] + "_" + rt["frame_number"]
        rt["sequence_category"] = sample["category"]
        rt["frame_timestamp"] = 0
        rt["image_size_hw"] = torch.tensor(
            [sample["img_size"][1], sample["img_size"][0]]
        ).long()
        rt["effective_image_size_hw"] = torch.tensor(
            [self.img_size, self.img_size]
        ).long()
        rt["image_path"] = (
            os.path.join(self.root_dir, sample["img"])
            if not self.processed
            else os.path.join(self.processed_root_dir, sample["img"])
        )
        rt["image_rgb"] = img_cropped_tensor
        rt["mask_crop"] = None
        rt["depth_path"] = None
        rt["depth_map"] = None
        rt["depth_mask"] = None
        rt["mask_path"] = None
        rt["fg_probability"] = None
        rt["bbox_xywh"] = None
        rt["crop_bbox_xywh"] = None
        rt["camera"] = camera
        rt["camera_quality_score"] = None
        rt["point_cloud_quality_score"] = None
        rt["sequence_point_cloud_path"] = (
            os.path.join(self.root_dir, sample["model"])
            if not self.processed
            else os.path.join(self.processed_root_dir, sample["model"])
        )
        rt["sequence_point_cloud"] = torch.tensor(pts_v1).float()
        rt["sequence_point_cloud_idx"] = 0
        rt["frame_type"] = "real"
        rt["meta"] = {}

        return rt


def get_dataset_pix3d(cfg: ProjectConfig):
    dataset_cfg: Pix3DConfig = cfg.dataset
    dataloader_cfg: DataloaderConfig = cfg.dataloader

    if "sample" in cfg.run.job:
        dataloader_train = None
    else:
        dataset_train = Pix3D(
            root_dir=dataset_cfg.root,
            pc_dict=dataset_cfg.pc_dict,
            category=dataset_cfg.category,
            split="train",
            sample_size=dataset_cfg.max_points,
            img_size=dataset_cfg.image_size,
            subset_ratio=dataset_cfg.subset_ratio,
            processed=dataset_cfg.processed,
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=dataloader_cfg.batch_size,
            shuffle=True,
            num_workers=int(dataloader_cfg.num_workers),
            drop_last=True,
            collate_fn=custom_collate,
        )

    dataset_val = Pix3D(
        root_dir=dataset_cfg.root,
        pc_dict=dataset_cfg.pc_dict,
        category=dataset_cfg.category,
        split="test",
        sample_size=dataset_cfg.max_points,
        img_size=dataset_cfg.image_size,
        processed=dataset_cfg.processed,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=dataloader_cfg.batch_size,
        shuffle=False,
        num_workers=int(dataloader_cfg.num_workers),
        drop_last=False,
        collate_fn=custom_collate,
    )

    dataloader_vis = dataloader_val

    return dataloader_train, dataloader_val, dataloader_vis


def custom_collate(batch):
    data = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], PerspectiveCameras):
            data[key] = [sample[key] for sample in batch]
        elif batch[0][key] is None:
            data[key] = None
        else:
            data[key] = torch.utils.data.dataloader.default_collate(
                [sample[key] for sample in batch]
            )
    return data


if __name__ == "__main__":
    dataset_train = Pix3D()
