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
from torch.utils import data

from pytorch3d.structures import Pointclouds
from pytorch3d.renderer.cameras import PerspectiveCameras

import open3d as o3d
from PIL import Image
from collections import OrderedDict

from tqdm.auto import tqdm

from config.structured import ShapeNetR2N2Config, DataloaderConfig, ProjectConfig

from .utils import compute_extrinsic_matrix, compute_camera_calibration

R2N2_cate = {
    "02691156": "airplane",
    "02828884": "bench",
    "02933112": "cabinet",
    "02958343": "car",
    "03001627": "chair",
    "03211117": "display",
    "03636649": "lamp",
    "03691459": "loudspeaker",
    "04090263": "rifle",
    "04256520": "sofa",
    "04379243": "table",
    "04401088": "telephone",
    "04530566": "watercraft",
}
R2N2_synsetid = {v: k for k, v in R2N2_cate.items()}

K = torch.tensor(
    [
        [2.1875, 0.0, 0.0, 0.0],
        [0.0, 2.1875, 0.0, 0.0],
        [0.0, 0.0, -1.002002, -0.2002002],
        [0.0, 0.0, -1.0, 0.0],
    ]
)


def transform_v2_to_v1(point_cloud_v2):
    point_cloud = point_cloud_v2.clone()
    point_cloud[:, 0] = -point_cloud_v2[:, 2]
    point_cloud[:, 1] = point_cloud_v2[:, 1]
    point_cloud[:, 2] = -point_cloud_v2[:, 0]

    return point_cloud.float()


def build_camera_from_R2N2(Rs, Ts, mean, std):
    pose = torch.cat([Rs, Ts[None]], dim=0)
    r = torch.Tensor([[0, 0, 0, 1]]).to(pose)
    extrin = torch.cat([pose, r.T], dim=1)
    shapenet_to_pytorch3d = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    ).to(pose)
    RT = extrin @ shapenet_to_pytorch3d
    R = RT[:3, :3].clone()
    camera_R = (R * std).float()
    t = RT[3, :3].clone()
    camera_T = (mean @ R / std + t).float()
    camera_R[:, :2] *= -1
    camera_T[:2] *= -1

    focal = torch.Tensor([K[0, 0], K[1, 1]]).to(pose)
    pp = torch.Tensor([0, 0]).to(pose)
    camera = PerspectiveCameras(
        focal_length=focal[None],
        principal_point=pp[None],
        R=camera_R[None],
        T=camera_T[None],
    )

    return camera


class ShapeNet_R2N2(Dataset):
    """
    this class should fulfill the need of loading the shapenet data, and pack it in the format as FrameData.
    Salient elements include sequence point cloud, camera, img_rgb and fg_probability(which is likely to be set as None)

    ideal process is :
    in the initial funciton
    1) we load the split json and take out the id of the train and test file
    in the build_data function:
    2) we find the point cloud with corresponding id
    3) we find the camera txt and calculate the corresponding R,T,K
    in the get item function:
    1) we should just load the data and return to the dataloader.
    """

    def __init__(
        self,
        root_dir="/mnt/sphere/hax027/ShapeNetCore.v2.PC15k",
        r2n2_dir="/mnt/sphere/hax027/ShapeNet.R2N2",
        pc_dict="pc_dict_v2.json",
        split_file="R2N2_split.json",
        views_rel_path="ShapeNetRendering",
        which_view_from24=["00"],
        categories=["chair"],
        split="train",
        sample_size=4096,
        img_size=224,
        scale_factor=1.0,
        random_subsample=True,
        normalize_per_shape=False,
        box_per_shape=False,
        subset_ratio=1.0,
        start_ratio=0.0,
        input_dim=3,
    ):
        self.root_dir = root_dir
        self.r2n2_dir = r2n2_dir
        self.views_rel_path = views_rel_path
        self.split = split
        if split not in ["train", "test"]:
            raise ValueError("split has to be one of (train, test).")

        # The list stores the category id of args.
        self.cates = categories
        if "all" in categories:
            self.cate_id = list(R2N2_synsetid.values())
        else:
            self.cate_id = [R2N2_synsetid[c] for c in self.cates]

        # Loading the split json.
        with open(os.path.join(r2n2_dir, split_file)) as split:
            self.split_dict = json.load(split)

        # Loading the pc split json to mark the point clouds' location in the train/test/val direction.
        with open(os.path.join(r2n2_dir, pc_dict)) as pc_split:
            self.pc_subdir = json.load(pc_split)

        self.img_size = img_size
        self.scale_factor = scale_factor

        self.view_rel_path = views_rel_path
        # Check if the folder containing R2N2 renderings is included in r2n2_dir.
        assert os.path.isdir(os.path.join(r2n2_dir, views_rel_path))

        self.sample_size = sample_size
        self.which_view_from24 = which_view_from24
        self.scale_factor = scale_factor

        self.normalize_per_shape = normalize_per_shape
        self.box_per_shape = box_per_shape
        self.random_subsample = random_subsample
        self.input_dim = input_dim
        self.subset_ratio = subset_ratio
        self.start_ratio = start_ratio

        # self.build_data_parallel()
        self.build_data()

    def build_data_chunk(self, object_ids_chunk):
        img_path_chunk = []
        img_rgb_chunk = []
        point_clouds_path_chunk = []
        all_point_clouds_chunk = []
        Rs_chunk = []
        Ts_chunk = []

        for object_id in object_ids_chunk:
            if object_id not in self.pc_subdir[self.split][self.cur_cate_id].keys():
                continue
            pc_subdir = self.pc_subdir[self.split][self.cur_cate_id][object_id]
            point_clouds_path = os.path.join(
                self.root_dir, self.cur_cate_id, pc_subdir, object_id + ".npy"
            )
            rendering_path = os.path.join(
                self.r2n2_dir,
                self.views_rel_path,
                self.cur_cate_id,
                object_id,
                "rendering",
            )
            with open(os.path.join(rendering_path, "rendering_metadata.txt"), "r") as f:
                metadata_lines = f.readlines()
            for i in self.which_view_from24:
                img_path, img, pc_path, pc, Rs, Ts = self.load_data(
                    point_clouds_path, rendering_path, metadata_lines, i
                )
                img_path_chunk.append(img_path)
                img_rgb_chunk.append(img)
                point_clouds_path_chunk.append(pc_path)
                all_point_clouds_chunk.append(pc)
                Rs_chunk.append(Rs)
                Ts_chunk.append(Ts)

        return (
            img_path_chunk,
            img_rgb_chunk,
            point_clouds_path_chunk,
            all_point_clouds_chunk,
            Rs_chunk,
            Ts_chunk,
        )

    def build_data_parallel(self):
        num_processes = cpu_count() // 2

        self.MAX_CAMERA_DISTANCE = 1.75
        self.camera = []
        self.img_rgb = []
        self.img_path = []
        self.all_point_clouds = []
        self.point_clouds_path = []
        self.Rs = []
        self.Ts = []

        # NOTE:the structure of these lists only support training on one category;we can modify them afterwards.
        for i, cate_id in enumerate(self.cate_id):
            self.cur_cate_id = cate_id
            if not (cate_id in R2N2_cate.keys()):
                print(
                    f"the category of {R2N2_cate[cate_id]} is not included in 13 categories of R2N2"
                )
            object_id_data = self.split_dict[self.split][cate_id]

            object_ids = list(object_id_data.keys())
            if self.start_ratio == 0.0:
                object_ids = object_ids[: int(len(object_ids) * self.subset_ratio)]
                print(
                    f"Start to load {self.split} data of {R2N2_cate[cate_id]}, {self.subset_ratio} of {len(object_ids)/self.subset_ratio}"
                )
            else:
                object_ids = object_ids[int(len(object_ids) * self.start_ratio): int(len(object_ids) * self.subset_ratio)]
                print(f"Start to load {self.split} data of {R2N2_cate[cate_id]}, {self.subset_ratio - self.start_ratio} of {len(object_ids)/(self.subset_ratio-self.start_ratio)}")

            chunk_size = len(object_ids) // num_processes
            object_id_chunks = [object_ids[i: i + chunk_size]for i in range(0, len(object_ids), chunk_size)]
            print(f"Split the object_ids into {len(object_id_chunks)} chunks")

            print(f"Start to load data in parallel with {num_processes} processes")
            with Pool(processes=num_processes) as pool:
                results = list(
                    tqdm(
                        pool.imap(self.build_data_chunk, object_id_chunks),
                        total=len(object_id_chunks),
                    )
                )

            for result in results:
                self.img_path.extend(result[0])
                self.img_rgb.extend(result[1])
                self.point_clouds_path.extend(result[2])
                self.all_point_clouds.extend(result[3])
                self.Rs.extend(result[4])
                self.Ts.extend(result[5])

        # NOTE:I don't know whether we need to do this step of shuffle...But I believe it has its some worthwhile effect.
        self.shuffle_idx = list(range(len(self.all_point_clouds)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.all_point_clouds = [
            self.all_point_clouds[i][None] for i in self.shuffle_idx
        ]

        self.point_clouds_path = [self.point_clouds_path[i] for i in self.shuffle_idx]
        self.img_rgb = [self.img_rgb[i] for i in self.shuffle_idx]
        self.img_path = [self.img_path[i] for i in self.shuffle_idx]
        self.Rs = [self.Rs[i] for i in self.shuffle_idx]
        self.Ts = [self.Ts[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = torch.cat(self.all_point_clouds, dim=0)
        if self.normalize_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                B, 1, self.input_dim
            )
            self.all_points_std = (
                self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
            )
        else:
            # normalize across the whole dataset
            self.all_points_mean = (
                self.all_points.reshape(-1, self.input_dim)
                .mean(axis=0)
                .reshape(1, 1, self.input_dim)
            )
            self.all_points_std = (
                self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
            )

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        print("Start to load point clouds and build camera...")
        for i in tqdm(range(len(self.all_point_clouds))):
            if self.random_subsample:
                point_cloud = self.all_points[i, :, :]
                point_idxs = np.random.choice(point_cloud.shape[0], self.sample_size)
                self.all_point_clouds[i] = point_cloud[point_idxs, :].float()

            # begin to process the Rs and Ts to create camera
            Rs = self.Rs[i].clone()
            Ts = self.Ts[i].clone()
            if self.normalize_per_shape:
                m = self.all_points_mean[i, 0, :]  # (input_dim)
                s = self.all_points_std[i, 0, :]  # (1)
            else:
                m = self.all_points_mean[0, 0, :]
                s = self.all_points_std[0, 0, :]
            camera = build_camera_from_R2N2(Rs=Rs, Ts=Ts, mean=m, std=s)
            self.camera.append(camera)

        print(
            "Finish creating {} dataset with {} of {} in total".format(
                self.split, len(self.img_path), self.cates
            )
        )

    def load_data(
        self,
        point_clouds_path,
        rendering_path,
        metadata_lines,
        view_number,
    ):
        """
        we should note that:
        we shouldn't normalize the point cloud here.
        our method is to return the Rs,Ts first...
        after we read all the point cloud then we can calculate the mean and std
        we can adjust the Rs\Ts then turn it into camera.
        """
        # Read image
        image_path = os.path.join(rendering_path, view_number + ".png")
        raw_img = Image.open(image_path)
        try:
            R, G, B, A = raw_img.split()
        except:
            R, G, B = raw_img.split()
        raw_img = Image.merge("RGB", (R, G, B)).resize(
            (self.img_size, self.img_size), Image.BILINEAR
        )
        image = (
            (torch.from_numpy(np.array(raw_img) / 255.0)[..., :3])
            .permute(2, 0, 1)
            .float()
        )

        # Load point clouds from Shapenet V2
        try:
            point_cloud_v2 = torch.tensor(np.load(point_clouds_path))
            assert point_cloud_v2.shape[0] == 15000
        except:
            raise FileNotFoundError

        # Tranform the point cloud from v2 to v1
        point_cloud = transform_v2_to_v1(point_cloud_v2)

        # Get camera calibration.
        azim, elev, yaw, dist_ratio, fov = [
            float(v) for v in metadata_lines[int(view_number)].strip().split(" ")
        ]
        dist = dist_ratio * self.MAX_CAMERA_DISTANCE
        RT = compute_extrinsic_matrix(azim, elev, dist)
        Rs, Ts = compute_camera_calibration(RT)

        return image_path, image, point_clouds_path, point_cloud, Rs, Ts

    def build_data(self):
        self.MAX_CAMERA_DISTANCE = 1.75
        self.camera = []
        self.img_rgb = []
        self.img_path = []
        self.all_point_clouds = []
        self.point_clouds_path = []
        self.Rs = []
        self.Ts = []

        # NOTE:the structure of these lists only support training on one category;we can modify them afterwards.
        for i, cate_id in enumerate(self.cate_id):
            if not (cate_id in R2N2_cate.keys()):
                print(
                    f"the category of {R2N2_cate[cate_id]} is not included in 13 categories of R2N2"
                )
            object_id_data = self.split_dict[self.split][cate_id]

            object_ids = list(object_id_data.keys())
            object_ids = object_ids[: int(len(object_ids) * self.subset_ratio)]
            print(
                f"Start to load {self.split} data of {R2N2_cate[cate_id]}, {self.subset_ratio} of {len(object_ids)/self.subset_ratio}"
            )
            for object_id in tqdm(object_ids):
                # Check if the object in the split file is included in the ShapeNetCorev2
                if object_id not in self.pc_subdir[self.split][cate_id].keys():
                    continue
                pc_subdir = self.pc_subdir[self.split][cate_id][
                    object_id
                ]  # e.g. XXXXXXX is in the dir of 'train/test/val' of the ShapeNetCorev2
                point_clouds_path = os.path.join(
                    self.root_dir, cate_id, pc_subdir, object_id + ".npy"
                )

                rendering_path = os.path.join(
                    self.r2n2_dir,
                    self.views_rel_path,
                    cate_id,
                    object_id,
                    "rendering",
                )

                # Read metadata file to obtain params of calibration matrics.
                with open(
                    os.path.join(rendering_path, "rendering_metadata.txt"), "r"
                ) as f:
                    metadata_lines = f.readlines()
                for i in self.which_view_from24:
                    img_path, img, pc_path, pc, Rs, Ts = self.load_data(
                        point_clouds_path,
                        rendering_path,
                        metadata_lines,
                        i,
                    )
                    self.img_path.append(img_path)
                    self.img_rgb.append(img)
                    self.point_clouds_path.append(pc_path)
                    self.all_point_clouds.append(pc)
                    self.Rs.append(Rs)
                    self.Ts.append(Ts)

        # NOTE: I don't know whether we need to do this step of shuffle... But I believe it has its some worthwhile effect.
        self.shuffle_idx = list(range(len(self.all_point_clouds)))
        random.Random(38383).shuffle(self.shuffle_idx)
        self.all_point_clouds = [
            self.all_point_clouds[i][None] for i in self.shuffle_idx
        ]

        self.point_clouds_path = [self.point_clouds_path[i] for i in self.shuffle_idx]
        self.img_rgb = [self.img_rgb[i] for i in self.shuffle_idx]
        self.img_path = [self.img_path[i] for i in self.shuffle_idx]
        self.Rs = [self.Rs[i] for i in self.shuffle_idx]
        self.Ts = [self.Ts[i] for i in self.shuffle_idx]

        # Normalization
        self.all_points = torch.cat(self.all_point_clouds, dim=0)
        if self.normalize_per_shape:
            B, N = self.all_points.shape[:2]
            self.all_points_mean = self.all_points.mean(axis=1).reshape(
                B, 1, self.input_dim
            )
            self.all_points_std = (
                self.all_points.reshape(B, -1).std(axis=1).reshape(B, 1, 1)
            )
        else:
            # normalize across the whole dataset
            self.all_points_mean = (
                self.all_points.reshape(-1, self.input_dim)
                .mean(axis=0)
                .reshape(1, 1, self.input_dim)
            )
            self.all_points_std = (
                self.all_points.reshape(-1).std(axis=0).reshape(1, 1, 1)
            )

        self.all_points = (self.all_points - self.all_points_mean) / self.all_points_std

        print("Start to subsample point clouds and build camera...")
        for i in tqdm(range(len(self.all_point_clouds))):
            if self.random_subsample:
                point_cloud = self.all_points[i, :, :]
                point_idxs = np.random.choice(point_cloud.shape[0], self.sample_size)
                self.all_point_clouds[i] = point_cloud[point_idxs, :].float()

            # begin to process the Rs and Ts to create camera
            Rs = self.Rs[i].clone()
            Ts = self.Ts[i].clone()
            if self.normalize_per_shape:
                m = self.all_points_mean[i, 0, :]  # (input_dim)
                s = self.all_points_std[i, 0, :]  # (1)
            else:
                m = self.all_points_mean[0, 0, :]
                s = self.all_points_std[0, 0, :]
            camera = build_camera_from_R2N2(Rs=Rs, Ts=Ts, mean=m, std=s)
            self.camera.append(camera)

        print(
            "Finish creating {} dataset with {} of {} in total".format(
                self.split, len(self.img_path), self.cates
            )
        )

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        sample = OrderedDict()

        sample["frame_number"] = self.img_path[idx].split("/")[-1].split(".")[0]
        sample["sequence_name"] = (self.img_path[idx].split("/")[-3] + "_" + sample["frame_number"])
        sample["sequence_category"] = R2N2_cate[self.img_path[idx].split("/")[-4]]
        sample["frame_timestamp"] = 0
        sample["image_size_hw"] = torch.tensor(self.img_rgb[idx].shape[1:]).long()
        sample["effective_image_size_hw"] = torch.tensor(self.img_rgb[idx].shape[1:]).long()
        sample["image_path"] = self.img_path[idx]
        sample["image_rgb"] = self.img_rgb[idx]
        sample["mask_crop"] = None
        sample["depth_path"] = None
        sample["depth_map"] = None
        sample["depth_mask"] = None
        sample["mask_path"] = None
        sample["fg_probability"] = None
        sample["bbox_xywh"] = None
        sample["crop_bbox_xywh"] = None
        sample["camera"] = self.camera[idx]
        sample["camera_quality_score"] = None
        sample["point_cloud_quality_score"] = None
        sample["sequence_point_cloud_path"] = self.point_clouds_path[idx]
        sample["sequence_point_cloud"] = self.all_point_clouds[idx]
        sample["sequence_point_cloud_idx"] = 0
        sample["frame_type"] = "real"
        sample["meta"] = {}

        return sample


def get_dataset_shapenet_r2n2(cfg: ProjectConfig):
    dataset_cfg: ShapeNetR2N2Config = cfg.dataset
    dataloader_cfg: DataloaderConfig = cfg.dataloader

    if "sample" in cfg.run.job:
        dataloader_train = None
    else:
        dataset_train = ShapeNet_R2N2(
            root_dir=dataset_cfg.root,
            r2n2_dir=dataset_cfg.r2n2_dir,
            pc_dict=dataset_cfg.pc_dict,
            split_file=dataset_cfg.split_file,
            views_rel_path=dataset_cfg.views_rel_path,
            which_view_from24=[dataset_cfg.which_view_from24],
            categories=[dataset_cfg.category],
            split="train",
            sample_size=dataset_cfg.max_points,
            img_size=dataset_cfg.image_size,
            scale_factor=dataset_cfg.scale_factor,
            subset_ratio=dataset_cfg.subset_ratio,
            start_ratio=dataset_cfg.start_ratio,
            random_subsample=True,  # randomly pick dataset_cfg.max_points points from the point cloud
        )

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=dataloader_cfg.batch_size,
            shuffle=True,
            num_workers=int(dataloader_cfg.num_workers),
            drop_last=True,
            collate_fn=custom_collate,
        )

    dataset_val = ShapeNet_R2N2(
        root_dir=dataset_cfg.root,
        r2n2_dir=dataset_cfg.r2n2_dir,
        pc_dict=dataset_cfg.pc_dict,
        split_file=dataset_cfg.split_file,
        views_rel_path=dataset_cfg.views_rel_path,
        which_view_from24=[dataset_cfg.which_view_from24],
        categories=[dataset_cfg.category],
        sample_size=dataset_cfg.max_points,
        split="test",
        img_size=dataset_cfg.image_size,
        scale_factor=dataset_cfg.scale_factor,
        random_subsample=True,
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
    dataset_train = ShapeNet_R2N2()
