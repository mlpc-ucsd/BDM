from tqdm.auto import tqdm
import os
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import open3d as o3d
import trimesh

import argparse
import logging

from pytorch3d.io import load_ply
from pytorch3d.loss import chamfer_distance as CD
from pytorch3d.ops import iterative_closest_point as ICP

import ipdb

import warnings
warnings.filterwarnings("ignore")


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Process and generate point cloud data."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        help="PointCloud directory",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        help="GroundTruth directory",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for inference"
    )
    args = parser.parse_args()
    return args


def find_ply_files(directory):
    ply_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ply"):
                ply_files.append(os.path.join(root, file))
    ply_files.sort()
    return ply_files


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(args):
    script_name = os.path.basename(__file__)
    script_name_without_extension = os.path.splitext(script_name)[0]
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                "./logs/{}_seed{}_evaluation_{}.log".format(
                    script_name_without_extension,
                    args.seed,
                    time.strftime("%Y-%m-%d--%H-%M-%S"),
                )
            ),
        ],
    )
    logger = logging.getLogger(__name__)
    return logger


def main(args):
    os.makedirs("./logs", exist_ok=True)

    logger = get_logger(args)
    set_seed(args.seed)

    pred_pcd_list = find_ply_files(args.pred_dir)
    logger.info("Evaluating on {} pointclouds".format(len(pred_pcd_list)))

    error_list = []
    cd_list = []
    for pred_pcd_path in tqdm(pred_pcd_list):
        logger.debug("Processing {}".format(pred_pcd_path))

        file_name = pred_pcd_path.split("/")[-1]
        gt_pcd_path = os.path.join(args.gt_dir, file_name)
        if not os.path.exists(gt_pcd_path):
            logger.debug("GT file not found: {}".format(gt_pcd_path))
            error_list.append(pred_pcd_path)
            continue

        gt_pcd = o3d.io.read_point_cloud(gt_pcd_path)
        gt_pcd_tensor = (
            torch.tensor(np.array(gt_pcd.points)).unsqueeze(0).to(args.device)
        )
        gt_pcd_tensor = gt_pcd_tensor - gt_pcd_tensor.mean(dim=1, keepdim=True)

        pred_pcd = o3d.io.read_point_cloud(pred_pcd_path)
        pred_pcd_tensor = (
            torch.tensor(np.array(pred_pcd.points)
                         ).unsqueeze(0).to(args.device)
        )
        pred_pcd_tensor = pred_pcd_tensor - \
            pred_pcd_tensor.mean(dim=1, keepdim=True)

        chamfer_distance_numpy = CD(pred_pcd_tensor, gt_pcd_tensor)[
            0].cpu().numpy()
        if np.isnan(chamfer_distance_numpy):
            error_list.append(pred_pcd_path)
            continue
        else:
            chamfer_distance = float(chamfer_distance_numpy) * 1000
            cd_list.append(chamfer_distance)
        logger.debug(
            "CD: {} e-3, Mean CD: {} e-3".format(
                chamfer_distance, np.mean(cd_list))
        )

    logger.info("Mean CD: {} e-3".format(np.mean(cd_list)))
    logger.info("Error list: {}".format(error_list))


if __name__ == "__main__":
    args = arg_parser()
    main(args)
