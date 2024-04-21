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


def compute_pc_to_pc_dist(src, tgt):
    # the source and predicted pc both should be like (N, 3)
    dist = -2 * (src @ tgt.transpose(1, 0))
    dist += torch.sum(src**2, dim=-1)[:, None]
    dist += torch.sum(tgt**2, dim=-1)[None, :]
    # shape (N, M)
    dist = torch.clamp(dist, min=1e-12, max=None)
    dist = (torch.min(dist, dim=1, keepdim=True)[0]).squeeze().tolist()
    return dist


def cal_fscore(gt, pred, thr: float = 0.01):
    d1 = compute_pc_to_pc_dist(gt, pred)
    d2 = compute_pc_to_pc_dist(pred, gt)

    assert len(d1) and len(d2), "Check the point clouds!"
    precision = float(sum(d < thr for d in d1)) / float(len(d1))
    recall = float(sum(d < thr for d in d2)) / float(len(d2))

    fscore = 2 * recall * precision / (recall + precision + 1e-12)
    return fscore


def main(args):
    os.makedirs("./logs", exist_ok=True)

    logger = get_logger(args)
    set_seed(args.seed)

    pred_pcd_list = find_ply_files(args.pred_dir)
    logger.info("Evaluating on {} pointclouds".format(len(pred_pcd_list)))

    error_list = []
    fscore_list = []
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
        gt_pcd_tensor = gt_pcd_tensor - gt_pcd_tensor.mean(1, keepdim=True)

        pred_pcd = o3d.io.read_point_cloud(pred_pcd_path)
        pred_pcd_tensor = (
            torch.tensor(np.array(pred_pcd.points)
                         ).unsqueeze(0).to(args.device)
        )
        pred_pcd_tensor = pred_pcd_tensor - \
            pred_pcd_tensor.mean(1, keepdim=True)

        pred_pcd_t = pred_pcd_tensor.squeeze(0)
        gt_pcd = gt_pcd_tensor.squeeze(0)
        fscore = cal_fscore(gt_pcd, pred_pcd_t)
        fscore_list.append(fscore)
        logger.debug(
            "F-Score: {}, Mean F-Score: {}".format(
                fscore, np.mean(fscore_list))
        )

    logger.info("Mean F-Score: {}".format(np.mean(fscore_list)))
    logger.info("Error list: {}".format(error_list))


if __name__ == "__main__":
    args = arg_parser()
    main(args)
