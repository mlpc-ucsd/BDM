from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import math

import torch


def show_item(item: Dict):
    for key in item.keys():
        value = item[key]
        if torch.is_tensor(value) and value.numel() < 5:
            value_str = value
        elif torch.is_tensor(value):
            value_str = value.shape
        elif isinstance(value, str):
            value_str = ("..." + value[-52:]) if len(value) > 50 else value
        elif isinstance(value, dict):
            value_str = str({k: type(v) for k, v in value.items()})
        else:
            value_str = type(value)
        print(f"{key:<30} {value_str}")


def normalize_to_zero_one(x: torch.Tensor):
    return (x - x.min()) / (x.max() - x.min())


def default(x, d):
    return d if x is None else x


@dataclass
class DatasetMap:
    train: Optional[Iterable] = None
    val: Optional[Iterable] = None
    test: Optional[Iterable] = None


def compute_extrinsic_matrix(
    azimuth: float, elevation: float, distance: float
):  # pragma: no cover
    """
    Copied from meshrcnn codebase:
    https://github.com/facebookresearch/meshrcnn/blob/main/shapenet/utils/coords.py#L96

    Compute 4x4 extrinsic matrix that converts from homogeneous world coordinates
    to homogeneous camera coordinates. We assume that the camera is looking at the
    origin.
    Used in R2N2 Dataset when computing calibration matrices.

    Args:
        azimuth: Rotation about the z-axis, in degrees.
        elevation: Rotation above the xy-plane, in degrees.
        distance: Distance from the origin.

    Returns:
        FloatTensor of shape (4, 4).
    """
    azimuth, elevation, distance = float(azimuth), float(elevation), float(distance)

    az_rad = -math.pi * azimuth / 180.0
    el_rad = -math.pi * elevation / 180.0
    sa = math.sin(az_rad)
    ca = math.cos(az_rad)
    se = math.sin(el_rad)
    ce = math.cos(el_rad)
    R_world2obj = torch.tensor(
        [[ca * ce, sa * ce, -se], [-sa, ca, 0], [ca * se, sa * se, ce]]
    )
    R_obj2cam = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    R_world2cam = R_obj2cam.mm(R_world2obj)
    cam_location = torch.tensor([[distance, 0, 0]]).t()
    T_world2cam = -(R_obj2cam.mm(cam_location))
    RT = torch.cat([R_world2cam, T_world2cam], dim=1)
    RT = torch.cat([RT, torch.tensor([[0.0, 0, 0, 1]])])

    # Georgia: For some reason I cannot fathom, when Blender loads a .obj file it
    # rotates the model 90 degrees about the x axis. To compensate for this quirk we
    # roll that rotation into the extrinsic matrix here
    rot = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    RT = RT.mm(rot.to(RT))

    return RT


def compute_camera_calibration(RT):
    """
    Helper function for calculating rotation and translation matrices from ShapeNet
    to camera transformation and ShapeNet to PyTorch3D transformation.

    Args:
        RT: Extrinsic matrix that performs ShapeNet world view to camera view
            transformation.

    Returns:
        R: Rotation matrix of shape (3, 3).
        T: Translation matrix of shape (3).
    """
    # Transform the mesh vertices from shapenet world to pytorch3d world.
    shapenet_to_pytorch3d = torch.tensor(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    RT = torch.transpose(RT, 0, 1).mm(shapenet_to_pytorch3d)  # (4, 4)
    # Extract rotation and translation matrices from RT.
    R = RT[:3, :3]
    T = RT[3, :3]
    return R, T
