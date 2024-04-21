import os
import json

import numpy as np
from torch.utils.data import Dataset

from pytorch3d.ops import sample_points_from_meshes
import pytorch3d.io as p3dio
import trimesh
from PIL import Image

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings("ignore")

class Pix3D(Dataset):
    """
    Pix3D dataset.
    """
    def __init__(
        self,
        root_dir="./pix3d",
        split="train",
        sample_size=4096,
        img_size=224,
        pc_dict="pix3d.json",
        category="chair",
        subset_ratio=1.0,
    ):
        json_file = json.load(open(os.path.join(root_dir, pc_dict), "r"))

        # split out the 'category', and random 4:1 split for train/test
        cat_json = [x for x in json_file if x['category'] == category]
        print(f"Found {len(cat_json)} samples for category {category}")
        if split == "train":
            json_file = cat_json[:int(len(cat_json)*0.8)]
            if subset_ratio != 1.0:
                json_file = json_file[:int(len(json_file)*subset_ratio)]
            print(f"Using {len(json_file)} samples for training")
        elif split == "test":
            json_file = cat_json[int(len(cat_json)*0.8):]
            print(f"Using {len(json_file)} samples for testing")
        elif split == "all":
            json_file = cat_json
            print(f"Using {len(json_file)} samples for all")
        else:
            raise ValueError("split must be 'train' or 'test'")
        self.data = json_file

        self.root_dir = root_dir
        self.sample_size = sample_size
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # pts
        mesh = p3dio.load_objs_as_meshes([os.path.join(self.root_dir, sample['model'])])
        pointcloud = sample_points_from_meshes(mesh, self.sample_size).squeeze()
        pts = np.array(pointcloud)
        pts_v2_obj = trimesh.Trimesh(vertices=pts)

        # img
        img = Image.open(os.path.join(self.root_dir, sample['img']))
        x0, y0, x1, y1 = sample['bbox']
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        half_w = max(y1 - y0, x1 - x0) / 2
        x0, y0, x1, y1 = cx - half_w, cy - half_w, cx + half_w, cy + half_w
        img_cropped = img.crop((x0, y0, x1, y1)).resize((self.img_size, self.img_size))
        
        # pts path
        raw_pts_path = os.path.join(self.root_dir, sample['model'])
        new_pts_path = raw_pts_path.replace("pix3d", "pix3d_processed")
        new_pts_dir = os.path.dirname(new_pts_path)
        raw_img_path = os.path.join(self.root_dir, sample['img'])
        new_img_path = raw_img_path.replace("pix3d", "pix3d_processed")
        new_img_dir = os.path.dirname(new_img_path)
        if not os.path.exists(new_pts_dir):
            os.makedirs(new_pts_dir)
        if not os.path.exists(new_img_dir):
            os.makedirs(new_img_dir)
        # export obj
        pts_v2_obj.export(new_pts_path)
        # export img as JPEG, and take care of non-RGB images
        if img_cropped.mode != "RGB":
            img_cropped = img_cropped.convert("RGB")
        img_cropped.save(new_img_path)


if __name__ == "__main__":

    dataset = Pix3D(split="all", category="chair")
    for i in tqdm(range(len(dataset))):
        dataset[i]

    dataset = Pix3D(split="all", category="sofa")
    for i in tqdm(range(len(dataset))):
        dataset[i]

    dataset = Pix3D(split="all", category="table")
    for i in tqdm(range(len(dataset))):
        dataset[i]