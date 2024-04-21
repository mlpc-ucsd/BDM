from config.structured import ProjectConfig
from .shapenet_r2n2 import get_dataset_shapenet_r2n2
from .pix3d import get_dataset_pix3d


def get_dataset(cfg: ProjectConfig):
    if cfg.dataset.type == "shapenet_r2n2":
        dataloader_train, dataloader_val, dataloader_vis = get_dataset_shapenet_r2n2(cfg)

    elif cfg.dataset.type == "pix3d":
        dataloader_train, dataloader_val, dataloader_vis = get_dataset_pix3d(cfg)

    else:
        raise NotImplementedError(cfg.dataset.type)

    return dataloader_train, dataloader_val, dataloader_vis
