from config.structured import ProjectConfig
from .model import ConditionalPointCloudDiffusionModel, PointCloudFusionModel
from .model_coloring import PointCloudColoringModel
from .model_utils import set_requires_grad


def get_model(cfg: ProjectConfig):
    model = ConditionalPointCloudDiffusionModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model


def get_coloring_model(cfg: ProjectConfig):
    model = PointCloudColoringModel(**cfg.model)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model


def get_fusion_model(cfg: ProjectConfig, pvd_model, pc2_model):
    """
    1)in this PointCloudFusionModel, we need to fulfill its function of training and sampling
    2)this model is similar with the ConditionalPointCloudDiffusionModel
    """
    model = PointCloudFusionModel(pvd_model, pc2_model, **cfg.model)
    set_requires_grad(model.fusion_model.model.pvd_model_sa_layers, False)
    set_requires_grad(model.fusion_model.model.pvd_model_global_att, False)
    set_requires_grad(model.fusion_model.model.pc2_model_sa_layers, False)
    set_requires_grad(model.fusion_model.model.pc2_model_fp_layers, False)
    set_requires_grad(model.fusion_model.model.pc2_model_global_att, False)
    set_requires_grad(model.fusion_model.model.pc2_model_classiifier, False)
    set_requires_grad(model.fusion_model.model.pc2_model_embedf, False)
    if cfg.run.freeze_feature_model:
        set_requires_grad(model.feature_model, False)
    return model
