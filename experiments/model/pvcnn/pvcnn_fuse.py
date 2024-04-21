import numpy as np
import torch
import torch.nn as nn

from model.pvcnn.modules import Attention
from model.pvcnn.pvcnn_utils import (
    create_mlp_components,
    create_pointnet2_sa_components,
    create_pointnet2_fp_modules,
)
from model.pvcnn.pvcnn_utils import get_timestep_embedding


class PVCNNBase_fuse(nn.Module):
    def __init__(
        self,
        pvd_model: torch.nn.Module,
        pc2_model: torch.nn.Module,
        num_classes: int,
        embed_dim: int,
        use_att: bool = True,
        dropout: float = 0.1,
        extra_feature_channels: int = 3,
        width_multiplier: int = 1,
        voxel_resolution_multiplier: int = 1,
    ):
        super().__init__()
        assert extra_feature_channels >= 0
        # NOTE:load the encoder part of model from pvd_model and pc2_model, and also load their weights
        self.pvd_model_sa_layers = pvd_model.model.module.sa_layers
        self.pvd_model_global_att = pvd_model.model.module.global_att
        self.pc2_model_sa_layers = pc2_model.point_cloud_model.model.sa_layers
        self.pc2_model_global_att = pc2_model.point_cloud_model.model.global_att
        self.pc2_model_fp_layers = pc2_model.point_cloud_model.model.fp_layers
        self.pc2_model_classiifier = pc2_model.point_cloud_model.model.classifier
        self.pc2_model_embedf = pc2_model.point_cloud_model.model.embedf

        # ---------------------------------------To Create Our Fusion Decoder-----------------------------------#
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.width_multiplier = width_multiplier

        self.in_channels = extra_feature_channels + 3

        # Create PointNet-2 model
        (
            sa_layers,
            sa_in_channels,
            channels_sa_features,
            _,
        ) = create_pointnet2_sa_components(
            sa_blocks=self.sa_blocks,
            extra_feature_channels=extra_feature_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        # self.sa_layers = nn.ModuleList(sa_layers)

        # # Additional global attention module
        # self.global_att = None if not use_att else Attention(channels_sa_features, 8, D=1)

        # Only use extra features in the last fp module
        sa_in_channels[0] = extra_feature_channels
        fp_layers, channels_fp_features = create_pointnet2_fp_modules(
            fp_blocks=self.fp_blocks,
            in_channels=channels_sa_features,
            sa_in_channels=sa_in_channels,
            with_se=True,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
        self.fusion_decoder_fp_layers = nn.ModuleList(fp_layers)

        # NOTE: we also should notice that we should load the parameters of these fp layers !!!
        # Create MLP layers
        self.channels_fp_features = channels_fp_features
        layers, _ = create_mlp_components(
            in_channels=channels_fp_features,
            out_channels=[128, dropout, num_classes],  # was 0.5
            classifier=True,
            dim=2,
            width_multiplier=width_multiplier,
        )
        self.classifier = nn.Sequential(*layers)

        # Time embedding function
        self.embedf = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
        self.embedf.load_state_dict(self.pc2_model_embedf.state_dict())
        # breakpoint()
        # NOTE: the initialization of decoder and classifier. RIGHT ??
        self.fusion_decoder_fp_layers.load_state_dict(
            self.pc2_model_fp_layers.state_dict()
        )
        self.classifier.load_state_dict(self.pc2_model_classiifier.state_dict())
        # breakpoint()
        # prior in_features_list: [(16, 64, 1024), (16, 128, 256), (16, 256, 64)]
        # prior middle_feature: (16, 512, 16)
        # NOTE: fuse these features by applying a 1x1 conv layer and add them together via zero_conv
        # HACK: hard code dim here
        projs = []
        for dim in [64, 128, 256, 512]:
            conv1 = nn.Conv1d(dim, dim, 1)
            act1 = nn.LeakyReLU(0.02, inplace=True)
            conv2 = nn.Conv1d(dim, dim, 1)
            zero_conv = nn.Conv1d(dim, dim, 1)
            for p in [conv1, conv2]:
                nn.init.normal_(p.weight, mean=0.0, std=np.sqrt(2 / dim))
                nn.init.constant_(p.bias, 0)
            for p in zero_conv.parameters():
                p.detach().zero_()  # NOTE: why detach here ???
            projs.append(nn.Sequential(conv1, act1, conv2, zero_conv))
        self.projs = nn.ModuleList(projs)

    def forward(
        self,
        recon_inputs_with_cond: torch.Tensor,
        input_from_prior:torch.Tensor,
        t: torch.Tensor,
        mode: str = 'fusion_nstep',
    ):
        """
        The inputs have size (B, 3 + S, N), where S is the number of additional
        feature channels and N is the number of points. The timesteps t can be either
        continuous or discrete. This model has a sort of U-Net-like structure I think,
        which is why it first goes down and then up in terms of resolution (?)
        """

        # Embed timesteps
        t_emb = get_timestep_embedding(self.embed_dim, t, recon_inputs_with_cond.device).float()
        t_emb = self.embedf(t_emb)[:, :, None].expand(
            -1, -1, recon_inputs_with_cond.shape[-1]
        )  # (16, 64, 4096)

        # Separate input coordinates and features
        coords_pc2 = recon_inputs_with_cond[:, :3, :].contiguous()  # (B, 3, N)
        features_pc2 = recon_inputs_with_cond  # (B, 390, N)

        if mode == 'fusion_nstep':
            coords_pvd = input_from_prior.clone()  # (B, 3, N)
            features_pvd = coords_pvd.clone()  # (B, 3, N)
        else:
            coords_pvd = coords_pc2.clone()  # (B, 3, N)
            features_pvd = coords_pvd.clone()  # (B, 3, N)

        # Downscaling layers
        coords_pc2_list = []
        in_pc2_features_list = []
        for i, sa_blocks in enumerate(self.pc2_model_sa_layers):
            in_pc2_features_list.append(features_pc2)
            coords_pc2_list.append(coords_pc2)
            if i == 0:
                features_pc2, coords_pc2, t_emb = sa_blocks(
                    (features_pc2, coords_pc2, t_emb)
                )
            else:
                features_pc2, coords_pc2, t_emb = sa_blocks(
                    (torch.cat([features_pc2, t_emb], dim=1), coords_pc2, t_emb)
                )
        # Replace the input features
        in_pc2_features_list[0] = recon_inputs_with_cond[:, 3:, :].contiguous()

        # Get the global attention of the output of pc2
        if self.pc2_model_global_att is not None:
            features_pc2 = self.pc2_model_global_att(features_pc2)

        # Downscaling layers
        coords_pvd_list = []
        in_pvd_features_list = []
        for i, sa_blocks in enumerate(self.pvd_model_sa_layers):
            in_pvd_features_list.append(features_pvd)
            coords_pvd_list.append(coords_pvd)
            if i == 0:
                features_pvd, coords_pvd, t_emb = sa_blocks(
                    (features_pvd, coords_pvd, t_emb)
                )
            else:
                features_pvd, coords_pvd, t_emb = sa_blocks(
                    (torch.cat([features_pvd, t_emb], dim=1), coords_pvd, t_emb)
                )
        # Replace the input features
        in_pvd_features_list[0] = input_from_prior[:, 3:, :].contiguous()

        # Get the global attention of the output of pvd
        if self.pvd_model_global_att is not None:
            features_pvd = self.pvd_model_global_att(features_pvd)
        # So what is the shape of feature after the global attention module???
        # (16, 64, 1024), (16, 3, 1024), (16, 64, 1024)
        # (16, 128, 256), (16, 3, 256), (16, 64, 256)
        # (16, 256, 64), (16, 3, 64), (16, 64, 64)
        # (16, 512, 16), (16, 3, 16), (16, 64, 16)

        features = self.projs[-1](features_pvd) + features_pc2

        fused_in_features_list = []
        fused_in_features_list.append(in_pc2_features_list[0])
        # prior in_features_list: [(16, 64, 1024), (16, 128, 256), (16, 256, 64)]
        for i, (in_pc2_features, in_pvd_feature, proj) in enumerate(
            zip(in_pc2_features_list[1:], in_pvd_features_list[1:], self.projs)
        ):
            fused_in_features = proj(in_pvd_feature) + in_pc2_features
            fused_in_features_list.append(fused_in_features)

        # Upscaling layers
        for fp_idx, fp_blocks in enumerate(self.fusion_decoder_fp_layers):
            features, coords_pc2, t_emb = fp_blocks(
                (  # this is a tuple because of nn.Sequential
                    coords_pc2_list[-1 - fp_idx],  # reverse coords list from above
                    coords_pc2,  # original point coordinates
                    torch.cat(
                        [features, t_emb], dim=1
                    ),  # keep concatenating upsampled features with timesteps
                    fused_in_features_list[
                        -1 - fp_idx
                    ],  # reverse features list from above
                    t_emb,  # original timestep embedding
                )
            )
        # (16, 256, 64), (16, 3, 64), (16, 64, 64)
        # (16, 256, 256), (16, 3, 256), (16, 64, 256)
        # (16, 128, 1024), (16, 3, 1024), (16, 64, 1024)
        # (16, 64, 4096), (16, 3, 4096), (16, 64, 4096)

        # Output MLP layers
        output = self.classifier(features)

        return output


class PVCNN_fuse(PVCNNBase_fuse):
    sa_blocks = [
        ((32, 2, 32), (1024, 0.1, 32, (32, 64))),
        ((64, 3, 16), (256, 0.2, 32, (64, 128))),
        ((128, 3, 8), (64, 0.4, 32, (128, 256))),
        (None, (16, 0.8, 32, (256, 256, 512))),
    ]
    fp_blocks = [
        ((256, 256), (256, 3, 8)),
        ((256, 256), (256, 3, 8)),
        ((256, 128), (128, 2, 16)),
        ((128, 128, 64), (64, 2, 32)),
    ]

    def __init__(
        self,
        pvd_model,
        pc2_model,
        num_classes,
        embed_dim,
        use_att=True,
        dropout=0.1,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__(
            pvd_model=pvd_model,
            pc2_model=pc2_model,
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )
