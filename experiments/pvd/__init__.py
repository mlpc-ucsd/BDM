import os

import torch
import torch.nn as nn
import torch.utils.data

import argparse

from pvd.utils.file_utils import *
from pvd.utils.visualize import *
from pvd.model.pvcnn_generation import PVCNN2Base_PVD

from tqdm.auto import tqdm

import ipdb


class GaussianDiffusion:
    def __init__(self, betas, loss_type, model_mean_type, model_var_type):
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        assert isinstance(betas, np.ndarray)
        self.np_betas = betas = betas.astype(
            np.float64
        )  # computations here in float64 for accuracy
        assert (betas > 0).all() and (betas <= 1).all()
        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)

        # initialize twice the actual length so we can keep running for eval
        # betas = np.concatenate([betas, np.full_like(betas[:int(0.2*len(betas))], betas[-1])])

        alphas = 1.0 - betas
        alphas_cumprod = torch.from_numpy(np.cumprod(alphas, axis=0)).float()
        alphas_cumprod_prev = torch.from_numpy(
            np.append(1.0, alphas_cumprod[:-1])
        ).float()

        self.betas = torch.from_numpy(betas).float()
        self.alphas_cumprod = alphas_cumprod.float()
        self.alphas_cumprod_prev = alphas_cumprod_prev.float()

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).float()
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - alphas_cumprod).float()
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod).float()
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1).float()

        betas = torch.from_numpy(betas).float()
        alphas = torch.from_numpy(alphas).float()
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = posterior_variance
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(
            torch.max(posterior_variance, 1e-20 * torch.ones_like(posterior_variance))
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    @staticmethod
    def _extract(a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        (bs,) = t.shape
        assert x_shape[0] == bs
        out = torch.gather(a, 0, t)
        assert out.shape == torch.Size([bs])
        return torch.reshape(out, [bs] + ((len(x_shape) - 1) * [1]))

    def q_mean_variance(self, x_start, t):
        mean = (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
        )
        variance = self._extract(
            1.0 - self.alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        log_variance = self._extract(
            self.log_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t == 0 means diffused for 1 step)
        """
        if noise is None:
            noise = torch.randn(x_start.shape, device=x_start.device)
        assert noise.shape == x_start.shape
        return (
            self._extract(self.sqrt_alphas_cumprod.to(x_start.device), t, x_start.shape)
            * x_start
            + self._extract(
                self.sqrt_one_minus_alphas_cumprod.to(x_start.device), t, x_start.shape
            )
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            self._extract(self.posterior_mean_coef1.to(x_start.device), t, x_t.shape)
            * x_start
            + self._extract(self.posterior_mean_coef2.to(x_start.device), t, x_t.shape)
            * x_t
        )
        posterior_variance = self._extract(
            self.posterior_variance.to(x_start.device), t, x_t.shape
        )
        posterior_log_variance_clipped = self._extract(
            self.posterior_log_variance_clipped.to(x_start.device), t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, denoise_fn, data, t, clip_denoised: bool, return_pred_xstart: bool
    ):
        model_output = denoise_fn(data, t)

        if self.model_var_type in ["fixedsmall", "fixedlarge"]:
            # below: only log_variance is used in the KL computations
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so to get a better decoder log likelihood
                "fixedlarge": (
                    self.betas.to(data.device),
                    torch.log(
                        torch.cat([self.posterior_variance[1:2], self.betas[1:]])
                    ).to(data.device),
                ),
                "fixedsmall": (
                    self.posterior_variance.to(data.device),
                    self.posterior_log_variance_clipped.to(data.device),
                ),
            }[self.model_var_type]
            model_variance = self._extract(
                model_variance, t, data.shape
            ) * torch.ones_like(data)
            model_log_variance = self._extract(
                model_log_variance, t, data.shape
            ) * torch.ones_like(data)
        else:
            raise NotImplementedError(self.model_var_type)

        if self.model_mean_type == "eps":
            x_recon = self._predict_xstart_from_eps(data, t=t, eps=model_output)

            if clip_denoised:
                x_recon = torch.clamp(x_recon, -0.5, 0.5)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=x_recon, x_t=data, t=t
            )
        else:
            raise NotImplementedError(self.loss_type)

        assert model_mean.shape == x_recon.shape == data.shape
        assert model_variance.shape == model_log_variance.shape == data.shape
        if return_pred_xstart:
            return model_mean, model_variance, model_log_variance, x_recon
        else:
            return model_mean, model_variance, model_log_variance

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            self._extract(self.sqrt_recip_alphas_cumprod.to(x_t.device), t, x_t.shape)
            * x_t
            - self._extract(
                self.sqrt_recipm1_alphas_cumprod.to(x_t.device), t, x_t.shape
            )
            * eps
        )

    # ------------------ sample ------------------
    def p_sample(
        self,
        denoise_fn,
        data,
        t,
        noise_fn,
        clip_denoised=False,
        return_pred_xstart=False,
        use_var=True,
    ):
        model_mean, _, model_log_variance, pred_xstart = self.p_mean_variance(
            denoise_fn,
            data=data,
            t=t,
            clip_denoised=clip_denoised,
            return_pred_xstart=True,
        )
        noise = noise_fn(size=data.shape, dtype=data.dtype, device=data.device)
        assert noise.shape == data.shape
        # no noise when t == 0
        nonzero_mask = torch.reshape(
            1 - (t == 0).float(), [data.shape[0]] + [1] * (len(data.shape) - 1)
        )

        sample = model_mean
        if use_var:
            sample = sample + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        assert sample.shape == pred_xstart.shape
        return (sample, pred_xstart) if return_pred_xstart else sample

    def p_sample_loop(
        self,
        data,
        denoise_fn,
        shape,
        device,
        noise_fn=torch.randn,
        constrain_fn=lambda x, t: x,
        clip_denoised=True,
        start_time=None,
        final_time=None,
        keep_running=False,
    ):
        """
        Generate samples
        keep_running: True if we run 2 x num_timesteps, False if we just run num_timesteps
        """
        if start_time is None:
            start_time = self.num_timesteps

        if final_time is None:
            final_time = 0

        assert isinstance(shape, (tuple, list))

        img_t = data

        for t in tqdm(
            reversed(
                range(final_time, start_time if not keep_running else len(self.betas))
            )
        ):
            img_t = constrain_fn(img_t, t)
            t_ = torch.empty(shape[0], dtype=torch.int64, device=device).fill_(t)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                clip_denoised=clip_denoised,
                return_pred_xstart=False,
            ).detach()

        assert img_t.shape == shape
        return img_t

    def reconstruct(
        self, x0, t, denoise_fn, noise_fn=torch.randn, constrain_fn=lambda x, t: x
    ):
        assert t >= 1

        t_vec = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(
            t - 1
        )
        encoding = self.q_sample(x0, t_vec)

        img_t = encoding

        for k in reversed(range(0, t)):
            img_t = constrain_fn(img_t, k)
            t_ = torch.empty(x0.shape[0], dtype=torch.int64, device=x0.device).fill_(k)
            img_t = self.p_sample(
                denoise_fn=denoise_fn,
                data=img_t,
                t=t_,
                noise_fn=noise_fn,
                clip_denoised=False,
                return_pred_xstart=False,
                use_var=True,
            ).detach()

        return img_t


class PVCNN2_PVD(PVCNN2Base_PVD):
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
        num_classes,
        embed_dim,
        use_att,
        dropout,
        extra_feature_channels=3,
        width_multiplier=1,
        voxel_resolution_multiplier=1,
    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            use_att=use_att,
            dropout=dropout,
            extra_feature_channels=extra_feature_channels,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
        )


class Model(nn.Module):
    def __init__(
        self, args, betas, loss_type: str, model_mean_type: str, model_var_type: str
    ):
        super(Model, self).__init__()
        self.diffusion = GaussianDiffusion(
            betas, loss_type, model_mean_type, model_var_type
        )
        self.model = PVCNN2_PVD(
            num_classes=args["nc"],
            embed_dim=args["embed_dim"],
            use_att=args["attention"],
            dropout=args["dropout"],
            extra_feature_channels=0,
        )

    def prior_kl(self, x0):
        return self.diffusion._prior_bpd(x0)

    def all_kl(self, x0, clip_denoised=True):
        total_bpd_b, vals_bt, prior_bpd_b, mse_bt = self.diffusion.calc_bpd_loop(
            self._denoise, x0, clip_denoised
        )
        return {
            "total_bpd_b": total_bpd_b,
            "terms_bpd": vals_bt,
            "prior_bpd_b": prior_bpd_b,
            "mse_bt": mse_bt,
        }

    def _denoise(self, data, t):
        B, D, N = data.shape
        assert data.dtype == torch.float
        assert t.shape == torch.Size([B]) and t.dtype == torch.int64

        out = self.model(data, t)

        assert out.shape == torch.Size([B, D, N])
        return out

    def get_loss_iter(self, data, noises=None):
        B, D, N = data.shape
        t = torch.randint(
            0, self.diffusion.num_timesteps, size=(B,), device=data.device
        )

        if noises is not None:
            noises[t != 0] = torch.randn((t != 0).sum(), *noises.shape[1:]).to(noises)

        losses = self.diffusion.p_losses(
            denoise_fn=self._denoise, data_start=data, t=t, noise=noises
        )
        assert losses.shape == t.shape == torch.Size([B])
        return losses

    def gen_samples(
        self,
        data,
        shape,
        device,
        noise_fn=torch.randn,
        constrain_fn=lambda x, t: x,
        clip_denoised=False,
        start_time=None,
        final_time=None,
        keep_running=False,
    ):
        return self.diffusion.p_sample_loop(
            data=data,
            denoise_fn=self._denoise,
            shape=shape,
            device=device,
            noise_fn=noise_fn,
            constrain_fn=constrain_fn,
            clip_denoised=clip_denoised,
            start_time=start_time,
            final_time=final_time,
            keep_running=keep_running,
        )

    def reconstruct(self, x0, t, constrain_fn=lambda x, t: x):
        return self.diffusion.reconstruct(
            x0, t, self._denoise, constrain_fn=constrain_fn
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)


def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == "linear":
        betas = np.linspace(b_start, b_end, time_num)
    elif schedule_type == "warm0.1":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.1)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.2":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.2)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    elif schedule_type == "warm0.5":
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warmup_time = int(time_num * 0.5)
        betas[:warmup_time] = np.linspace(b_start, b_end, warmup_time, dtype=np.float64)
    else:
        raise NotImplementedError(schedule_type)
    return betas


def generate_pvd_xyz(
    model: Model,
    x: torch.Tensor,
    start_time: int,
    final_time: int,
    *args, **kwargs
):
    """
    Input: x: (bsz, 3, num_points)
    Output: gen: (bsz, 3, num_points)
    """
    start_time = int(start_time)
    final_time = int(final_time)

    with torch.no_grad():
        gen = model.gen_samples(
            data=x,
            shape=x.shape,
            device=x.device,
            start_time=start_time,
            final_time=final_time,
        )

        return gen


def prepare_pvd_model(opt, device):
    betas = get_betas("linear", 0.0001, 0.02, 1000)
    model = Model(opt, betas, "mse", "eps", "fixedsmall")

    def _transform_(m):
        return nn.parallel.DataParallel(m)

    model = model.to(device)
    model.multi_gpu_wrapper(_transform_)

    model.eval()

    with torch.no_grad():
        print("PVD model resume path:%s" % opt["model"])
        resumed_param = torch.load(opt["model"])
        try:
            model.load_state_dict(resumed_param["model_state"])
        except:
            model.load_state_dict(resumed_param["prior_model"])

    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", default="/mnt/sphere/datasets/3d/shapenet/ShapeNetCore.v2.PC15k/"
    )
    parser.add_argument("--category", default="bed")

    parser.add_argument("--batch_size", type=int, default=1, help="input batch size")
    parser.add_argument("--workers", type=int, default=8, help="workers")

    parser.add_argument("--nc", default=3)
    parser.add_argument("--npoints", default=4096)

    # models
    parser.add_argument("--beta_start", default=0.0001)
    parser.add_argument("--beta_end", default=0.02)
    parser.add_argument("--schedule_type", default="linear")
    parser.add_argument("--time_num", default=1000)  # 1000

    # params
    parser.add_argument("--attention", default=True)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--loss_type", default="mse")
    parser.add_argument("--model_mean_type", default="eps")
    parser.add_argument("--model_var_type", default="fixedsmall")
    parser.add_argument(
        "--model",
        default="/mnt/sphere/hax027/pvd/ckpts/chair_1799.pth",
        help="path to model (to continue training)",
    )

    # eval
    parser.add_argument("--eval_path", default="")
    parser.add_argument("--manualSeed", default=42, type=int, help="random seed")
    parser.add_argument(
        "--gpu", type=int, default=1, metavar="S", help="gpu id (default: 0)"
    )

    opt = parser.parse_args()

    return opt
