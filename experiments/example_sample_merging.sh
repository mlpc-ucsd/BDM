export wandb_name="bdm"
export save_dir="./outputs"

export root="absolute-path-to-your-ShapeNetCore.v2.PC15k"
export r2n2_dir="absolute-path-to-your-ShapeNet.R2N2"

export category="chair"
export subset_ratio=0.1
export save_name="bdm-merging_sample_chair_pc2_r2n2_0.1"

export prior_ckpt="path-to-the-pvd-checkpoint-of-chair"
export recon_ckpt="path-to-the-pc2-checkpoint-of-0.1chair"
export fusion_ckpt="path-to-the-bdm-merging-checkpoint-of-0.1chair"

# bayesian denoising steps
export roll_step=16

# milestones for interaction
export milestones="[1000,968,936,872,128,64,32,0]"

python main_merging.py \
    logging.wandb_project=${wandb_name} \
    run.job=sample_bdm_merging \
    run.save_dir=${save_dir} \
    run.num_inference_steps=1000 \
    run.diffusion_scheduler=ddpm \
    run.name=${save_name} \
    run.checkpoint_freq=4000 \
    run.val_freq=4000 \
    run.vis_freq=4000 \
    run.max_fusion_steps=20000 \
    scheduler="fusion" \
    dataset=shapenet_r2n2 \
    dataset.root=${root} \
    dataset.r2n2_dir=${r2n2_dir} \
    dataset.image_size=224 \
    dataset.category=${category} \
    dataset.max_points=4096 \
    dataset.subset_ratio=${subset_ratio} \
    dataloader.batch_size=16 \
    dataloader.num_workers=8 \
    aux_run.roll_step=${roll_step} \
    aux_run.milestones=${milestones} \
    aux_run.prior_ckpt=${prior_ckpt} \
    aux_run.recon_ckpt=${recon_ckpt}