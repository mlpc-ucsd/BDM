export wandb_name="bdm"
export save_dir="./outputs"

export root="absolute-path-to-your-ShapeNetCore.v2.PC15k"
export r2n2_dir="absolute-path-to-your-ShapeNet.R2N2"

export category="chair"
# we maintain the same epoch number for different subset_ratio, 
# so subset_ratio:max_steps should be 
# 0.1:10000, 0.5:50000 and 1.0:100000. 
export subset_ratio=0.1
export max_steps=10000
export save_name="train_chair_pc2_r2n2_0.1"

python main.py \
    logging.wandb_project=${wandb_name} \
    run.job=train \
    run.save_dir=${save_dir} \
    run.num_inference_steps=1000 \
    run.diffusion_scheduler=ddpm \
    run.name=${save_name} \
    run.checkpoint_freq=5000 \
    run.val_freq=5000 \
    run.vis_freq=5000 \
    dataset.subset_ratio=${subset_ratio} \
    run.max_steps=${max_steps} \
    dataset=shapenet_r2n2 \
    dataset.root=${root} \
    dataset.r2n2_dir=${r2n2_dir} \
    dataset.image_size=224 \
    dataset.category=${category} \
    dataset.max_points=4096 \
    dataloader.batch_size=16 \
    dataloader.num_workers=8