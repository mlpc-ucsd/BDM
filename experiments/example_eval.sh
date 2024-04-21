cd evaluation

export pred_dir="absolute-path-to-the-sampled-pred-dir" # (e.g., ".../sample/pred/chair")
export gt_dir="absolute-path-to-the-sampled-gt-dir" # (e.g., ".../sample/gt/chair")

echo "----------"
echo "sample_chair_pc2_r2n2_0.1"

python evaluation_cd.py \
    --pred_dir ${pred_dir} \
    --gt_dir ${gt_dir} \
    --seed 2003

python evaluation_f1.py \
    --pred_dir ${pred_dir} \
    --gt_dir ${gt_dir} \
    --seed 2003