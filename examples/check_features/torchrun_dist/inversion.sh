torchrun --nproc_per_node=1 \
../../../seistorch_dist.py \
configs/forward.yml \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--step-per-epoch 1 \
--mode inversion \
--save-path ./results/towed \
--grad-cut \
--grad-smooth \
--use-cuda