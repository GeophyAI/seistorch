torchrun --nproc_per_node=1 \
../../../../seistorch_dist.py forward.yml \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/towed_torch \
--grad-cut \
--use-cuda