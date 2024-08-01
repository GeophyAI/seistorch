torchrun --nproc_per_node=1 \
../../../../seistorch_dist.py configs/inversion.yml \
--opt adam \
--loss vp=envelope \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/envelope \
--use-cuda