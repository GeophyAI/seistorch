torchrun --nproc_per_node=1 \
../../../../seistorch_dist.py configs/inversion.yml \
--opt adam \
--loss vp=l2 \
--lr vp=10.0 \
--mode inversion \
--save-path ./results/l2 \
--filteratfirst \
--use-cuda