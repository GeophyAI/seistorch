torchrun --nproc_per_node=1 \
../../../../seistorch_dist.py configs/inversion.yml \
--opt adam \
--loss vp=cs \
--lr vp=20.0 \
--mode inversion \
--save-path ./results/towed_cs \
--filteratfirst \
--grad-cut \
--use-cuda